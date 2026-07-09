import streamlit as st
from streamlit_drawable_canvas import st_canvas
import re
import pandas as pd
import csv
import cv2
import matplotlib.pyplot as plt
import json
import datetime
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image, ImageOps, ImageDraw, ImageFont
from core.models import SmallCharCNN
from core.locks import training_lock

from pathlib import Path

from core.config import (
    DEFAULT_ALPHABET,
    MIN_EXAMPLES_PER_CLASS,
    RECOMMENDED_EXAMPLES_PER_CLASS,
    LOCK_FILE,
)
from core.storage import (
    create_project,
    list_projects,
    load_project_config,
    save_project_config,
    save_uploaded_files,
)
from core.locks import is_locked, read_lock
from core.utils import load_first_page_image, crop_relative, draw_roi_overlay, split_fixed_slots, crop_box_pixels, draw_char_boxes_overlay, resize_for_zoom, trim_to_ink_bbox, slot_ink_ratio
from core.vlm import get_char_boxes_from_vlm

st.set_page_config(
    page_title="EVO Skill Builder",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 EVO Skill Builder")
st.caption("Crea skills de lectura visual de camps documentals. MVP controlat, anti-Gowex.")

with st.sidebar:
    st.header("Navegació")
    page = st.radio(
        "Pantalla",
        [
            "00 · Estat",
            "01 · Crear projecte",
            "02 · Upload documents",
            "03 · ROI bàsica",
            "04 · Dataset / labeling",
            "05 · Training",
            "06 · Test skill",
            "07 · Run batch",
        ],
    )

    st.markdown("---")
    st.subheader("Training lock")
    if is_locked():
        st.error("Entrenament en curs")
        st.code(read_lock())
    else:
        st.success("GPU lliure")

projects = list_projects()

def select_project():
    projects = list_projects()
    if not projects:
        st.warning("Encara no hi ha projectes.")
        return None, None

    labels = [p.name for p in projects]
    selected = st.selectbox("Projecte", labels)
    pdir = next(p for p in projects if p.name == selected)
    return pdir, load_project_config(pdir)



# ---------- Training/Test helpers Anti-Gowex ----------

def normalize_char_pil(img: Image.Image) -> torch.Tensor:
    """
    Normalització consistent per training i inferència.
    Crop RGB/gray -> tensor 1x32x32 amb tinta alta sobre fons baix.
    """
    img = img.convert("L")
    img = ImageOps.autocontrast(img)

    w, h = img.size
    if w <= 0 or h <= 0:
        img = Image.new("L", (32, 32), 255)
    else:
        pad = max(2, int(max(w, h) * 0.20))
        canvas = Image.new("L", (w + 2 * pad, h + 2 * pad), 255)
        canvas.paste(img, (pad, pad))

        w2, h2 = canvas.size
        side = max(w2, h2)
        square = Image.new("L", (side, side), 255)
        square.paste(canvas, ((side - w2) // 2, (side - h2) // 2))
        img = square.resize((32, 32), Image.Resampling.LANCZOS)

    arr = np.asarray(img).astype("float32") / 255.0
    arr = 1.0 - arr  # fons blanc -> 0, tinta -> alt
    return torch.from_numpy(arr).unsqueeze(0)


class CharSamplesDataset(Dataset):
    def __init__(self, samples_dir: Path, alphabet: str):
        self.samples_dir = Path(samples_dir)
        self.alphabet = alphabet
        self.char_to_idx = {ch: i for i, ch in enumerate(alphabet)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.items = []

        for ch in alphabet:
            cdir = self.samples_dir / ch
            if not cdir.exists():
                continue
            for p in sorted(cdir.glob("*.png")):
                self.items.append((p, self.char_to_idx[ch], ch))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label_idx, ch = self.items[idx]
        img = Image.open(path).convert("RGB")
        x = normalize_char_pil(img)
        y = torch.tensor(label_idx, dtype=torch.long)
        return x, y, str(path), ch


def collect_sample_counts(samples_dir: Path, alphabet: str) -> dict:
    out = {}
    for ch in alphabet:
        cdir = samples_dir / ch
        out[ch] = len(list(cdir.glob("*.png"))) if cdir.exists() else 0
    return out


def train_char_cnn_for_project(pdir: Path, config: dict, epochs: int = 15, batch_size: int = 16, lr: float = 1e-3) -> dict:
    """
    Entrena CNN petita amb les mostres actuals.
    Permet dataset incomplet per validar pipeline.
    """
    samples_dir = pdir / "samples"
    models_dir = pdir / "models"
    reports_dir = pdir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    alphabet = str(config.get("alphabet", DEFAULT_ALPHABET))
    dataset = CharSamplesDataset(samples_dir, alphabet)

    if len(dataset) < 2:
        raise RuntimeError("No hi ha prou mostres per entrenar. Necessito almenys 2 crops.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCharCNN(num_classes=len(alphabet)).to(device)

    # Split simple per demo
    if len(dataset) >= 5:
        val_size = max(1, int(len(dataset) * 0.20))
    else:
        val_size = 1

    train_size = len(dataset) - val_size
    if train_size < 1:
        train_size = 1
        val_size = len(dataset) - 1

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=min(batch_size, max(1, train_size)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, max(1, val_size)), shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_ok = 0
        train_total = 0

        for x, y, _, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item()) * x.size(0)
            pred = logits.argmax(dim=1)
            train_ok += int((pred == y).sum().item())
            train_total += int(y.numel())

        model.eval()
        val_loss = 0.0
        val_ok = 0
        val_total = 0

        with torch.no_grad():
            for x, y, _, _ in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits, y)

                val_loss += float(loss.item()) * x.size(0)
                pred = logits.argmax(dim=1)
                val_ok += int((pred == y).sum().item())
                val_total += int(y.numel())

        history.append({
            "epoch": epoch,
            "train_loss": train_loss / max(1, train_total),
            "train_acc": train_ok / max(1, train_total),
            "val_loss": val_loss / max(1, val_total),
            "val_acc": val_ok / max(1, val_total),
        })

    model_path = models_dir / "model_latest.pt"
    metrics_path = reports_dir / "metrics_latest.json"

    payload = {
        "model_state_dict": model.state_dict(),
        "alphabet": alphabet,
        "project_id": config.get("project_id"),
        "field_name": config.get("field_name"),
        "fixed_length": config.get("fixed_length"),
        "roi": config.get("roi"),
        "normalization": "pil_grayscale_autocontrast_invert_32x32",
    }

    torch.save(payload, model_path)

    metrics = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "cuda": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "num_samples": len(dataset),
        "alphabet": alphabet,
        "counts": collect_sample_counts(samples_dir, alphabet),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "history": history,
        "model_path": str(model_path),
    }

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def load_trained_model_for_project(pdir: Path):
    config = load_project_config(pdir)
    alphabet = str(config.get("alphabet", DEFAULT_ALPHABET))
    model_path = pdir / "models" / "model_latest.pt"

    if not model_path.exists():
        raise RuntimeError("No hi ha model_latest.pt. Primer entrena a 05 · Training.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)

    alphabet = ckpt.get("alphabet", alphabet)
    model = SmallCharCNN(num_classes=len(alphabet)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, alphabet, device, ckpt


def predict_char_image(model, alphabet: str, device, img: Image.Image) -> dict:
    x = normalize_char_pil(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

    order = np.argsort(probs)[::-1]
    top = []
    for idx in order[:5]:
        top.append({
            "classe": alphabet[int(idx)],
            "prob": float(probs[int(idx)]),
        })

    best = top[0]
    return {
        "pred": best["classe"],
        "confidence": best["prob"],
        "top": top,
    }




# ---------- Runtime Batch helpers ----------

def build_box_template_from_labels(pdir: Path, fixed_length: int) -> list:
    """
    Construeix una plantilla de caixes per posició a partir de labels.csv.

    Usa els crops manuals canvas:
    - pos / slot_idx
    - box_x0, box_y0, box_x1, box_y1

    Retorna llista ordenada pos 1..N.
    """
    labels_csv = pdir / "labels.csv"

    if not labels_csv.exists():
        raise RuntimeError("No existeix labels.csv. Primer etiqueta caràcters a 04 · Dataset / labeling.")

    df = pd.read_csv(labels_csv)

    # Compatibilitat: canvas_manual usa 'pos'; versions anteriors poden usar 'slot_idx'
    if "pos" not in df.columns and "slot_idx" in df.columns:
        df["pos"] = df["slot_idx"]

    required = ["pos", "box_x0", "box_y0", "box_x1", "box_y1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"labels.csv no té columnes de caixa necessàries: {missing}")

    out = []

    for pos in range(1, int(fixed_length) + 1):
        sub = df[df["pos"].astype(int) == pos]

        if sub.empty:
            raise RuntimeError(f"No hi ha cap caixa etiquetada per la posició {pos}.")

        box = {
            "pos": pos,
            "x0": int(round(float(sub["box_x0"].mean()))),
            "y0": int(round(float(sub["box_y0"].mean()))),
            "x1": int(round(float(sub["box_x1"].mean()))),
            "y1": int(round(float(sub["box_y1"].mean()))),
            "n": int(len(sub)),
        }
        out.append(box)

    return out


def infer_default_trim_from_labels(pdir: Path) -> dict:
    """
    Recupera configuració d'auto-trim més recent del labels.csv.
    Si no existeix, retorna defaults.
    """
    labels_csv = pdir / "labels.csv"

    defaults = {
        "auto_trim": True,
        "trim_threshold": 245,
        "trim_pad": 4,
    }

    if not labels_csv.exists():
        return defaults

    try:
        df = pd.read_csv(labels_csv)
        if df.empty:
            return defaults

        last = df.iloc[-1]

        return {
            "auto_trim": bool(last.get("auto_trim", defaults["auto_trim"])),
            "trim_threshold": int(last.get("trim_threshold", defaults["trim_threshold"])),
            "trim_pad": int(last.get("trim_pad", defaults["trim_pad"])),
        }

    except Exception:
        return defaults


def predict_document_code_from_path(
    path: Path,
    config: dict,
    model,
    alphabet: str,
    device,
    box_template: list,
    use_auto_trim: bool = True,
    trim_threshold: int = 245,
    trim_pad: int = 4,
) -> dict:
    """
    Prediu el codi complet d'un document nou:
    document -> ROI -> auto-trim -> caixes plantilla -> CNN per caràcter.
    """
    roi = config.get("roi")
    fixed_length = int(config.get("fixed_length", 0) or 0)

    if not roi:
        raise RuntimeError("Aquest projecte no té ROI guardada.")

    img = load_first_page_image(path, dpi=150)
    crop_raw = crop_relative(img, roi)

    if use_auto_trim:
        crop = trim_to_ink_bbox(crop_raw, threshold=int(trim_threshold), pad_px=int(trim_pad))
    else:
        crop = crop_raw.convert("RGB")

    chars = []
    confs = []
    pos_details = []
    char_crops = []

    for b in box_template:
        box = {
            "x0": b["x0"],
            "y0": b["y0"],
            "x1": b["x1"],
            "y1": b["y1"],
        }

        ch_img = crop_box_pixels(crop, box)
        pred = predict_char_image(model, alphabet, device, ch_img)

        chars.append(pred["pred"])
        confs.append(float(pred["confidence"]))
        char_crops.append(ch_img)

        pos_details.append({
            "pos": int(b["pos"]),
            "pred": pred["pred"],
            "confidence": float(pred["confidence"]),
            "box": box,
            "top": pred["top"],
        })

    code = "".join(chars)

    min_conf = min(confs) if confs else 0.0
    avg_conf = float(sum(confs) / max(1, len(confs)))

    return {
        "file": path.name,
        "code": code,
        "min_conf": min_conf,
        "avg_conf": avg_conf,
        "pos_details": pos_details,
        "crop_raw": crop_raw,
        "crop": crop,
        "char_crops": char_crops,
        "fixed_length": fixed_length,
    }


def save_uploaded_batch_files(pdir: Path, uploaded_files) -> Path:
    """
    Desa batch uploads dins del projecte per poder processar-los com Path.
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = pdir / "runtime_batches" / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    for uf in uploaded_files:
        target = run_dir / uf.name
        target.write_bytes(uf.getbuffer())

    return run_dir


# ---------- Auto-segmentation helpers ----------

def auto_split_characters_by_x_histogram(
    roi_img: Image.Image,
    threshold_method="otsu",
    min_char_width=3,
    min_ink_per_col=1,
    gap_merge_px=2,
    pad_x=2,
    pad_y=2,
    invert_if_needed=True,
    apply_morph_open=True,
    apply_morph_close=False,
):
    """
    Segmenta automàticament una ROI en caràcters mitjançant histograma X.
    """
    debug = {}
    if roi_img is None or roi_img.size[0] == 0 or roi_img.size[1] == 0:
        return [], [], {"error": "ROI image is empty."}

    img_np = np.array(roi_img.copy())

    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    elif len(img_np.shape) == 3 and img_np.shape[2] == 4: # RGBA
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGBA2GRAY)
    else:
        gray = img_np
    debug['gray'] = Image.fromarray(gray)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    if threshold_method == "adaptive":
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    if invert_if_needed:
        if cv2.countNonZero(binary) > (binary.shape[0] * binary.shape[1]) // 2:
            binary = cv2.bitwise_not(binary)
    debug['binary_pre_morph'] = Image.fromarray(binary)

    if apply_morph_open:
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    if apply_morph_close:
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    debug['binary'] = Image.fromarray(binary)

    ink = binary > 0
    x_projection = np.sum(ink, axis=0)
    debug['x_projection'] = x_projection

    active_cols = x_projection >= min_ink_per_col
    raw_segments = []
    is_in_segment = False
    start_x = 0
    for x, is_active in enumerate(active_cols):
        if is_active and not is_in_segment:
            is_in_segment = True
            start_x = x
        elif not is_active and is_in_segment:
            is_in_segment = False
            raw_segments.append((start_x, x - 1))
    if is_in_segment:
        raw_segments.append((start_x, len(active_cols) - 1))
    debug['raw_segments'] = raw_segments

    if not raw_segments:
        return [], [], debug

    merged_segments = [raw_segments[0]]
    for i in range(1, len(raw_segments)):
        prev_end = merged_segments[-1][1]
        curr_start = raw_segments[i][0]
        if (curr_start - prev_end - 1) <= gap_merge_px:
            merged_segments[-1] = (merged_segments[-1][0], raw_segments[i][1])
        else:
            merged_segments.append(raw_segments[i])
    debug['merged_segments'] = merged_segments

    crops, boxes = [], []
    h, w = gray.shape
    for x1, x2 in merged_segments:
        if (x2 - x1 + 1) >= min_char_width:
            segment_slice = ink[:, x1:x2+1]
            y_projection = np.sum(segment_slice, axis=1)
            active_rows = np.where(y_projection > 0)[0]
            if len(active_rows) > 0:
                y1, y2 = active_rows[0], active_rows[-1]
                box_x1, box_y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                box_x2, box_y2 = min(w, x2 + pad_x + 1), min(h, y2 + pad_y + 1)
                boxes.append((box_x1, box_y1, box_x2, box_y2))
                crops.append(roi_img.crop((box_x1, box_y1, box_x2, box_y2)))

    # TODO: Implement split_wide_segments_by_valleys for connected characters.
    return crops, boxes, debug

def draw_detected_boxes(roi_img: Image.Image, boxes: list):
    if not boxes: return roi_img.copy()
    draw_img = roi_img.copy().convert("RGB")
    draw = ImageDraw.Draw(draw_img)
    try: font = ImageFont.truetype("arial.ttf", 15)
    except IOError: font = ImageFont.load_default()
    for i, box in enumerate(boxes):
        draw.rectangle(box, outline="green", width=1)
        draw.text((box[0] + 2, box[1] - 16 if box[1]>16 else box[1]+2), str(i + 1), fill="red", font=font)
    return draw_img

if page == "00 · Estat":
    st.subheader("Estat general")

    st.write("Projectes trobats:", len(projects))

    if projects:
        rows = []
        for p in projects:
            try:
                c = load_project_config(p)
                rows.append({
                    "project_id": c.get("project_id"),
                    "nom": c.get("project_name"),
                    "departament": c.get("department"),
                    "camp": c.get("field_name"),
                    "alfabet": c.get("alphabet"),
                    "longitud": c.get("fixed_length"),
                    "status": c.get("status"),
                })
            except Exception as e:
                rows.append({"project_id": p.name, "status": f"ERROR: {e}"})

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.info("""
MVP v0.1:
- ROI fixa
- longitud fixa
- alfabet inicial recomanat: 0-9
- una feina d'entrenament a la vegada
- validació humana obligatòria
""")

elif page == "01 · Crear projecte":
    st.subheader("01 · Crear projecte")

    with st.form("create_project_form"):
        project_name = st.text_input("Nom del projecte", placeholder="SAT_partes_codigo_instalacion")
        department = st.text_input("Departament", placeholder="SAT / RRHH / Qualitat / Logística")
        field_name = st.text_input("Camp a llegir", placeholder="numero_parte / codigo_instalacion / empleado")
        alphabet = st.text_input("Alfabet permès", value=DEFAULT_ALPHABET)
        fixed_length = st.number_input("Longitud fixa del camp", min_value=1, max_value=64, value=7, step=1)

        st.markdown("### Constraints")
        st.warning("""
Per aquesta primera versió:
- el camp ha d'estar sempre a la mateixa zona
- el camp ha de tenir longitud fixa
- recomanat començar amb només dígits
- mínim 15 exemples per classe
""")

        submitted = st.form_submit_button("Crear projecte", type="primary")

    if submitted:
        if not project_name or not field_name or not alphabet:
            st.error("Falten dades obligatòries.")
        else:
            pdir = create_project(project_name, department, field_name, alphabet, int(fixed_length))
            st.success(f"Projecte creat: {pdir.name}")
            st.code(str(pdir))

elif page == "02 · Upload documents":
    st.subheader("02 · Upload documents")
    pdir, config = select_project()

    if pdir:
        st.json(config)
        uploaded = st.file_uploader(
            "Puja documents d'exemple",
            type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            accept_multiple_files=True,
        )

        if uploaded and st.button("Guardar uploads", type="primary"):
            saved = save_uploaded_files(pdir, uploaded)
            config["status"] = "documents_uploaded"
            save_project_config(pdir, config)
            st.success(f"Guardats {len(saved)} documents.")
            st.write(saved)

elif page == "03 · ROI bàsica":
    st.subheader("03 · ROI bàsica")
    st.caption("Mode anti-Gowex: rectangle visible + crop zoom. Els sliders ja no van a cegues.")

    pdir, config = select_project()

    if pdir:
        upload_dir = pdir / "uploads"
        docs = sorted([p for p in upload_dir.iterdir() if p.is_file()]) if upload_dir.exists() else []

        if not docs:
            st.warning("Primer puja documents.")
        else:
            doc = st.selectbox("Document per calibrar ROI", [p.name for p in docs])
            path = next(p for p in docs if p.name == doc)

            img = load_first_page_image(path, dpi=130)
            w, h = img.size

            current_roi = config.get("roi") or {
                "x0": 0.50,
                "y0": 0.00,
                "x1": 0.98,
                "y1": 0.25,
            }

            st.markdown("### 1) Ajust visual de ROI")
            st.info("Mou sliders i mira el rectangle vermell i el crop ampliat. Quan el crop contingui només el camp/codi, guarda ROI.")

            # Presets ràpids per no començar des de zero
            p1, p2, p3, p4 = st.columns(4)
            if p1.button("Preset: superior dreta"):
                current_roi = {"x0": 0.55, "y0": 0.00, "x1": 0.98, "y1": 0.22}
            if p2.button("Preset: capçalera ampla"):
                current_roi = {"x0": 0.25, "y0": 0.00, "x1": 0.98, "y1": 0.28}
            if p3.button("Preset: centre dreta"):
                current_roi = {"x0": 0.45, "y0": 0.20, "x1": 0.98, "y1": 0.55}
            if p4.button("Preset: full ample"):
                current_roi = {"x0": 0.00, "y0": 0.00, "x1": 1.00, "y1": 0.35}

            c_img, c_sliders = st.columns([1.25, 1.0], gap="large")

            with c_sliders:
                st.markdown("#### Coordenades ROI")

                x0 = st.slider("x0 · esquerra", 0.0, 1.0, float(current_roi.get("x0", 0.50)), 0.005)
                y0 = st.slider("y0 · dalt",     0.0, 1.0, float(current_roi.get("y0", 0.00)), 0.005)
                x1 = st.slider("x1 · dreta",    0.0, 1.0, float(current_roi.get("x1", 0.98)), 0.005)
                y1 = st.slider("y1 · baix",     0.0, 1.0, float(current_roi.get("y1", 0.25)), 0.005)

                # Normalitzem perquè no peti si es creuen sliders
                rx0, rx1 = sorted([x0, x1])
                ry0, ry1 = sorted([y0, y1])
                roi = {"x0": rx0, "y0": ry0, "x1": rx1, "y1": ry1}

                px = {
                    "x0_px": int(w * rx0),
                    "y0_px": int(h * ry0),
                    "x1_px": int(w * rx1),
                    "y1_px": int(h * ry1),
                    "ample_px": int(w * (rx1 - rx0)),
                    "alt_px": int(h * (ry1 - ry0)),
                }

                st.markdown("#### ROI en píxels")
                st.json(px)

                if px["ample_px"] < 10 or px["alt_px"] < 10:
                    st.error("ROI massa petita. Mou x1/y1 o separa x0/x1.")
                elif px["ample_px"] > w * 0.95 and px["alt_px"] > h * 0.8:
                    st.warning("ROI massa gran. Això entrenarà fum.")

                if st.button("💾 Guardar ROI", type="primary", use_container_width=True):
                    config["roi"] = roi
                    config["status"] = "roi_defined"
                    save_project_config(pdir, config)
                    st.success("ROI guardada.")
                    st.json(roi)

            with c_img:
                st.markdown("#### Pàgina amb ROI marcada")
                try:
                    overlay = draw_roi_overlay(img, roi)
                    st.image(overlay, caption="Rectangle vermell = ROI seleccionada", use_container_width=True)
                except Exception as e:
                    st.error(f"No he pogut dibuixar overlay: {e}")
                    st.image(img, caption="Pàgina completa", use_container_width=True)

            st.markdown("---")
            st.markdown("### 2) Crop resultant ampliat")
            try:
                crop = crop_relative(img, roi)
                st.image(crop, caption="Això és el que veurà la CNN / segmentador", use_container_width=False)

                cw, ch = crop.size
                st.caption(f"Crop size: {cw} x {ch} px")

                if cw < 40 or ch < 20:
                    st.error("Crop massa petit per entrenar bé. Amplia una mica la ROI.")
                elif ch > 250:
                    st.warning("Crop molt alt. Intenta retallar només la línia/camp.")
                else:
                    st.success("ROI visualment usable si el camp/codi queda dins del crop.")
            except Exception as e:
                st.error(f"No he pogut generar crop: {e}")

            st.markdown("### 3) Preview de la mateixa ROI en altres documents")
            other_docs = [p for p in docs if p != path][:5]
            if other_docs:
                cols = st.columns(min(3, len(other_docs)))
                for i, op in enumerate(other_docs):
                    with cols[i % len(cols)]:
                        try:
                            oimg = load_first_page_image(op, dpi=100)
                            ocrop = crop_relative(oimg, roi)
                            st.image(ocrop, caption=op.name[:45], use_container_width=True)
                        except Exception as e:
                            st.caption(f"{op.name}: {e}")
            else:
                st.caption("No hi ha altres documents per validar generalització de ROI.")


elif page == "04 · Dataset / labeling":
    st.subheader("04 · Dataset / labeling")
    st.caption(
        "Tria un mode: manual per control total, histograma per auto-detecció, o VLM per assistència AI."
    )

    import csv
    from datetime import datetime

    pdir, config = select_project()

    if pdir:
        st.json(config)

        roi = config.get("roi")
        alphabet = str(config.get("alphabet", DEFAULT_ALPHABET))
        fixed_length = int(config.get("fixed_length", 0) or 0)

        if not roi:
            st.error("Primer has de definir i guardar la ROI a 03 · ROI bàsica.")
            st.stop()

        if fixed_length <= 0:
            st.error("La longitud fixa del camp no és vàlida.")
            st.stop()

        upload_dir = pdir / "uploads"
        docs = sorted([p for p in upload_dir.iterdir() if p.is_file()]) if upload_dir.exists() else []

        if not docs:
            st.warning("Primer puja documents a 02 · Upload documents.")
            st.stop()

        samples_dir = pdir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        labels_csv = pdir / "labels.csv"

        # ========================================================
        # Estat dataset
        # ========================================================
        st.markdown("### 1) Estat del dataset")

        counts = {}
        for ch in alphabet:
            cdir = samples_dir / ch
            counts[ch] = len(list(cdir.glob("*.png"))) if cdir.exists() else 0

        rows = []
        for ch in alphabet:
            n = counts.get(ch, 0)
            rows.append({
                "classe": ch,
                "mostres": n,
                "mínim": MIN_EXAMPLES_PER_CLASS,
                "recomanat": RECOMMENDED_EXAMPLES_PER_CLASS,
                "estat": "✅ mínim OK" if n >= MIN_EXAMPLES_PER_CLASS else "🟠 falta",
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        
        segmentation_mode = st.radio(
            "Mode de segmentació de caràcters",
            ["Manual amb Canvas", "Auto per Histograma-X", "Assistit per VLM"],
            horizontal=True,
            key=f"seg_mode_{pdir.name}"
        )
        st.markdown("---")

        if segmentation_mode == "Assistit per VLM":
            st.markdown("### 🤖 Etiquetar amb assistència VLM")
            vlm_api_key = st.text_input("Clau API del VLM (p.ex. OpenAI)", type="password", key="vlm_api_key")
            doc_vlm = st.selectbox("Document per VLM", [p.name for p in docs], key="vlm_doc_select")
            vlm_prompt = st.text_area(
                "Prompt per al VLM",
                "Ets un expert en OCR. A la imatge adjunta hi ha un codi alfanumèric. Identifica cada caràcter individualment i retorna les seves coordenades."
            )

            if st.button("Executar VLM per obtenir caixes", disabled=not vlm_api_key):
                path_vlm = next(p for p in docs if p.name == doc_vlm)
                img_vlm = load_first_page_image(path_vlm, dpi=150)
                crop_vlm = crop_relative(img_vlm, roi)

                try:
                    with st.spinner("El VLM està pensant..."):
                        vlm_result = get_char_boxes_from_vlm(crop_vlm, vlm_prompt, vlm_api_key, fixed_length)

                    st.success("VLM ha retornat les caixes!")
                    st.json(vlm_result)

                    if "characters" in vlm_result and len(vlm_result["characters"]) == fixed_length:
                        st.info("Les dades semblen correctes. Pots guardar-les.")
                        if st.button("💾 Guardar tots els caràcters del VLM"):
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            saved_count = 0
                            for i, item in enumerate(vlm_result["characters"]):
                                char_label = str(item["char"]).strip().upper()
                                if char_label not in alphabet:
                                    st.warning(f"Caràcter '{char_label}' ignorat (fora de l'alfabet).")
                                    continue

                                box_coords = item["box"]
                                box_dict = {"x0": box_coords[0], "y0": box_coords[1], "x1": box_coords[2], "y1": box_coords[3]}
                                char_crop = crop_box_pixels(crop_vlm, box_dict)

                                class_dir = samples_dir / char_label
                                class_dir.mkdir(parents=True, exist_ok=True)
                                safe_stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", path_vlm.stem)[:80]
                                out_name = f"{safe_stem}_{ts}_vlm_pos{i+1:02d}_{char_label}.png"
                                out_path = class_dir / out_name
                                char_crop.save(out_path)
                                saved_count += 1
                            st.success(f"Guardats {saved_count} caràcters etiquetats pel VLM.")
                            st.rerun()
                except Exception as e:
                    st.error(f"Error amb el VLM: {e}")

        elif segmentation_mode == "Auto per Histograma-X":
            st.markdown("### 🤖 Etiquetar amb Auto-segmentació per Histograma-X")
            st.caption("Ajusta la ROI a la pàgina 03, després ajusta els paràmetres d'aquí per detectar els caràcters.")

            doc = st.selectbox("Document a etiquetar", [p.name for p in docs], key=f"doc_auto_{pdir.name}")
            path = next(p for p in docs if p.name == doc)

            img = load_first_page_image(path, dpi=150)
            crop_raw = crop_relative(img, roi)

            st.markdown("#### Paràmetres de segmentació")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                threshold_method = st.selectbox("Threshold", ["otsu", "adaptive"], key=f"th_auto_{pdir.name}")
                apply_morph_open = st.checkbox("Neteja soroll", value=True, key=f"mo_auto_{pdir.name}")
            with c2:
                min_char_width = st.slider("Amplada min", 1, 30, 3, key=f"mcw_auto_{pdir.name}")
                apply_morph_close = st.checkbox("Unir fragments", value=False, key=f"mc_auto_{pdir.name}")
            with c3:
                gap_merge_px = st.slider("Fusionar gaps <=", 0, 20, 2, key=f"gmp_auto_{pdir.name}")
                min_ink_per_col = st.slider("Tinta min/col", 1, 30, 1, key=f"mipc_auto_{pdir.name}")
            with c4:
                pad_x = st.slider("Padding X", 0, 20, 2, key=f"px_auto_{pdir.name}")
                pad_y = st.slider("Padding Y", 0, 20, 2, key=f"py_auto_{pdir.name}")

            try:
                crops, boxes, debug = auto_split_characters_by_x_histogram(
                    roi_img=crop_raw, threshold_method=threshold_method, min_char_width=min_char_width,
                    min_ink_per_col=min_ink_per_col, gap_merge_px=gap_merge_px, pad_x=pad_x, pad_y=pad_y,
                    apply_morph_open=apply_morph_open, apply_morph_close=apply_morph_close,
                )
                preview = draw_detected_boxes(crop_raw, boxes)
                
                c_left, c_right = st.columns(2, gap="large")
                with c_left:
                    st.image(preview, caption=f"Auto-detecció: {len(crops)} caràcters trobats", use_container_width=True)
                
                with c_right:
                    truth_raw = st.text_input(
                        f"Valor real complet ({len(crops)} caràcters detectats)",
                        key=f"truth_auto_{pdir.name}_{path.name}",
                        placeholder="Escriu el text que veus a la ROI",
                    )
                    truth = "".join([c for c in truth_raw.strip().upper() if c in alphabet])

                    if truth_raw and truth != truth_raw.strip().upper():
                        st.warning(f"S'han eliminat caràcters fora de l'alfabet. Valor net: {truth}")

                    labels, can_save, manual_labels_ok = [], False, False
                    manual_labels = {}

                    if truth:
                        if len(truth) == len(crops):
                            st.success(f"El text coincideix amb el número de crops ({len(truth)}). Llest per guardar.")
                            labels = list(truth)
                            can_save = True
                        else:
                            st.error(f"El text té {len(truth)} caràcters, però s'han detectat {len(crops)} crops. Ajusta paràmetres o etiqueta manualment.")
                    
                    if not can_save and len(crops) > 0:
                        st.markdown("##### Etiquetatge manual per crop")
                        cols = st.columns(min(len(crops), 5))
                        for i in range(len(crops)):
                            with cols[i % 5]:
                                manual_labels[i] = st.text_input(f"Crop {i+1}", key=f"ml_{pdir.name}_{path.name}_{i}", max_chars=1).strip().upper()
                        
                        provided_labels = [manual_labels.get(i, "") for i in range(len(crops))]
                        if all(provided_labels) and all(l in alphabet for l in provided_labels):
                            labels = provided_labels
                            truth = "".join(labels)
                            can_save = True
                            manual_labels_ok = True
                            st.info("Tots els crops tenen una etiqueta manual. Llest per guardar.")
                        elif any(provided_labels):
                            st.warning("Completa totes les etiquetes manuals amb caràcters de l'alfabet.")

                    allow_duplicate = st.checkbox("Permetre duplicats (mateix doc/valor)", value=False, key=f"dup_auto_{pdir.name}")
                    if st.button("💾 Guardar crops etiquetats", type="primary", disabled=not can_save, use_container_width=True):
                        already = False
                        if labels_csv.exists() and not allow_duplicate:
                            try:
                                prev = pd.read_csv(labels_csv)
                                if "source_doc" in prev.columns and "truth" in prev.columns:
                                    already = ((prev["source_doc"] == path.name) & (prev["truth"] == truth)).any()
                            except Exception: already = False

                        if already:
                            st.error("Aquest document amb aquest valor ja està etiquetat. Marca 'Permetre duplicats' si vols repetir.")
                        else:
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            saved_rows = []
                            for i, (crop_img, label, box) in enumerate(zip(crops, labels, boxes)):
                                class_dir = samples_dir / label
                                class_dir.mkdir(parents=True, exist_ok=True)
                                safe_stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", path.stem)[:80]
                                out_name = f"{safe_stem}_{ts}_auto_pos{i+1:02d}_{label}.png"
                                out_path = class_dir / out_name
                                crop_img.save(out_path)

                                row = {
                                    "timestamp": ts, "project_id": config.get("project_id"), "source_doc": path.name,
                                    "truth": truth, "pos": i + 1, "label": label, "image_path": str(out_path),
                                    "mode": "auto_histogram", "ink_ratio": slot_ink_ratio(crop_img, threshold=200),
                                    "box_x0": box[0], "box_y0": box[1], "box_x1": box[2], "box_y2": box[3],
                                    "roi_x0": roi.get("x0"), "roi_y0": roi.get("y0"), "roi_x1": roi.get("x1"), "roi_y1": roi.get("y1"),
                                    "hist_threshold_method": threshold_method, "hist_min_char_width": min_char_width,
                                    "hist_min_ink_per_col": min_ink_per_col, "hist_gap_merge_px": gap_merge_px,
                                    "hist_pad_x": pad_x, "hist_pad_y": pad_y, "hist_apply_morph_open": apply_morph_open,
                                    "hist_apply_morph_close": apply_morph_close,
                                }
                                saved_rows.append(row)

                            if saved_rows:
                                fieldnames = list(saved_rows[0].keys())
                                write_header = not labels_csv.exists()
                                with labels_csv.open("a", newline="", encoding="utf-8") as f:
                                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                                    if write_header: writer.writeheader()
                                    writer.writerows(saved_rows)

                                config["status"] = "samples_labeling_auto"
                                save_project_config(pdir, config)
                                st.success(f"Guardats {len(saved_rows)} caràcters etiquetats per `{truth}`.")
                                st.rerun()

                if crops:
                    st.markdown("#### Crops detectats")
                    cols = st.columns(min(len(crops), 10))
                    for i, crop in enumerate(crops):
                        with cols[i % 10]:
                            caption = f"Crop {i+1}"
                            if labels and i < len(labels): caption += f" -> '{labels[i]}'"
                            st.image(crop, caption=caption, use_container_width=True)

                with st.expander("Debug histograma"):
                    if 'binary' in debug:
                        st.image(debug['binary'], caption="Imatge binària usada per a la projecció", use_container_width=True)
                    if 'x_projection' in debug:
                        st.markdown("Projecció X (suma de píxels de tinta per columna)")
                        fig, ax = plt.subplots()
                        ax.plot(debug['x_projection'])
                        ax.set_title("Projecció Vertical (Histograma X)")
                        ax.set_xlabel("Columna X"); ax.set_ylabel("Píxels de tinta")
                        st.pyplot(fig)
                    st.json({"raw_segments": debug.get('raw_segments', []), "merged_segments": debug.get('merged_segments', [])})

            except Exception as e:
                st.error(f"Error en la segmentació automàtica: {e}")
                st.exception(e)

        elif segmentation_mode == "Manual amb Canvas":
            st.markdown("### ✍️ Etiquetar caràcters amb mouse (Manual)")
            st.caption("Dibuixa amb el mouse el retall de cada caràcter. Zero geometria Gowex.")

            doc = st.selectbox("Document a etiquetar", [p.name for p in docs], key=f"doc_manual_{pdir.name}")

        path = next(p for p in docs if p.name == doc)

        img = load_first_page_image(path, dpi=150)
        crop_raw = crop_relative(img, roi)

        c_cfg1, c_cfg2, c_cfg3, c_cfg4 = st.columns(4)

        with c_cfg1:
            use_auto_trim = st.checkbox("Auto-trim ROI", value=True)
        with c_cfg2:
            trim_threshold = st.slider("Threshold tinta", 180, 255, 245, 1)
        with c_cfg3:
            trim_pad = st.slider("Padding trim px", 0, 30, 4, 1)
        with c_cfg4:
            zoom = st.slider("Zoom canvas", 3, 16, 8, 1)

        if use_auto_trim:
            crop = trim_to_ink_bbox(crop_raw, threshold=trim_threshold, pad_px=trim_pad)
        else:
            crop = crop_raw.convert("RGB")

        cw, ch = crop.size

        truth_raw = st.text_input(
            f"Valor real complet ({fixed_length} caràcters)",
            key=f"truth_canvas_{pdir.name}_{path.name}",
            placeholder="7000177418",
        )

        truth = "".join([c for c in truth_raw.strip().upper() if c in alphabet])

        if truth_raw and truth != truth_raw.strip().upper():
            st.warning(f"S'han eliminat caràcters fora de l'alfabet. Valor net: {truth}")

        valid_truth = len(truth) == fixed_length

        if truth and not valid_truth:
            st.error(f"Longitud incorrecta: {len(truth)}. Aquest projecte espera {fixed_length}.")
        elif valid_truth:
            st.success(f"Valor vàlid: {truth}")

        # Estat posició activa
        active_key = f"active_char_idx_canvas_{pdir.name}_{path.name}"
        if active_key not in st.session_state:
            st.session_state[active_key] = 0

        active_idx = int(st.session_state[active_key])
        active_idx = max(0, min(fixed_length - 1, active_idx))
        expected_label = truth[active_idx] if valid_truth else ""

        c_left, c_right = st.columns([1.35, 1.0], gap="large")

        with c_left:
            st.markdown("#### ROI raw")
            st.image(crop_raw, caption=f"ROI original · {crop_raw.size[0]} x {crop_raw.size[1]} px", use_container_width=False)

            st.markdown("#### Canvas ampliat")
            st.caption("Dibuixa UN rectangle al voltant del caràcter actiu. Si surt malament, usa 'Reset canvas'.")

            canvas_img = resize_for_zoom(crop, zoom=zoom)
            canvas_w, canvas_h = canvas_img.size

            canvas_rev_key = f"canvas_rev_{pdir.name}_{path.name}_{active_idx}"
            if canvas_rev_key not in st.session_state:
                st.session_state[canvas_rev_key] = 0

            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.15)",
                stroke_width=2,
                stroke_color="#ff0000",
                background_image=canvas_img,
                update_streamlit=True,
                height=canvas_h,
                width=canvas_w,
                drawing_mode="rect",
                key=f"canvas_{pdir.name}_{path.name}_{active_idx}_{st.session_state[canvas_rev_key]}",
            )

        with c_right:
            st.markdown("#### Caràcter actiu")

            pos = st.number_input(
                "Posició",
                min_value=1,
                max_value=fixed_length,
                value=active_idx + 1,
                step=1,
            )

            new_idx = int(pos) - 1
            if new_idx != active_idx:
                st.session_state[active_key] = new_idx
                st.rerun()

            if valid_truth:
                st.info(f"Posició {active_idx + 1}/{fixed_length} → etiqueta: `{expected_label}`")
            else:
                st.warning("Escriu primer el valor real complet per saber l'etiqueta del caràcter.")

            # Llegir rectangle del canvas
            box = None
            if canvas_result.json_data and canvas_result.json_data.get("objects"):
                objs = canvas_result.json_data["objects"]
                # agafem l'últim rectangle dibuixat
                obj = objs[-1]
                left = float(obj.get("left", 0))
                top = float(obj.get("top", 0))
                width = float(obj.get("width", 0)) * float(obj.get("scaleX", 1))
                height = float(obj.get("height", 0)) * float(obj.get("scaleY", 1))

                # Convertim canvas pixels -> crop original pixels
                x0 = int(round(left / zoom))
                y0 = int(round(top / zoom))
                x1 = int(round((left + width) / zoom))
                y1 = int(round((top + height) / zoom))

                box = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

            if box:
                st.markdown("##### Preview crop caràcter")
                char_crop = crop_box_pixels(crop, box)
                char_zoom = resize_for_zoom(char_crop, zoom=max(10, zoom))
                st.image(char_zoom, caption=f"Crop pos {active_idx+1} · etiqueta `{expected_label or '?'}`", use_container_width=False)

                ink = slot_ink_ratio(char_crop, threshold=trim_threshold)
                st.caption(f"box={box} · size={char_crop.size[0]}x{char_crop.size[1]} · ink={ink*100:.2f}%")

                if ink < 0.005:
                    st.error("Aquest crop sembla buit. Dibuixa millor el rectangle.")
                elif char_crop.size[0] < 3 or char_crop.size[1] < 8:
                    st.warning("Crop molt petit. Potser cal redibuixar.")
                else:
                    st.success("Crop usable si visualment només conté aquest caràcter.")
            else:
                st.warning("Dibuixa un rectangle al canvas per obtenir el crop.")

            nav1, nav2, nav3 = st.columns(3)

            if nav1.button("⬅️ Anterior", use_container_width=True, disabled=active_idx <= 0):
                st.session_state[active_key] = max(0, active_idx - 1)
                st.rerun()

            if nav2.button("➡️ Següent", use_container_width=True, disabled=active_idx >= fixed_length - 1):
                st.session_state[active_key] = min(fixed_length - 1, active_idx + 1)
                st.rerun()

            if nav3.button("🧽 Reset canvas", use_container_width=True):
                st.session_state[canvas_rev_key] += 1
                st.rerun()

            can_save = bool(valid_truth and expected_label and box)

            if st.button("💾 Guardar caràcter actiu", type="primary", disabled=not can_save, use_container_width=True):
                char_crop = crop_box_pixels(crop, box)
                ink = slot_ink_ratio(char_crop, threshold=trim_threshold)

                if ink < 0.005:
                    st.error("No guardo: crop buit.")
                else:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    class_dir = samples_dir / expected_label
                    class_dir.mkdir(parents=True, exist_ok=True)

                    safe_stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", path.stem)[:80]
                    out_name = f"{safe_stem}_{ts}_pos{active_idx+1:02d}_{expected_label}.png"
                    out_path = class_dir / out_name

                    char_crop.save(out_path)

                    row = {
                        "timestamp": ts,
                        "project_id": config.get("project_id"),
                        "source_doc": path.name,
                        "truth": truth,
                        "pos": active_idx + 1,
                        "label": expected_label,
                        "image_path": str(out_path),
                        "mode": "canvas_manual",
                        "ink_ratio": ink,
                        "box_x0": box["x0"],
                        "box_y0": box["y0"],
                        "box_x1": box["x1"],
                        "box_y1": box["y1"],
                        "roi_x0": roi.get("x0"),
                        "roi_y0": roi.get("y0"),
                        "roi_x1": roi.get("x1"),
                        "roi_y1": roi.get("y1"),
                        "auto_trim": use_auto_trim,
                        "trim_threshold": trim_threshold,
                        "trim_pad": trim_pad,
                        "zoom": zoom,
                    }

                    write_header = not labels_csv.exists()
                    with labels_csv.open("a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                        if write_header:
                            writer.writeheader()
                        writer.writerow(row)

                    config["status"] = "samples_labeling_canvas"
                    save_project_config(pdir, config)

                    st.success(f"Guardat pos {active_idx+1}: `{expected_label}`")

                    # auto-next i canvas net
                    if active_idx < fixed_length - 1:
                        st.session_state[active_key] = active_idx + 1
                    st.session_state[canvas_rev_key] += 1
                    st.rerun()

        st.markdown("---")
        st.markdown("### Resum de mostres guardades")

        preview_cols = st.columns(min(len(alphabet), 10))
        for idx, ch in enumerate(alphabet):
            with preview_cols[idx % min(len(alphabet), 10)]:
                cdir = samples_dir / ch
                imgs = sorted(cdir.glob("*.png"))[-4:] if cdir.exists() else []
                st.markdown(f"**{ch}** · {len(list(cdir.glob('*.png'))) if cdir.exists() else 0}")
                for im in imgs:
                    st.image(str(im), use_container_width=True)


elif page == "05 · Training":
    st.subheader("05 · Training")
    st.caption("Entrenament de prova. Encara que sigui Gowex-Fake-Fum, valida el pipeline complet.")

    pdir, config = select_project()

    if pdir:
        samples_dir = pdir / "samples"
        alphabet = str(config.get("alphabet", DEFAULT_ALPHABET))
        counts = collect_sample_counts(samples_dir, alphabet)

        st.markdown("### Estat del dataset")

        rows = []
        for ch in alphabet:
            n = counts.get(ch, 0)
            rows.append({
                "classe": ch,
                "mostres": n,
                "mínim recomanat": MIN_EXAMPLES_PER_CLASS,
                "estat": "✅ OK" if n >= MIN_EXAMPLES_PER_CLASS else "🧪 demo/falta",
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        total_samples = sum(counts.values())
        st.metric("Total samples", total_samples)

        if is_locked():
            st.error("Ara mateix hi ha un entrenament en curs.")
            st.code(read_lock())
        else:
            st.success("GPU disponible.")

        st.markdown("### Paràmetres training")

        c1, c2, c3 = st.columns(3)
        with c1:
            epochs = st.slider("Epochs", 1, 100, 20, 1)
        with c2:
            batch_size = st.selectbox("Batch size", [4, 8, 16, 32], index=1)
        with c3:
            lr = st.selectbox(
                "Learning rate",
                [1e-2, 3e-3, 1e-3, 3e-4],
                index=2,
                format_func=lambda x: str(x)
            )

        allow_gowex = st.checkbox("🧪 Permetre training encara que faltin mostres per classe", value=True)

        missing = [ch for ch, n in counts.items() if n < MIN_EXAMPLES_PER_CLASS]

        if missing and not allow_gowex:
            st.error(f"Falten mostres per: {', '.join(missing)}")
        elif total_samples < 2:
            st.error("Necessites almenys 2 crops guardats.")
        else:
            st.warning("Mode demo: si falten mostres, el model pot ser fum. Però serveix per validar pipeline.")

            if st.button("🚀 Entrenar CNN ara", type="primary", use_container_width=True):
                try:
                    with training_lock(owner=str(pdir.name)):
                        with st.spinner("Entrenant CNN..."):
                            metrics = train_char_cnn_for_project(
                                pdir=pdir,
                                config=config,
                                epochs=int(epochs),
                                batch_size=int(batch_size),
                                lr=float(lr),
                            )

                    config["status"] = "trained_demo"
                    save_project_config(pdir, config)

                    st.success("Training completat.")
                    st.json({
                        "device": metrics.get("device"),
                        "cuda": metrics.get("cuda"),
                        "gpu": metrics.get("gpu"),
                        "num_samples": metrics.get("num_samples"),
                        "model_path": metrics.get("model_path"),
                    })

                    hist = pd.DataFrame(metrics["history"])
                    st.line_chart(hist.set_index("epoch")[["train_loss", "val_loss"]])
                    st.line_chart(hist.set_index("epoch")[["train_acc", "val_acc"]])

                except Exception as e:
                    st.error(f"Error training: {e}")

        metrics_path = pdir / "reports" / "metrics_latest.json"
        if metrics_path.exists():
            st.markdown("### Últimes mètriques")
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                st.json({
                    "timestamp": metrics.get("timestamp"),
                    "device": metrics.get("device"),
                    "cuda": metrics.get("cuda"),
                    "gpu": metrics.get("gpu"),
                    "num_samples": metrics.get("num_samples"),
                    "model_path": metrics.get("model_path"),
                })

                hist = pd.DataFrame(metrics.get("history", []))
                if not hist.empty:
                    st.dataframe(hist.tail(10), use_container_width=True, hide_index=True)

            except Exception as e:
                st.warning(f"No puc llegir mètriques: {e}")


elif page == "06 · Test skill":
    st.subheader("06 · Test skill")
    st.caption("Test mínim: carregar model_latest.pt i predir sobre crops guardats o una imatge pujada.")

    pdir, config = select_project()

    if pdir:
        model_path = pdir / "models" / "model_latest.pt"

        if not model_path.exists():
            st.error("No hi ha model entrenat. Ves a 05 · Training.")
            st.stop()

        try:
            model, alphabet, device, ckpt = load_trained_model_for_project(pdir)
            st.success(f"Model carregat: {model_path}")
            st.caption(f"Device: {device} · alphabet: {alphabet}")
        except Exception as e:
            st.error(f"No puc carregar model: {e}")
            st.stop()

        samples_dir = pdir / "samples"

        st.markdown("### 1) Test amb una mostra guardada")

        sample_files = []
        for ch in alphabet:
            cdir = samples_dir / ch
            if cdir.exists():
                for p in sorted(cdir.glob("*.png")):
                    sample_files.append(p)

        if sample_files:
            sample_label = st.selectbox(
                "Mostra guardada",
                [str(p.relative_to(samples_dir)) for p in sample_files]
            )
            sample_path = samples_dir / sample_label

            img = Image.open(sample_path).convert("RGB")
            pred = predict_char_image(model, alphabet, device, img)

            c1, c2 = st.columns([1, 1])
            with c1:
                st.image(img, caption=f"Sample: {sample_label}", use_container_width=False)
            with c2:
                st.metric("Predicció", pred["pred"])
                st.metric("Confiança", f"{pred['confidence']*100:.1f}%")
                st.dataframe(pd.DataFrame(pred["top"]), use_container_width=True, hide_index=True)
        else:
            st.warning("No hi ha mostres guardades.")

        st.markdown("---")
        st.markdown("### 2) Test amb imatge/crop pujat")

        uploaded = st.file_uploader("Puja un crop de caràcter PNG/JPG", type=["png", "jpg", "jpeg"])

        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            pred = predict_char_image(model, alphabet, device, img)

            c1, c2 = st.columns([1, 1])
            with c1:
                st.image(img, caption="Crop pujat", use_container_width=False)
            with c2:
                st.metric("Predicció", pred["pred"])
                st.metric("Confiança", f"{pred['confidence']*100:.1f}%")
                st.dataframe(pd.DataFrame(pred["top"]), use_container_width=True, hide_index=True)


elif page == "07 · Run batch":
    st.subheader("07 · Run batch")
    st.caption("Runtime per Urkos: puja documents nous i la skill retorna el codi complet.")

    pdir, config = select_project()

    if pdir:
        model_path = pdir / "models" / "model_latest.pt"

        if not model_path.exists():
            st.error("No hi ha model entrenat. Ves primer a 05 · Training.")
            st.stop()

        try:
            model, alphabet, device, ckpt = load_trained_model_for_project(pdir)
            st.success(f"Model carregat: {model_path}")
            st.caption(f"Device: {device} · alphabet: {alphabet}")
        except Exception as e:
            st.error(f"No puc carregar el model: {e}")
            st.stop()

        fixed_length = int(config.get("fixed_length", 0) or 0)

        try:
            box_template = build_box_template_from_labels(pdir, fixed_length)
        except Exception as e:
            st.error(f"No puc construir plantilla de caixes: {e}")
            st.stop()

        trim_defaults = infer_default_trim_from_labels(pdir)

        st.markdown("### 1) Plantilla de caixes runtime")
        st.dataframe(pd.DataFrame(box_template), use_container_width=True, hide_index=True)

        st.markdown("### 2) Configuració runtime")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            use_auto_trim = st.checkbox("Auto-trim", value=trim_defaults.get("auto_trim", True))
        with c2:
            trim_threshold = st.slider("Threshold tinta", 180, 255, int(trim_defaults.get("trim_threshold", 245)), 1)
        with c3:
            trim_pad = st.slider("Padding trim px", 0, 30, int(trim_defaults.get("trim_pad", 4)), 1)
        with c4:
            min_conf_threshold = st.slider("Min conf OK", 0.0, 1.0, 0.70, 0.05)

        avg_conf_threshold = st.slider("Avg conf OK", 0.0, 1.0, 0.85, 0.05)

        st.markdown("### 3) Upload documents nous")
        uploaded = st.file_uploader(
            "Puja documents nous per processar",
            type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            accept_multiple_files=True,
        )

        if uploaded:
            st.info(f"Documents pujats: {len(uploaded)}")

            if st.button("▶️ Processar batch", type="primary", use_container_width=True):
                try:
                    run_dir = save_uploaded_batch_files(pdir, uploaded)

                    results = []
                    detail_store = {}

                    progress = st.progress(0.0, text="Processant batch...")

                    files = sorted([p for p in run_dir.iterdir() if p.is_file()])

                    for i, path in enumerate(files, start=1):
                        try:
                            pred = predict_document_code_from_path(
                                path=path,
                                config=config,
                                model=model,
                                alphabet=alphabet,
                                device=device,
                                box_template=box_template,
                                use_auto_trim=use_auto_trim,
                                trim_threshold=trim_threshold,
                                trim_pad=trim_pad,
                            )

                            status = "OK" if (
                                pred["min_conf"] >= min_conf_threshold
                                and pred["avg_conf"] >= avg_conf_threshold
                            ) else "REVIEW"

                            row = {
                                "fitxer": pred["file"],
                                "codi": pred["code"],
                                "min_conf": round(pred["min_conf"], 4),
                                "avg_conf": round(pred["avg_conf"], 4),
                                "status": status,
                            }

                            results.append(row)
                            detail_store[pred["file"]] = pred

                        except Exception as e:
                            results.append({
                                "fitxer": path.name,
                                "codi": "",
                                "min_conf": 0.0,
                                "avg_conf": 0.0,
                                "status": "ERROR",
                                "error": str(e),
                            })

                        progress.progress(i / max(1, len(files)), text=f"Processat {i}/{len(files)}")

                    st.session_state["run_batch_results"] = results
                    st.session_state["run_batch_details"] = detail_store
                    st.session_state["run_batch_dir"] = str(run_dir)

                    st.success("Batch processat.")

                except Exception as e:
                    st.error(f"Error processant batch: {e}")

        results = st.session_state.get("run_batch_results", [])
        details = st.session_state.get("run_batch_details", {})

        if results:
            st.markdown("---")
            st.markdown("### 4) Resultats batch")

            df_res = pd.DataFrame(results)
            st.dataframe(df_res, use_container_width=True, hide_index=True)

            csv_bytes = df_res.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "⬇️ Descarregar resultats CSV",
                data=csv_bytes,
                file_name="evo_skill_batch_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.markdown("### 5) Detall visual")

            ok_files = [r["fitxer"] for r in results if r.get("fitxer") in details]

            if ok_files:
                selected_file = st.selectbox("Fitxer per inspeccionar", ok_files)
                pred = details[selected_file]

                st.markdown(f"#### `{selected_file}` → `{pred['code']}`")
                st.caption(f"min_conf={pred['min_conf']:.3f} · avg_conf={pred['avg_conf']:.3f}")

                c_left, c_right = st.columns([1.0, 1.2], gap="large")

                with c_left:
                    st.image(pred["crop"], caption="ROI processada", use_container_width=False)

                with c_right:
                    rows = []
                    for d in pred["pos_details"]:
                        rows.append({
                            "pos": d["pos"],
                            "pred": d["pred"],
                            "confidence": round(d["confidence"], 4),
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                st.markdown("#### Crops per posició")
                cols = st.columns(min(len(pred["char_crops"]), 10))
                for i, ch_img in enumerate(pred["char_crops"]):
                    detail = pred["pos_details"][i]
                    with cols[i % min(len(pred["char_crops"]), 10)]:
                        st.image(ch_img, caption=f"{detail['pos']} · {detail['pred']} · {detail['confidence']*100:.0f}%", use_container_width=True)
