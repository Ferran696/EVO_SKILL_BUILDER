import streamlit as st
import streamlit.elements.image as st_image

# Patch de compatibilitat per a streamlit-drawable-canvas amb versions modernes de Streamlit
if not hasattr(st_image, "image_to_url"):
    try:
        from streamlit.elements.lib import image_utils

        def _compat_image_to_url(image, width_or_layout, clamp, channels, output_format, image_id):
            if isinstance(width_or_layout, (int, float)):
                layout_config = st_image.create_layout_config(width=int(width_or_layout))
            else:
                layout_config = width_or_layout
            return image_utils.image_to_url(image, layout_config, clamp, channels, output_format, image_id)

        st_image.image_to_url = _compat_image_to_url
    except Exception:
        pass

from streamlit_drawable_canvas import st_canvas
import re
import pandas as pd
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
import importlib
import core.storage as _storage_mod
importlib.reload(_storage_mod)
from core.storage import (
    create_project,
    list_projects,
    load_project_config,
    save_project_config,
    save_uploaded_files,
    build_renamed_filenames,
    build_batch_export_zip,
    export_batch_to_folder,
    cleanup_runtime_batch_dir,
    reset_batch_session_state,
)
from core.locks import is_locked, read_lock
from core.utils import load_first_page_image, crop_relative, draw_roi_overlay, split_fixed_slots, crop_box_pixels, draw_char_boxes_overlay, resize_for_zoom, trim_to_ink_bbox, slot_ink_ratio
import core.router as _router_mod
import core.segmentation as _segmentation_mod
importlib.reload(_router_mod)
importlib.reload(_segmentation_mod)
from core.router import classify_document_format, FORMAT_DEFINITIONS, locate_anchor_and_adjust_roi
from core.segmentation import (
    auto_split_characters_by_x_histogram,
    extract_characters_by_connected_components,
    segment_roi_characters,
    split_wide_segments_by_valleys,
    deskew_image,
)
from core.vlm import get_char_boxes_from_vlm
import core.labels as _labels_mod
importlib.reload(_labels_mod)
from core.labels import (
    append_label_rows,
    build_box_template_from_labels,
    apply_manual_review_update,
    save_corrected_crops_for_training,
)

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
    box_template: list = None,
    use_auto_trim: bool = True,
    trim_threshold: int = 245,
    trim_pad: int = 4,
    use_router: bool = True,
    segmentation_method: str = "connected_components",
    use_anchor: bool = True,
    use_deskew: bool = False,
    override_roi: dict = None,
) -> dict:
    """
    Prediu el codi complet d'un document nou.
    Suporta dos modes:
    1) Multi-format router (classificació dinàmica de plantilla + segmentació per components connectats / histograma + ancoratge)
    2) Template fixa (legacy: document -> ROI estàtica -> caixes plantilla -> CNN)
    Si es passa `override_roi`, s'utilitza aquesta ROI manual per a aquest document concret.
    """
    img = load_first_page_image(path, dpi=150)
    w, h = img.size

    deskew_angle = 0.0
    if use_deskew:
        img, deskew_angle = deskew_image(img)

    if use_router:
        format_result = classify_document_format(
            img,
            custom_rois=config.get("custom_rois"),
            auto_anchor=(use_anchor and override_roi is None),
        )
        roi = override_roi if override_roi is not None else format_result["roi"]
        expected_len = format_result["expected_length"]
        filter_dots = format_result["filter_dots"]
        crop_raw = crop_relative(img, roi)

        crops, boxes, debug = segment_roi_characters(
            roi_img=crop_raw,
            method=segmentation_method,
            threshold_method="otsu",
            min_char_width=3,
            min_ink_per_col=1,
            gap_merge_px=2,
            pad_x=1,
            pad_y=1,
            enable_valley_split=True,
            expected_count=expected_len,
            filter_dots=filter_dots,
        )

        chars = []
        confs = []
        pos_details = []
        char_crops = []

        for idx, (ch_img, b) in enumerate(zip(crops, boxes), start=1):
            pred = predict_char_image(model, alphabet, device, ch_img)
            chars.append(pred["pred"])
            confs.append(float(pred["confidence"]))
            char_crops.append(ch_img)
            pos_details.append({
                "pos": idx,
                "pred": pred["pred"],
                "confidence": float(pred["confidence"]),
                "box": {"x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3]},
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
            "crop": crop_raw,
            "char_crops": char_crops,
            "fixed_length": expected_len,
            "roi": roi,
            "format_id": format_result["format_id"],
            "format_name": format_result["format_info"]["name"],
            "format_confidence": format_result["confidence"],
            "format_details": format_result,
            "segmentation_method": segmentation_method,
            "use_anchor": use_anchor,
            "deskew_angle": deskew_angle,
        }

    # Legacy template mode
    roi = override_roi if override_roi is not None else config.get("roi")
    fixed_length = int(config.get("fixed_length", 0) or 0)

    if not roi:
        raise RuntimeError("Aquest projecte no té ROI guardada.")

    crop_raw = crop_relative(img, roi)
    if use_auto_trim:
        crop = trim_to_ink_bbox(crop_raw, threshold=int(trim_threshold), pad_px=int(trim_pad))
    else:
        crop = crop_raw.convert("RGB")

    chars = []
    confs = []
    pos_details = []
    char_crops = []

    template_boxes = box_template or []
    for b in template_boxes:
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
        "roi": roi,
        "format_id": "legacy_fixed_template",
        "format_name": "Plantilla fixa estàndard",
        "format_confidence": 1.0,
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

            doc_fmt = classify_document_format(img, custom_rois=config.get("custom_rois"))
            fmt_info = doc_fmt["format_info"]
            st.info(f"Format detectat pel Router: **{fmt_info['name']}** (Longitud esperada: **{doc_fmt['expected_length']}** caràcters)")

            current_roi = doc_fmt["roi"] or config.get("roi") or {
                "x0": 0.50,
                "y0": 0.00,
                "x1": 0.98,
                "y1": 0.25,
            }

            st.markdown("### 1) Ajust visual de ROI")
            st.info("Mou sliders i mira el rectangle vermell i el crop ampliat. Quan el crop contingui només el camp/codi, guarda ROI.")

            # Presets ràpids per no començar des de zero
            p0, p1, p2, p3 = st.columns(4)
            if p0.button("Preset: Recomanat format"):
                current_roi = doc_fmt["roi"]
            if p1.button("Preset: superior dreta"):
                current_roi = {"x0": 0.55, "y0": 0.00, "x1": 0.98, "y1": 0.22}
            if p2.button("Preset: capçalera ampla"):
                current_roi = {"x0": 0.25, "y0": 0.00, "x1": 0.98, "y1": 0.28}
            if p3.button("Preset: full ample"):
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
                    if "custom_rois" not in config or not isinstance(config["custom_rois"], dict):
                        config["custom_rois"] = {}
                    config["custom_rois"][doc_fmt["format_id"]] = roi
                    config["status"] = "roi_defined"
                    save_project_config(pdir, config)
                    st.success(f"ROI guardada per al format {fmt_info['name']}.")
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
                vlm_fmt = classify_document_format(img_vlm, custom_rois=config.get("custom_rois"))
                vlm_expected_len = vlm_fmt.get("expected_length", fixed_length)
                vlm_roi = vlm_fmt.get("roi", roi)
                crop_vlm = crop_relative(img_vlm, vlm_roi)

                try:
                    with st.spinner("El VLM està pensant..."):
                        vlm_result = get_char_boxes_from_vlm(crop_vlm, vlm_prompt, vlm_api_key, vlm_expected_len)

                    st.success("VLM ha retornat les caixes!")
                    st.json(vlm_result)

                    if "characters" in vlm_result and len(vlm_result["characters"]) == vlm_expected_len:
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
            doc_fmt = classify_document_format(img, custom_rois=config.get("custom_rois"))
            doc_expected_len = doc_fmt.get("expected_length", fixed_length)
            doc_roi = doc_fmt.get("roi", roi)
            st.info(f"Format detectat: **{doc_fmt['format_info']['name']}** (Esperats: **{doc_expected_len}** caràcters)")

            crop_raw = crop_relative(img, doc_roi)

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
                    expected_count=doc_expected_len,
                    filter_dots=doc_fmt.get("filter_dots", False),
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
                                    "box_x0": box[0], "box_y0": box[1], "box_x1": box[2], "box_y1": box[3],
                                    "roi_x0": roi.get("x0"), "roi_y0": roi.get("y0"), "roi_x1": roi.get("x1"), "roi_y1": roi.get("y1"),
                                    "hist_threshold_method": threshold_method, "hist_min_char_width": min_char_width,
                                    "hist_min_ink_per_col": min_ink_per_col, "hist_gap_merge_px": gap_merge_px,
                                    "hist_pad_x": pad_x, "hist_pad_y": pad_y, "hist_apply_morph_open": apply_morph_open,
                                    "hist_apply_morph_close": apply_morph_close,
                                }
                                saved_rows.append(row)

                            if saved_rows:
                                append_label_rows(labels_csv, saved_rows)

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
        doc_fmt = classify_document_format(img, custom_rois=config.get("custom_rois"))
        doc_expected_len = doc_fmt.get("expected_length", fixed_length)
        doc_roi = doc_fmt.get("roi", roi)
        st.info(f"Format detectat: **{doc_fmt['format_info']['name']}** (Esperats: **{doc_expected_len}** caràcters)")

        crop_raw = crop_relative(img, doc_roi)

        c_cfg1, c_cfg2, c_cfg3, c_cfg4 = st.columns(4)

        with c_cfg1:
            use_auto_trim = st.checkbox("Auto-trim ROI", value=True)
        with c_cfg2:
            trim_threshold = st.slider("Threshold tinta", 180, 255, 245, 1)
        with c_cfg3:
            trim_pad = st.slider("Padding trim px", 0, 30, 4, 1)
        with c_cfg4:
            # Zoom adaptatiu per defecte per evitar que el canvas es talli a la columna
            crop_sample = trim_to_ink_bbox(crop_raw, threshold=trim_threshold, pad_px=trim_pad) if use_auto_trim else crop_raw
            w_sample = max(1, crop_sample.size[0])
            ideal_zoom = min(6, max(2, int(680 / w_sample)))
            zoom = st.slider("Zoom canvas", 1, 12, ideal_zoom, 1)

        if use_auto_trim:
            crop = trim_to_ink_bbox(crop_raw, threshold=trim_threshold, pad_px=trim_pad)
        else:
            crop = crop_raw.convert("RGB")

        cw, ch = crop.size

        truth_raw = st.text_input(
            f"Valor real complet ({doc_expected_len} caràcters per a {doc_fmt['format_info']['name']})",
            key=f"truth_canvas_{pdir.name}_{path.name}",
            placeholder=f"P.ex. {'0528158' if doc_expected_len == 7 else '515159'}",
        )

        truth = "".join([c for c in truth_raw.strip().upper() if c in alphabet])

        if truth_raw and truth != truth_raw.strip().upper():
            st.warning(f"S'han eliminat caràcters fora de l'alfabet. Valor net: {truth}")

        valid_truth = len(truth) > 0 and (len(truth) == doc_expected_len or len(truth) in [6, 7] or len(truth) == fixed_length)
        target_length = len(truth) if (truth and len(truth) > 0) else doc_expected_len

        if truth and not valid_truth:
            st.error(f"Longitud incorrecta: {len(truth)}. S'esperen {doc_expected_len} caràcters per a {doc_fmt['format_info']['name']}.")
        elif valid_truth:
            st.success(f"Valor vàlid: {truth} ({len(truth)} caràcters)")

        # Estat posició activa
        active_key = f"active_char_idx_canvas_{pdir.name}_{path.name}"
        if active_key not in st.session_state:
            st.session_state[active_key] = 0

        active_idx = int(st.session_state[active_key])
        active_idx = max(0, min(target_length - 1, active_idx))
        expected_label = truth[active_idx] if (valid_truth and active_idx < len(truth)) else ""

        canvas_img = resize_for_zoom(crop, zoom=zoom)
        canvas_w, canvas_h = canvas_img.size

        c_left, c_right = st.columns([1.6, 1.0], gap="large")

        with c_left:
            st.markdown("#### ROI raw")
            st.image(crop_raw, caption=f"ROI original · {crop_raw.size[0]} x {crop_raw.size[1]} px", use_container_width=False)

            st.markdown("#### Canvas ampliat")
            st.caption(f"Dibuixa UN rectangle al voltant del caràcter actiu (Canvas: {canvas_w}x{canvas_h} px). Si queda tallat, redueix el 'Zoom canvas'.")

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
                max_value=max(1, target_length),
                value=active_idx + 1,
                step=1,
            )

            new_idx = int(pos) - 1
            if new_idx != active_idx:
                st.session_state[active_key] = new_idx
                st.rerun()

            if valid_truth:
                st.info(f"Posició {active_idx + 1}/{target_length} → etiqueta: `{expected_label}`")
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

            if nav2.button("➡️ Següent", use_container_width=True, disabled=active_idx >= target_length - 1):
                st.session_state[active_key] = min(target_length - 1, active_idx + 1)
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

                    append_label_rows(labels_csv, [row])

                    config["status"] = "samples_labeling_canvas"
                    save_project_config(pdir, config)

                    st.success(f"Guardat pos {active_idx+1}: `{expected_label}`")

                    # auto-next i canvas net
                    if active_idx < target_length - 1:
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

        st.markdown("### 1) Plantilla de caixes runtime i Mode d'execució")
        inference_mode = st.radio(
            "Estratègia de lectura",
            [
                "🔄 Enrutador multi-format automàtic (Router + Segmentació Dinàmica)",
                "📐 Plantilla de caixes fixa (Legacy)"
            ],
            index=0,
            help="L'enrutador classifica automàticament si és Tipus 1 (Barcode), Tipus 2 (Logo Carburos) o Tipus 3 (Excepció amb punts filtrats)."
        )
        use_router = inference_mode.startswith("🔄")

        if not use_router:
            st.dataframe(pd.DataFrame(box_template), use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ Mode Multi-Format actiu: s'identificarà cada document per plantilla (Tipus 1: 7 caràcters, Tipus 2: 6 caràcters, Tipus 3: 6 caràcters nets filtrant punts).")

        st.markdown("### 2) Configuració runtime")

        if use_router:
            with st.expander("⚙️ Opcions avançades de lectura (Segmentació i Alineació)", expanded=True):
                f_col1, f_col2, f_col3 = st.columns(3)
                with f_col1:
                    seg_label = st.selectbox(
                        "Mètode de segmentació",
                        [
                            "🔲 Components Connectats (Robust a desplaçaments)",
                            "📊 Histograma X clàssic (Projecció vertical)",
                        ],
                        index=0,
                        help="Components Connectats analitza formes i descarta taques o textos paràsits de manera intel·ligent."
                    )
                    segmentation_method = "connected_components" if seg_label.startswith("🔲") else "histogram"
                with f_col2:
                    use_anchor = st.checkbox(
                        "🎯 Ancoratge visual automàtic",
                        value=True,
                        help="Re-centra la ROI automàticament trobant el codi de barres, logo o línia de text si l'escaneig està mogut."
                    )
                with f_col3:
                    use_deskew = st.checkbox(
                        "📐 Deskewing (Alineació d'inclinació)",
                        value=False,
                        help="Detecta i corregeix petits angles de rotació en documents escanejats torts."
                    )
        else:
            segmentation_method = "histogram"
            use_anchor = False
            use_deskew = False

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            use_auto_trim = st.checkbox("Auto-trim", value=trim_defaults.get("auto_trim", True), disabled=use_router)
        with c2:
            trim_threshold = st.slider("Threshold tinta", 180, 255, int(trim_defaults.get("trim_threshold", 245)), 1, disabled=use_router)
        with c3:
            trim_pad = st.slider("Padding trim px", 0, 30, int(trim_defaults.get("trim_pad", 4)), 1, disabled=use_router)
        with c4:
            min_conf_threshold = st.slider("Min conf OK", 0.0, 1.0, 0.70, 0.05)

        avg_conf_threshold = st.slider("Avg conf OK", 0.0, 1.0, 0.85, 0.05)

        st.markdown("### 3) Upload documents nous")

        reset_flash = st.session_state.pop("batch_reset_flash", None)
        if reset_flash:
            st.success(reset_flash)

        uploader_rev = int(st.session_state.get("batch_uploader_rev", 0))
        uploaded = st.file_uploader(
            "Puja documents nous per processar",
            type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            accept_multiple_files=True,
            key=f"batch_uploader_{pdir.name}_{uploader_rev}",
        )

        has_active_batch = bool(st.session_state.get("run_batch_results")) or bool(uploaded)

        if uploaded:
            st.info(f"Documents pujats: {len(uploaded)}")

        act_col1, act_col2 = st.columns([2.2, 1.0], gap="medium")
        with act_col1:
            run_clicked = st.button(
                "▶️ Processar batch",
                type="primary",
                disabled=(not uploaded),
                use_container_width=True,
                key="btn_process_batch_main",
            )
        with act_col2:
            if st.button(
                "🧹 Netejar / Següent Batch (Reset)",
                disabled=(not has_active_batch),
                use_container_width=True,
                help=(
                    "Buida la llista de fitxers pujats, neteja la taula de resultats i el visor, "
                    "i esborra els temporals d'execució mantenint intactes les exportacions i mostres."
                ),
                key="btn_reset_batch_top",
            ):
                cleanup_info = reset_batch_session_state(st.session_state, pdir=pdir, clean_disk_temp=True)
                st.session_state["batch_reset_flash"] = (
                    f"🧹 Lot netejat! Carregador buidat i eliminades {cleanup_info['removed_count']} carpeta(es) temporal(s). "
                    "Ja pots arrossegar el següent batch de documents."
                )
                st.rerun()

        if uploaded and run_clicked:
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
                            use_router=use_router,
                            segmentation_method=segmentation_method,
                            use_anchor=use_anchor,
                            use_deskew=use_deskew,
                        )

                        status = "OK" if (
                            pred["min_conf"] >= min_conf_threshold
                            and pred["avg_conf"] >= avg_conf_threshold
                        ) else "REVIEW"

                        row = {
                            "exportar": status == "OK",
                            "fitxer": pred["file"],
                            "codi": pred["code"],
                            "format": pred.get("format_name", "N/A"),
                            "min_conf": round(pred["min_conf"], 4),
                            "avg_conf": round(pred["avg_conf"], 4),
                            "status": status,
                        }

                        results.append(row)
                        detail_store[pred["file"]] = pred

                    except Exception as e:
                        results.append({
                            "exportar": False,
                            "fitxer": path.name,
                            "codi": "",
                            "format": "ERROR",
                            "min_conf": 0.0,
                            "avg_conf": 0.0,
                            "status": "ERROR",
                            "error": str(e),
                        })

                    progress.progress(i / max(1, len(files)), text=f"Processat {i}/{len(files)}")

                st.session_state["run_batch_results"] = results
                st.session_state["run_batch_details"] = detail_store
                st.session_state["run_batch_dir"] = str(run_dir)
                st.session_state["run_batch_editor_rev"] = st.session_state.get("run_batch_editor_rev", 0) + 1

                st.success("Batch processat.")

            except Exception as e:
                st.error(f"Error processant batch: {e}")

        results = st.session_state.get("run_batch_results", [])
        details = st.session_state.get("run_batch_details", {})
        run_batch_dir = st.session_state.get("run_batch_dir", "")

        if results:
            st.markdown("---")
            st.markdown("### 4) Resultats batch, Reindexació i Exportació")
            st.caption("Revisa o edita els codis detectats, marca els documents que vols reindexar i exporta'ls ja reanomenats.")

            # Garantir compatibilitat si hi ha resultats previs en sessió sense camp 'exportar'
            for r in results:
                if "exportar" not in r:
                    r["exportar"] = (r.get("status") == "OK" and bool(r.get("codi")))

            cfg_c1, cfg_c2, sel_c1, sel_c2, sel_c3 = st.columns([1.6, 1.2, 0.9, 0.9, 0.9])
            with cfg_c1:
                pattern_label = st.selectbox(
                    "Patró de nou nom",
                    [
                        "{codi}.ext (p.ex. 0529820.pdf)",
                        "{prefix}{codi}.ext (p.ex. ALB_0529820.pdf)",
                        "{codi}_{original}.ext (p.ex. 0529820_18.pdf)",
                    ],
                    index=0,
                    key="batch_rename_pattern_select",
                )
                if pattern_label.startswith("{prefix}"):
                    pattern_key = "{prefix}{codi}"
                elif pattern_label.startswith("{codi}_{original}"):
                    pattern_key = "{codi}_{original}"
                else:
                    pattern_key = "{codi}"

            with cfg_c2:
                prefix_val = st.text_input(
                    "Prefix opcional",
                    value="ALB_",
                    disabled=(pattern_key != "{prefix}{codi}"),
                    key="batch_rename_prefix_input",
                )

            with sel_c1:
                st.write("")
                if st.button("✅ Seleccionar tots", use_container_width=True):
                    for r in st.session_state["run_batch_results"]:
                        r["exportar"] = (r.get("status") != "ERROR" and bool(str(r.get("codi", "")).strip()))
                    st.session_state["run_batch_editor_rev"] = st.session_state.get("run_batch_editor_rev", 0) + 1
                    st.rerun()

            with sel_c2:
                st.write("")
                if st.button("☑️ Només OK", use_container_width=True):
                    for r in st.session_state["run_batch_results"]:
                        r["exportar"] = (r.get("status") in ("OK", "MANUAL_OK"))
                    st.session_state["run_batch_editor_rev"] = st.session_state.get("run_batch_editor_rev", 0) + 1
                    st.rerun()

            with sel_c3:
                st.write("")
                if st.button("🚫 Desmarcar tots", use_container_width=True):
                    for r in st.session_state["run_batch_results"]:
                        r["exportar"] = False
                    st.session_state["run_batch_editor_rev"] = st.session_state.get("run_batch_editor_rev", 0) + 1
                    st.rerun()

            rows_for_editor = build_renamed_filenames(
                results,
                pattern=pattern_key,
                prefix=prefix_val if pattern_key == "{prefix}{codi}" else "",
            )

            col_order = ["exportar", "fitxer", "codi", "nou_nom", "format", "min_conf", "avg_conf", "status"]
            if any("error" in r for r in rows_for_editor):
                col_order.append("error")

            df_editor_in = pd.DataFrame(rows_for_editor)
            for c in col_order:
                if c not in df_editor_in.columns:
                    df_editor_in[c] = ""
            df_editor_in = df_editor_in[col_order]

            editor_rev = st.session_state.get("run_batch_editor_rev", 0)
            edited_df = st.data_editor(
                df_editor_in,
                column_config={
                    "exportar": st.column_config.CheckboxColumn(
                        "Exportar",
                        help="Marca per reindexar i exportar aquest document",
                        default=True,
                    ),
                    "fitxer": st.column_config.TextColumn("Fitxer original", disabled=True),
                    "codi": st.column_config.TextColumn(
                        "Codi (editable)",
                        help="Pots corregir manualment el codi predit abans d'exportar",
                    ),
                    "nou_nom": st.column_config.TextColumn(
                        "Nou nom fitxer",
                        help="Nom final amb gestió automàtica de col·lisions",
                        disabled=True,
                    ),
                    "format": st.column_config.TextColumn("Format", disabled=True),
                    "min_conf": st.column_config.NumberColumn("Min conf", format="%.4f", disabled=True),
                    "avg_conf": st.column_config.NumberColumn("Avg conf", format="%.4f", disabled=True),
                    "status": st.column_config.TextColumn("Status", disabled=True),
                },
                use_container_width=True,
                hide_index=True,
                key=f"batch_results_editor_{editor_rev}",
            )

            # Sincronitzar edicions d'usuari (checkbox 'exportar' i correccions de 'codi')
            edited_records = edited_df.to_dict(orient="records")
            code_changed = False
            for idx, rec in enumerate(edited_records):
                if idx < len(results):
                    new_exp = bool(rec.get("exportar", False))
                    new_code = str(rec.get("codi", "") or "").strip()
                    if results[idx].get("codi") != new_code:
                        code_changed = True
                        if new_code:
                            new_exp = True
                            if results[idx].get("status") in ("REVIEW", "ERROR"):
                                results[idx]["status"] = "MANUAL_OK"
                    results[idx]["exportar"] = new_exp
                    results[idx]["codi"] = new_code
                    fname = results[idx].get("fitxer")
                    if fname and fname in details:
                        details[fname]["code"] = new_code

            final_rows = build_renamed_filenames(
                results,
                pattern=pattern_key,
                prefix=prefix_val if pattern_key == "{prefix}{codi}" else "",
            )
            if code_changed:
                st.session_state["run_batch_editor_rev"] = editor_rev + 1
                st.rerun()

            df_res = pd.DataFrame(final_rows)
            selected_rows = [r for r in final_rows if bool(r.get("exportar", False))]
            n_selected = len(selected_rows)

            st.info(f"📌 **{n_selected} de {len(final_rows)}** documents seleccionats per reindexar i exportar.")

            exp_col1, exp_col2, exp_col3, exp_col4 = st.columns([1.2, 1.1, 1.0, 1.1], gap="medium")

            with exp_col1:
                if run_batch_dir and Path(run_batch_dir).exists() and n_selected > 0:
                    zip_bytes = build_batch_export_zip(Path(run_batch_dir), final_rows)
                    batch_ts = Path(run_batch_dir).name
                    st.download_button(
                        f"📦 Descarregar ZIP reanomenat ({n_selected} docs)",
                        data=zip_bytes,
                        file_name=f"export_{pdir.name}_{batch_ts}.zip",
                        mime="application/zip",
                        type="primary",
                        use_container_width=True,
                    )
                else:
                    st.button(
                        "📦 Descarregar ZIP reanomenat (0 docs)",
                        disabled=True,
                        use_container_width=True,
                    )

            with exp_col2:
                csv_bytes = df_res.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    "⬇️ Descarregar taula resultats CSV",
                    data=csv_bytes,
                    file_name="evo_skill_batch_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with exp_col3:
                default_export_dir = str(pdir / "exports" / (Path(run_batch_dir).name if run_batch_dir else "latest"))
                with st.popover("📁 Exportar a carpeta local", use_container_width=True):
                    target_dir_str = st.text_input(
                        "Ruta carpeta destinació",
                        value=default_export_dir,
                        key="batch_export_target_dir",
                    )
                    if st.button(
                        f"💾 Copiar {n_selected} fitxers reanomenats",
                        type="primary",
                        disabled=(n_selected == 0 or not run_batch_dir or not Path(run_batch_dir).exists()),
                        use_container_width=True,
                    ):
                        try:
                            summary = export_batch_to_folder(
                                source_dir=Path(run_batch_dir),
                                target_dir=Path(target_dir_str),
                                rows=final_rows,
                            )
                            st.success(f"Exportats {summary['copied_count']} fitxers a:\n`{summary['target_dir']}`")
                        except Exception as e:
                            st.error(f"Error exportant a carpeta: {e}")

            with exp_col4:
                if st.button(
                    "🧹 Següent Batch (Reset)",
                    use_container_width=True,
                    help=(
                        "Un cop exportat el lot, prem aquí per buidar el carregador de fitxers, "
                        "netejar la taula i el visor, i eliminar els temporals per començar el següent batch."
                    ),
                    key="btn_reset_batch_bottom",
                ):
                    cleanup_info = reset_batch_session_state(st.session_state, pdir=pdir, clean_disk_temp=True)
                    st.session_state["batch_reset_flash"] = (
                        f"🧹 Lot finalitzat i netejat! Carregador buidat i eliminades {cleanup_info['removed_count']} carpeta(es) temporal(s). "
                        "Ja pots arrossegar el següent batch de documents."
                    )
                    st.rerun()

            st.markdown("---")
            st.markdown("### 5) Estació de Revisió Ràpida (`REVIEW`) i Detall Visual")
            st.caption("Inspecciona, corregeix amb un clic, reajusta la ROI de documents rebels o guarda dígits corregits per entrenar.")

            flash_msg = st.session_state.pop("batch_review_flash", None)
            if flash_msg:
                st.success(flash_msg)

            pending_rows = [r for r in results if r.get("status") in ("REVIEW", "ERROR")]
            n_pending = len(pending_rows)
            n_manual_ok = sum(1 for r in results if r.get("status") == "MANUAL_OK")

            if n_pending > 0:
                st.warning(
                    f"⚠️ Hi ha **{n_pending} document(s) pendent(s) de revisió** (`REVIEW` / `ERROR`). "
                    + (f"({n_manual_ok} ja validats manualment)" if n_manual_ok else "")
                )
            else:
                st.success(
                    f"🎉 Tots els documents estan validats (`OK` / `MANUAL_OK`)"
                    + (f" · {n_manual_ok} corregits manualment." if n_manual_ok else ".")
                )

            rev_f1, rev_f2 = st.columns([1.4, 2.0])
            with rev_f1:
                filter_mode = st.radio(
                    "🔍 Mostrar documents",
                    [
                        f"⚠️ Només pendents ({n_pending})",
                        f"📋 Tots els documents ({len(results)})",
                    ],
                    index=0 if n_pending > 0 else 1,
                    horizontal=True,
                    key="batch_review_filter_radio",
                )
                only_pending = filter_mode.startswith("⚠️")

            filtered_rows = pending_rows if only_pending else results
            filtered_files = [r["fitxer"] for r in filtered_rows if r.get("fitxer")]

            if not filtered_files:
                st.info("✅ No queda cap document pendent a la cua `REVIEW`. Canvia a `📋 Tots els documents` si vols inspeccionar la resta.")
            else:
                cur_sel = st.session_state.get("batch_review_selected_file")
                if cur_sel not in filtered_files:
                    cur_sel = filtered_files[0]
                    st.session_state["batch_review_selected_file"] = cur_sel

                cur_idx = filtered_files.index(cur_sel)

                nav_c1, nav_c2, nav_c3 = st.columns([0.6, 2.2, 0.6])
                with nav_c1:
                    st.write("")
                    if st.button("⬅️ Anterior", disabled=(cur_idx <= 0), use_container_width=True, key="btn_rev_prev"):
                        st.session_state["batch_review_selected_file"] = filtered_files[max(0, cur_idx - 1)]
                        st.rerun()
                with nav_c2:
                    def _fmt_doc_option(fname: str) -> str:
                        r_match = next((x for x in results if x.get("fitxer") == fname), {})
                        st_tag = r_match.get("status", "")
                        icon = "⚠️" if st_tag in ("REVIEW", "ERROR") else ("🛠️" if st_tag == "MANUAL_OK" else "✅")
                        return f"{icon} {fname}  →  {r_match.get('codi') or '(buit)'}  [{st_tag}]"

                    selected_file = st.selectbox(
                        f"Document ({cur_idx + 1}/{len(filtered_files)})",
                        filtered_files,
                        index=cur_idx,
                        format_func=_fmt_doc_option,
                        key="batch_review_selectbox_widget",
                    )
                    if selected_file != cur_sel:
                        st.session_state["batch_review_selected_file"] = selected_file
                        st.rerun()
                with nav_c3:
                    st.write("")
                    if st.button("➡️ Següent", disabled=(cur_idx >= len(filtered_files) - 1), use_container_width=True, key="btn_rev_next"):
                        st.session_state["batch_review_selected_file"] = filtered_files[min(len(filtered_files) - 1, cur_idx + 1)]
                        st.rerun()

                row_obj = next((r for r in results if r.get("fitxer") == selected_file), {})
                pred = details.get(selected_file, {})
                row_status = row_obj.get("status", "REVIEW")
                doc_path = Path(run_batch_dir) / selected_file if run_batch_dir else None
                has_doc_file = bool(doc_path and doc_path.exists())

                c_vis, c_edit = st.columns([1.25, 1.0], gap="large")

                with c_vis:
                    st.markdown(f"#### 👁️ Visor: `{selected_file}`")
                    if pred.get("crop") is not None:
                        st.image(pred["crop"], caption="ROI processada per la CNN", use_container_width=False)
                    else:
                        st.warning("No hi ha imatge de ROI disponible per a aquest document.")

                    char_crops_list = pred.get("char_crops", [])
                    pos_details_list = pred.get("pos_details", [])
                    if char_crops_list:
                        st.markdown("**Crops segmentats per posició:**")
                        cols = st.columns(min(max(1, len(char_crops_list)), 10))
                        for i, ch_img in enumerate(char_crops_list):
                            detail = pos_details_list[i] if i < len(pos_details_list) else {"pos": i + 1, "pred": "?", "confidence": 0.0}
                            with cols[i % min(max(1, len(char_crops_list)), 10)]:
                                st.image(
                                    ch_img,
                                    caption=f"#{detail['pos']} · `{detail['pred']}` ({detail['confidence']*100:.0f}%)",
                                    use_container_width=True,
                                )
                    else:
                        st.info("No s'ha extret cap caràcter d'aquesta ROI.")

                    if has_doc_file:
                        expand_header = (row_status in ("REVIEW", "ERROR") or not char_crops_list)
                        with st.expander("📄 Veure capçalera del document original (amb ROI marcada)", expanded=expand_header):
                            try:
                                full_img = load_first_page_image(doc_path, dpi=130)
                                used_roi = pred.get("roi") or config.get("roi") or {"x0": 0.5, "y0": 0.0, "x1": 0.98, "y1": 0.25}
                                overlay_img = draw_roi_overlay(full_img, used_roi)
                                header_y1 = min(1.0, max(0.32, float(used_roi.get("y1", 0.25)) + 0.10))
                                header_crop = crop_relative(overlay_img, {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": header_y1})
                                st.image(header_crop, caption="Capçalera del document (rectangle vermell = ROI actual)", use_container_width=True)
                            except Exception as e:
                                st.warning(f"No s'ha pogut mostrar la capçalera: {e}")

                with c_edit:
                    st.markdown("#### ✏️ Correcció i Validació")
                    st_badge = {
                        "OK": "🟢 **OK** (Automàtic)",
                        "MANUAL_OK": "🔵 **MANUAL_OK** (Validat manualment)",
                        "REVIEW": "🟠 **REVIEW** (Pendent de revisar)",
                        "ERROR": "🔴 **ERROR**",
                    }.get(row_status, row_status)
                    st.markdown(f"**Estat actual:** {st_badge}")
                    if "format_name" in pred:
                        st.caption(
                            f"Format: **{pred['format_name']}** · "
                            f"min_conf=`{pred.get('min_conf', 0.0):.3f}` · avg_conf=`{pred.get('avg_conf', 0.0):.3f}`"
                        )
                    if row_obj.get("error"):
                        st.error(f"Detall error: {row_obj['error']}")

                    manual_code_val = st.text_input(
                        "Codi del document",
                        value=str(row_obj.get("codi", "") or ""),
                        placeholder="Escriu o corregeix el codi aquí...",
                        key=f"manual_code_input_{selected_file}_{editor_rev}",
                    )
                    clean_manual_code = manual_code_val.strip()

                    def _advance_to_next_pending(current_fname: str):
                        remaining = [
                            r["fitxer"]
                            for r in st.session_state.get("run_batch_results", [])
                            if r.get("status") in ("REVIEW", "ERROR") and r.get("fitxer") != current_fname
                        ]
                        if remaining:
                            st.session_state["batch_review_selected_file"] = remaining[0]
                        elif cur_idx + 1 < len(filtered_files):
                            st.session_state["batch_review_selected_file"] = filtered_files[cur_idx + 1]

                    if st.button(
                        "✅ Validar codi i marcar per exportar",
                        type="primary",
                        disabled=(not clean_manual_code),
                        use_container_width=True,
                        key=f"btn_validate_manual_{selected_file}",
                    ):
                        apply_manual_review_update(
                            results=st.session_state["run_batch_results"],
                            details=st.session_state["run_batch_details"],
                            filename=selected_file,
                            new_code=clean_manual_code,
                            mark_export=True,
                        )
                        st.session_state["run_batch_editor_rev"] = editor_rev + 1
                        st.session_state["batch_review_flash"] = (
                            f"✅ Document `{selected_file}` validat amb el codi `{clean_manual_code}` (MANUAL_OK i marcat per exportar)."
                        )
                        _advance_to_next_pending(selected_file)
                        st.rerun()

                    can_save_samples = (
                        len(char_crops_list) > 0
                        and len(clean_manual_code) == len(char_crops_list)
                        and all(c.upper() in alphabet for c in clean_manual_code)
                    )
                    if st.button(
                        "🧠 Validar + Guardar dígits per entrenar (Active Learning)",
                        disabled=(not can_save_samples),
                        use_container_width=True,
                        help=(
                            "Valida el codi per exportar i alhora guarda els retalls d'aquests dígits a samples/ i labels.csv "
                            "perquè el model aprengui en el pròxim entrenament."
                        ),
                        key=f"btn_validate_and_train_{selected_file}",
                    ):
                        try:
                            n_saved = save_corrected_crops_for_training(
                                pdir=pdir,
                                config=config,
                                source_doc=selected_file,
                                truth_code=clean_manual_code,
                                char_crops=char_crops_list,
                                pos_details=pos_details_list,
                                roi=pred.get("roi") or config.get("roi"),
                                alphabet=alphabet,
                            )
                            apply_manual_review_update(
                                results=st.session_state["run_batch_results"],
                                details=st.session_state["run_batch_details"],
                                filename=selected_file,
                                new_code=clean_manual_code,
                                mark_export=True,
                            )
                            st.session_state["run_batch_editor_rev"] = editor_rev + 1
                            st.session_state["batch_review_flash"] = (
                                f"🧠 Validat `{selected_file}` → `{clean_manual_code}` i guardats **{n_saved} dígits** a `samples/` per entrenar!"
                            )
                            _advance_to_next_pending(selected_file)
                            st.rerun()
                        except Exception as e:
                            st.error(f"No s'han pogut guardar les mostres: {e}")

                    if char_crops_list and len(clean_manual_code) != len(char_crops_list):
                        st.caption(
                            f"ℹ️ Per guardar dígits d'entrenament cal que la longitud del codi ({len(clean_manual_code)}) "
                            f"coincideixi amb els crops detectats ({len(char_crops_list)})."
                        )

                    if pos_details_list:
                        with st.expander("📊 Detall de confiança per dígit", expanded=False):
                            rows_conf = [
                                {
                                    "pos": d["pos"],
                                    "pred": d["pred"],
                                    "confidence": round(d["confidence"], 4),
                                }
                                for d in pos_details_list
                            ]
                            st.dataframe(pd.DataFrame(rows_conf), use_container_width=True, hide_index=True)

                if has_doc_file:
                    with st.expander("📐 Reajustar ROI d'aquest document rebel i re-llegir amb la CNN", expanded=False):
                        st.caption(
                            "Si aquest PDF s'ha escanejat tan desplaçat que la ROI no ha agafat bé el número, "
                            "ajusta les coordenades aquí i prem **Re-llegir**."
                        )
                        base_doc_roi = pred.get("roi") or config.get("roi") or {"x0": 0.75, "y0": 0.05, "x1": 0.96, "y1": 0.14}
                        r_c1, r_c2 = st.columns([1.0, 1.2], gap="large")
                        with r_c1:
                            rx0 = st.slider(
                                "x0 · esquerra", 0.0, 1.0, float(base_doc_roi.get("x0", 0.75)), 0.005,
                                key=f"rev_roi_x0_{selected_file}",
                            )
                            ry0 = st.slider(
                                "y0 · dalt", 0.0, 1.0, float(base_doc_roi.get("y0", 0.05)), 0.005,
                                key=f"rev_roi_y0_{selected_file}",
                            )
                            rx1 = st.slider(
                                "x1 · dreta", 0.0, 1.0, float(base_doc_roi.get("x1", 0.96)), 0.005,
                                key=f"rev_roi_x1_{selected_file}",
                            )
                            ry1 = st.slider(
                                "y1 · baix", 0.0, 1.0, float(base_doc_roi.get("y1", 0.14)), 0.005,
                                key=f"rev_roi_y1_{selected_file}",
                            )
                            sx0, sx1 = sorted([rx0, rx1])
                            sy0, sy1 = sorted([ry0, ry1])
                            custom_doc_roi = {"x0": sx0, "y0": sy0, "x1": sx1, "y1": sy1}

                            save_as_format_default = st.checkbox(
                                "Guardar també aquesta ROI com a plantilla per al format",
                                value=False,
                                key=f"rev_roi_save_fmt_{selected_file}",
                            )

                            if st.button(
                                "🔄 Re-llegir aquest document amb la nova ROI",
                                type="primary",
                                use_container_width=True,
                                key=f"btn_repredict_roi_{selected_file}",
                            ):
                                try:
                                    new_pred = predict_document_code_from_path(
                                        path=doc_path,
                                        config=config,
                                        model=model,
                                        alphabet=alphabet,
                                        device=device,
                                        box_template=box_template,
                                        use_auto_trim=use_auto_trim,
                                        trim_threshold=trim_threshold,
                                        trim_pad=trim_pad,
                                        use_router=use_router,
                                        segmentation_method=segmentation_method,
                                        use_anchor=False,
                                        use_deskew=use_deskew,
                                        override_roi=custom_doc_roi,
                                    )
                                    new_status = "OK" if (
                                        new_pred["min_conf"] >= min_conf_threshold
                                        and new_pred["avg_conf"] >= avg_conf_threshold
                                    ) else "REVIEW"

                                    st.session_state["run_batch_details"][selected_file] = new_pred
                                    for r in st.session_state["run_batch_results"]:
                                        if r.get("fitxer") == selected_file:
                                            r["codi"] = new_pred["code"]
                                            r["format"] = new_pred.get("format_name", "N/A")
                                            r["min_conf"] = round(new_pred["min_conf"], 4)
                                            r["avg_conf"] = round(new_pred["avg_conf"], 4)
                                            r["status"] = new_status
                                            r["exportar"] = (new_status == "OK" and bool(new_pred["code"]))
                                            r.pop("error", None)

                                    if save_as_format_default and new_pred.get("format_id"):
                                        if "custom_rois" not in config or not isinstance(config["custom_rois"], dict):
                                            config["custom_rois"] = {}
                                        config["custom_rois"][new_pred["format_id"]] = custom_doc_roi
                                        save_project_config(pdir, config)

                                    st.session_state["run_batch_editor_rev"] = editor_rev + 1
                                    st.session_state["batch_review_flash"] = (
                                        f"🔄 Re-llegit `{selected_file}` amb nova ROI → Codi detectat: `{new_pred['code']}` "
                                        f"(min_conf={new_pred['min_conf']:.2f}, estat={new_status})."
                                    )
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error en re-llegir el document: {e}")

                        with r_c2:
                            try:
                                preview_page = load_first_page_image(doc_path, dpi=130)
                                preview_crop = crop_relative(preview_page, custom_doc_roi)
                                st.image(preview_crop, caption="Previsualització en viu del nou retall ROI", use_container_width=False)
                            except Exception as e:
                                st.warning(f"No es pot previsualitzar el retall: {e}")
