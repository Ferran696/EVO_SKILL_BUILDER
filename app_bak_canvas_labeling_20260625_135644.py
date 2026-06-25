import streamlit as st
import re
import pandas as pd
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
from core.utils import load_first_page_image, crop_relative, draw_roi_overlay, split_fixed_slots, crop_box_pixels, draw_char_boxes_overlay, resize_for_zoom, trim_to_ink_bbox, slot_ink_ratio, crop_box_pixels, draw_char_boxes_overlay, resize_for_zoom

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
    st.caption("Mode reliable: etiquetatge manual caràcter per caràcter amb zoom. Zero slots Gowex.")

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
        st.markdown("### 2) Etiquetar caràcters manualment")

        doc = st.selectbox("Document a etiquetar", [p.name for p in docs])
        path = next(p for p in docs if p.name == doc)

        img = load_first_page_image(path, dpi=150)
        crop_raw = crop_relative(img, roi)

        st.markdown("#### Configuració visual")

        c_cfg1, c_cfg2, c_cfg3, c_cfg4 = st.columns(4)
        with c_cfg1:
            use_auto_trim = st.checkbox("Auto-trim tinta", value=True)
        with c_cfg2:
            trim_threshold = st.slider("Threshold tinta", 180, 255, 245, 1)
        with c_cfg3:
            trim_pad = st.slider("Padding trim px", 0, 20, 4, 1)
        with c_cfg4:
            zoom = st.slider("Zoom visual", 2, 12, 6, 1)

        if use_auto_trim:
            crop = trim_to_ink_bbox(crop_raw, threshold=trim_threshold, pad_px=trim_pad)
        else:
            crop = crop_raw

        cw, ch = crop.size

        truth_raw = st.text_input(
            f"Valor real complet ({fixed_length} caràcters)",
            key=f"truth_manual_{pdir.name}_{path.name}",
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

        # Inicialitzar caixes per document/valor.
        boxes_key = f"manual_boxes_{pdir.name}_{path.name}_{truth or 'EMPTY'}"

        if boxes_key not in st.session_state:
            boxes = []
            for i in range(fixed_length):
                x0 = int(round(i * cw / fixed_length))
                x1 = int(round((i + 1) * cw / fixed_length))
                boxes.append({
                    "x0": max(0, x0),
                    "y0": 0,
                    "x1": min(cw, x1),
                    "y1": ch,
                })
            st.session_state[boxes_key] = boxes

        boxes = st.session_state[boxes_key]

        c_left, c_right = st.columns([1.25, 1.0], gap="large")

        with c_left:
            st.markdown("#### ROI raw")
            st.image(crop_raw, caption=f"ROI original · {crop_raw.size[0]} x {crop_raw.size[1]} px", use_container_width=False)

            st.markdown("#### ROI neta ampliada amb caixes")
            overlay = draw_char_boxes_overlay(crop, boxes, active_idx=int(st.session_state.get("active_char_idx", 0)))
            overlay_zoom = resize_for_zoom(overlay, zoom=zoom)
            st.image(overlay_zoom, caption=f"ROI neta x{zoom} · vermell = caràcter actiu · groc = resta", use_container_width=False)

        with c_right:
            st.markdown("#### Caràcter actiu")

            if "active_char_idx" not in st.session_state:
                st.session_state.active_char_idx = 0

            active_pos = st.number_input(
                "Posició del caràcter",
                min_value=1,
                max_value=fixed_length,
                value=int(st.session_state.active_char_idx) + 1,
                step=1,
            )

            active_idx = int(active_pos) - 1
            st.session_state.active_char_idx = active_idx

            expected_label = truth[active_idx] if valid_truth else ""

            if valid_truth:
                st.info(f"Posició {active_pos} → etiqueta esperada: `{expected_label}`")
            else:
                st.warning("Escriu primer el valor real complet per saber l'etiqueta del caràcter.")

            b = boxes[active_idx]

            st.markdown("##### Ajust caixa en píxels")
            x0 = st.slider("x0", 0, max(1, cw - 1), int(b["x0"]), 1, key=f"x0_{boxes_key}_{active_idx}")
            x1 = st.slider("x1", 1, max(1, cw), int(b["x1"]), 1, key=f"x1_{boxes_key}_{active_idx}")
            y0 = st.slider("y0", 0, max(1, ch - 1), int(b["y0"]), 1, key=f"y0_{boxes_key}_{active_idx}")
            y1 = st.slider("y1", 1, max(1, ch), int(b["y1"]), 1, key=f"y1_{boxes_key}_{active_idx}")

            if x1 <= x0:
                x1 = min(cw, x0 + 1)
            if y1 <= y0:
                y1 = min(ch, y0 + 1)

            boxes[active_idx] = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
            st.session_state[boxes_key] = boxes

            char_crop = crop_box_pixels(crop, boxes[active_idx])
            char_zoom = resize_for_zoom(char_crop, zoom=max(8, zoom))

            st.markdown("##### Crop del caràcter actiu")
            st.image(char_zoom, caption=f"pos {active_pos} · etiqueta `{expected_label or '?'}`", use_container_width=False)

            ink = slot_ink_ratio(char_crop, threshold=trim_threshold)
            st.caption(f"ink ratio: {ink*100:.2f}% · size: {char_crop.size[0]} x {char_crop.size[1]} px")

            if ink < 0.005:
                st.error("Aquest crop sembla buit. Ajusta la caixa abans de guardar.")
            elif char_crop.size[0] < 3 or char_crop.size[1] < 8:
                st.warning("Crop massa petit. Ajusta la caixa.")
            else:
                st.success("Crop sembla usable.")

            st.markdown("---")
            nav1, nav2, nav3 = st.columns(3)

            if nav1.button("⬅️ Anterior", use_container_width=True, disabled=active_idx <= 0):
                st.session_state.active_char_idx = max(0, active_idx - 1)
                st.rerun()

            if nav2.button("➡️ Següent", use_container_width=True, disabled=active_idx >= fixed_length - 1):
                st.session_state.active_char_idx = min(fixed_length - 1, active_idx + 1)
                st.rerun()

            if nav3.button("Reset caixes", use_container_width=True):
                del st.session_state[boxes_key]
                st.rerun()

            can_save_char = valid_truth and expected_label in alphabet and ink >= 0.005

            if st.button("💾 Guardar caràcter actiu", type="primary", disabled=not can_save_char, use_container_width=True):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                class_dir = samples_dir / expected_label
                class_dir.mkdir(parents=True, exist_ok=True)

                safe_stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", path.stem)[:80]
                out_name = f"{safe_stem}_{ts}_pos{active_pos:02d}_{expected_label}.png"
                out_path = class_dir / out_name

                char_crop.save(out_path)

                row = {
                    "timestamp": ts,
                    "project_id": config.get("project_id"),
                    "source_doc": path.name,
                    "truth": truth,
                    "slot_idx": active_pos,
                    "label": expected_label,
                    "ink_ratio": ink,
                    "image_path": str(out_path),
                    "mode": "manual_char",
                    "box_x0": x0,
                    "box_y0": y0,
                    "box_x1": x1,
                    "box_y1": y1,
                    "roi_x0": roi.get("x0"),
                    "roi_y0": roi.get("y0"),
                    "roi_x1": roi.get("x1"),
                    "roi_y1": roi.get("y1"),
                    "auto_trim": use_auto_trim,
                    "trim_threshold": trim_threshold,
                    "trim_pad": trim_pad,
                }

                write_header = not labels_csv.exists()
                with labels_csv.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)

                config["status"] = "samples_labeling_manual"
                save_project_config(pdir, config)

                st.success(f"Guardat caràcter pos {active_pos}: `{expected_label}`")
                if active_idx < fixed_length - 1:
                    st.session_state.active_char_idx = active_idx + 1
                st.rerun()

        st.markdown("---")
        st.markdown("### 3) Mostres guardades")

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
    st.info("Següent fase: entrenament CNN dinàmic amb lock anti-solapaments.")
    if is_locked():
        st.error("Ara mateix hi ha un entrenament en curs.")
        st.code(read_lock())
    else:
        st.success("GPU disponible.")

elif page == "06 · Test skill":
    st.subheader("06 · Test skill")
    st.info("Següent fase: provar skill publicada amb documents nous i mostrar confiança per caràcter.")

