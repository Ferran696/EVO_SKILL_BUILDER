import streamlit as st
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
from core.utils import load_first_page_image, crop_relative, draw_roi_overlay

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
    st.info("Següent fase: aquí generarem crops per slots i els etiquetarem amb comptadors per classe.")
    pdir, config = select_project()
    if pdir:
        st.json(config)
        st.warning(f"Objectiu mínim: {MIN_EXAMPLES_PER_CLASS} exemples per classe. Recomanat: {RECOMMENDED_EXAMPLES_PER_CLASS}.")

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
