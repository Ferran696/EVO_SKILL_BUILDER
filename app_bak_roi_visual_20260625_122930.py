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
from core.utils import load_first_page_image, crop_relative

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

            st.markdown("Ajusta ROI amb sliders. V0.1 fiable > selector fancy.")
            c1, c2 = st.columns([1, 1])

            with c2:
                x0 = st.slider("x0", 0.0, 1.0, config.get("roi", {}).get("x0", 0.60) if config.get("roi") else 0.60, 0.01)
                y0 = st.slider("y0", 0.0, 1.0, config.get("roi", {}).get("y0", 0.00) if config.get("roi") else 0.00, 0.01)
                x1 = st.slider("x1", 0.0, 1.0, config.get("roi", {}).get("x1", 0.95) if config.get("roi") else 0.95, 0.01)
                y1 = st.slider("y1", 0.0, 1.0, config.get("roi", {}).get("y1", 0.20) if config.get("roi") else 0.20, 0.01)

                roi = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

                if st.button("Guardar ROI", type="primary"):
                    config["roi"] = roi
                    config["status"] = "roi_defined"
                    save_project_config(pdir, config)
                    st.success("ROI guardada.")

            with c1:
                st.image(img, caption="Pàgina completa", use_container_width=True)
                try:
                    crop = crop_relative(img, roi)
                    st.image(crop, caption="ROI seleccionada", use_container_width=True)
                except Exception as e:
                    st.error(e)

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
