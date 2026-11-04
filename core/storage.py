import json
import csv
import io
import re
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
from core.config import PROJECTS_DIR, SKILLS_DIR, REGISTRY_FILE

def safe_slug(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "project"

def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_project(project_name: str, department: str, field_name: str, alphabet: str, fixed_length: int) -> Path:
    slug = safe_slug(project_name)
    project_id = f"{slug}_{now_stamp()}"
    pdir = PROJECTS_DIR / project_id

    (pdir / "uploads").mkdir(parents=True, exist_ok=True)
    (pdir / "samples").mkdir(parents=True, exist_ok=True)
    (pdir / "models").mkdir(parents=True, exist_ok=True)
    (pdir / "reports").mkdir(parents=True, exist_ok=True)

    config = {
        "project_id": project_id,
        "project_name": project_name,
        "department": department,
        "field_name": field_name,
        "alphabet": alphabet,
        "fixed_length": fixed_length,
        "roi": None,
        "segmentation_mode": "fixed_slots",
        "status": "created",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    (pdir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return pdir

def load_project_config(pdir: Path) -> dict:
    return json.loads((pdir / "config.json").read_text(encoding="utf-8"))

def save_project_config(pdir: Path, config: dict):
    (pdir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

def list_projects():
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted([p for p in PROJECTS_DIR.iterdir() if p.is_dir()], reverse=True)

def save_uploaded_files(pdir: Path, uploaded_files):
    out = []
    upload_dir = pdir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    for uf in uploaded_files:
        target = upload_dir / uf.name
        target.write_bytes(uf.getbuffer())
        out.append(str(target))

    return out

def load_registry() -> list:
    if not REGISTRY_FILE.exists():
        return []
    try:
        return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []

def save_registry(items: list):
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def sanitize_filename_token(text: str) -> str:
    """
    Neteja caràcters invàlids per a noms de fitxer a Windows/Linux.
    """
    if text is None:
        return ""
    s = str(text).strip()
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", s)
    s = re.sub(r"\s+", "_", s)
    s = s.strip(" .")
    return s


def build_renamed_filenames(
    rows: list,
    pattern: str = "{codi}",
    prefix: str = "",
) -> list:
    """
    Calcula el camp `nou_nom` per a cada fila de resultats del batch,
    aplicant el patró escollit i resolent col·lisions (duplicats) amb sufixos `_2`, `_3`, etc.
    """
    clean_prefix = sanitize_filename_token(prefix)
    used_names = {}
    enriched = []

    for r in rows:
        row_copy = dict(r)
        orig_file = str(row_copy.get("fitxer", "") or "")
        orig_path = Path(orig_file)
        ext = orig_path.suffix
        orig_stem = sanitize_filename_token(orig_path.stem) or "document"
        clean_code = sanitize_filename_token(row_copy.get("codi", ""))

        if not clean_code:
            base_stem = orig_stem
        else:
            if pattern == "{prefix}{codi}":
                base_stem = f"{clean_prefix}{clean_code}"
            elif pattern == "{codi}_{original}":
                base_stem = f"{clean_code}_{orig_stem}"
            else:
                base_stem = clean_code

        candidate = f"{base_stem}{ext}"
        key = candidate.lower()

        if key not in used_names:
            used_names[key] = 1
            final_name = candidate
        else:
            idx = used_names[key] + 1
            while f"{base_stem}_{idx}{ext}".lower() in used_names:
                idx += 1
            used_names[key] = idx
            final_name = f"{base_stem}_{idx}{ext}"
            used_names[final_name.lower()] = 1

        row_copy["nou_nom"] = final_name
        enriched.append(row_copy)

    return enriched


def _build_export_index_csv_bytes(exported_items: list) -> bytes:
    buf = io.StringIO()
    fieldnames = [
        "fitxer_original",
        "nou_nom",
        "codi",
        "format",
        "min_conf",
        "avg_conf",
        "status",
        "timestamp",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    ts = datetime.now().isoformat(timespec="seconds")
    for item in exported_items:
        writer.writerow({
            "fitxer_original": item.get("fitxer", ""),
            "nou_nom": item.get("nou_nom", ""),
            "codi": item.get("codi", ""),
            "format": item.get("format", ""),
            "min_conf": item.get("min_conf", ""),
            "avg_conf": item.get("avg_conf", ""),
            "status": item.get("status", ""),
            "timestamp": ts,
        })
    return buf.getvalue().encode("utf-8-sig")


def build_batch_export_zip(source_dir: Path, rows: list) -> bytes:
    """
    Genera en memòria un fitxer .ZIP amb els documents seleccionats (`exportar=True`)
    reanomenats segons `nou_nom`, més un fitxer `index_exportacio.csv`.
    """
    source_dir = Path(source_dir)
    selected = [
        r for r in rows
        if bool(r.get("exportar", False)) and str(r.get("fitxer", "")).strip()
    ]

    mem_zip = io.BytesIO()
    exported_items = []

    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for r in selected:
            src_path = source_dir / str(r["fitxer"])
            if not src_path.exists() or not src_path.is_file():
                continue
            dst_name = str(r.get("nou_nom") or r["fitxer"])
            zf.writestr(dst_name, src_path.read_bytes())
            exported_items.append(r)

        index_bytes = _build_export_index_csv_bytes(exported_items)
        zf.writestr("index_exportacio.csv", index_bytes)

    return mem_zip.getvalue()


def export_batch_to_folder(source_dir: Path, target_dir: Path, rows: list) -> dict:
    """
    Copia els documents seleccionats (`exportar=True`) de `source_dir` a `target_dir`
    amb el seu `nou_nom`, i desa `index_exportacio.csv` a la carpeta de destinació.
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    selected = [
        r for r in rows
        if bool(r.get("exportar", False)) and str(r.get("fitxer", "")).strip()
    ]

    copied_files = []
    exported_items = []

    for r in selected:
        src_path = source_dir / str(r["fitxer"])
        if not src_path.exists() or not src_path.is_file():
            continue
        dst_name = str(r.get("nou_nom") or r["fitxer"])
        dst_path = target_dir / dst_name
        shutil.copy2(src_path, dst_path)
        copied_files.append(str(dst_path))
        exported_items.append(r)

    index_path = target_dir / "index_exportacio.csv"
    index_path.write_bytes(_build_export_index_csv_bytes(exported_items))

    return {
        "copied_count": len(copied_files),
        "target_dir": str(target_dir),
        "files": copied_files,
        "index_csv": str(index_path),
    }


def cleanup_runtime_batch_dir(
    pdir: Path,
    run_batch_dir: str | Path = None,
    clean_all_runtime: bool = False,
) -> dict:
    """
    Elimina de forma segura els fitxers temporals d'execució de `runtime_batches`
    sense tocar mai `exports/`, `samples/`, `uploads/` ni `labels.csv`.
    """
    pdir = Path(pdir).resolve()
    runtime_root = (pdir / "runtime_batches").resolve()
    removed_dirs = []

    if not runtime_root.exists():
        return {"removed_count": 0, "removed_dirs": []}

    if clean_all_runtime:
        for child in sorted(runtime_root.iterdir()):
            if child.is_dir() and child.resolve().parent == runtime_root:
                shutil.rmtree(child, ignore_errors=True)
                removed_dirs.append(str(child))
    elif run_batch_dir:
        candidate = Path(run_batch_dir).resolve()
        if candidate.exists() and candidate.is_dir() and candidate.parent == runtime_root:
            shutil.rmtree(candidate, ignore_errors=True)
            removed_dirs.append(str(candidate))

    return {
        "removed_count": len(removed_dirs),
        "removed_dirs": removed_dirs,
    }


def reset_batch_session_state(
    session_state,
    pdir: Path = None,
    clean_disk_temp: bool = True,
) -> dict:
    """
    Reseteja tot l'estat del batch a `session_state` (taula de resultats, visor REVIEW,
    comptador de revisió del `file_uploader` perquè quedi buit) i opcionalment
    elimina els temporals de `runtime_batches`.
    """
    old_run_dir = session_state.get("run_batch_dir", "")
    cleanup_info = {"removed_count": 0, "removed_dirs": []}

    if clean_disk_temp and pdir is not None:
        cleanup_info = cleanup_runtime_batch_dir(
            pdir=Path(pdir),
            run_batch_dir=old_run_dir if old_run_dir else None,
            clean_all_runtime=True,
        )

    session_state.pop("run_batch_results", None)
    session_state.pop("run_batch_details", None)
    session_state.pop("run_batch_dir", None)
    session_state.pop("batch_review_selected_file", None)
    session_state.pop("batch_review_flash", None)

    session_state["batch_uploader_rev"] = int(session_state.get("batch_uploader_rev", 0)) + 1
    session_state["run_batch_editor_rev"] = int(session_state.get("run_batch_editor_rev", 0)) + 1

    return cleanup_info
