import json
import re
import shutil
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
