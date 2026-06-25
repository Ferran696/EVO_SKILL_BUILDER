from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

UPLOADS_DIR = DATA_DIR / "uploads"
PROJECTS_DIR = DATA_DIR / "projects"
SKILLS_DIR = DATA_DIR / "skills"
JOBS_DIR = DATA_DIR / "jobs"
LOGS_DIR = DATA_DIR / "logs"

LOCK_FILE = JOBS_DIR / "training.lock"
REGISTRY_FILE = SKILLS_DIR / "skills_registry.json"

DEFAULT_ALPHABET = "0123456789"
MIN_EXAMPLES_PER_CLASS = 15
RECOMMENDED_EXAMPLES_PER_CLASS = 30
