import os
import time
from pathlib import Path
from contextlib import contextmanager
from core.config import LOCK_FILE

def is_locked() -> bool:
    return LOCK_FILE.exists()

def read_lock() -> str:
    if not LOCK_FILE.exists():
        return ""
    try:
        return LOCK_FILE.read_text(encoding="utf-8")
    except Exception:
        return ""

@contextmanager
def training_lock(owner: str):
    """
    Lock simple anti-solapaments.
    Només permet un entrenament a la vegada.
    """
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)

    try:
        fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"owner={owner}\ntime={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        acquired = True
    except FileExistsError:
        acquired = False

    if not acquired:
        raise RuntimeError("Ja hi ha un entrenament en curs. Torna-ho a provar més tard.")

    try:
        yield
    finally:
        try:
            LOCK_FILE.unlink(missing_ok=True)
        except Exception:
            pass
