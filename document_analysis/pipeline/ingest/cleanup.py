from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parents[2]

PATHS_TO_CLEAR = [
    BASE_DIR / "data" / "raw",
    BASE_DIR / "data" / "parsed",
    BASE_DIR / "data" / "indexes",
]

DB_PATH = BASE_DIR / "db" / "rag.db"


def clear_workspace():
    # remove data folders
    for p in PATHS_TO_CLEAR:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
        p.mkdir(parents=True, exist_ok=True)


    return {
        "status": "cleared",
        "paths_cleared": [str(p) for p in PATHS_TO_CLEAR],
        "db_removed": False
    }