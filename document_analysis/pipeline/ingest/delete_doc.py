from __future__ import annotations
from pathlib import Path
import shutil
from typing import Dict, Any

BASE_DIR = Path(__file__).resolve().parents[2]

def delete_doc_artifacts(doc_id: str) -> Dict[str, Any]:
    """
    Deletes raw + parsed artifacts for a document.
    NOTE: Does not edit FAISS index. We'll rebuild separately if needed.
    """
    removed = []

    raw_dir = BASE_DIR / "data" / "raw" / doc_id
    parsed_dir = BASE_DIR / "data" / "parsed" / doc_id

    # if you store raw file directly as data/raw/{doc_id}.pdf etc, adjust here.
    # We'll remove both possible layouts safely:
    raw_file_glob = list((BASE_DIR / "data" / "raw").glob(f"{doc_id}.*"))

    if raw_dir.exists():
        shutil.rmtree(raw_dir)
        removed.append(str(raw_dir))

    for f in raw_file_glob:
        try:
            f.unlink()
            removed.append(str(f))
        except Exception:
            pass

    if parsed_dir.exists():
        shutil.rmtree(parsed_dir)
        removed.append(str(parsed_dir))

    return {"doc_id": doc_id, "removed_paths": removed}
