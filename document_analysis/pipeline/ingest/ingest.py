from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Tuple

from .db import SessionLocal, Document, init_db

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def infer_type(filename: str) -> str:
    ext = filename.lower().split(".")[-1]
    if ext in {"pdf"}:
        return "pdf"
    if ext in {"png", "jpg", "jpeg", "webp"}:
        return "image"
    if ext in {"docx"}:
        return "docx"
    return "unknown"


def save_document(file_bytes: bytes, filename: str) -> Tuple[str, str]:
    """
    Saves the raw file and registers it in SQLite.
    Returns (doc_id, sha256).
    doc_id is sha256 for simplicity (dedupe-friendly).
    """
    init_db()

    digest = sha256_bytes(file_bytes)
    doc_id = digest
    file_type = infer_type(filename)

    doc_folder = RAW_DIR / doc_id
    doc_folder.mkdir(parents=True, exist_ok=True)

    raw_path = doc_folder / f"original.{filename.split('.')[-1]}"
    raw_path.write_bytes(file_bytes)

    with SessionLocal() as db:
        existing = db.get(Document, doc_id)
        if existing is None:
            db.add(
                Document(
                    doc_id=doc_id,
                    filename=filename,
                    file_type=file_type,
                    sha256=digest,
                    status="queued",
                )
            )
            db.commit()

    return doc_id, digest
