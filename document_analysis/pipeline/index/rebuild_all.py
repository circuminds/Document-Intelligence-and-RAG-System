from __future__ import annotations
from pathlib import Path
import shutil
import json

BASE_DIR = Path(__file__).resolve().parents[2]
INDEX_DIR = BASE_DIR / "data" / "indexes"
PARSED_DIR = BASE_DIR / "data" / "parsed"

def rebuild_all_indexes() -> None:
    """
    Deletes existing FAISS indexes + meta and rebuilds by re-indexing all docs found under data/parsed/*.
    Assumes chunks.jsonl exists for text indexing and pages/ images exist for image indexing.
    """
    # 1) wipe indexes
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # 2) re-index all documents that still have parsed artifacts
    doc_ids = sorted([p.name for p in PARSED_DIR.iterdir() if p.is_dir()])

    # Text index
    try:
        from pipeline.index.faiss_index import index_document_chunks, DEFAULT_MODEL
        for doc_id in doc_ids:
            chunks_path = PARSED_DIR / doc_id / "chunks.jsonl"
            if chunks_path.exists():
                index_document_chunks(doc_id=doc_id, model_name=DEFAULT_MODEL)
    except Exception:
        # If text index modules aren't present yet or no docs, ignore
        pass

    # Image index
    try:
        from pipeline.index.faiss_image_index import index_document_page_images, DEFAULT_CLIP
        for doc_id in doc_ids:
            pages_dir = PARSED_DIR / doc_id / "pages"
            if pages_dir.exists():
                # only index images if pages exist
                index_document_page_images(doc_id=doc_id, clip_model_name=DEFAULT_CLIP)
    except Exception:
        pass
