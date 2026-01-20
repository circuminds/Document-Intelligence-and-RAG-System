from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[2]
INDEX_DIR = BASE_DIR / "data" / "indexes"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

FAISS_PATH = INDEX_DIR / "faiss_text.index"
META_PATH = INDEX_DIR / "faiss_text_meta.jsonl"
MODEL_NAME_PATH = INDEX_DIR / "faiss_text_model.txt"

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # fast + decent baseline


@dataclass
class ChunkMeta:
    vector_id: int
    chunk_id: str
    doc_id: str
    page_number: int
    chunk_type: str
    text: str
    bbox: Optional[Dict[str, int]] = None
    ocr_mean_conf: Optional[float] = None


def _load_chunks(doc_id: str) -> List[Dict[str, Any]]:
    chunks_path = BASE_DIR / "data" / "parsed" / doc_id / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.jsonl not found for doc_id={doc_id}. Run Step 6 chunking first.")
    lines = chunks_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def _load_or_create_index(dim: int) -> faiss.Index:
    if FAISS_PATH.exists():
        idx = faiss.read_index(str(FAISS_PATH))
        if idx.d != dim:
            raise ValueError(f"FAISS dim mismatch: existing={idx.d} new={dim}. Delete index or use same model.")
        return idx
    # cosine similarity via inner product on normalized vectors
    return faiss.IndexFlatIP(dim)


def _count_existing_meta() -> int:
    if not META_PATH.exists():
        return 0
    return sum(1 for _ in META_PATH.open("r", encoding="utf-8"))


def _append_meta(rows: List[ChunkMeta]) -> None:
    with META_PATH.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")


def _save_index(idx: faiss.Index) -> None:
    faiss.write_index(idx, str(FAISS_PATH))


def _save_model_name(model_name: str) -> None:
    MODEL_NAME_PATH.write_text(model_name, encoding="utf-8")


def _load_model_name() -> Optional[str]:
    if MODEL_NAME_PATH.exists():
        return MODEL_NAME_PATH.read_text(encoding="utf-8").strip()
    return None


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


def index_document_chunks(
    doc_id: str,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    Reads data/parsed/{doc_id}/chunks.jsonl, embeds chunk texts,
    appends to a GLOBAL FAISS index (data/indexes/faiss_text.index),
    and writes metadata rows to faiss_text_meta.jsonl.
    """
    chunks = _load_chunks(doc_id)

    # Guard: If model differs from existing, block (keeps your index consistent)
    existing_model = _load_model_name()
    if existing_model and existing_model != model_name:
        raise ValueError(
            f"Index already built with model '{existing_model}'. "
            f"You requested '{model_name}'. Use same model or delete data/indexes/ to rebuild."
        )

    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()

    idx = _load_or_create_index(dim)
    start_vector_id = _count_existing_meta()

    texts = [c["text"] for c in chunks]
    metas: List[ChunkMeta] = []

    # embed in batches
    all_vecs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        emb = emb.astype("float32")
        emb = _normalize(emb)
        all_vecs.append(emb)

    vecs = np.vstack(all_vecs) if all_vecs else np.zeros((0, dim), dtype="float32")

    # add to index
    if vecs.shape[0] > 0:
        idx.add(vecs)

    # build metas aligned with vectors
    for j, c in enumerate(chunks):
        vid = start_vector_id + j
        metas.append(
            ChunkMeta(
                vector_id=vid,
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                page_number=int(c["page_number"]),
                chunk_type=c.get("chunk_type", "text_layer"),
                text=c["text"],
                bbox=c.get("bbox"),
                ocr_mean_conf=c.get("ocr_mean_conf"),
            )
        )

    _append_meta(metas)
    _save_index(idx)
    _save_model_name(model_name)

    return {
        "doc_id": doc_id,
        "model_name": model_name,
        "added_vectors": int(vecs.shape[0]),
        "index_total_vectors": int(idx.ntotal),
        "faiss_path": str(FAISS_PATH),
        "meta_path": str(META_PATH),
    }


def _load_meta_by_ids(vector_ids: List[int]) -> List[Dict[str, Any]]:
    """
    Naive but fine for portfolio scale:
    read meta jsonl into memory and return by vector_id.
    Later we can replace with sqlite or mmap index.
    """
    if not META_PATH.exists():
        raise FileNotFoundError("Meta file not found. Build index first.")
    rows = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    lookup = {int(r["vector_id"]): r for r in rows}
    return [lookup[i] for i in vector_ids if i in lookup]


def search_index(
    query: str,
    top_k: int = 8,
    model_name: Optional[str] = None,
    filter_doc_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Dense retrieval from the global FAISS index.
    Optional filter_doc_id (applied after retrieval for now).
    """
    if not FAISS_PATH.exists():
        raise FileNotFoundError("FAISS index not found. Index at least one document first.")

    existing_model = _load_model_name() or DEFAULT_MODEL
    use_model = model_name or existing_model
    if existing_model and use_model != existing_model:
        raise ValueError(f"Index model is '{existing_model}', but you requested '{use_model}'.")

    model = SentenceTransformer(use_model)
    idx = faiss.read_index(str(FAISS_PATH))

    qv = model.encode([query], show_progress_bar=False, convert_to_numpy=True).astype("float32")
    qv = _normalize(qv)

    scores, ids = idx.search(qv, top_k * 5)  # overfetch, then filter
    ids_list = [int(i) for i in ids[0].tolist() if int(i) >= 0]
    scores_list = [float(s) for s in scores[0].tolist()]

    metas = _load_meta_by_ids(ids_list)

    # attach scores, apply optional doc filter
    scored = []
    meta_by_id = {int(m["vector_id"]): m for m in metas}
    for vid, sc in zip(ids_list, scores_list):
        m = meta_by_id.get(vid)
        if not m:
            continue
        if filter_doc_id and m["doc_id"] != filter_doc_id:
            continue
        m2 = dict(m)
        m2["score"] = sc
        scored.append(m2)
        if len(scored) >= top_k:
            break

    return {
        "query": query,
        "top_k": top_k,
        "model_name": use_model,
        "results": scored,
    }
