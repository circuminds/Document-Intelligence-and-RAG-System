from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable, Tuple

from pipeline.ingest.db import SessionLocal, Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page_number: int
    chunk_index: int
    chunk_type: str  # "text_layer" | "ocr" | "mixed"
    text: str
    # citation signals
    bbox: Optional[Dict[str, int]] = None  # {"left":..,"top":..,"width":..,"height":..}
    ocr_mean_conf: Optional[float] = None


def _base_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _pages_dir(doc_id: str) -> Path:
    return _base_dir() / "data" / "parsed" / doc_id / "pages"


def _iter_page_json(doc_id: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    pages_dir = _pages_dir(doc_id)
    if not pages_dir.exists():
        raise FileNotFoundError(f"Parsed pages not found: {pages_dir}")

    page_files = sorted(pages_dir.glob("page_*.json"))
    if not page_files:
        raise FileNotFoundError(f"No page_*.json found in {pages_dir}")

    for pj in page_files:
        data = json.loads(pj.read_text())
        yield int(data["page_number"]), data


_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")
_WS = re.compile(r"\s+")


def _clean_text(t: str) -> str:
    t = t.replace("\u00ad", "")  # soft hyphen
    t = t.replace("-\n", "")     # dehyphenate line breaks
    t = t.replace("\n", " ")
    t = _WS.sub(" ", t).strip()
    return t


def _split_into_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    # lightweight sentence split
    sents = _SENT_SPLIT.split(text)
    sents = [s.strip() for s in sents if s.strip()]
    return sents


def semantic_chunk_text(
    text: str,
    max_chars: int = 1600,
    overlap_sents: int = 1,
) -> List[str]:
    """
    Simple semantic-ish chunker:
    - split into sentences
    - pack sentences into chunks up to max_chars
    - overlap by N sentences
    """
    text = _clean_text(text)
    sents = _split_into_sentences(text)
    if not sents:
        return []

    chunks: List[str] = []
    i = 0
    while i < len(sents):
        cur: List[str] = []
        cur_len = 0
        j = i
        while j < len(sents):
            add = sents[j]
            if cur_len + len(add) + 1 > max_chars and cur:
                break
            cur.append(add)
            cur_len += len(add) + 1
            j += 1
        chunk = " ".join(cur).strip()
        if chunk:
            chunks.append(chunk)

        if j >= len(sents):
            break
        # overlap
        i = max(i + 1, j - overlap_sents)

    return chunks


def _ocr_boxes_to_lines(ocr_boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group word boxes by (block_num, par_num, line_num) and return line objects:
      {
        "text": "...",
        "bbox": {"left":..,"top":..,"width":..,"height":..},
        "mean_conf": ...
      }
    """
    if not ocr_boxes:
        return []

    groups: Dict[Tuple[int, int, int], List[Dict[str, Any]]] = {}
    for b in ocr_boxes:
        key = (int(b.get("block_num", 0)), int(b.get("par_num", 0)), int(b.get("line_num", 0)))
        groups.setdefault(key, []).append(b)

    lines: List[Dict[str, Any]] = []
    for key, words in groups.items():
        words = sorted(words, key=lambda w: w["left"])
        text = " ".join([w["text"] for w in words]).strip()

        left = min(w["left"] for w in words)
        top = min(w["top"] for w in words)
        right = max(w["left"] + w["width"] for w in words)
        bottom = max(w["top"] + w["height"] for w in words)

        confs = [float(w.get("conf", -1)) for w in words if float(w.get("conf", -1)) >= 0]
        mean_conf = sum(confs) / max(1, len(confs))

        lines.append({
            "text": text,
            "bbox": {"left": int(left), "top": int(top), "width": int(right-left), "height": int(bottom-top)},
            "mean_conf": float(mean_conf),
        })

    # sort top-to-bottom
    lines.sort(key=lambda x: (x["bbox"]["top"], x["bbox"]["left"]))
    return lines


def layoutish_chunk_from_ocr(
    ocr_boxes: List[Dict[str, Any]],
    max_chars: int = 1400,
) -> List[Dict[str, Any]]:
    """
    v1.5 layout-ish chunking:
    - group OCR by lines (using OCR box line numbers)
    - pack lines into chunks top-to-bottom
    - each chunk has an approximate bbox = union of line bboxes
    Returns list of {"text":..., "bbox":..., "mean_conf":...}
    """
    lines = _ocr_boxes_to_lines(ocr_boxes)
    chunks: List[Dict[str, Any]] = []

    cur_lines: List[Dict[str, Any]] = []
    cur_len = 0

    def flush():
        nonlocal cur_lines, cur_len
        if not cur_lines:
            return
        text = _clean_text(" ".join([l["text"] for l in cur_lines]))
        left = min(l["bbox"]["left"] for l in cur_lines)
        top = min(l["bbox"]["top"] for l in cur_lines)
        right = max(l["bbox"]["left"] + l["bbox"]["width"] for l in cur_lines)
        bottom = max(l["bbox"]["top"] + l["bbox"]["height"] for l in cur_lines)
        confs = [l["mean_conf"] for l in cur_lines]
        mean_conf = sum(confs) / max(1, len(confs))
        if text:
            chunks.append({
                "text": text,
                "bbox": {"left": int(left), "top": int(top), "width": int(right-left), "height": int(bottom-top)},
                "mean_conf": float(mean_conf),
            })
        cur_lines = []
        cur_len = 0

    for ln in lines:
        t = ln["text"]
        if not t:
            continue
        if cur_len + len(t) + 1 > max_chars and cur_lines:
            flush()
        cur_lines.append(ln)
        cur_len += len(t) + 1

    flush()
    return chunks


def build_chunks_for_doc(
    doc_id: str,
    mode: str = "auto",  # "auto" | "semantic" | "layout_ocr"
    max_chars: int = 1600,
    overlap_sents: int = 1,
) -> Dict[str, Any]:
    """
    Creates chunks.jsonl under data/parsed/{doc_id}/
    """
    base = _base_dir()
    out_path = base / "data" / "parsed" / doc_id / "chunks.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunks_out: List[Chunk] = []
    chunk_counter = 0

    for page_number, page in _iter_page_json(doc_id):
        text_layer = page.get("text_layer") or ""
        used_ocr = bool(page.get("used_ocr"))
        ocr_text = page.get("ocr_text") or ""
        ocr_mean_conf = float(page.get("ocr_mean_conf") or 0.0)
        ocr_boxes = page.get("ocr_boxes") or []

        # Choose source text based on mode + availability
                # Decide chunking strategy per page (AUTO)
        page_mode = mode
        if mode == "auto":
            # If OCR was used and we have boxes, use layout chunking for that page
            if used_ocr and ocr_boxes:
                page_mode = "layout_ocr"
            else:
                page_mode = "semantic"

        if page_mode == "layout_ocr" and ocr_boxes:
            packed = layoutish_chunk_from_ocr(ocr_boxes, max_chars=max_chars)
            for idx, obj in enumerate(packed):
                ch = Chunk(
                    chunk_id=f"{doc_id}_p{page_number:04d}_c{idx:04d}",
                    doc_id=doc_id,
                    page_number=page_number,
                    chunk_index=idx,
                    chunk_type="ocr",
                    text=obj["text"],
                    bbox=obj.get("bbox"),
                    ocr_mean_conf=obj.get("mean_conf", ocr_mean_conf),
                )
                chunks_out.append(ch)
                chunk_counter += 1
        else:
            # Semantic chunking on best available source
            source = text_layer if len(text_layer) >= len(ocr_text) else ocr_text
            texts = semantic_chunk_text(source, max_chars=max_chars, overlap_sents=overlap_sents)
            for idx, t in enumerate(texts):
                ch = Chunk(
                    chunk_id=f"{doc_id}_p{page_number:04d}_c{idx:04d}",
                    doc_id=doc_id,
                    page_number=page_number,
                    chunk_index=idx,
                    chunk_type="mixed" if used_ocr else "text_layer",
                    text=t,
                    bbox=None,
                    ocr_mean_conf=ocr_mean_conf if used_ocr else None,
                )
                chunks_out.append(ch)
                chunk_counter += 1


    # Write JSONL
    with out_path.open("w", encoding="utf-8") as f:
        for ch in chunks_out:
            f.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")

    # Update DB status
    with SessionLocal() as db:
        d = db.get(Document, doc_id)
        if d:
            d.status = "chunked"
            # if you added num_chunks column and reset db, this will work:
            if hasattr(d, "num_chunks"):
                try:
                    d.num_chunks = len(chunks_out)
                except Exception:
                    pass
            d.error = None
            db.commit()

    return {
        "doc_id": doc_id,
        "mode": mode,
        "chunks_path": str(out_path),
        "num_chunks": len(chunks_out),
        "max_chars": max_chars,
        "overlap_sents": overlap_sents,
    }
