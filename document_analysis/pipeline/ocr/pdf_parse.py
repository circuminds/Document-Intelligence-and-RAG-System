from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import pytesseract

from pipeline.ingest.db import SessionLocal, Document


@dataclass
class PageParseResult:
    page_number: int
    page_image_path: str
    text_layer: str
    ocr_text: str
    ocr_mean_conf: float
    ocr_boxes: List[Dict[str, Any]]
    used_ocr: bool


def _pdf_raw_path(base_dir: Path, doc_id: str) -> Path:
    raw_dir = base_dir / "data" / "raw" / doc_id
    # find original.*
    matches = list(raw_dir.glob("original.*"))
    if not matches:
        raise FileNotFoundError(f"No raw file found for doc_id={doc_id} in {raw_dir}")
    return matches[0]


def _parsed_dir(base_dir: Path, doc_id: str) -> Path:
    pdir = base_dir / "data" / "parsed" / doc_id
    (pdir / "pages").mkdir(parents=True, exist_ok=True)
    return pdir


def _render_page_to_pil(page: fitz.Page, dpi: int = 200) -> Image.Image:
    # dpi -> scaling factor relative to 72 dpi
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def _extract_text_layer(page: fitz.Page) -> str:
    # Text layer extraction. If scanned, this will be empty/short.
    text = page.get_text("text") or ""
    # Normalize whitespace lightly
    text = "\n".join([ln.rstrip() for ln in text.splitlines()]).strip()
    return text


def _ocr_page(pil_img: Image.Image) -> Tuple[str, float, List[Dict[str, Any]]]:
    """
    Returns:
      - ocr_text
      - mean_confidence (0..100, or 0 if none)
      - boxes: list of word-level boxes with confidence
    """
    # pytesseract image_to_data gives word boxes + conf + text
    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)

    n = len(data.get("text", []))
    boxes: List[Dict[str, Any]] = []
    confs: List[float] = []
    words: List[str] = []

    for i in range(n):
        txt = (data["text"][i] or "").strip()
        conf_str = data["conf"][i]
        try:
            conf = float(conf_str)
        except Exception:
            conf = -1.0

        if txt:
            box = {
                "text": txt,
                "conf": conf,
                "left": int(data["left"][i]),
                "top": int(data["top"][i]),
                "width": int(data["width"][i]),
                "height": int(data["height"][i]),
                "line_num": int(data.get("line_num", [0] * n)[i]),
                "block_num": int(data.get("block_num", [0] * n)[i]),
                "par_num": int(data.get("par_num", [0] * n)[i]),
            }
            boxes.append(box)
            words.append(txt)
            if conf >= 0:
                confs.append(conf)

    ocr_text = " ".join(words).strip()
    mean_conf = float(np.mean(confs)) if confs else 0.0
    return ocr_text, mean_conf, boxes


def parse_pdf_document(
    doc_id: str,
    base_dir: Optional[Path] = None,
    dpi: int = 200,
    ocr_if_text_shorter_than: int = 60,
    ocr_always: bool = False,
) -> Dict[str, Any]:
    """
    Parses a PDF into per-page images + JSON artifacts with:
    - text_layer
    - OCR text + boxes + confidence
    - a simple decision on whether OCR was used

    Returns a summary dict with counts/paths.
    """
    base_dir = base_dir or Path(__file__).resolve().parents[2]
    raw_path = _pdf_raw_path(base_dir, doc_id)
    parsed_dir = _parsed_dir(base_dir, doc_id)
    pages_dir = parsed_dir / "pages"

    doc = fitz.open(str(raw_path))
    num_pages = doc.page_count

    results: List[PageParseResult] = []

    for pno in range(num_pages):
        page = doc.load_page(pno)
        page_number = pno + 1

        # Render page to image
        pil_img = _render_page_to_pil(page, dpi=dpi)

        img_name = f"page_{page_number:04d}.png"
        img_path = pages_dir / img_name
        pil_img.save(img_path)

        # Text layer
        text_layer = _extract_text_layer(page)

        # Decide OCR
        used_ocr = ocr_always or (len(text_layer) < ocr_if_text_shorter_than)

        ocr_text = ""
        mean_conf = 0.0
        ocr_boxes: List[Dict[str, Any]] = []

        if used_ocr:
            ocr_text, mean_conf, ocr_boxes = _ocr_page(pil_img)

        result = PageParseResult(
            page_number=page_number,
            page_image_path=str(img_path),
            text_layer=text_layer,
            ocr_text=ocr_text,
            ocr_mean_conf=mean_conf,
            ocr_boxes=ocr_boxes,
            used_ocr=used_ocr,
        )
        results.append(result)

        # Write per-page JSON
        page_json_path = pages_dir / f"page_{page_number:04d}.json"
        payload = {
            "doc_id": doc_id,
            "page_number": page_number,
            "page_image_path": str(img_path),
            "text_layer": text_layer,
            "used_ocr": used_ocr,
            "ocr_text": ocr_text,
            "ocr_mean_conf": mean_conf,
            "ocr_boxes": ocr_boxes,
            "dpi": dpi,
        }
        page_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    # Update DB
    with SessionLocal() as db:
        d = db.get(Document, doc_id)
        if d:
            d.num_pages = num_pages
            d.status = "parsed"
            d.error = None
            db.commit()

    summary = {
        "doc_id": doc_id,
        "raw_path": str(raw_path),
        "parsed_dir": str(parsed_dir),
        "num_pages": num_pages,
        "dpi": dpi,
        "ocr_if_text_shorter_than": ocr_if_text_shorter_than,
        "ocr_always": ocr_always,
    }
    (parsed_dir / "parse_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
