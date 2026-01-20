from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware

from pipeline.ingest.db import SessionLocal, Document, init_db
from pipeline.ingest.ingest import save_document

from pipeline.ocr.pdf_parse import parse_pdf_document
from pathlib import Path

from fastapi.responses import FileResponse
import json

from pipeline.chunking.chunker import build_chunks_for_doc

from pipeline.index.faiss_index import index_document_chunks, search_index

from pipeline.retrieval.llm_ollama import build_prompt, ollama_generate


from pipeline.ingest.cleanup import clear_workspace
from pipeline.ingest.db import init_db

from pipeline.ingest.delete_doc import delete_doc_artifacts

from pydantic import BaseModel

from sqlalchemy import text



class SearchRequest(BaseModel):
    query: str
    top_k: int = 8
    filter_doc_id: str | None = None

class AnswerRequest(BaseModel):
    query: str
    top_k: int = 8
    filter_doc_id: str | None = None
    model: str = "mistral"
    temperature: float = 0.2



app = FastAPI(title="Multimodal RAG API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    data = await file.read()
    doc_id, digest = save_document(data, file.filename)

    return {"doc_id": doc_id, "sha256": digest, "filename": file.filename}


@app.get("/documents")
def list_documents():
    with SessionLocal() as db:
        docs = db.query(Document).order_by(Document.created_at.desc()).all()
        return [
            {
                "doc_id": d.doc_id,
                "filename": d.filename,
                "file_type": d.file_type,
                "status": d.status,
                "num_pages": d.num_pages,
                "created_at": d.created_at.isoformat(),
            }
            for d in docs
        ]


@app.post("/parse/{doc_id}")
def parse(doc_id: str, dpi: int = 200, ocr_always: bool = False):
    """
    Parses the PDF into page images + OCR artifacts.
    """
    # mark status parsing
    with SessionLocal() as db:
        d = db.get(Document, doc_id)
        if not d:
            raise HTTPException(status_code=404, detail="doc_id not found")
        if d.file_type != "pdf":
            raise HTTPException(status_code=400, detail="Only PDF parsing implemented in Step 5")
        d.status = "parsing"
        db.commit()

    try:
        summary = parse_pdf_document(doc_id=doc_id, dpi=dpi, ocr_always=ocr_always)
        return {"status": "ok", "summary": summary}
    except Exception as e:
        with SessionLocal() as db:
            d = db.get(Document, doc_id)
            if d:
                d.status = "failed"
                d.error = str(e)
                db.commit()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{doc_id}")
def get_document(doc_id: str):
    with SessionLocal() as db:
        d = db.get(Document, doc_id)
        if not d:
            raise HTTPException(status_code=404, detail="doc_id not found")
        return {
            "doc_id": d.doc_id,
            "filename": d.filename,
            "file_type": d.file_type,
            "status": d.status,
            "num_pages": d.num_pages,
            "error": d.error,
            "created_at": d.created_at.isoformat(),
        }


@app.get("/parsed/{doc_id}/pages")
def list_parsed_pages(doc_id: str):
    base_dir = Path(__file__).resolve().parents[2]
    pages_dir = base_dir / "data" / "parsed" / doc_id / "pages"
    if not pages_dir.exists():
        return {"doc_id": doc_id, "pages": []}

    pages = sorted([p.name for p in pages_dir.glob("page_*.json")])
    return {"doc_id": doc_id, "pages": pages}


@app.get("/parsed/{doc_id}/page/{page_number}")
def get_parsed_page(doc_id: str, page_number: int):
    base_dir = Path(__file__).resolve().parents[2]
    page_json = base_dir / "data" / "parsed" / doc_id / "pages" / f"page_{page_number:04d}.json"
    if not page_json.exists():
        raise HTTPException(status_code=404, detail="page json not found")
    return json.loads(page_json.read_text())

@app.get("/parsed/{doc_id}/page/{page_number}/image")
def get_page_image(doc_id: str, page_number: int):
    base_dir = Path(__file__).resolve().parents[2]
    img_path = base_dir / "data" / "parsed" / doc_id / "pages" / f"page_{page_number:04d}.png"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="page image not found")
    return FileResponse(str(img_path))


# Build Chunks
@app.post("/chunk/{doc_id}")
def chunk_doc(
    doc_id: str,
    mode: str = "semantic",        # "semantic" or "layout_ocr"
    max_chars: int = 1600,
    overlap_sents: int = 1,
):
    with SessionLocal() as db:
        d = db.get(Document, doc_id)
        if not d:
            raise HTTPException(status_code=404, detail="doc_id not found")
        if d.status not in {"parsed", "chunked"}:
            # allow re-chunking, but require parsed
            raise HTTPException(status_code=400, detail=f"Document must be parsed first. Current status={d.status}")
        d.status = "chunking"
        db.commit()

    try:
        out = build_chunks_for_doc(doc_id=doc_id, mode=mode, max_chars=max_chars, overlap_sents=overlap_sents)
        return {"status": "ok", "summary": out}
    except Exception as e:
        with SessionLocal() as db:
            d = db.get(Document, doc_id)
            if d:
                d.status = "failed"
                d.error = str(e)
                db.commit()
        raise HTTPException(status_code=500, detail=str(e))

# List Chunks
@app.get("/chunks/{doc_id}")
def list_chunks(doc_id: str, page: int = 1, page_size: int = 20, page_number: int | None = None):
    """
    Lists chunks from chunks.jsonl with pagination (for UI).
    Optional: filter by page_number.
    """
    base_dir = Path(__file__).resolve().parents[2]
    chunks_path = base_dir / "data" / "parsed" / doc_id / "chunks.jsonl"
    if not chunks_path.exists():
        raise HTTPException(status_code=404, detail="chunks.jsonl not found. Run /chunk first.")

    # Read all lines (ok for portfolio size; later we’ll optimize)
    lines = chunks_path.read_text(encoding="utf-8").splitlines()
    items = [json.loads(l) for l in lines if l.strip()]

    if page_number is not None:
        items = [x for x in items if int(x["page_number"]) == int(page_number)]

    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "doc_id": doc_id,
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": items[start:end],
    }

# Index a document
@app.post("/index/{doc_id}")
def index_doc(doc_id: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Embed chunks.jsonl and append to GLOBAL FAISS index.
    """
    with SessionLocal() as db:
        d = db.get(Document, doc_id)
        if not d:
            raise HTTPException(status_code=404, detail="doc_id not found")
        if d.status not in {"chunked", "indexed"}:
            raise HTTPException(status_code=400, detail=f"Document must be chunked first. Current status={d.status}")
        d.status = "indexing"
        db.commit()

    try:
        out = index_document_chunks(doc_id=doc_id, model_name=model_name)
        with SessionLocal() as db:
            d = db.get(Document, doc_id)
            if d:
                d.status = "indexed"
                d.error = None
                db.commit()
        return {"status": "ok", "summary": out}
    except Exception as e:
        with SessionLocal() as db:
            d = db.get(Document, doc_id)
            if d:
                d.status = "failed"
                d.error = str(e)
                db.commit()
        raise HTTPException(status_code=500, detail=str(e))


# Search

@app.post("/search")
def search(req: SearchRequest):
    try:
        return search_index(
            query=req.query,
            top_k=req.top_k,
            filter_doc_id=req.filter_doc_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        

# Answer


@app.post("/answer")
def answer(req: AnswerRequest):
    """
    Full RAG: retrieve evidence chunks -> generate grounded answer with citations.
    """
    # 1) retrieve
    retrieved = search_index(
        query=req.query,
        top_k=req.top_k,
        filter_doc_id=req.filter_doc_id,
    )
    evidence = retrieved.get("results", [])

    if not evidence:
        return {
            "query": req.query,
            "model": req.model,
            "answer": "I couldn't retrieve any relevant chunks. Make sure documents are indexed.",
            "evidence": [],
        }

    # Retrieve visual evidence
    try:
        img = search_image_index(query=req.query, top_k=3, filter_doc_id=req.filter_doc_id)
        image_evidence = img.get("results", [])
    except Exception:
        image_evidence = []

    # 2) build prompt
    prompt = build_prompt(req.query, evidence)

    # 3) generate
    try:
        ans = ollama_generate(model=req.model, prompt=prompt, temperature=req.temperature)
    except Exception as e:
        # helpful error for UI
        return {
            "query": req.query,
            "model": req.model,
            "answer": f"LLM call failed: {str(e)}. Is Ollama running on localhost:11434 and is the model pulled?",
            "evidence": evidence,
        }


    return {
        "query": req.query,
        "model": req.model,
        "answer": ans,
        "evidence": evidence,
        "image_evidence": image_evidence,
    }




# Index page image
@app.post("/index_images/{doc_id}")
def index_images(doc_id: str):
    try:
        out = index_document_page_images(doc_id=doc_id)
        return {"status": "ok", "summary": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Multimodal retrieval 
class SearchMMRequest(BaseModel):
    query: str
    text_top_k: int = 8
    image_top_k: int = 4
    filter_doc_id: str | None = None


@app.post("/search_mm")
def search_mm(req: SearchMMRequest):
    """
    Returns text evidence (SentenceTransformer+FAISS) + visual evidence (CLIP+FAISS).
    """
    text = search_index(query=req.query, top_k=req.text_top_k, filter_doc_id=req.filter_doc_id)
    images = search_image_index(query=req.query, top_k=req.image_top_k, filter_doc_id=req.filter_doc_id)
    return {"query": req.query, "text": text["results"], "images": images["results"]}

# Reset workspace

@app.post("/reset")
def reset_workspace():
    # 1) Clear DB rows
    with SessionLocal() as db:
        # Fast and safe (resets table content)
        db.query(Document).delete()
        db.commit()

    # 2) Clear filesystem workspace (parsed files, uploads, indexes, etc.)
    out = clear_workspace()

    return {"status": "ok", **out, "db_cleared": True}


# Delete a document and all its artifacts

@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str, rebuild_indexes: bool = True):
    """
    Deletes a document and its artifacts. Optionally rebuilds indexes to remove vectors.
    """
    # 1) delete DB row
    with SessionLocal() as db:
        d = db.get(Document, doc_id)
        if not d:
            raise HTTPException(status_code=404, detail="doc_id not found")
        db.delete(d)
        db.commit()

    # 2) delete files/artifacts
    out = delete_doc_artifacts(doc_id)

    # 3) optionally rebuild indexes
    if rebuild_indexes:
        try:
            # We'll call a helper you add in Part B
            from pipeline.index.rebuild_all import rebuild_all_indexes
            rebuild_all_indexes()
            out["indexes_rebuilt"] = True
        except Exception as e:
            out["indexes_rebuilt"] = False
            out["rebuild_error"] = str(e)

    return {"status": "deleted", **out}

