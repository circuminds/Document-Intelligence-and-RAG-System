# Document-Intelligence-and-RAG-System
**PDF + OCR + Chunking + FAISS + Retrieval + Chat**

> An end-to-end, production-oriented Retrieval-Augmented Generation (RAG) system designed to work with **real-world documents** - scanned PDFs, noisy OCR, long reports, and text retrieval.

---

## Key Capabilities

### ✅ Document Ingestion
- Upload digital or scanned PDFs
- SHA-based deduplication
- Persistent document metadata tracking

### ✅ OCR & Page Parsing
- Page-level PDF rendering
- Automatic OCR detection
- Optional forced OCR for all pages
- OCR confidence tracking per page

### ✅ Intelligent Chunking
- Semantic chunking for digital text
- Layout-aware chunking for OCR-heavy pages
- Context overlap control
- Page-aligned chunks for precise citations

### ✅ Vector Indexing (FAISS)
- Sentence-Transformer embeddings
- Global FAISS index across documents
- Document-scoped filtering at query time

### ✅ Retrieval-Augmented Generation (RAG)
- Evidence-first retrieval
- Grounded answers with citations
- Configurable top-K retrieval
- Local LLM inference using Ollama (LLaMA-3, Mistral)

### ✅ Interactive UI
- Streamlit-based workspace
- Document lifecycle tracking
- One-click document processing
- Page preview panel
- Chat interface with citations

---

## 🏗️ System Architecture

<img width="776" height="464" alt="image" src="https://github.com/user-attachments/assets/82e1598e-e6a5-43df-95ce-331eb8589a51" />

---

## Processing Pipeline

1. **Parse + OCR**
   - Render PDF pages
   - Detect OCR necessity
   - Extract text, bounding boxes, and confidence

2. **Chunking**
   - Adaptive strategy (semantic vs layout-aware)
   - Page aware chunking
   - Context overlap preservation

3. **Indexing**
   - Dense text embeddings
   - FAISS index update
   - Document becomes query-ready

Once processed, the document is immediately available for chat and retrieval.

---

## Retrieval & Citations

Every answer:
- retrieves top-K evidence chunks
- tracks document ID, page number, and chunk ID
- exposes citations in the UI

This makes hallucinations **inspectable and debuggable**, not hidden.

---

## Text Retrieval

- Text queries retrieve semantically relevant page images

---

## Tech Stack

| Layer | Technology |
|-----|-----------|
| UI | Streamlit |
| API | FastAPI |
| OCR | Tesseract |
| Embeddings | Sentence-Transformers |
| Vector DB | FAISS |
| LLM | Ollama (local) |
| Database | SQLite |
| Language | Python |


---

## ▶️ How to Run

### 1️⃣ Environment Setup
```bash
conda create -n rag python=3.10
conda activate rag
pip install -r requirements.txt

```bash

### 2️⃣ Start Backend
```bash
uvicorn app.api:app --reload
```bash

### 3️⃣ Start UI
```bash
streamlit run app/ui/app.py
```bash




