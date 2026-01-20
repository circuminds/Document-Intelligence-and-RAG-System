
## Overview

This project implements a production-style document analysis pipeline:

1. Documents are uploaded and stored with metadata
2. PDFs are parsed into page images and text layers
3. OCR is applied where digital text is unavailable
4. Text is chunked using semantic and layout-aware strategies
5. Chunks are embedded and indexed for fast retrieval
6. Users interact through a chat interface with grounded citations


## Project Structure

document_analysis/
├── app/
│   ├── ui/                     # Streamlit frontend (document UI + chat)
│   └── api.py                  # FastAPI backend (RAG APIs)
│
├── pipeline/
│   ├── ingest/                 # Upload, delete, cleanup, DB logic
│   ├── ocr/                    # PDF parsing and OCR pipeline
│   ├── chunking/               # Semantic & layout-aware chunking
│   ├── retrieval/              # Retrieval + RAG orchestration
│
├── data/
│   ├── raw/                    # Uploaded documents
│   ├── parsed/                 # Parsed pages, OCR output, chunks
│   └── indexes/                # Vector indexes & metadata
│
├── db/
│   └── rag.db                  # SQLite metadata database
│
├── requirements.txt
└── README.md

