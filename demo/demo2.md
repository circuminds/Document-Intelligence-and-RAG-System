## Demo 2 — Scanned PDF (OCR-Heavy)

### Input
- **Type:** Scanned PDF
- **Characteristics:**
  - No embedded text
  - Mixed scan quality
  - OCR required for all pages

### Processing Flow
1. Upload scanned PDF
2. Page images rendered
3. OCR forced on all pages
4. OCR confidence computed per page
5. Layout-aware chunking applied
6. Vector indexing completed

### Sample Queries
1. What personal details are mentioned in the document?
2. What dates are referenced?
3. Extract key entities from the report.


### Observations
- OCR fallback worked automatically
- Low-confidence pages identifiable in UI
- Retrieval robust despite OCR noise
- Answers grounded to page-level citations
