import requests
import streamlit as st
from typing import Dict, Any, List, Set
import time

API = "http://localhost:8000"

st.set_page_config(page_title="Production RAG Workspace", layout="wide")
st.title("Document Intelligence Pipeline")

st.markdown("""
<style>
/* Sidebar-like right panel */
div[data-testid="stHorizontalBlock"] > div:nth-child(2) {
  background: var(--secondary-background-color);
  padding: 0rem 1rem 1.25rem 1rem;
  border-left: 1px solid rgba(49, 51, 63, 0.15);
  border-radius: 8px;
  min-height: 100%;
}

/* Make images/expanders sit nicely */
div[data-testid="stHorizontalBlock"] > div:nth-child(2) img {
  border-radius: 8px;
}
            
.processing-success {
    background-color: rgba(46, 204, 113, 0.12);
    border: 1px solid rgba(46, 204, 113, 0.35);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
    font-weight: 500;
    color: #1e7e34;
}
</style>
""", unsafe_allow_html=True)





# =========================================================
# Helpers
# =========================================================
def api_get(path: str, timeout: int = 60):
    r = requests.get(f"{API}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()

def api_post(path: str, timeout: int = 60, **kwargs):
    r = requests.post(f"{API}{path}", timeout=timeout, **kwargs)
    r.raise_for_status()
    return r.json()

def api_delete(path: str, timeout: int = 120, **kwargs):
    r = requests.delete(f"{API}{path}", timeout=timeout, **kwargs)
    return r

def fetch_documents() -> List[Dict[str, Any]]:
    r = requests.get(f"{API}/documents", timeout=20)
    r.raise_for_status()
    return r.json()

def safe_rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

def ping(url: str) -> bool:
    try:
        r = requests.get(url, timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def filter_results(results: List[Dict[str, Any]], allowed: Set[str]):
    if not allowed:
        return results
    return [r for r in results if r.get("doc_id") in allowed]

def can_show_pages(doc_meta: Dict[str, Any]) -> bool:
    return doc_meta.get("status") in {"parsed", "chunked", "indexed"} and doc_meta.get("file_type") == "pdf"


if "last_processed_doc" not in st.session_state:
    st.session_state.last_processed_doc = None

# =========================================================
# Sidebar — System + Upload
# =========================================================
with st.sidebar.expander("System status", expanded=True):
    st.write("API:", "✅" if ping(f"{API}/health") else "❌")

st.sidebar.header("Upload")
uploaded = st.sidebar.file_uploader("PDF", type=["pdf"], key="uploader_file")

upload_status = st.sidebar.empty()
upload_progress_slot = st.sidebar.empty()

st.sidebar.write("Selected:", uploaded.name if uploaded else "—")
upload_clicked = st.sidebar.button("⬆️ Upload", use_container_width=True, key="upload_btn")

if upload_clicked:
    if uploaded is None:
        upload_status.error("Please select a PDF first.")
    else:
        progress = upload_progress_slot.progress(0)
        t0 = time.time()
        try:
            upload_status.info("Preparing upload...")

            file_bytes = uploaded.read()
            size_mb = len(file_bytes) / (1024 * 1024)
            upload_status.info(f"Uploading `{uploaded.name}` ({size_mb:.2f} MB)...")

            # Indeterminate-ish progress (Streamlit/requests can't show real upload progress)
            for i in range(1, 35):
                progress.progress(min(i * 2, 70))
                time.sleep(0.02)

            files = {"file": (uploaded.name, file_bytes, uploaded.type)}
            resp = requests.post(f"{API}/upload", files=files, timeout=300)

            progress.progress(95)
            upload_status.write(f"Server responded in {time.time() - t0:.2f}s")

            if resp.status_code != 200:
                upload_status.error(f"Upload failed ({resp.status_code})")
                upload_status.code(resp.text)
                progress.progress(0)
            else:
                progress.progress(100)
                upload_status.success("Upload complete ✅")
                upload_status.json(resp.json())

                # Clear uploader (best effort)
                try:
                    st.session_state["uploader_file"] = None
                except Exception:
                    pass
                time.sleep(0.15)
                safe_rerun()

        except Exception as e:
            upload_status.error("Upload crashed / timed out")
            upload_status.code(str(e))
        finally:
            upload_progress_slot.empty()

st.sidebar.divider()
st.sidebar.caption("Tip: Upload a scanned PDF to see OCR kick in automatically.")


# =========================================================
# Sidebar — Workspace reset
# =========================================================
st.sidebar.divider()
if "confirm_reset" not in st.session_state:
    st.session_state.confirm_reset = False

if st.sidebar.button("🧹 Clear workspace", use_container_width=True):
    st.session_state.confirm_reset = True

if st.session_state.confirm_reset:
    st.sidebar.warning("Deletes ALL documents and indexes.")
    c1, c2 = st.sidebar.columns(2)
    if c1.button("Confirm", use_container_width=True):
        requests.post(f"{API}/reset", timeout=60)
        st.session_state.clear()
        safe_rerun()
    if c2.button("Cancel", use_container_width=True):
        st.session_state.confirm_reset = False
        safe_rerun()

# =========================================================
# Main — Documents Table
# =========================================================
st.subheader("Document Library")

docs = fetch_documents()
if not docs:
    st.info("Upload a document to begin.")
    st.stop()

if "active_doc_id" not in st.session_state:
    st.session_state.active_doc_id = docs[0]["doc_id"]

if "doc_include" not in st.session_state:
    st.session_state.doc_include = {d["doc_id"]: True for d in docs}

current_ids = {d["doc_id"] for d in docs}
st.session_state.doc_include = {
    k: v for k, v in st.session_state.doc_include.items() if k in current_ids
}

if st.session_state.active_doc_id not in current_ids:
    st.session_state.active_doc_id = docs[0]["doc_id"]

hdr = st.columns([1, 1, 4, 1, 1, 1, 1])
hdr[0].markdown("**Primary**")
hdr[1].markdown("**Included in Search**")
hdr[2].markdown("**Document Name**")
hdr[3].markdown("**Format**")
hdr[4].markdown("**Processing State**")
hdr[5].markdown("**Page Count**")
hdr[6].markdown("**Delete Document**")

for d in docs:
    row = st.columns([1, 1, 4, 1, 1, 1, 1])
    if row[0].button("✅" if d["doc_id"] == st.session_state.active_doc_id else "Set", key=f"act_{d['doc_id']}"):
        st.session_state.active_doc_id = d["doc_id"]
        safe_rerun()

    inc_key = f"use_{d['doc_id']}"
    # default include=True for new docs
    
    if inc_key not in st.session_state:
        st.session_state.doc_include[d["doc_id"]] = True
        #st.session_state[inc_key] = st.session_state.doc_include[d["doc_id"]]
    st.session_state.doc_include[d["doc_id"]] = row[1].checkbox("", key=inc_key)

    row[2].write(d["filename"])
    row[3].write(d["file_type"])
    row[4].write(d["status"])
    row[5].write(d.get("num_pages") or "-")

    if row[6].button("🗑️", key=f"del_{d["doc_id"]}", help="Delete this document"):
        st.session_state["delete_target"] = d["doc_id"]
        st.session_state["show_delete_modal"] = True
        safe_rerun()

active_doc_id = st.session_state.active_doc_id
included_docs = {k for k, v in st.session_state.doc_include.items() if v}

doc_meta = api_get(f"/documents/{active_doc_id}")

st.divider()


# ----------------------------
# Delete modal (fixed)
# ----------------------------
if "show_delete_modal" not in st.session_state:
    st.session_state.show_delete_modal = False
if "delete_target" not in st.session_state:
    st.session_state.delete_target = None

if st.session_state.show_delete_modal and st.session_state.delete_target:
    st.warning(f"Delete doc `{st.session_state.delete_target}`? This cannot be undone.")

    rebuild = st.checkbox("Rebuild indexes after delete (recommended)", value=True, key="rebuild_after_delete")
    c1, c2 = st.columns(2)

    if c1.button("✅ Confirm delete", use_container_width=True):
        target = st.session_state.delete_target
        try:
            with st.spinner("Deleting document..."):
                r = requests.delete(
                    f"{API}/documents/{target}",
                    params={"rebuild_indexes": rebuild},
                    timeout=120,
                )

            if r.status_code != 200:
                st.error("Delete failed")
                st.code(r.text)
            else:
                st.success("Deleted ✅")
                # clean include state
                st.session_state.doc_include.pop(target, None)
                st.session_state.pop(f"inc_{target}", None)

            st.session_state.show_delete_modal = False
            st.session_state.delete_target = None

            # if active deleted, pick another
            try:
                remaining = fetch_documents()
                if remaining:
                    st.session_state.active_doc_id = remaining[0]["doc_id"]
            except Exception:
                pass

            safe_rerun()

        except Exception as e:
            st.error("Delete crashed")
            st.code(str(e))

    if c2.button("Cancel", use_container_width=True):
        st.session_state.show_delete_modal = False
        st.session_state.delete_target = None
        safe_rerun()

# ----------------------------------
# Processing completion banner
# ----------------------------------
lp = st.session_state.get("last_processed_doc")
if lp and lp["doc_id"] == active_doc_id:
    st.markdown(
        f"""
        <div class="processing-success">
            ✅ Processing complete — <strong>{lp['filename']}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# Tabs — Process + Chat only
# =========================================================
tab_process, tab_chat = st.tabs(["Document processing", " Ask Questions"])

# =========================================================
# PROCESS TAB (Parse → Chunk → Index)
# =========================================================
with tab_process:
    st.subheader("Document Processing Settings")
    
    

    left, right = st.columns([1.4, 1])  # left = processing controls, right = pages panel

    # ============================
    # LEFT: Processing controls
    # ============================
    with left:
        dpi = st.slider("Page Rendering Resolution", 100, 300, 200, 25)
        ocr_always = st.checkbox("Force OCR on All Pages")

        chunk_mode = st.selectbox("Chunking Strategy", ["auto", "semantic", "layout_ocr"])
        max_chars = st.slider("Maximum Chunk Size", 600, 2500, 1600, 100)
        overlap = st.slider("Context Overlap", 0, 3, 1)

        emb_model = st.selectbox(
            "Embedding Model",
            ["sentence-transformers/all-MiniLM-L6-v2"],
        )

        run = st.button("Start Document Processing", use_container_width=True)
        bar = st.progress(0)
        status = st.empty()

        if run:
            try:
                status.info("Running OCR and page parsing...")
                bar.progress(15)
                api_post(
                    f"/parse/{active_doc_id}?dpi={dpi}&ocr_always={str(ocr_always).lower()}",
                    timeout=900,
                )

                status.info("Building semantic chunks...")
                bar.progress(40)
                api_post(
                    f"/chunk/{active_doc_id}?mode={chunk_mode}&max_chars={max_chars}&overlap_sents={overlap}",
                    timeout=300,
                )

                status.info("Indexing content for retrieval...")
                bar.progress(65)
                api_post(f"/index/{active_doc_id}?model_name={emb_model}", timeout=900)

                bar.progress(100)
                status.success("Processing completed successfully ✅")
                
                # ✅ ADD THIS
                st.session_state.last_processed_doc = {
                    "doc_id": active_doc_id,
                    "filename": doc_meta["filename"],
                }

                safe_rerun()

            except Exception as e:
                status.error("Processing failed")
                st.code(str(e))
                bar.progress(0)

    # ============================
    # RIGHT: Pages panel
    # ============================
    with right:
        #st.markdown('<div class="right-panel">', unsafe_allow_html=True)
        st.markdown("### Page Preview")

        # Refresh doc_meta (because status might have changed after process)
        try:
            doc_meta_rt = api_get(f"/documents/{active_doc_id}", timeout=10)
        except Exception:
            st.info("Cannot load document metadata.")
            #st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        if doc_meta_rt.get("file_type") != "pdf":
            st.info("Pages viewer is only for PDFs.")
            #st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        if doc_meta_rt.get("status") not in {"parsed", "chunked", "indexed"}:
            st.info("Parse the document first to view pages.")
            #st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        # list pages
        try:
            pages_info = api_get(f"/parsed/{active_doc_id}/pages", timeout=20)
            page_files = pages_info.get("pages", [])
        except Exception as e:
            st.error("Failed to fetch parsed pages list")
            st.code(str(e))
            st.stop()

        if not page_files:
            st.info("No parsed pages found yet.")
            #st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        total_pages = len(page_files)

        # dropdown of page numbers (1..N)
        page_numbers = list(range(1, total_pages + 1))

        # keep selection stable per-doc
        page_key = f"page_select_{active_doc_id}"
        if page_key not in st.session_state:
            st.session_state[page_key] = 1

        selected_page = st.selectbox(
            "Select Page",
            page_numbers,
            index=page_numbers.index(st.session_state[page_key]),
            key=page_key,
            label_visibility="visible",
        )

        # preview image
        img_url = f"{API}/parsed/{active_doc_id}/page/{selected_page}/image"
        st.image(img_url, use_container_width=True)

        # optional page signals + text
        with st.expander("Show page text (debug)", expanded=False):
            try:
                page_data = api_get(f"/parsed/{active_doc_id}/page/{selected_page}", timeout=20)
                st.caption(f"Used OCR: {page_data.get('used_ocr')} | OCR mean conf: {page_data.get('ocr_mean_conf')}")
                st.text_area("Text layer", page_data.get("text_layer") or "", height=160)
                st.text_area("OCR text", page_data.get("ocr_text") or "", height=160)
            except Exception as e:
                st.error("Failed to load page JSON")
                st.code(str(e))

# =========================================================
# CHAT TAB (RAG)
# =========================================================
with tab_chat:
    st.subheader("Chat with your documents")

    llm = st.selectbox("LLM (Ollama)", ["llama3"])
    top_k = st.slider("Evidence top-k", 3, 20, 8)
    temp = st.slider("Temperature", 0.0, 1.0, 0.2)

    scope = st.selectbox(
        "Chat scope",
        ["Active document only", "Included documents only", "All indexed documents"],
    )

    q = st.text_area("Ask a question")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("Ask", use_container_width=True) and q.strip():
        filter_doc = None
        if scope == "Active document only":
            filter_doc = active_doc_id

        payload = {
            "query": q.strip(),
            "top_k": top_k,
            "filter_doc_id": filter_doc,
            "model": llm,
            "temperature": temp,
        }

        with st.spinner("Thinking…"):
            r = requests.post(f"{API}/answer", json=payload, timeout=180)
            r.raise_for_status()
            out = r.json()

        if scope == "Included documents only":
            out["evidence"] = filter_results(out.get("evidence", []), included_docs)

        st.session_state.chat_history.append(out)

    for turn in st.session_state.chat_history[::-1]:
        st.markdown("### ✅ Answer")
        st.write(turn["answer"])

        with st.expander("Evidence"):
            for ev in turn.get("evidence", []):
                st.code(ev["text"])
                st.caption(f"Doc {ev['doc_id']} • Page {ev['page_number']}")
