"""Data Ingestion page - Upload and manage your data sources here."""
import os
import uuid
from datetime import datetime
from utils.helpers import (
    load_documents,
    save_documents_atomic,
    sha256_bytes,
    ensure_unique_filename,
    upsert_by_name,
)
import streamlit as st
from config.settings import RESOURCE_CONFIG_PATH, DOCUMENT_PERSIST_DIR

st.header("Data Ingestion")
st.write("Upload and manage your data sources here.")

# Rotating key for resetting the file_uploader without touching its state key
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = uuid.uuid4().hex

def reset_uploader():
    st.session_state.uploader_key = uuid.uuid4().hex

# Load existing documents
documents = load_documents(RESOURCE_CONFIG_PATH)

# List uploaded files
st.subheader("Uploaded Files")

if not documents:
    st.info("No files have been uploaded yet.")
else:
    for r in documents:
        meta = r.get("metadata", {})

        st.markdown(
            f"- **{r.get('name','?')}** ({r.get('type','?')}) "
            f"- Uploaded by: {meta.get('uploaded_by','Unknown')} "
            f"on {meta.get('upload_date','—')}"
        )

# Uploader
uploaded_file = st.file_uploader(
    "Choose a file",
    type=["pdf", "txt", "png", "jpg", "jpeg", "md", "json"],
    key=st.session_state.uploader_key,
)

if uploaded_file is not None:
    # Read once (avoid re-reading during rerun)
    blob = uploaded_file.getvalue()

    digest = sha256_bytes(blob)

    size_bytes = len(blob)

    # Idempotency: if same hash exists, skip saving & queue
    same_hash = next((d for d in documents if d.get("sha256") == digest), None)
    
    if same_hash:
        st.info(f"Exact file already uploaded as **{same_hash.get('name')}** — skipping save and queue.")
        reset_uploader()
        st.rerun()

    # Prepare target path
    os.makedirs(DOCUMENT_PERSIST_DIR, exist_ok=True)
    
    target_name = uploaded_file.name

    # Same name, different content => auto-rename
    if any(d.get("name") == target_name for d in documents):
        target_name = ensure_unique_filename(DOCUMENT_PERSIST_DIR, target_name)
        
        st.warning(
            f"A different file with the same name exists. "
            f"Saving this upload as **{target_name}**."
        )

    disk_path = os.path.join(DOCUMENT_PERSIST_DIR, target_name)

    # Save file to disk
    with open(disk_path, "wb") as f:
        f.write(blob)

    # Upsert JSON (by final saved name) and write atomically
    entry = {
        "name": target_name,
        "type": uploaded_file.type,
        "path": f"./{DOCUMENT_PERSIST_DIR}/{target_name}",
        "sha256": digest,
        "size_bytes": size_bytes,
        "metadata": {
            "uploaded_by": st.session_state.get("username", "Unknown"),
            "upload_date": datetime.now().strftime("%Y-%m-%d"),
        },
        "status": "queued",
    }

    documents = upsert_by_name(documents, entry)
    
    save_documents_atomic(RESOURCE_CONFIG_PATH, documents)

    st.success(f"File '{target_name}' saved.")

    # enqueue for embeddings (file now exists on disk)
    if "embed_queue" in st.session_state:
        try:
            st.session_state.embed_queue.enqueue_file(
                src_path=disk_path,
                doc_id=target_name,
                meta={"uploader": st.session_state.get("username", "Unknown")},
            )
            st.toast(
                "Embedding job queued. It will be added to Chroma and removed from cache on success.",
                icon="✅",
            )
        except Exception as e:
            st.error(f"Queue failed: {e}")
    else:
        st.info("Embedding queue not initialized in session; file saved but not queued.")

    # Reset uploader and rerun to show updated list
    reset_uploader()

    st.rerun()

