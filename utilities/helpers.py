import os
import glob
import hashlib
import re
import json
import tempfile
from typing import List
from langchain_core.documents import Document
from config.app_config import CHUNK_OVERLAP, CHUNK_SIZE, RESOURCE_CONFIG_PATH

# ------- Chunking / IO -------
def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = normalize_ws(text)
    chunks, i, n = [], 0, len(text)
    step = max(1, chunk_size - overlap)
    while i < n:
        chunks.append(text[i:i+chunk_size])
        i += step
    return chunks

def read_file_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".json":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                obj = json.load(f)
            return json.dumps(obj, ensure_ascii=False)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        with open(path, "rb") as f:
            return f.read().decode("utf-8", errors="ignore")

def load_documents(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f) or {}
    docs = data.get("DOCUMENTS") or []
    return docs if isinstance(docs, list) else []


def save_documents_atomic(path: str, documents: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_resource_", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump({"DOCUMENTS": documents}, f, indent=4)
        os.replace(tmp, path)  # atomic on same filesystem
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def ensure_unique_filename(dirpath: str, filename: str) -> str:
    """Return a non-colliding filename under dirpath if filename exists."""
    base, ext = os.path.splitext(filename)
    i = 2
    candidate = filename
    while os.path.exists(os.path.join(dirpath, candidate)):
        candidate = f"{base}_v{i}{ext}"
        i += 1
    return candidate


def upsert_by_name(documents: list, entry: dict) -> list:
    """Replace existing record with same 'name'; otherwise append."""
    name = entry.get("name")
    for i, d in enumerate(documents):
        if d.get("name") == name:
            documents[i] = entry
            return documents
    documents.append(entry)
    return documents

def update_document_status(doc_name: str, status: str):
    """Update the status of a document in resource_config.json"""
    print(f"[INFO] Updating status for document '{doc_name}' to '{status}'")
    
    if not os.path.exists(RESOURCE_CONFIG_PATH):
        print(f"[WARN] Resource config not found: {RESOURCE_CONFIG_PATH}")
        return

    with open(RESOURCE_CONFIG_PATH, "r") as f:
        data = json.load(f) or {}

    documents = data.get("DOCUMENTS", [])
    updated = False
    for doc in documents:
        if doc.get("name") == doc_name:
            doc["status"] = status
            updated = True
            break

    if updated:
        with open(RESOURCE_CONFIG_PATH, "w") as f:
            json.dump({"DOCUMENTS": documents}, f, indent=4)
        print(f"[INFO] Updated status for '{doc_name}' → {status}")
    else:
        print(f"[WARN] Document '{doc_name}' not found in config.")

def initialize_data_json(path: str = RESOURCE_CONFIG_PATH) -> None:
    """Create an initial data JSON file if it does not exist."""
    data = {"DOCUMENTS": []}
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
    