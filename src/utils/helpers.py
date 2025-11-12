"""Helper utility functions for file operations and document management."""

import os
import re
import json
import hashlib
import tempfile
from typing import List

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE, RESOURCE_CONFIG_PATH


# ------- Text Processing -------
def normalize_ws(s: str) -> str:
    """Normalize whitespace in a string."""
    return re.sub(r"\s+", " ", s).strip()


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to chunk
        chunk_size: Size of each chunk
        overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    text = normalize_ws(text)
    chunks, i, n = [], 0, len(text)
    step = max(1, chunk_size - overlap)
    while i < n:
        chunks.append(text[i : i + chunk_size])
        i += step
    return chunks


# ------- File I/O -------
def read_file_text(path: str) -> str:
    """
    Read text from a file, handling various formats including PDF.

    Args:
        path: Path to the file

    Returns:
        File contents as string
    """
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".json":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                obj = json.load(f)
            return json.dumps(obj, ensure_ascii=False)
        elif ext == ".pdf":
            # Use pypdf to extract text from PDF
            try:
                from pypdf import PdfReader
                reader = PdfReader(path)
                text_parts = []
                for page in reader.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        print(f"[WARN] Could not extract text from PDF page: {e}")
                        continue
                if text_parts:
                    return "\n\n".join(text_parts)
                else:
                    raise ValueError("No text could be extracted from PDF")
            except ImportError:
                print("[WARN] pypdf not installed, falling back to binary read")
                raise
            except Exception as e:
                print(f"[ERROR] PDF extraction failed: {e}")
                raise
        else:
            # Try reading as text file
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception:
        # Last resort: try binary read and decode
        try:
            with open(path, "rb") as f:
                return f.read().decode("utf-8", errors="ignore")
        except Exception as e:
            raise ValueError(f"Could not read file {path}: {e}")


def load_documents(path: str) -> list:
    """
    Load documents from a JSON file.

    Args:
        path: Path to the JSON file

    Returns:
        List of documents
    """
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f) or {}
    docs = data.get("DOCUMENTS") or []
    return docs if isinstance(docs, list) else []


def save_documents_atomic(path: str, documents: list) -> None:
    """
    Save documents to a JSON file atomically.

    Args:
        path: Path to save the file
        documents: List of documents to save
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".tmp_resource_", dir=os.path.dirname(path)
    )
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


# ------- Hashing and File Management -------
def sha256_bytes(b: bytes) -> str:
    """
    Compute SHA256 hash of bytes.

    Args:
        b: Bytes to hash

    Returns:
        Hexadecimal hash string
    """
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def ensure_unique_filename(dirpath: str, filename: str) -> str:
    """
    Return a non-colliding filename under dirpath if filename exists.

    Args:
        dirpath: Directory path
        filename: Desired filename

    Returns:
        Unique filename
    """
    base, ext = os.path.splitext(filename)
    i = 2
    candidate = filename
    while os.path.exists(os.path.join(dirpath, candidate)):
        candidate = f"{base}_v{i}{ext}"
        i += 1
    return candidate


# ------- Document Management -------
def upsert_by_name(documents: list, entry: dict) -> list:
    """
    Replace existing record with same 'name'; otherwise append.

    Args:
        documents: List of documents
        entry: Document entry to upsert

    Returns:
        Updated list of documents
    """
    name = entry.get("name")
    for i, d in enumerate(documents):
        if d.get("name") == name:
            documents[i] = entry
            return documents
    documents.append(entry)
    return documents


def update_document_status(doc_name: str, status: str):
    """
    Update the status of a document in resource_config.json.

    Args:
        doc_name: Name of the document
        status: New status
    """
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
    """
    Create an initial data JSON file if it does not exist.

    Args:
        path: Path to the resource config file
    """
    data = {"DOCUMENTS": []}

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

