"""RAG retrieval service for context retrieval - refactored to use tools/context_retriever."""

import re
import streamlit as st
from typing import List, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config.settings import (
    COLLECTION_NAME,
    SEARCH_KWARGS,
    SENTENCE_TRANSFORMERS_MODEL,
    VECTOR_DB_PATH,
)
from core.database.vector_db import get_chroma_collection
from tools.context_retriever import context_retriever


@st.cache_resource
def get_vectorstore() -> Chroma:
    """Get or create a cached Chroma vectorstore."""
    embeddings = HuggingFaceEmbeddings(
        model_name=SENTENCE_TRANSFORMERS_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
    )


@st.cache_resource
def get_retriever_collection():
    """Get or create a cached Chroma collection for direct queries."""
    from config.settings import VECTOR_DB_PATH, COLLECTION_NAME
    return get_chroma_collection(
        path=VECTOR_DB_PATH,
        collection_name=COLLECTION_NAME
    )


def _is_pdf_metadata(text: str) -> bool:
    """
    Check if text appears to be PDF metadata or binary content.
    
    Args:
        text: Text to check
        
    Returns:
        True if text appears to be PDF metadata/binary
    """
    if not text or len(text.strip()) < 10:
        return True
    
    # Check for PDF metadata patterns
    pdf_patterns = [
        r'^\d+\s+\d+\s+n\s+\d+\s+\d+\s+n',  # PDF object references like "0000048679 00000 n"
        r'trailer\s*<<',  # PDF trailer
        r'/Size\s+\d+',  # PDF size
        r'/Root\s+\d+\s+\d+\s+R',  # PDF root
        r'/Info\s+\d+\s+\d+\s+R',  # PDF info
        r'startxref',  # PDF startxref
        r'endobj',  # PDF endobj
        r'stream\s*\n',  # PDF stream
        r'endstream',  # PDF endstream
        r'/Filter\s*\[',  # PDF filter
        r'/ASCII85Decode',  # ASCII85 encoding
        r'/FlateDecode',  # Flate encoding
        r'/BitsPerComponent',  # Image bits
        r'/ColorSpace',  # Color space
        r'/Width\s+\d+',  # Image width
        r'/Height\s+\d+',  # Image height
    ]
    
    # Check if text matches PDF patterns
    for pattern in pdf_patterns:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            return True
    
    # Check for repeated characters (likely encoded binary)
    if re.search(r'(.)\1{20,}', text):  # 20+ repeated characters
        return True
    
    # Check for high ratio of non-alphanumeric characters
    if len(text) > 50:
        alnum_ratio = sum(1 for c in text if c.isalnum() or c.isspace()) / len(text)
        if alnum_ratio < 0.3:  # Less than 30% alphanumeric
            return True
    
    # Check for ASCII85 encoded content patterns
    if re.search(r'[!-u]{20,}', text):  # ASCII85 encoding range
        ascii85_ratio = sum(1 for c in text if '!' <= c <= 'u') / max(len(text), 1)
        if ascii85_ratio > 0.7 and len(text) > 100:
            return True
    
    return False


def _clean_text(text: str) -> str:
    """Clean text by removing non-printable characters and PDF metadata."""
    if not text:
        return ""
    
    # First check if this looks like PDF metadata
    if _is_pdf_metadata(text):
        return ""
    
    # Remove non-printable characters but keep newlines and tabs
    text = re.sub(r'[^\x20-\x7E\n\t\r]+', '', text)
    
    # Remove excessive whitespace but keep structure
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double newline
    
    # Remove lines that are mostly non-alphanumeric
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        # Skip lines that are mostly symbols/numbers (likely PDF metadata)
        if len(line_stripped) > 10:
            alnum_count = sum(1 for c in line_stripped if c.isalnum())
            if alnum_count < len(line_stripped) * 0.3:  # Less than 30% alphanumeric
                continue
        cleaned_lines.append(line_stripped)
    
    text = '\n'.join(cleaned_lines)
    text = text.strip()
    
    # Replace double quotes with single quotes for JSON safety
    text = text.replace('"', "'")
    
    return text


def _format_source(metadata: Optional[dict]) -> str:
    """Format source metadata into a readable string."""
    metadata = metadata or {}
    return (
        metadata.get("source")
        or metadata.get("path")
        or metadata.get("file_path")
        or metadata.get("doc_id")
        or "Knowledge Base"
    )


def _dedupe_by_text(docs: List[Document]) -> List[Document]:
    """Remove duplicate documents based on text content."""
    seen, deduped = set(), []
    for doc in docs:
        content = getattr(doc, "page_content", "").strip()
        if not content or content in seen:
            continue
        seen.add(content)
        deduped.append(doc)
    return deduped


def retrieve_context_langchain(query: str, n_results: int = 3) -> str:
    """
    Retrieve context using LangChain vectorstore (refactored to use tools/context_retriever).

    Args:
        query: Search query
        n_results: Number of results to return

    Returns:
        Formatted context string
    """
    try:
        # Use direct vectorstore implementation for better control
        # This matches how documents are stored during ingestion
        vectorstore = get_vectorstore()
        k = max(1, n_results or 1)
        base_kwargs = dict(SEARCH_KWARGS)
        base_kwargs_k = base_kwargs.get("k")
        if base_kwargs_k is None or base_kwargs_k < k:
            base_kwargs["k"] = k

        documents: List[Document] = []
        try:
            # Try similarity search first
            documents = vectorstore.similarity_search(query, k=base_kwargs.get("k", k))
        except Exception as search_exc:
            # Fallback to retriever API
            try:
                retriever = vectorstore.as_retriever(search_kwargs=base_kwargs)
                documents = retriever.get_relevant_documents(query)
            except Exception as retriever_exc:
                # If both fail, try context_retriever tool as last resort
                try:
                    result = context_retriever.invoke({"query": query})
                    if result and not result.startswith("Error") and not result.startswith("No relevant"):
                        if result.startswith("Retrieved Context:"):
                            return result
                        return f"Retrieved Context:\n{result}"
                except Exception:
                    pass
                return f"RAG Retrieval Error: Similarity search failed ({search_exc}), retriever failed ({retriever_exc})"

        if not documents:
            # Check if collection is empty
            try:
                collection_count = vectorstore._collection.count() if hasattr(vectorstore, "_collection") else None
                if collection_count == 0:
                    return "No relevant context found in the knowledge base. The vector database appears to be empty. Please upload and process documents in the Data Ingestion page first."
            except Exception:
                pass
            return "No relevant context found in the knowledge base. Please ensure documents have been uploaded and processed in the Data Ingestion page."

        documents = _dedupe_by_text(documents)[:k]

        context_blocks: List[str] = []
        for doc in documents:
            raw_text = getattr(doc, "page_content", "")
            
            # Skip if text appears to be PDF metadata
            if _is_pdf_metadata(raw_text):
                continue
            
            cleaned_text = _clean_text(raw_text)
            
            # Skip if cleaned text is empty or has no alphabetic characters
            if not cleaned_text or len(cleaned_text.strip()) < 20:
                continue
            
            if not any(ch.isalpha() for ch in cleaned_text):
                continue
            
            # Skip if text is too short after cleaning (likely noise)
            if len(cleaned_text.strip()) < 30:
                continue
            
            source = _format_source(getattr(doc, "metadata", None))
            context_blocks.append(f"Source: {source}\n{cleaned_text}")

        if not context_blocks:
            return "No relevant context found in the knowledge base. The retrieved documents were empty or invalid."

        context = "\n---\n".join(context_blocks)
        return f"Retrieved Context:\n{context}"
    except Exception as exc:
        return f"RAG Retrieval Error: Could not load context. {exc}"


def retrieve_context_chroma(query: str, n_results: int = 1) -> str:
    """
    Retrieve context using direct Chroma collection query.
    
    Note: This method requires embedding the query first. For better results,
    use retrieve_context_langchain() which handles this automatically.

    Args:
        query: Search query
        n_results: Number of results to return

    Returns:
        Formatted context string
    """
    try:
        collection = get_retriever_collection()
        
        # Embed the query using the same embedding model used for storage
        from core.embeddings.embedder import STEmbedder
        embedder = STEmbedder()
        query_embedding = embedder.embed_batch([query])[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas']
        )

        if results and results.get('documents') and results['documents'][0]:
            cleaned_documents = []
            metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
            
            for idx, doc_text in enumerate(results['documents'][0]):
                # Skip if text appears to be PDF metadata
                if _is_pdf_metadata(doc_text):
                    continue
                
                cleaned_text = _clean_text(doc_text)
                
                # Skip if cleaned text is empty or too short
                if not cleaned_text or len(cleaned_text.strip()) < 30:
                    continue
                
                if not any(ch.isalpha() for ch in cleaned_text):
                    continue

                # Add source information if available
                metadata = metadatas[idx] if idx < len(metadatas) else {}
                source = _format_source(metadata)
                cleaned_documents.append(f"Source: {source}\n{cleaned_text}")

            if cleaned_documents:
                context = "\n---\n".join(cleaned_documents)
                return f"Retrieved Context:\n{context}"

        return "No relevant context found in the knowledge base."

    except Exception as e:
        return f"RAG Retrieval Error: Could not load context. {e}"

