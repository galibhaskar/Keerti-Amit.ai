"""RAG retrieval service for context retrieval."""

import re
import streamlit as st
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

from config.settings import (
    COLLECTION_NAME,
    SEARCH_KWARGS,
    SENTENCE_TRANSFORMERS_MODEL,
    VECTOR_DB_PATH,
)
from core.database.vector_db import get_chroma_collection


@st.cache_resource
def get_vectorstore() -> Chroma:
    """Get or create a cached Chroma vectorstore."""
    embeddings = SentenceTransformerEmbeddings(
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


def _clean_text(text: str) -> str:
    """Clean text by removing non-printable characters."""
    text = re.sub(r"[^\x20-\x7E\s]+", "", text or "").strip()
    return text.replace('"', "'")


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
    Retrieve context using LangChain vectorstore.

    Args:
        query: Search query
        n_results: Number of results to return

    Returns:
        Formatted context string
    """
    try:
        vectorstore = get_vectorstore()
        k = max(1, n_results or 1)
        base_kwargs = dict(SEARCH_KWARGS)
        base_kwargs_k = base_kwargs.get("k")
        if base_kwargs_k is None or base_kwargs_k < k:
            base_kwargs["k"] = k

        documents: List[Document] = []
        try:
            documents = vectorstore.similarity_search(query, k=base_kwargs.get("k", k))
        except Exception:
            retriever = vectorstore.as_retriever(search_kwargs=base_kwargs)
            documents = retriever.get_relevant_documents(query)

        if not documents:
            return "No relevant context found in the knowledge base."

        documents = _dedupe_by_text(documents)[:k]

        context_blocks: List[str] = []
        for doc in documents:
            raw_text = getattr(doc, "page_content", "")
            cleaned_text = _clean_text(raw_text)
            if not cleaned_text or not any(ch.isalpha() for ch in cleaned_text):
                continue
            source = _format_source(getattr(doc, "metadata", None))
            context_blocks.append(f"Source: {source}\n{cleaned_text}")

        if not context_blocks:
            return "No relevant context found in the knowledge base."

        context = "\n---\n".join(context_blocks)
        return f"Retrieved Context:\n{context}"
    except Exception as exc:
        return f"RAG Retrieval Error: Could not load context. {exc}"


def retrieve_context_chroma(query: str, n_results: int = 1) -> str:
    """
    Retrieve context using direct Chroma collection query.

    Args:
        query: Search query
        n_results: Number of results to return

    Returns:
        Formatted context string
    """
    try:
        collection = get_retriever_collection()

        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=['documents']
        )

        if results and results.get('documents') and results['documents'][0]:
            cleaned_documents = []
            for doc_text in results['documents'][0]:
                cleaned_text = re.sub(r'[^\x20-\x7E\s]+', '', doc_text)
                cleaned_text = cleaned_text.strip()

                if cleaned_text:
                    cleaned_documents.append(cleaned_text)

            if cleaned_documents:
                context = "\n---\n".join(cleaned_documents)
                return f"Retrieved Context:\n{context}"

        return "No relevant context found in the knowledge base."

    except Exception as e:
        return f"RAG Retrieval Error: Could not load context. {e}"

