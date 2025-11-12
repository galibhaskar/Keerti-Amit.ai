from typing import Any, List, Tuple

# pyright: reportMissingImports=false

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

from config.app_config import (
    COLLECTION_NAME,
    SEARCH_KWARGS,
    SENTENCE_TRANSFORMERS_MODEL,
    VECTOR_DB_PATH,
)

_vectorstore: Chroma | None = None
_retriever: Any | None = None


def _ensure_vectorstore() -> Tuple[Chroma, Any]:
    global _vectorstore, _retriever

    if _vectorstore is None:
        embeddings = SentenceTransformerEmbeddings(
            model_name=SENTENCE_TRANSFORMERS_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embeddings,
        )

    if _retriever is None:
        _retriever = _vectorstore.as_retriever(search_kwargs=SEARCH_KWARGS)
        _retriever.search_type = SEARCH_KWARGS.get("search_type", "similarity")

    return _vectorstore, _retriever

def _format_source(md: dict) -> str:
    md = md or {}
    # try common metadata keys
    return md.get("source") or md.get("path") or md.get("file_path") or md.get("doc_id") or "Unknown"

def _dedupe_by_text(docs: List[Document]) -> List[Document]:
    seen, out = set(), []
    for d in docs:
        key = d.page_content.strip()
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out

@tool
def context_retriever(query: str) -> str:
    """
    Fetch the most relevant context from the vector database for a given query.
    Returns a readable string with sources and chunk text.
    """
    try:
        vectorstore, retriever = _ensure_vectorstore()

        # Prefer results with scores if the vector store supports it
        docs: List[Document] = []

        try:
            # Many LangChain vectorstores (incl. Chroma) support this:
            scored: List[Tuple[Document, float]] = vectorstore.similarity_search_with_relevance_scores(
                query, k=SEARCH_KWARGS.get("k", 4)
            )

            # sort (high→low), extract docs
            scored.sort(key=lambda x: x[1], reverse=True)

            docs = [d for d, _ in scored]
        except Exception:
            # Fallback to retriever API
            docs = retriever.get_relevant_documents(query)

        if not docs:
            return "No relevant context found for this query."

        # Optional: dedupe near-identical chunks
        docs = _dedupe_by_text(docs)

        # Limit output length (avoid huge prompts)
        k = SEARCH_KWARGS.get("k", 4)

        docs = docs[:k]

        # Build readable context with sources
        parts = []

        for d in docs:
            source = _format_source(d.metadata if isinstance(d, Document) else {})
            
            text = d.page_content if isinstance(d, Document) else str(d)
            
            parts.append(f"Source: {source}\n{text}")

        context = "\n\n---\n\n".join(parts)

        print(f"[INFO] Retrieved {len(docs)} chunks for query: {query!r}")
        
        return context

    except Exception as e:
        print(f"[ERROR] Failed to retrieve context: {e}")
        return f"Error retrieving context: {e}. Please try again."
