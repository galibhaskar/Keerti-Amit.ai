"""Vector database client and collection management."""

import chromadb
from config.settings import COLLECTION_NAME, VECTOR_DB_PATH


def get_chroma_client(path: str = VECTOR_DB_PATH):
    """Get a ChromaDB persistent client."""
    return chromadb.PersistentClient(path=path)


def get_chroma_collection(
    path: str = VECTOR_DB_PATH,
    collection_name: str = COLLECTION_NAME
):
    """Get or create a ChromaDB collection."""
    client = get_chroma_client(path)
    return client.get_or_create_collection(name=collection_name)

