"""Database-related modules."""

from core.database.vector_db import get_chroma_collection

# Note: get_chroma_client is only used internally, not exported
__all__ = ["get_chroma_collection"]

