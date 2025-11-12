"""Configuration modules."""

# Import all settings and model configurations
from config import settings
from config import models

# Re-export commonly used constants
from config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    QUEUE_PROCESSING_DIR,
    QUEUE_FAILED_DIR,
    VECTOR_DB_PATH,
    COLLECTION_NAME,
    DOCUMENT_PERSIST_DIR,
    RESOURCE_CONFIG_PATH,
    EMBEDDING_MODEL,
    SENTENCE_TRANSFORMERS_MODEL,
    SEARCH_KWARGS,
)

from config.models import (
    PROVIDERS,
    SYSTEM_PROMPTS,
)

__all__ = [
    # Settings
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "QUEUE_PROCESSING_DIR",
    "QUEUE_FAILED_DIR",
    "VECTOR_DB_PATH",
    "COLLECTION_NAME",
    "DOCUMENT_PERSIST_DIR",
    "RESOURCE_CONFIG_PATH",
    "EMBEDDING_MODEL",
    "SENTENCE_TRANSFORMERS_MODEL",
    "SEARCH_KWARGS",
    # Models
    "PROVIDERS",
    "SYSTEM_PROMPTS",
]

