"""Application settings and configuration constants."""

# Text chunking configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Queue processing directories
QUEUE_PROCESSING_DIR = "./.cache/queue/queue_processing"
QUEUE_FAILED_DIR = "./.cache/queue/queue_failed"

# Vector database configuration
VECTOR_DB_PATH = "./.cache/vector_db/collections"
COLLECTION_NAME = "DocumentsCollection"

# Document storage
DOCUMENT_PERSIST_DIR = "./.cache/data/documents"
RESOURCE_CONFIG_PATH = "./.cache/resource_config.json"

# Embedding model configuration
EMBEDDING_MODEL = "nomic-embed-text"
SENTENCE_TRANSFORMERS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Search configuration
SEARCH_KWARGS = {"k": 5}

# Log files
FLASHCARD_LOG_FILE = "./.cache/flashcard_log.json"
BATTLE_HISTORY_FILE = "./.cache/battle_history.json"

