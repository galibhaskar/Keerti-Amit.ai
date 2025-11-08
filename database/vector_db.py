from config.app_config import COLLECTION_NAME, VECTOR_DB_PATH
import chromadb

def get_chroma_client(path: str):
    client = chromadb.PersistentClient(path=path)

    return client

def get_chroma_collection(path: str = VECTOR_DB_PATH, 
                collection_name: str = COLLECTION_NAME):
    client = get_chroma_client(path)

    return client.get_or_create_collection(name=collection_name)
    