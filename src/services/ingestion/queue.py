"""Background queue for processing document embeddings."""

import os
import shutil
import queue
import threading
from dataclasses import dataclass
from typing import List, Optional

from core.database.vector_db import get_chroma_collection
from core.embeddings.embedder import STEmbedder
from utils.helpers import (
    read_file_text,
    chunk_text,
    update_document_status,
)
from config.settings import (
    QUEUE_PROCESSING_DIR,
    VECTOR_DB_PATH,
    COLLECTION_NAME,
    QUEUE_FAILED_DIR,
    SENTENCE_TRANSFORMERS_MODEL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
)


@dataclass
class Job:
    """Represents a document embedding job."""

    doc_id: str
    cache_path: str
    meta: Optional[dict] = None
    chunk_size: int = CHUNK_SIZE
    overlap: int = CHUNK_OVERLAP


class EmbeddingQueue:
    """Background queue for processing document embeddings."""

    def __init__(
        self,
        cache_dir: str = QUEUE_PROCESSING_DIR,
        cache_failed_dir: str = QUEUE_FAILED_DIR,
        chroma_path: str = VECTOR_DB_PATH,
        collection_name: str = COLLECTION_NAME,
        st_model_name: str = SENTENCE_TRANSFORMERS_MODEL,
        st_device: Optional[str] = None,
        st_normalize: bool = True,
        max_workers: int = 2,
        queue_maxsize: int = 0,
    ):
        """
        Initialize the embedding queue.

        Args:
            cache_dir: Directory for processing files
            cache_failed_dir: Directory for failed files
            chroma_path: Path to ChromaDB
            collection_name: Name of the collection
            st_model_name: Sentence Transformers model name
            st_device: Device for embedding model
            st_normalize: Whether to normalize embeddings
            max_workers: Number of worker threads
            queue_maxsize: Maximum queue size (0 for unlimited)
        """
        self.cache_dir = cache_dir
        self.cache_failed_dir = cache_failed_dir
        self.collection_name = collection_name
        self.chroma_path = chroma_path

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.cache_failed_dir, exist_ok=True)
        os.makedirs(self.chroma_path, exist_ok=True)

        self.embedder = STEmbedder(st_model_name, device=st_device, normalize=st_normalize)

        self.q: "queue.Queue[Optional[Job]]" = queue.Queue(maxsize=queue_maxsize)
        self._threads: List[threading.Thread] = []
        self._stop = threading.Event()
        self.max_workers = max_workers

    def start(self, workers: Optional[int] = None):
        """Start worker threads."""
        if self._threads:
            return
        n = workers or self.max_workers
        for i in range(n):
            t = threading.Thread(
                target=self._worker,
                name=f"st-embed-worker-{i}",
                daemon=True
            )
            t.start()
            self._threads.append(t)

    def stop(self, timeout: Optional[float] = 5.0):
        """Stop worker threads."""
        self._stop.set()
        for _ in self._threads:
            self.q.put(None)
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()
        self._stop.clear()

    def enqueue_file(
        self,
        src_path: str,
        doc_id: Optional[str] = None,
        meta: Optional[dict] = None
    ) -> Job:
        """
        Enqueue a file for embedding processing.

        Args:
            src_path: Path to the source file
            doc_id: Optional document ID
            meta: Optional metadata

        Returns:
            The created Job instance
        """
        import uuid

        print(f"[INFO] Enqueuing file for embedding: {src_path}")

        if not os.path.isfile(src_path):
            raise FileNotFoundError(src_path)

        base = os.path.basename(src_path)
        uid = uuid.uuid4().hex[:8]
        cached_name = f"{uid}__{base}"
        dst_path = os.path.join(self.cache_dir, cached_name)
        shutil.copy2(src_path, dst_path)

        job = Job(
            doc_id=doc_id or base,
            cache_path=dst_path,
            meta=meta or {},
        )
        self.q.put(job)
        return job

    def join(self):
        """Wait for all queued jobs to complete."""
        self.q.join()

    def _worker(self):
        """Worker thread that processes jobs."""
        collection = get_chroma_collection(self.chroma_path, self.collection_name)

        while not self._stop.is_set():
            job = self.q.get()

            if job is None:
                break
            try:
                print(job)
                print(f"[INFO] Processing job for: {job.cache_path}")

                self._process_job(job, collection)

                update_document_status(job.doc_id, "completed")

                try:
                    os.remove(job.cache_path)
                except Exception as e:
                    print(f"[WARN] Could not remove cached file: {job.cache_path} ({e})")
            except Exception as e:
                print(f"[ERROR] Job failed for {job.cache_path}: {e}")
                try:
                    fail_to = os.path.join(
                        self.cache_failed_dir,
                        os.path.basename(job.cache_path)
                    )
                    shutil.move(job.cache_path, fail_to)
                    update_document_status(job.doc_id, "failed")
                except Exception as e2:
                    print(f"[WARN] Could not move failed file: {e2}")
            finally:
                self.q.task_done()

    def _process_job(self, job: Job, collection):
        """Process a single embedding job."""
        # 1) Read & chunk
        print(f"[INFO] Reading and chunking file: {job.cache_path}")

        raw = read_file_text(job.cache_path)
        chunks = chunk_text(raw, chunk_size=job.chunk_size, overlap=job.overlap)

        print(f"[INFO] Generated {len(chunks)} text chunks")

        if not chunks:
            raise ValueError("No text chunks generated")

        # 2) Generate embeddings
        embeddings = self.embedder.embed_batch(chunks)

        print(f"[INFO] Generated embeddings for {len(embeddings)} chunks")

        # 3) Upsert to Chroma
        ids = [f"{job.doc_id}:{i}" for i in range(len(chunks))]

        metadatas = [
            {"doc_id": job.doc_id, "chunk_id": i, **(job.meta or {})}
            for i in range(len(chunks))
        ]

        print(
            f"[INFO] Upserting {len(ids)} embeddings to Chroma collection "
            f"'{self.collection_name}'"
        )

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        print(f"[INFO] Upsert complete for document ID: {job.doc_id}")

