import os
import re
import json
import uuid
import time
import shutil
import queue
import threading
from dataclasses import dataclass
from typing import List, Optional
from database.vector_db import get_chroma_collection
from utilities.helpers import (
    read_file_text, chunk_text,
    update_document_status
)
from config.app_config import (
    QUEUE_PROCESSING_DIR,
    VECTOR_DB_PATH, 
    COLLECTION_NAME, 
    QUEUE_FAILED_DIR,
    SENTENCE_TRANSFORMERS_MODEL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
)
from sentence_transformers import SentenceTransformer

class STEmbedder:
    def __init__(
        self,
        model_name: str = SENTENCE_TRANSFORMERS_MODEL,
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        # device is auto if None
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Efficient batched encoding; returns list[list[float]]
        vecs = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]


# ------- Job + Queue -------
@dataclass
class Job:
    doc_id: str                
    cache_path: str
    meta: Optional[dict] = None
    chunk_size: int = CHUNK_SIZE
    overlap: int = CHUNK_OVERLAP

class EmbeddingQueue:
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

    # ---- Public API ----
    def start(self, workers: Optional[int] = None):
        if self._threads:
            return
        n = workers or self.max_workers
        for i in range(n):
            t = threading.Thread(target=self._worker, name=f"st-embed-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self, timeout: Optional[float] = 5.0):
        self._stop.set()
        for _ in self._threads:
            self.q.put(None)
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads.clear()
        self._stop.clear()

    def enqueue_file(self, src_path: str, doc_id: Optional[str] = None, meta: Optional[dict] = None) -> Job:
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
        self.q.join()

    # ---- Worker ----
    def _worker(self):
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
                    fail_to = os.path.join(self.cache_failed_dir, os.path.basename(job.cache_path))
                    
                    shutil.move(job.cache_path, fail_to)

                    update_document_status(job.doc_id, "failed")
                except Exception as e2:
                    print(f"[WARN] Could not move failed file: {e2}")
            finally:
                self.q.task_done()

    def _process_job(self, job: Job, collection):
        # 1) read & chunk
        print(f"[INFO] Reading and chunking file: {job.cache_path}")
        
        raw = read_file_text(job.cache_path)
        
        chunks = chunk_text(raw, chunk_size=job.chunk_size, overlap=job.overlap)
        
        print(f"[INFO] Generated {len(chunks)} text chunks")
        
        if not chunks:
            raise ValueError("No text chunks generated")

        # 2) SentenceTransformers embeddings (batched)
        embeddings = self.embedder.embed_batch(chunks)

        print(f"[INFO] Generated embeddings for {len(embeddings)} chunks")

        # 3) Upsert to Chroma
        ids = [f"{job.doc_id}:{i}" for i in range(len(chunks))]
        
        metadatas = [{"doc_id": job.doc_id, "chunk_id": i, **(job.meta or {})} for i in range(len(chunks))]

        print(f"[INFO] Upserting {len(ids)} embeddings to Chroma collection '{self.collection_name}'")

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        print(f"[INFO] Upsert complete for document ID: {job.doc_id}")
