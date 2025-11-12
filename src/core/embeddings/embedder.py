"""Sentence Transformers embedding model wrapper."""

from typing import List, Optional
from sentence_transformers import SentenceTransformer
from config.settings import SENTENCE_TRANSFORMERS_MODEL


class STEmbedder:
    """Sentence Transformers-based embedder for text chunks."""

    def __init__(
        self,
        model_name: str = SENTENCE_TRANSFORMERS_MODEL,
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        Initialize the embedder.

        Args:
            model_name: Name of the Sentence Transformers model
            device: Device to run on (None for auto)
            normalize: Whether to normalize embeddings
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (list of floats)
        """
        vecs = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]

