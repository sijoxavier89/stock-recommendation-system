"""
Embedding Service
Generates vector embeddings from text using sentence-transformers.
This implementation is locked to the "all-MiniLM-L6-v2" model.
"""
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import logging
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Generate embeddings for text chunks using the local sentence-transformer
    model all-MiniLM-L6-v2. Batching and normalization are supported.
    """

    FORCED_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = FORCED_MODEL,
        cache_size: int = 1000,
        device: str = "cpu"  # Use "cuda" if you have GPU
    ):
        """
        Initialize the embedding model. model_name parameter is accepted for API
        compatibility but will be overridden to FORCED_MODEL.
        """
        if model_name != self.FORCED_MODEL:
            logger.info(
                "EmbeddingService: overriding requested model '%s' -> using '%s'",
                model_name,
                self.FORCED_MODEL,
            )
        self.model_name = self.FORCED_MODEL
        self.cache_size = cache_size

        logger.info(f"Loading embedding model: {self.model_name} (device={device})")
        self.model = SentenceTransformer(self.model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def embed_text(
        self,
        text: str,
        normalize: bool = True
    ) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.embed_batch([text], normalize=normalize)[0]

    def embed_batch(
        self,
        texts: List[str],
        normalize: bool = True,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings for multiple texts efficiently."""
        logger.info(f"Generating embeddings for {len(texts)} texts (model={self.model_name})")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        return embeddings

    def embed_documents(
        self,
        chunks: List[dict],
        text_key: str = "text",
        batch_size: int = 32
    ) -> List[dict]:
        """Embed a list of document chunks and attach 'embedding' to each chunk."""
        texts = [chunk[text_key] for chunk in chunks]
        logger.info(f"Embedding {len(texts)} document chunks")
        embeddings = self.embed_batch(
            texts,
            normalize=True,
            batch_size=batch_size
        )
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
        logger.info(f"Successfully embedded {len(chunks)} chunks")
        return chunks

    @lru_cache(maxsize=1000)
    def embed_query_cached(self, query: str) -> np.ndarray:
        """Cache query embeddings for repeated queries."""
        return self.embed_text(query, normalize=True)

    def get_model_info(self) -> dict:
        """Get information about the current embedding model."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "max_sequence_length": getattr(self.model, "max_seq_length", None),
            "device": str(self.model.device),
        }

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two normalized embeddings."""
        return float(np.dot(embedding1, embedding2))

    def find_most_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[tuple]:
        """Find top-k most similar embeddings to a query (assumes normalized vectors)."""
        similarities = np.dot(candidate_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [
            (int(idx), float(similarities[idx]))
            for idx in top_indices
        ]
        return results


# Example usage (kept for local debugging)
if __name__ == "__main__":
    embedding_service = EmbeddingService(device="cpu")
    print("Model Info:", embedding_service.get_model_info())
    sample_texts = [
        "Apple reported revenue of $394 billion in fiscal 2023.",
        "The company's profit margin increased by 2.5% year-over-year.",
    ]
    embeddings = embedding_service.embed_batch(sample_texts)
    print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
