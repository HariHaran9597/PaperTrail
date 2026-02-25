"""
Embeddings & FAISS Vector Store
Manages the paper index using sentence-transformers for embeddings
and FAISS for fast similarity search.
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
import logging

logger = logging.getLogger(__name__)


class PaperIndex:
    """FAISS-based similarity search over academic paper abstracts."""

    def __init__(self, index_dir: str = "data/faiss_index", model_name: str = "all-MiniLM-L6-v2"):
        self.index_dir = index_dir
        self.model_name = model_name
        self.model = None  # Lazy load
        self.index = None
        self.metadata = []
        os.makedirs(index_dir, exist_ok=True)

    def _load_model(self):
        """Lazy-load the embedding model."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def build_index(self, papers: list[dict]):
        """
        Build a FAISS index from paper abstracts.
        
        Args:
            papers: List of dicts with 'title' and 'abstract' keys
        """
        model = self._load_model()

        logger.info(f"Building index from {len(papers)} papers...")

        # Create search texts combining title + abstract for better retrieval
        texts = [f"{p['title']}. {p['abstract']}" for p in papers]

        # Generate embeddings (normalized for cosine similarity via inner product)
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=64,
        )

        # Build FAISS index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine for normalized vectors
        self.index.add(embeddings.astype(np.float32))
        self.metadata = papers

        # Save index and metadata
        index_path = os.path.join(self.index_dir, "papers.index")
        metadata_path = os.path.join(self.index_dir, "metadata.json")

        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Indexed {len(papers)} papers. Dimension: {dim}")
        logger.info(f"Index saved to {index_path}")

    def load_index(self):
        """Load an existing FAISS index from disk."""
        index_path = os.path.join(self.index_dir, "papers.index")
        metadata_path = os.path.join(self.index_dir, "metadata.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                f"Run 'python scripts/build_seed_index.py' first."
            )

        logger.info(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(index_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        logger.info(f"Loaded index with {self.index.ntotal} vectors and {len(self.metadata)} metadata entries")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Search for papers similar to the query.
        
        Args:
            query: Search query (e.g., paper title + abstract)
            top_k: Number of results to return
            
        Returns:
            List of paper metadata dicts with added 'similarity_score' field
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load_index() or build_index() first.")

        model = self._load_model()

        # Encode query
        query_embedding = model.encode(
            [query],
            normalize_embeddings=True,
        ).astype(np.float32)

        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):  # FAISS returns -1 for empty results
                result = self.metadata[idx].copy()
                result["similarity_score"] = float(score)
                results.append(result)

        return results

    def get_stats(self) -> dict:
        """Return index statistics."""
        return {
            "total_papers": self.index.ntotal if self.index else 0,
            "metadata_count": len(self.metadata),
            "embedding_dim": self.index.d if self.index else 0,
            "index_dir": self.index_dir,
        }
