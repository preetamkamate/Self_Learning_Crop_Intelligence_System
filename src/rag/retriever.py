"""
RAG Retriever for Agricultural Guidelines.
Loads FAISS index and metadata to retrieve top-k authoritative ICAR/IIHR/UAS advice
with relevance scoring and document citations.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class AgriculturalRetriever:
    """Semantic search engine over regional Indian agricultural knowledge base."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        index_path: Path = config.FAISS_INDEX_PATH,
        metadata_path: Path = config.METADATA_PATH,
        model_name: str = config.EMBEDDING_MODEL_ID,
    ):
        if self._initialized:
            return

        if not (index_path.exists() and metadata_path.exists()):
            raise FileNotFoundError(
                f"FAISS index or metadata missing. Run indexer first: {index_path}"
            )

        logging.info("Loading FAISS index from: %s", index_path)
        self.index = faiss.read_index(str(index_path))

        logging.info("Loading metadata store from: %s", metadata_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        logging.info("Loading SentenceTransformer model: %s", model_name)
        self.embedder = SentenceTransformer(model_name)
        self._initialized = True

    def retrieve(
        self,
        query: str,
        top_k: int = config.RAG_TOP_K,
        min_score: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant chunks for a user query.

        Args:
            query: Agricultural query text
            top_k: Number of chunks to retrieve
            min_score: Minimum cosine similarity threshold

        Returns:
            List of ranked chunks with citation, header, content, and similarity score.
        """
        # Encode query
        query_vector = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_vector)

        # Search index
        scores, indices = self.index.search(query_vector, min(top_k * 2, self.index.ntotal))

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            if score < min_score:
                continue

            chunk_meta = self.metadata[idx]
            results.append({
                "source": chunk_meta["source"],
                "header": chunk_meta["header"],
                "content": chunk_meta["content"],
                "score": round(float(score), 4),
                "relevance_percent": round(float(score) * 100, 1),
            })
            if len(results) >= top_k:
                break

        return results


def get_retriever() -> AgriculturalRetriever:
    """Helper function to obtain cached AgriculturalRetriever instance."""
    return AgriculturalRetriever()


if __name__ == "__main__":
    retriever = get_retriever()
    test_q = "How to control early blight in tomato with chemical dosage?"
    matches = retriever.retrieve(test_q, top_k=2)
    print(f"\nQuery: {test_q}")
    for i, m in enumerate(matches, 1):
        print(f"\n[{i}] Source: {m['source']} | Score: {m['score']} ({m['relevance_percent']}%)")
        print(f"Header: {m['header']}")
        print(m['content'][:200] + "...")

