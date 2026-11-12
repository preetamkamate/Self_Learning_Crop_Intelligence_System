"""
RAG Indexer for Agricultural Guidelines.
Chunks ICAR/IIHR/UAS agricultural documents, generates embeddings using sentence-transformers,
and builds a persistent FAISS index with metadata store.
"""
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def chunk_text(
    text: str,
    source_file: str,
    target_words: int = config.CHUNK_SIZE_WORDS,
    overlap_words: int = config.CHUNK_OVERLAP_WORDS,
) -> List[Dict[str, str]]:
    """
    Split document text into overlapping word chunks preserving section headers.
    """
    sections = re.split(r"(^#+\s+.+$)", text, flags=re.MULTILINE)
    chunks: List[Dict[str, str]] = []
    current_header = Path(source_file).stem.replace("_", " ").title()

    full_text_blocks = []
    for section in sections:
        clean = section.strip()
        if not clean:
            continue
        if clean.startswith("#"):
            current_header = clean.lstrip("#").strip()
        else:
            full_text_blocks.append((current_header, clean))

    for header, block in full_text_blocks:
        words = block.split()
        if not words:
            continue

        if len(words) <= target_words:
            content = f"[{header}]\n{block}"
            chunks.append({
                "source": source_file,
                "header": header,
                "content": content,
                "word_count": len(words),
            })
        else:
            step = max(1, target_words - overlap_words)
            for i in range(0, len(words), step):
                chunk_slice = words[i : i + target_words]
                chunk_text_slice = " ".join(chunk_slice)
                content = f"[{header}]\n{chunk_text_slice}"
                chunks.append({
                    "source": source_file,
                    "header": header,
                    "content": content,
                    "word_count": len(chunk_slice),
                })
                if i + target_words >= len(words):
                    break

    return chunks


def build_faiss_index(
    docs_dir: Path = config.KNOWLEDGE_DOCS_DIR,
    index_path: Path = config.FAISS_INDEX_PATH,
    metadata_path: Path = config.METADATA_PATH,
    model_name: str = config.EMBEDDING_MODEL_ID,
):
    """
    Read markdown documents, embed with sentence-transformers, and save FAISS index.
    """
    logging.info("Indexing knowledge documents from: %s", docs_dir)
    doc_files = list(docs_dir.glob("*.md")) + list(docs_dir.glob("*.txt"))
    if not doc_files:
        raise FileNotFoundError(f"No documents found in {docs_dir}")

    all_chunks: List[Dict[str, str]] = []
    for filepath in sorted(doc_files):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = chunk_text(content, source_file=filepath.name)
        all_chunks.extend(chunks)
        logging.info("Processed %s: generated %d chunks", filepath.name, len(chunks))

    logging.info("Total chunks generated: %d", len(all_chunks))

    # Initialize SentenceTransformer
    logging.info("Loading embedding model: %s", model_name)
    embedder = SentenceTransformer(model_name)
    texts_to_embed = [chunk["content"] for chunk in all_chunks]

    logging.info("Encoding %d chunks...", len(texts_to_embed))
    embeddings = embedder.encode(texts_to_embed, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    # Normalize vectors for Cosine Similarity (IndexFlatIP)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    logging.info("Building FAISS IndexFlatIP (Cosine Similarity) with dimension %d", dim)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logging.info("Indexed %d vectors in FAISS.", index.ntotal)

    # Save artifacts
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    logging.info("FAISS index saved to: %s", index_path)
    logging.info("Metadata saved to: %s", metadata_path)
    return index, all_chunks


if __name__ == "__main__":
    build_faiss_index()

