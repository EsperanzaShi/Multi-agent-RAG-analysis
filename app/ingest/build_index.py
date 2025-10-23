from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROC_DIR = Path("data/processed")
INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)


def load_chunks() -> List[Dict[str, Any]]:
    """Load chunk metadata/text from the processed JSONL file."""
    src = PROC_DIR / "chunks.jsonl"
    if not src.exists():
        raise FileNotFoundError(
            "data/processed/chunks.jsonl not found. Run scripts/01_ingest.py first."
        )
    chunks: List[Dict[str, Any]] = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_and_save_index(
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    normalize: bool = True,
) -> None:
    """
    Build a FAISS index over chunk texts and persist the index + metadata.
    - Uses all-MiniLM-L6-v2 (free, local) by default
    - Normalizes embeddings for Inner Product (cosine) similarity
    """
    chunks = load_chunks()
    if not chunks:
        print("No chunks found. Did ingestion run correctly?")
        return

    texts = [c.get("text", "") for c in chunks]
    model = SentenceTransformer(embed_model)
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
    )
    embs = np.asarray(embs, dtype="float32")
    dim = embs.shape[1]

    index = faiss.IndexFlatIP(dim) if normalize else faiss.IndexFlatL2(dim)
    index.add(embs)

    # Persist index and metadata
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    with open(INDEX_DIR / "meta.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Saved FAISS index with {index.ntotal} vectors to {INDEX_DIR / 'index.faiss'}")
    print(f"Wrote metadata to {INDEX_DIR / 'meta.jsonl'}")


if __name__ == "__main__":
    build_and_save_index()