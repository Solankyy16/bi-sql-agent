"""
rag/retriever.py
────────────────
Retrieves the most relevant table schemas for a given user question.

This is the "R" in Schema RAG (Retrieval-Augmented Generation).

THE FLOW:
    User asks: "Which artist had the highest total sales?"
               ↓
    We embed the question: [0.12, -0.34, 0.87, ...]  (384 numbers)
               ↓
    FAISS finds the 3 closest table vectors:
        → Invoice (score: 0.91) ← most relevant
        → Artist  (score: 0.88)
        → Track   (score: 0.71)
               ↓
    We return those 3 CREATE TABLE blocks as a single string
               ↓
    That string gets injected into the LLM prompt as {schema}
    
    The LLM never sees the other 8 Chinook tables. Clean, focused prompt.
"""

import pickle
from pathlib import Path
from typing import Optional

from rag.schema_embedder import (
    INDEX_FILE,
    METADATA_FILE,
    build_index,
    index_exists,
)

# Singleton cache — load the model and index once per app session, not per query
_model = None
_index = None
_metadata: Optional[list[dict]] = None


def _load_resources():
    """
    Loads the FAISS index and embedding model into module-level cache.
    Called lazily on first retrieval — avoids slow startup.
    """
    global _model, _index, _metadata

    import faiss
    from sentence_transformers import SentenceTransformer

    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")

    if _index is None:
        _index = faiss.read_index(str(INDEX_FILE))

    if _metadata is None:
        with open(METADATA_FILE, "rb") as f:
            _metadata = pickle.load(f)


def invalidate_cache() -> None:
    """
    Clears the in-memory cache.
    Must be called when the user switches databases — otherwise the old
    index stays in memory even after a new index file is written to disk.
    """
    global _model, _index, _metadata
    # Keep _model — the embedding model is database-independent
    _index    = None
    _metadata = None


def get_relevant_schema(
    question: str,
    full_schema: str,
    table_names: list[str],
    top_k: int = 3,
) -> str:
    """
    Main entry point. Returns a schema string containing only the tables
    most relevant to the user's question.

    Args:
        question:     The user's natural language question
        full_schema:  db.get_table_info() output (used if index needs rebuilding)
        table_names:  db.get_usable_table_names() output
        top_k:        How many tables to retrieve. 3 is the sweet spot:
                      - Too few (1-2): LLM might miss a JOIN it needs
                      - Too many (5+): Approaches the original token count problem

    Returns:
        A string of CREATE TABLE statements for the top_k most relevant tables.
        This goes directly into the LLM prompt as {schema}.
    """
    import numpy as np
    import faiss

    # Build the index if it doesn't exist yet (first run, or after DB switch)
    if not index_exists():
        build_index(full_schema, table_names)

    _load_resources()

    # Embed the user's question using the same model that embedded the tables
    # CRITICAL: must use the same model — different models produce incompatible vectors
    question_vector = _model.encode([question], show_progress_bar=False)
    question_vector = question_vector.astype("float32")
    faiss.normalize_L2(question_vector)

    # Search: returns (distances, indices) for the top_k nearest vectors
    # distances[0] = similarity scores, indices[0] = positions in _metadata
    distances, indices = _index.search(question_vector, min(top_k, len(_metadata)))

    # Build the schema string from the retrieved table metadata
    retrieved_schemas = []
    for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
        if idx == -1:  # FAISS returns -1 for "no result" slots
            continue
        table_doc = _metadata[idx]
        retrieved_schemas.append(
            f"-- Table: {table_doc['table_name']} (relevance: {score:.2f})\n"
            f"{table_doc['schema']}"
        )

    return "\n\n".join(retrieved_schemas)
