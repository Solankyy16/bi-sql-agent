"""
rag/schema_embedder.py
──────────────────────
Builds and saves the FAISS vector index from database table descriptions.

WHY THIS EXISTS (The Schema RAG Problem):
    Chinook has 11 tables. Northwind has 13. If a user uploads a Kaggle DB,
    it might have 30+ tables. get_table_info() on all of them = ~6,000+ tokens.

    That's expensive (slow, costly) and actually HURTS the LLM — too much
    irrelevant context causes hallucinations.

    Instead, we:
      1. Write a natural-language description of each table
      2. Embed those descriptions as vectors (numbers that capture meaning)
      3. Store in FAISS (a fast similarity search index)
      4. At query time, find the 2-3 tables most relevant to the user's question
      5. Inject ONLY those schemas into the prompt

    Result: "Who are the top customers?" → retrieves Customer + Invoice tables.
            The Playlist and Genre tables never appear in the prompt.

HOW EMBEDDINGS WORK (simple version):
    An embedding model converts text into a list of numbers (a "vector").
    Similar texts → similar vectors. "customer" and "client" end up close together.
    FAISS stores these vectors and can find the closest ones to any query vector.
    
    We use all-MiniLM-L6-v2 — a small, fast, free model that runs locally.
    No API needed. It downloads once (~90MB) and caches in ~/.cache/huggingface/
"""

import os
import json
import pickle
from pathlib import Path
from typing import Optional

# These imports happen at function call time to avoid slow startup
# (sentence_transformers takes ~2s to import)


# ── Paths ─────────────────────────────────────────────────────────────────────
RAG_DIR = Path(__file__).parent
INDEX_DIR = RAG_DIR / "faiss_index"

# We store both the FAISS index and the metadata (table names + schemas)
# as separate files. FAISS handles the vectors; we handle the text lookup.
INDEX_FILE    = INDEX_DIR / "index.faiss"
METADATA_FILE = INDEX_DIR / "metadata.pkl"


def _build_table_descriptions(full_schema: str, table_names: list[str]) -> list[dict]:
    """
    Converts raw schema SQL into human-readable descriptions for embedding.

    WHY NOT EMBED THE RAW SQL?
        Raw SQL: "CREATE TABLE Invoice (InvoiceId INTEGER PRIMARY KEY, CustomerId INTEGER...)"
        This works but is noisy. The embedding model does better with natural language.

        Better: "The Invoice table stores sales transaction records. It contains columns:
                 InvoiceId (primary key), CustomerId (foreign key to Customer),
                 InvoiceDate, BillingAddress, Total..."

    For now, we use a hybrid: table name + the raw CREATE TABLE block.
    This is good enough for a portfolio project. In production you'd write
    richer descriptions or use an LLM to auto-generate them.
    """
    # Split the full schema string by double newlines to get per-table blocks
    # LangChain's get_table_info() separates tables this way
    table_blocks = [b.strip() for b in full_schema.split("\n\n") if b.strip()]

    documents = []
    for table_name in table_names:
        # Find the CREATE TABLE block for this specific table
        schema_block = next(
            (b for b in table_blocks if f"CREATE TABLE {table_name}" in b
             or f'CREATE TABLE "{table_name}"' in b),
            f"Table: {table_name}"  # fallback if parsing fails
        )

        documents.append({
            "table_name":  table_name,
            "description": f"Table name: {table_name}\n\n{schema_block}",
            "schema":      schema_block,
        })

    return documents


def build_index(full_schema: str, table_names: list[str]) -> None:
    """
    Builds and saves the FAISS index for the given database schema.

    Called by: rag/retriever.py when no index exists yet, or when the DB changes.

    Args:
        full_schema:  Output of db.get_table_info() — full CREATE TABLE statements
        table_names:  Output of db.get_usable_table_names()
    """
    # Lazy import — only load the heavy libraries when actually needed
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np

    print("🔨 Building Schema RAG index...")
    print(f"   Tables to index: {len(table_names)}")

    # Step 1: Create the text documents we'll embed
    documents = _build_table_descriptions(full_schema, table_names)

    # Step 2: Load the embedding model
    # all-MiniLM-L6-v2: 384-dimensional vectors, fast, good quality
    # Downloads ~90MB on first run, then cached locally
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 3: Embed all table descriptions
    # encode() returns a numpy array of shape (num_tables, 384)
    descriptions = [doc["description"] for doc in documents]
    embeddings = model.encode(descriptions, show_progress_bar=False)

    # Step 4: Normalise vectors for cosine similarity
    # FAISS IndexFlatIP uses inner product — normalised vectors make it cosine similarity
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    # Step 5: Build the FAISS index
    # IndexFlatIP = "Flat Inner Product" — exact search, no approximation
    # For 30 tables this is instant. For 10,000+ tables, use IndexIVFFlat instead.
    dimension = embeddings.shape[1]  # 384 for all-MiniLM-L6-v2
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Step 6: Save to disk
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))

    # Save metadata (table names + schemas) so we can look up results by index
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(documents, f)

    print(f"   ✓ Index saved to {INDEX_DIR} ({len(documents)} tables indexed)")


def index_exists() -> bool:
    """Returns True if a saved index exists and can be loaded."""
    return INDEX_FILE.exists() and METADATA_FILE.exists()


def clear_index() -> None:
    """Deletes the saved index. Called when the user switches databases."""
    if INDEX_FILE.exists():
        INDEX_FILE.unlink()
    if METADATA_FILE.exists():
        METADATA_FILE.unlink()
