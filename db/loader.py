"""
db/loader.py
────────────
The database connection factory. This is the module that makes our app
"database-agnostic" — it doesn't care if you upload Chinook, Northwind,
a hospital DB, or a Kaggle dataset. It connects to anything.

Key design decisions:
  1. Demo DBs are loaded directly from db/ folder (committed to repo)
  2. Uploaded DBs are written to OS temp dir, never the project root
  3. atexit cleanup means temp files vanish when the app stops
  4. Connection objects are cached in session_state — not recreated on every rerun
"""

import os
import tempfile
import atexit
from pathlib import Path
from typing import Optional

from langchain_community.utilities import SQLDatabase

# ── Demo database registry ────────────────────────────────────────────────────
# Add new bundled databases here. The key is the display label in the sidebar;
# the value is the path relative to this file's location.
#
# WHY RELATIVE PATH: Using Path(__file__).parent means this works correctly
# whether you run from the project root, a subfolder, or a CI environment.
DEMO_DBS: dict[str, str] = {
    "🎵 Chinook  (Music Sales)":    str(Path(__file__).parent / "chinook.db"),
    "📦 Northwind (Logistics)":     str(Path(__file__).parent / "northwind.db"),
}

# ── Temp file tracking ────────────────────────────────────────────────────────
# Every uploaded DB we write to disk gets registered here.
# atexit calls _cleanup_temp_files() when the Streamlit process exits,
# deleting all temp files automatically. No stale files left behind.
_temp_files: list[str] = []


def _cleanup_temp_files() -> None:
    """Deletes all temp DB files created during this session."""
    for path in _temp_files:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass  # Already deleted — fine


atexit.register(_cleanup_temp_files)


# ── Public API ────────────────────────────────────────────────────────────────

def load_demo_db(label: str) -> SQLDatabase:
    """
    Load one of the bundled demo databases by its display label.

    Example:
        db = load_demo_db("🎵 Chinook  (Music Sales)")

    Raises FileNotFoundError if setup_dbs.py hasn't been run yet.
    """
    path = DEMO_DBS[label]

    if not Path(path).exists():
        raise FileNotFoundError(
            f"Database not found: {path}\n"
            f"Run setup first:  python db/setup_dbs.py"
        )

    # SQLDatabase.from_uri() is LangChain's wrapper around SQLAlchemy.
    # It gives us: db.run(sql), db.get_table_info(), db.get_usable_table_names()
    # The sqlite:/// prefix is SQLAlchemy's URI format for SQLite files.
    return SQLDatabase.from_uri(f"sqlite:///{path}")


def load_db_from_upload(uploaded_file) -> tuple[SQLDatabase, str]:
    """
    Takes a Streamlit UploadedFile object (from st.file_uploader),
    writes it safely to a temp directory, and returns a connected SQLDatabase.

    Returns:
        (db, temp_path) — the connection AND the path, so the caller
        can display the filename or pass it to other utilities.

    WHY tempfile.NamedTemporaryFile?
        - Writes to the OS temp dir (/tmp on Linux/Mac, %TEMP% on Windows)
        - Not the project root — won't pollute your repo
        - delete=False because we need to read it back after closing
          (on Windows, you can't read an open NamedTemporaryFile)
        - We handle deletion ourselves via atexit
    """
    suffix = Path(uploaded_file.name).suffix  # preserves .db or .sqlite

    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        prefix="bi_agent_upload_",
    )
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()

    _temp_files.append(tmp.name)  # register for cleanup on exit

    db = SQLDatabase.from_uri(f"sqlite:///{tmp.name}")
    return db, tmp.name


def get_schema_summary(db: SQLDatabase) -> dict:
    """
    Returns a summary dict used by:
      1. The sidebar UI (table count + names)
      2. Schema RAG (full_info injected into the LLM prompt)

    Returns:
        {
            "table_names": ["Artist", "Album", ...],
            "table_count": 11,
            "full_info":   "CREATE TABLE Artist (...)\n\nCREATE TABLE Album (...)\n..."
        }

    WHY get_table_info() instead of get_usable_table_names()?
        get_table_info() returns the full CREATE TABLE statements + 3 sample rows.
        This is what we inject into the LLM prompt — the LLM needs to see column
        names and types to write valid SQL, not just table names.
    """
    tables = db.get_usable_table_names()

    return {
        "table_names": tables,
        "table_count": len(tables),
        "full_info":   db.get_table_info(),
    }
