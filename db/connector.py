"""
db/connector.py
───────────────
Universal database connection factory.

THE CORE INSIGHT:
    SQLAlchemy uses a "connection URI" string to connect to any database.
    LangChain's SQLDatabase.from_uri() accepts any SQLAlchemy URI.
    So we don't need to change the agent at all — we just need to build
    the right URI string from whatever the user gives us.

    SQLite:     sqlite:///path/to/file.db
    PostgreSQL: postgresql+psycopg2://user:pass@host:port/dbname
    MySQL:      mysql+pymysql://user:pass@host:port/dbname
    SQL Server: mssql+pyodbc://user:pass@host:port/dbname?driver=...

    MongoDB is the ONLY exception — it's not SQL, so it cannot use
    SQLDatabase at all. It gets its own separate handler.

PLUGIN ARCHITECTURE:
    Each DB type is a "connector" — a function that takes credentials
    and returns a (SQLDatabase, label) tuple. Adding a new DB type
    means adding one function + one entry in DB_TYPES. Nothing else changes.

SECURITY:
    Credentials are NEVER stored in session_state beyond what's needed
    to display a label. The actual password never gets written to disk
    or committed anywhere. Connection objects hold the live connection;
    credentials are discarded after the connection is made.
"""

from __future__ import annotations

import tempfile
import os
import atexit
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable

from langchain_community.utilities import SQLDatabase


# ── Temp file registry (same pattern as loader.py) ───────────────────────────
_temp_files: list[str] = []

def _cleanup() -> None:
    for p in _temp_files:
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass

atexit.register(_cleanup)


# ── DB type registry ──────────────────────────────────────────────────────────

@dataclass
class DBType:
    """
    Describes one supported database type.

    Fields:
        label:       Display name shown in the sidebar dropdown
        icon:        Emoji shown next to the label
        uri_template: SQLAlchemy URI with {placeholders} for credentials
        fields:      Ordered list of credential field names the user must fill
        field_labels: Human-readable labels for each field
        field_types:  "text" or "password" — controls input masking
        requires_driver: pip package the user must install (shown as a hint)
        is_nosql:    True for MongoDB — triggers a different agent branch
    """
    label:           str
    icon:            str
    uri_template:    str
    fields:          list[str]          = field(default_factory=list)
    field_labels:    list[str]          = field(default_factory=list)
    field_types:     list[str]          = field(default_factory=list)
    requires_driver: Optional[str]      = None
    is_nosql:        bool               = False


# Registry of all supported database types.
# To add a new one: add an entry here. Nothing else in the codebase changes.
DB_TYPES: dict[str, DBType] = {

    "sqlite_file": DBType(
        label="SQLite (upload file)",
        icon="📁",
        uri_template="",          # handled separately via file uploader
        fields=[],
        field_labels=[],
        field_types=[],
    ),

    "sqlite_url": DBType(
        label="SQLite (local path)",
        icon="🗄",
        uri_template="sqlite:///{path}",
        fields=["path"],
        field_labels=["Absolute file path"],
        field_types=["text"],
    ),

    "postgresql": DBType(
        label="PostgreSQL",
        icon="🐘",
        uri_template="postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}",
        fields=["host", "port", "database", "user", "password"],
        field_labels=["Host", "Port", "Database name", "Username", "Password"],
        field_types=["text", "text", "text", "text", "password"],
        requires_driver="psycopg2-binary",
    ),

    "mysql": DBType(
        label="MySQL / MariaDB",
        icon="🐬",
        uri_template="mysql+pymysql://{user}:{password}@{host}:{port}/{database}",
        fields=["host", "port", "database", "user", "password"],
        field_labels=["Host", "Port", "Database name", "Username", "Password"],
        field_types=["text", "text", "text", "text", "password"],
        requires_driver="pymysql",
    ),

    "mssql": DBType(
        label="SQL Server (MSSQL)",
        icon="🪟",
        uri_template="mssql+pyodbc://{user}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server",
        fields=["host", "port", "database", "user", "password"],
        field_labels=["Host", "Port", "Database name", "Username", "Password"],
        field_types=["text", "text", "text", "text", "password"],
        requires_driver="pyodbc",
    ),

    "mongodb": DBType(
        label="MongoDB (self-hosted)",
        icon="🍃",
        uri_template="mongodb://{user}:{password}@{host}:{port}/{database}",
        fields=["host", "port", "database", "user", "password"],
        field_labels=["Host", "Port", "Database name", "Username", "Password"],
        field_types=["text", "text", "text", "text", "password"],
        requires_driver="pymongo",
        is_nosql=True,
    ),

    "mongodb_atlas": DBType(
        label="MongoDB Atlas (cloud)",
        icon="☁️",
        # Atlas uses SRV format — no port, cluster hostname looks like:
        # cluster0.abc12.mongodb.net
        # Full URI becomes: mongodb+srv://user:pass@cluster0.abc12.mongodb.net/dbname
        uri_template="mongodb+srv://{user}:{password}@{host}/{database}?retryWrites=true&w=majority",
        fields=["host", "database", "user", "password"],
        field_labels=[
            "Cluster hostname (e.g. cluster0.abc12.mongodb.net)",
            "Database name",
            "Username",
            "Password",
        ],
        field_types=["text", "text", "text", "password"],
        requires_driver="pymongo",
        is_nosql=True,
    ),

    "csv": DBType(
        label="CSV file (auto-convert)",
        icon="📊",
        uri_template="",          # handled separately — converts CSV → in-memory SQLite
        fields=[],
        field_labels=[],
        field_types=[],
    ),
}


# ── Connection builders ───────────────────────────────────────────────────────

def connect_from_credentials(db_type_key: str, credentials: dict) -> tuple[SQLDatabase, str]:
    """
    Main entry point. Takes a DB type key and a dict of credentials,
    returns a connected SQLDatabase and a display label.

    This is the only function app.py needs to call for credential-based DBs.

    Args:
        db_type_key:  Key from DB_TYPES (e.g. "postgresql", "mysql")
        credentials:  Dict matching the fields defined in DBType
                      e.g. {"host": "localhost", "port": "5432", ...}

    Returns:
        (db, label)  where label is safe to display (no password)

    Raises:
        ValueError: if db_type_key is unknown or required field is missing
        ImportError: if the required driver package isn't installed
        Exception:  if the database connection fails (wrong credentials, etc.)
    """
    if db_type_key not in DB_TYPES:
        raise ValueError(f"Unknown database type: {db_type_key}")

    db_type = DB_TYPES[db_type_key]

    # Check required driver is installed
    if db_type.requires_driver:
        _check_driver(db_type.requires_driver, db_type.label)

    # Build the URI by substituting credentials into the template
    uri = db_type.uri_template.format(**credentials)

    # Build a safe display label (no password)
    safe_label = _build_safe_label(db_type_key, credentials)

    # Connect — this is where wrong credentials will raise an exception
    db = SQLDatabase.from_uri(uri)

    return db, safe_label


def connect_from_csv(uploaded_file) -> tuple[SQLDatabase, str]:
    """
    Converts an uploaded CSV file to an in-memory SQLite database.

    WHY THIS IS POWERFUL:
        The user uploads a CSV → we load it into a Pandas DataFrame →
        we write it to a temp SQLite file → agent queries it with SQL.
        The user never knows SQL is involved. They just upload a CSV
        and ask questions.

    The table name inside SQLite is derived from the CSV filename,
    so the LLM can write: SELECT * FROM spotify_tracks LIMIT 5
    """
    import pandas as pd
    import sqlite3

    # Read the CSV into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Derive a clean table name from the filename
    # "Spotify Tracks 2024.csv" → "spotify_tracks_2024"
    raw_name = Path(uploaded_file.name).stem
    table_name = _sanitize_table_name(raw_name)

    # Write to a temp SQLite file
    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".db",
        prefix="bi_agent_csv_",
    )
    tmp.close()
    _temp_files.append(tmp.name)

    conn = sqlite3.connect(tmp.name)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()

    db = SQLDatabase.from_uri(f"sqlite:///{tmp.name}")
    label = f"{uploaded_file.name} → table: {table_name}"

    return db, label


def connect_mongodb(credentials: dict, db_type_key: str = "mongodb"):
    """
    Connects to MongoDB (self-hosted or Atlas) and returns a pymongo Database.

    Supports two URI formats:
      Standard:  mongodb://user:pass@host:port/db
      Atlas SRV: mongodb+srv://user:pass@cluster.host/db?retryWrites=true&w=majority

    Atlas requires db_type_key="mongodb_atlas" — no port field, SRV scheme.
    """
    _check_driver("pymongo", "MongoDB")

    from pymongo import MongoClient

    db_type = DB_TYPES.get(db_type_key, DB_TYPES["mongodb"])
    uri = db_type.uri_template.format(**credentials)

    # Atlas SRV connections need TLS — pymongo handles this automatically
    # with the mongodb+srv:// scheme. We just need a longer timeout for
    # the initial DNS SRV lookup (Atlas resolves via DNS, not direct TCP).
    timeout = 15000 if "atlas" in db_type_key else 5000

    client = MongoClient(uri, serverSelectionTimeoutMS=timeout)

    # Ping to verify credentials before returning
    try:
        client.admin.command("ping")
    except Exception as e:
        raise ConnectionError(
            f"MongoDB connection failed: {e}\n\n"
            f"Atlas checklist:\n"
            f"  1. Whitelist your IP in Atlas Network Access\n"
            f"  2. Username/password are correct\n"
            f"  3. Cluster hostname copied exactly (no mongodb+srv:// prefix needed here)"
        )

    return client[credentials["database"]]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_driver(package: str, db_label: str) -> None:
    """
    Checks if a required driver package is installed.
    Raises ImportError with a helpful install instruction if not.
    """
    import importlib
    # Map pip package names to importable module names
    import_map = {
        "psycopg2-binary": "psycopg2",
        "pymysql":         "pymysql",
        "pyodbc":          "pyodbc",
        "pymongo":         "pymongo",
    }
    module_name = import_map.get(package, package)
    try:
        importlib.import_module(module_name)
    except ImportError:
        raise ImportError(
            f"Connecting to {db_label} requires the '{package}' package.\n"
            f"Install it with:  pip install {package}"
        )


def _build_safe_label(db_type_key: str, credentials: dict) -> str:
    """
    Builds a display label from credentials that NEVER includes the password.
    This label is safe to store in session_state and display in the UI.

    Examples:
        postgresql → "PostgreSQL: myuser@localhost:5432/mydb"
        mysql      → "MySQL: root@192.168.1.10:3306/shop"
        sqlite_url → "SQLite: /home/user/data.db"
    """
    db_type = DB_TYPES[db_type_key]

    if db_type_key == "sqlite_url":
        return f"SQLite: {credentials.get('path', '')}"

    parts = []
    if "user" in credentials:
        parts.append(credentials["user"])
    if "host" in credentials:
        host_port = credentials["host"]
        if "port" in credentials:
            host_port += f":{credentials['port']}"
        parts.append(f"@{host_port}")
    if "database" in credentials:
        parts.append(f"/{credentials['database']}")

    return f"{db_type.label}: {''.join(parts)}"


def _sanitize_table_name(raw: str) -> str:
    """
    Converts a filename stem into a valid SQL table name.
    "Spotify Tracks 2024" → "spotify_tracks_2024"
    """
    import re
    name = raw.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)  # replace non-alphanumeric with _
    name = name.strip("_")
    if name and name[0].isdigit():
        name = "t_" + name              # SQL table names can't start with digits
    return name or "uploaded_data"


def get_db_type_options() -> list[tuple[str, str]]:
    """
    Returns display options for the sidebar selectbox.
    Each tuple is (key, display_string).
    """
    return [
        (key, f"{dt.icon}  {dt.label}")
        for key, dt in DB_TYPES.items()
    ]
