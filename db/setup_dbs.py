"""
db/setup_dbs.py
───────────────
One-time setup script. Downloads the two demo databases from their official
GitHub releases into the db/ folder.

Run once before starting the app:
    python db/setup_dbs.py

Both databases are:
  - Open-source and publicly licensed (safe to commit to GitHub)
  - SQLite format — no server, no installation, just a file
  - Rich enough to test complex SQL JOINs across multiple tables

Chinook  — a digital music store (like iTunes). 11 tables.
           Great for: revenue analysis, artist rankings, customer geography.

Northwind — a food import/export company. 13 tables.
            Great for: order analysis, shipping, supplier relationships.
"""

import urllib.request
import os
from pathlib import Path

# ── Database sources ──────────────────────────────────────────────────────────
# These are the official, canonical SQLite versions of each database.
DBS = {
    "chinook.db": (
        "https://github.com/lerocha/chinook-database/raw/master/"
        "ChinookDatabase/DataSources/Chinook_Sqlite.sqlite",
        "Music store — 11 tables (Artist, Album, Track, Invoice, Customer…)",
    ),
    "northwind.db": (
        "https://github.com/jpwhite3/northwind-SQLite3/raw/main/"
        "dist/northwind.db",
        "Logistics company — 13 tables (Orders, Products, Customers, Suppliers…)",
    ),
}

def download_db(filename: str, url: str, description: str) -> None:
    """Downloads a database file if it doesn't already exist."""
    # Build the path relative to THIS script's location (db/ folder)
    # Using Path(__file__).parent means this works from any working directory
    target = Path(__file__).parent / filename

    if target.exists():
        print(f"  ✓ {filename} already exists — skipping download")
        return

    print(f"  ↓ Downloading {filename}  ({description})")
    print(f"    Source: {url}")

    try:
        # urllib.request is built into Python — no extra dependency needed
        urllib.request.urlretrieve(url, target)
        size_kb = target.stat().st_size // 1024
        print(f"  ✓ {filename} saved ({size_kb} KB)")
    except Exception as e:
        print(f"  ✗ Failed to download {filename}: {e}")
        print(f"    Try downloading manually from: {url}")
        raise


if __name__ == "__main__":
    print("\n🗄  Setting up demo databases...\n")

    for filename, (url, description) in DBS.items():
        download_db(filename, url, description)

    print("\n✅ All databases ready. You can now run:  streamlit run app.py\n")
