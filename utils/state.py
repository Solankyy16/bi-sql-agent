"""
utils/state.py
──────────────
Centralised session state management for the Streamlit app.

THE PROBLEM THIS SOLVES:
    Streamlit reruns your entire script on every user interaction.
    Without session_state, every variable resets to its default.
    
    The naive approach scatters guards like this everywhere:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "active_db" not in st.session_state:
            st.session_state.active_db = None
        # ... repeated 6+ times across the file
    
    Problems: easy to miss a key, hard to see the full state contract,
    no single place to add new keys.

THE SOLUTION:
    AppState dataclass defines the full contract in one place.
    init_session_state() iterates over it and initialises every key once.
    Add a new key? Add one line to AppState. Done.
"""

from __future__ import annotations

import streamlit as st
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AppState:
    """
    Defines every session_state key and its default value.
    Each field maps directly to a st.session_state key.
    """
    messages:        list[dict] = field(default_factory=list)
    active_db:       Any        = None
    active_db_label: str        = ""
    is_nosql:        bool       = False
    mongo_db:        Any        = None
    last_sql:        str        = ""
    last_df:         Any        = None
    voice_enabled:   bool       = True
    retry_count:     int        = 0
    last_audio_hash: str        = ""


def init_session_state() -> None:
    """
    Call this ONCE at the very top of app.py, before any other st.* calls.

    For each key in AppState:
      - If the key doesn't exist in session_state yet → initialise it
      - If it already exists → leave it alone (preserves data across reruns)

    This is idempotent: safe to call multiple times, only acts on first run.
    """
    defaults = AppState()
    # vars() converts the dataclass to a plain dict: {"messages": [], ...}
    for key, value in vars(defaults).items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_chat() -> None:
    """
    Clears chat history and last query artifacts.
    KEEPS the active DB connection — user doesn't need to re-upload.

    Called when: user clicks "Clear chat" in the sidebar,
    or when the user switches to a different database.
    """
    st.session_state.messages   = []
    st.session_state.last_sql   = ""
    st.session_state.last_df    = None
    st.session_state.retry_count = 0
    st.rerun()  # Force Streamlit to re-render with the cleared state


def full_reset() -> None:
    """
    Nuclear option — wipes ALL session state and reruns.
    Used when something is truly broken and needs a clean slate.
    Not exposed in the normal UI, but useful during development.
    """
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
