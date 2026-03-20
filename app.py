"""
app.py — Enterprise BI Agent
"""
import os
import hashlib
from pathlib import Path

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from utils.state import init_session_state, reset_chat
from db.loader import DEMO_DBS, load_demo_db, load_db_from_upload, get_schema_summary
from db.connector import (
    DB_TYPES, get_db_type_options,
    connect_from_credentials, connect_from_csv, connect_mongodb,
)
from rag.retriever import get_relevant_schema, invalidate_cache
from rag.schema_embedder import clear_index
from agent.graph import run_agent
from charts.auto_chart import auto_chart
from voice.stt import transcribe_audio
from voice.tts import synthesize_speech

st.set_page_config(
    page_title="BI Agent",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar polish */
[data-testid="stSidebar"] { border-right: 0.5px solid rgba(136,135,128,0.18); }

/* Chat bubbles */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 4px 0;
}

/* SQL expander accent */
[data-testid="stExpander"] {
    border-left: 3px solid #7c6af5 !important;
    border-radius: 0 8px 8px 0 !important;
}

/* Demo DB buttons */
.demo-btn button {
    background: transparent !important;
    border: 0.5px solid rgba(124,106,245,0.4) !important;
    border-radius: 8px !important;
    color: #7c6af5 !important;
    font-size: 13px !important;
    transition: all 0.15s !important;
}
.demo-btn button:hover {
    background: rgba(124,106,245,0.08) !important;
    border-color: #7c6af5 !important;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 10px;
    margin-bottom: 12px;
}
.metric-card {
    flex: 1;
    background: rgba(124,106,245,0.06);
    border: 0.5px solid rgba(124,106,245,0.2);
    border-radius: 10px;
    padding: 10px 14px;
}
.metric-card .val { font-size: 20px; font-weight: 500; color: #7c6af5; }
.metric-card .lbl { font-size: 11px; color: #888780; margin-top: 2px; }

/* Connected badge */
.connected-badge {
    display: inline-block;
    background: rgba(29,158,117,0.12);
    border: 0.5px solid rgba(29,158,117,0.35);
    border-radius: 6px;
    color: #0F6E56;
    font-size: 12px;
    padding: 3px 10px;
    margin-bottom: 8px;
}

/* Retry caption */
.retry-note { font-size: 12px; color: #BA7517; }
</style>
""", unsafe_allow_html=True)

init_session_state()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _clear_and_reset() -> None:
    clear_index()
    invalidate_cache()
    st.session_state.active_db       = None
    st.session_state.active_db_label = ""
    st.session_state.is_nosql        = False
    st.session_state.mongo_db        = None
    st.session_state.messages        = []
    st.session_state.last_sql        = ""
    st.session_state.last_df         = None


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> None:
    with st.sidebar:
        # ── Brand header ──────────────────────────────────────────────────────
        st.markdown("""
        <div style='padding:8px 0 4px'>
          <div style='font-size:20px;font-weight:600;letter-spacing:-0.3px'>BI Agent</div>
          <div style='font-size:12px;color:#888780;margin-top:2px'>
            Talk to any database in plain English
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        # ── Active connection badge ────────────────────────────────────────────
        if st.session_state.active_db is not None or st.session_state.mongo_db is not None:
            label = st.session_state.active_db_label or "Connected"
            # Show just the filename/db name, not the full fingerprint
            display = label.split("_")[0] if "_" in label and label[0].islower() else label
            st.markdown(
                f'<div class="connected-badge">Connected: {display[:32]}</div>',
                unsafe_allow_html=True
            )
            # Disconnect button
            if st.button("Disconnect", use_container_width=True, type="secondary"):
                _clear_and_reset()
                st.rerun()
            st.divider()

        # ── Source type selector ──────────────────────────────────────────────
        st.markdown("**Connect a database**")

        options = get_db_type_options()
        keys    = [k for k, _ in options]
        labels  = [l for _, l in options]

        selected_idx = st.selectbox(
            "db_type", range(len(labels)),
            format_func=lambda i: labels[i],
            label_visibility="collapsed",
        )
        db_type_key = keys[selected_idx]
        db_type     = DB_TYPES[db_type_key]

        if db_type.requires_driver:
            st.caption(f"Requires: `pip install {db_type.requires_driver}`")

        # ── Demo databases — clickable cards ──────────────────────────────────
        if db_type_key == "sqlite_file":
            _sidebar_demos()
            st.markdown("<div style='margin:8px 0 4px;font-size:12px;color:#888780'>or upload your own</div>",
                        unsafe_allow_html=True)
            _sidebar_sqlite_upload()

        elif db_type_key == "csv":
            _sidebar_csv_upload()

        elif db_type_key in ("mongodb", "mongodb_atlas"):
            _sidebar_credential_form(db_type_key, is_nosql=True)

        else:
            _sidebar_credential_form(db_type_key, is_nosql=False)

        # ── Schema inspector ──────────────────────────────────────────────────
        if st.session_state.active_db is not None:
            st.divider()
            _render_schema_inspector()
        elif st.session_state.mongo_db is not None:
            st.divider()
            _render_mongo_inspector()

        # ── Settings ──────────────────────────────────────────────────────────
        st.divider()
        st.markdown("**Settings**")
        st.session_state.voice_enabled = st.toggle(
            "Voice output (gTTS)",
            value=st.session_state.voice_enabled,
        )

        # ── Last SQL + clear chat ──────────────────────────────────────────────
        if st.session_state.last_sql:
            st.divider()
            st.markdown("**Last SQL**")
            st.code(st.session_state.last_sql, language="sql")

        if st.session_state.messages:
            st.divider()
            if st.button("Clear chat", use_container_width=True):
                reset_chat()


def _sidebar_demos() -> None:
    """Clickable demo database buttons — actually loads and connects them."""
    st.markdown("<div style='font-size:12px;color:#888780;margin-bottom:6px'>Demo databases</div>",
                unsafe_allow_html=True)

    demo_meta = {
        "Chinook (Music Sales)": {
            "key": "chinook",
            "desc": "11 tables — artists, albums, invoices, customers",
            "questions": [
                "Who are the top 5 customers by total spend?",
                "Which genre sold the most tracks?",
                "Show monthly revenue for 2013",
            ],
        },
        "Northwind (Logistics)": {
            "key": "northwind",
            "desc": "13 tables — orders, products, suppliers, employees",
            "questions": [
                "Which product category has highest revenue?",
                "Who are the top 5 employees by orders handled?",
                "Show orders shipped to Germany",
            ],
        },
    }

    for demo_label, meta in demo_meta.items():
        # Find the matching key in DEMO_DBS
        db_key = next((k for k in DEMO_DBS if meta["key"] in k.lower()), None)
        if db_key is None:
            continue

        # Check if this demo is currently active
        is_active = meta["key"] in st.session_state.active_db_label.lower()

        col_btn, col_info = st.columns([3, 1])
        with col_btn:
            btn_label = f"{'✓ ' if is_active else ''}{demo_label}"
            if st.button(btn_label, key=f"demo_{meta['key']}", use_container_width=True):
                if not is_active:
                    _clear_and_reset()
                    try:
                        with st.spinner(f"Loading {demo_label}..."):
                            db = load_demo_db(db_key)
                        st.session_state.active_db       = db
                        st.session_state.active_db_label = meta["key"]
                        st.rerun()
                    except FileNotFoundError:
                        st.error("Run `python db/setup_dbs.py` first.")
        with col_info:
            st.caption(meta["desc"][:20] + "…")


def _sidebar_sqlite_upload() -> None:
    uploaded = st.file_uploader(
        "SQLite file", type=["db", "sqlite"],
        label_visibility="collapsed",
    )
    if uploaded is None:
        # Show delete button if a custom file was previously loaded
        if st.session_state.active_db is not None and st.session_state.active_db_label not in ("chinook", "northwind"):
            st.warning(f"Using: {st.session_state.active_db_label.split('_')[0]}")
            if st.button("Remove database", use_container_width=True, type="secondary"):
                _clear_and_reset()
                st.rerun()
        return

    fingerprint = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get("active_db_label") != fingerprint:
        _clear_and_reset()
        with st.spinner(f"Loading {uploaded.name}..."):
            db, _ = load_db_from_upload(uploaded)
        st.session_state.active_db       = db
        st.session_state.active_db_label = fingerprint
        st.rerun()


def _sidebar_csv_upload() -> None:
    uploaded = st.file_uploader(
        "CSV file", type=["csv"],
        label_visibility="collapsed",
        help="Auto-converts to a queryable SQL table.",
    )
    if uploaded is None:
        st.info("Upload a CSV — becomes a SQL table automatically.")
        return
    fingerprint = f"csv_{uploaded.name}_{uploaded.size}"
    if st.session_state.get("active_db_label") != fingerprint:
        _clear_and_reset()
        with st.spinner(f"Converting {uploaded.name}..."):
            db, label = connect_from_csv(uploaded)
        st.session_state.active_db       = db
        st.session_state.active_db_label = fingerprint
        st.success(f"Ready: {label}")
        st.rerun()


def _sidebar_credential_form(db_type_key: str, is_nosql: bool) -> None:
    db_type = DB_TYPES[db_type_key]

    # ── Saved profiles from .streamlit/secrets.toml ───────────────────────────
    # If the user has set up saved connections, show a one-click selector.
    # Format in secrets.toml:
    #   [connections.my_postgres]
    #   type = "postgresql"
    #   host = "localhost"
    #   port = "5432"
    #   database = "mydb"
    #   user = "analyst"
    #   password = "secret"
    saved_creds = None
    try:
        conns = st.secrets.get("connections", {})
        matching = {
            k: dict(v) for k, v in conns.items()
            if v.get("type") == db_type_key
        }
        if matching:
            st.markdown("<div style='font-size:12px;color:#888780;margin-bottom:4px'>Saved profiles</div>",
                        unsafe_allow_html=True)
            profile_key = st.selectbox(
                "saved_profile", ["— manual entry —"] + list(matching.keys()),
                label_visibility="collapsed",
            )
            if profile_key != "— manual entry —":
                saved_creds = matching[profile_key]
                saved_creds.pop("type", None)
                st.success(f"Profile loaded: {profile_key}")
    except Exception:
        pass   # secrets not configured — silently skip

    defaults = {
        "host": "localhost",
        "port": "5432" if db_type_key=="postgresql" else
                "3306" if db_type_key=="mysql" else
                "1433" if db_type_key=="mssql" else "27017",
        "database": "", "user": "", "password": "", "path": "",
    }
    credentials = {}
    for fname, flabel, ftype in zip(db_type.fields, db_type.field_labels, db_type.field_types):
        # Pre-fill from saved profile if available
        prefill = (saved_creds or {}).get(fname, defaults.get(fname, ""))
        if ftype == "password":
            credentials[fname] = st.text_input(
                flabel, type="password",
                value=prefill if saved_creds else "",
                placeholder=flabel,
            )
        else:
            credentials[fname] = st.text_input(
                flabel, value=prefill, placeholder=flabel,
            )

    if st.button(f"Connect", use_container_width=True, type="primary"):
        missing = [db_type.field_labels[i] for i, f in enumerate(db_type.fields)
                   if not credentials.get(f,"").strip()]
        if missing:
            st.error(f"Fill in: {', '.join(missing)}")
            return
        try:
            with st.spinner("Connecting..."):
                if is_nosql:
                    mongo_db = connect_mongodb(credentials, db_type_key=db_type_key)
                    _clear_and_reset()
                    st.session_state.mongo_db        = mongo_db
                    st.session_state.is_nosql        = True
                    st.session_state.active_db_label = f"mongodb_{credentials.get('database','')}"
                else:
                    db, label = connect_from_credentials(db_type_key, credentials)
                    # Connection test
                    try:
                        db.run("SELECT 1")
                    except Exception as e:
                        raise ConnectionError(f"Connection test failed: {e}")
                    _clear_and_reset()
                    st.session_state.active_db       = db
                    st.session_state.active_db_label = label
            st.rerun()
        except ImportError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Failed: {e}")


def _render_schema_inspector() -> None:
    db = st.session_state.active_db
    summary = get_schema_summary(db)
    st.markdown(
        f"<div class='metric-row'>"
        f"<div class='metric-card'><div class='val'>{summary['table_count']}</div>"
        f"<div class='lbl'>Tables</div></div>"
        f"</div>",
        unsafe_allow_html=True
    )
    with st.expander("View tables", expanded=False):
        for t in summary["table_names"]:
            st.markdown(f"<div style='font-size:12px;font-family:monospace;padding:2px 0'>• {t}</div>",
                        unsafe_allow_html=True)


def _render_mongo_inspector() -> None:
    mongo_db = st.session_state.mongo_db
    try:
        collections = mongo_db.list_collection_names()
        st.caption(f"{len(collections)} collections")
        with st.expander("Collections", expanded=False):
            for c in collections:
                st.text(f"• {c}")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

def render_main() -> None:
    has_sql   = st.session_state.active_db is not None
    has_mongo = st.session_state.mongo_db is not None

    if not has_sql and not has_mongo:
        _render_welcome_screen()
        return

    if has_mongo and not has_sql:
        # MongoDB is now fully functional — same chat UI, MQL branch in agent
        db = None   # no SQL db — run_agent will use mongo_db from session_state

    # Header
    db_name = st.session_state.active_db_label
    display_name = db_name.split("_")[0].title() if "_" in db_name else db_name
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:4px'>"
        f"<span style='font-size:22px;font-weight:500'>Ask your data</span>"
        f"<span style='font-size:12px;background:rgba(124,106,245,0.1);"
        f"border:0.5px solid rgba(124,106,245,0.3);border-radius:6px;"
        f"padding:3px 10px;color:#534AB7'>{display_name}</span>"
        f"</div>"
        f"<div style='font-size:13px;color:#888780;margin-bottom:16px'>"
        f"Type or speak a question — the agent writes SQL, runs it, and explains the result."
        f"</div>",
        unsafe_allow_html=True
    )

    # Replay chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sql"):
                with st.expander("Generated SQL", expanded=False):
                    st.code(msg["sql"], language="sql")
            if msg.get("dataframe") is not None and not msg["dataframe"].empty:
                st.dataframe(msg["dataframe"], use_container_width=True, hide_index=True)
            if msg.get("chart") is not None:
                st.plotly_chart(msg["chart"], use_container_width=True)

    # Suggested questions (shown only when no chat history)
    if not st.session_state.messages:
        _render_suggested_questions()

    # Input
    question = _get_user_input()
    if question:
        _process_question(question)


def _render_welcome_screen() -> None:
    st.markdown("""
    <div style='max-width:680px;margin:60px auto 0;text-align:center;padding:0 20px'>
      <div style='font-size:40px;margin-bottom:16px'>🏢</div>
      <div style='font-size:26px;font-weight:500;margin-bottom:10px'>Enterprise BI Agent</div>
      <div style='font-size:15px;color:#888780;line-height:1.7;margin-bottom:32px'>
        Connect a database from the sidebar — SQLite, PostgreSQL, MySQL, CSV, or MongoDB.
        Ask questions in plain English. Get SQL, data, charts, and voice answers instantly.
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    features = [
        ("Text + Voice input", "Type or speak your question"),
        ("Self-correcting SQL", "Agent retries on errors automatically"),
        ("Auto charts", "Bar, line, pie, scatter — auto-selected"),
    ]
    for col, (title, desc) in zip([col1,col2,col3], features):
        with col:
            st.markdown(
                f"<div style='background:rgba(124,106,245,0.05);border:0.5px solid "
                f"rgba(124,106,245,0.2);border-radius:12px;padding:16px;text-align:center'>"
                f"<div style='font-size:13px;font-weight:500;margin-bottom:4px'>{title}</div>"
                f"<div style='font-size:12px;color:#888780'>{desc}</div></div>",
                unsafe_allow_html=True
            )

    st.markdown("<div style='text-align:center;margin-top:32px;font-size:13px;color:#888780'>"
                "← Select a database from the sidebar to begin</div>",
                unsafe_allow_html=True)


def _render_suggested_questions() -> None:
    """Clickable suggested questions based on active DB."""
    db_label = st.session_state.active_db_label.lower()

    if "chinook" in db_label:
        questions = [
            "Who are the top 5 customers by total spend?",
            "Which genre sold the most tracks?",
            "Show monthly revenue for 2013",
            "Which artist has the most albums?",
        ]
    elif "northwind" in db_label:
        questions = [
            "Which product category has the highest revenue?",
            "Show the top 5 employees by orders processed",
            "Which country places the most orders?",
            "What are the top 10 most ordered products?",
        ]
    else:
        # Generic questions for unknown DB
        try:
            tables = st.session_state.active_db.get_usable_table_names()
            t = tables[0] if tables else "data"
            questions = [
                f"How many rows are in the {t} table?",
                f"Show me a sample of 5 rows from {t}",
                f"What columns does the {t} table have?",
            ]
        except Exception:
            return

    st.markdown(
        "<div style='font-size:12px;color:#888780;margin-bottom:8px'>Try asking:</div>",
        unsafe_allow_html=True
    )
    cols = st.columns(2)
    for i, q in enumerate(questions):
        with cols[i % 2]:
            if st.button(q, key=f"sugg_{i}", use_container_width=True):
                _process_question(q)


def _get_user_input() -> str | None:
    """
    Dual input: voice recorder sits ABOVE the chat input bar, not beside it.

    WHY NOT st.columns?
        st.chat_input() must be a direct child of the main page container —
        it anchors to the bottom of the screen. When placed inside st.columns,
        it loses this anchoring and conflicts with audio_recorder's rerun
        mechanism: the mic button goes red, captures audio, but the rerun
        triggered by audio_recorder clears the columns layout before the
        bytes can be read. Voice silently drops.

    FIX: render the mic widget in its own row above the chat input.
         Both widgets are now direct page children — no column wrapping.
    """
    question = None

    # ── Voice input row (above chat bar) ──────────────────────────────────────
    try:
        from audio_recorder_streamlit import audio_recorder

        # Show mic + status in a clean inline layout
        mic_col, status_col = st.columns([1, 8])
        with mic_col:
            audio_bytes = audio_recorder(
                text="",
                icon_size="2x",
                pause_threshold=2.0,
                key="voice_recorder",
                recording_color="#e85d30",   # coral red when active
                neutral_color="#7c6af5",     # purple when idle
            )
        with status_col:
            if audio_bytes and len(audio_bytes) > 1000:
                audio_hash = hashlib.md5(audio_bytes).hexdigest()
                if audio_hash != st.session_state.get("last_audio_hash"):
                    # New recording — process immediately
                    st.session_state["last_audio_hash"] = audio_hash
                    with st.spinner("Transcribing..."):
                        transcribed = transcribe_audio(audio_bytes)
                    if transcribed:
                        question = transcribed
                        st.caption(f'Heard: "{transcribed}"')
                    else:
                        st.warning("Transcription failed — check your mic and try again.")
                else:
                    st.caption("Click mic to record a new question")
            else:
                st.caption("Click the mic to ask with your voice")

    except ImportError:
        pass   # voice not installed — degrade silently, text input still works

    # ── Text input (anchored to page bottom by Streamlit) ─────────────────────
    # This MUST be called as a direct page child — do not wrap in columns/containers
    if text_input := st.chat_input("Ask about your data…"):
        question = text_input

    return question


# ═══════════════════════════════════════════════════════════════════════════════
# QUESTION PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def _process_question(question: str) -> None:
    db = st.session_state.active_db

    st.chat_message("user").write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        # Step 1: Schema RAG (SQL only — MongoDB samples its own schema)
        relevant_schema = ""
        if not st.session_state.is_nosql:
            with st.spinner("Finding relevant tables..."):
                schema_summary  = get_schema_summary(db)
                relevant_schema = get_relevant_schema(
                    question=question,
                    full_schema=schema_summary["full_info"],
                    table_names=schema_summary["table_names"],
                    top_k=3,
                )

        # Step 2: LangGraph agent
        with st.spinner("Running agent..."):
            if st.session_state.is_nosql:
                result = run_agent(
                    question=question,
                    mongo_db=st.session_state.mongo_db,
                )
            else:
                result = run_agent(
                    question=question,
                    db=db,
                    schema=relevant_schema,
                )

        # Step 3: Handle failure
        if result["error"] and not result["result"]:
            st.error(
                f"Could not generate a valid query after {result['retries']} attempts.\n\n"
                f"**Error:** {result['error']}\n\nTry rephrasing your question."
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Failed: {result['error']}",
            })
            return

        # Step 4: Display results
        if result["summary"]:
            st.markdown(f"**{result['summary']}**")

        if result["retries"] > 0:
            st.markdown(
                f"<span class='retry-note'>Agent self-corrected {result['retries']} time(s)</span>",
                unsafe_allow_html=True
            )
        if result.get("is_nosql") and result.get("collection"):
            st.caption(f"Collection queried: `{result['collection']}`")

        if result["sql"]:
            with st.expander("Generated SQL", expanded=True):
                st.code(result["sql"], language="sql")

        df = pd.DataFrame(result["result"]) if result["result"] else pd.DataFrame()
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Plotly chart
        fig = auto_chart(df) if not df.empty else None
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Voice output
        if st.session_state.voice_enabled and result["summary"]:
            audio_path = synthesize_speech(result["summary"])
            if audio_path:
                st.audio(audio_path, format="audio/mp3", autoplay=True)

    # Save to history
    st.session_state.messages.append({
        "role":      "assistant",
        "content":   result["summary"] or "Query completed.",
        "sql":       result["sql"],
        "dataframe": df,
        "chart":     fig,
    })
    if result["sql"]:
        st.session_state.last_sql = result["sql"]


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════
render_sidebar()
render_main()
