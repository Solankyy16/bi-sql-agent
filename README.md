<div align="center">

# Enterprise BI Agent

### Talk to any database in plain English.
### Get SQL, data, interactive charts, and voice answers — in seconds.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20Loop-1C3C3C?style=flat)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-Llama%203%20%2B%20Whisper-F55036?style=flat)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat)](LICENSE)

</div>

---

## What is this?

A production-grade, **database-agnostic** Business Intelligence agent built with LangGraph and Groq. Connect any database — SQLite, PostgreSQL, MySQL, MongoDB Atlas, or a plain CSV — and ask questions in plain English. The agent writes the query, executes it, self-corrects if it fails, visualises the result as an interactive chart, and reads the answer aloud.

No SQL knowledge required. No BI tool licence. No data ever leaves your machine (except the LLM API call to Groq).

---

## Live Demo

| Step | What happens |
|------|-------------|
| Select a demo database or upload your own | Agent reads schema automatically |
| Type or speak a question | Groq Whisper transcribes voice in <0.3s |
| Agent retrieves relevant tables via Schema RAG | Only the 2-3 relevant schemas injected into the prompt |
| Llama 3 writes a query | SQL or MongoDB aggregation pipeline, dialect-aware |
| Query fails? | LangGraph catches the error and self-corrects — up to 3 retries |
| Query succeeds | DataFrame + Plotly interactive chart rendered inline |
| Voice output enabled | Agent speaks the one-sentence summary via gTTS |

---

## Features

- **Dual input** — type your question or click the mic and speak
- **Dual output** — visual results in the UI + spoken summary via TTS
- **Full query transparency** — every generated SQL or MQL pipeline is shown, never hidden
- **Self-correcting agent** — LangGraph state machine retries on errors with the error message fed back to the LLM
- **Schema RAG** — FAISS vector index over table descriptions; retrieves only the 3 most relevant tables per question — handles 200-table databases without prompt overflow
- **7 chart types** — time series, ranked bar, donut pie, grouped bar, scatter, histogram, value counts — auto-selected from data shape
- **Universal connector** — SQLite file, SQLite path, PostgreSQL, MySQL, SQL Server, MongoDB (self-hosted), MongoDB Atlas, CSV (auto-converts to SQL table)
- **Dialect-aware SQL** — detects SQLite / PostgreSQL / MySQL / MSSQL dialect from the connection and adjusts syntax accordingly
- **Saved connection profiles** — store credentials in `.streamlit/secrets.toml`, one-click connect, no retyping
- **Plug-and-play databases** — upload any `.db`, `.sqlite`, or `.csv` file and start querying immediately
- **100% free to run** — Groq free tier powers both LLM and voice transcription

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Solankyy16/bi-sql-agent
cd bi-sql-agent

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate        # Mac / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. API key  —  get free at console.groq.com
cp .env.example .env
# Open .env and set: GROQ_API_KEY=gsk_...

# 5. Download demo databases
python db/setup_dbs.py

# 6. Run
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Example Questions

**Chinook — Music Sales (11 tables)**
```
Who are the top 5 customers by total spend?
Which artist generated the most revenue overall?
Show monthly revenue for 2013
Which genre sold the most tracks?
What country has the most customers?
```

**Northwind — Logistics (13 tables)**
```
Which product category has the highest total revenue?
Show orders shipped to Germany
Who processed the most orders?
What is the average order value by country?
Which supplier provides the most products?
```

**MongoDB Atlas — sample_supplies**
```
Which purchase method generates the most revenue?
What are the top 5 most purchased items?
How many orders were placed online vs in-store?
Show me the 10 most recent sales
```

**Your own database**
Upload any `.db`, `.sqlite`, or `.csv` file in the sidebar and ask anything.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Input Layer                                │
│   Text (st.chat_input) ──────────┬──── Voice (Groq Whisper STT)    │
│                                  │                                  │
│                        plain-text question                          │
└──────────────────────────────────┼──────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        Universal Connector                          │
│   SQLite · PostgreSQL · MySQL · MSSQL · MongoDB · Atlas · CSV       │
│   SQLAlchemy URI builder + pymongo for NoSQL                        │
└──────────────────────────────────┬──────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                          Schema RAG                                 │
│   FAISS vector index over table / collection descriptions           │
│   Retrieves top-3 most relevant schemas per question                │
│   (skipped for MongoDB — samples live documents instead)            │
└──────────────────────────────────┬──────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│              LangGraph State Machine  (agent/graph.py)              │
│                                                                     │
│   ┌─────────┐    SQL path:                                          │
│   │ router  │──→ generate_sql → execute_sql ──→ synthesize → END   │
│   │  node   │         ↑── error? retry (×3) ──┘                    │
│   └─────────┘                                                       │
│        │         MongoDB path:                                      │
│        └──────→ sample_schema → generate_mql → execute_mql ──→ ↑  │
│                                      ↑── error? retry (×3) ────┘  │
└──────────────────────────────────┬──────────────────────────────────┘
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         Output Layer                                │
│   Generated query (st.code)  +  DataFrame (st.dataframe)           │
│   Plotly interactive chart   +  Voice summary (gTTS autoplay)      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Version | Why |
|-------|-----------|---------|-----|
| **UI** | Streamlit | ≥1.35 | Pure Python web app, native chat interface, Plotly support |
| **LLM** | Groq + Llama 3.1 8B | latest | 800+ tokens/sec inference, free tier, temperature=0 for deterministic SQL |
| **Agent** | LangGraph | ≥0.1 | Cyclic state machine — enables the self-correction retry loop, SQL + MongoDB branches |
| **Orchestration** | LangChain | ≥0.2 | Prompt templates, LCEL chains, SQLDatabase utility |
| **Retrieval** | FAISS + Sentence Transformers | cpu / ≥3.0 | Local embeddings (all-MiniLM-L6-v2), no API cost, sub-10ms similarity search |
| **SQL Databases** | SQLAlchemy + LangChain SQLDatabase | ≥2.0 | Unified interface for SQLite / PostgreSQL / MySQL / MSSQL |
| **NoSQL** | pymongo | ≥4.6 | Direct MongoDB + Atlas connection, aggregation pipeline execution |
| **Voice Input** | Groq Whisper API | whisper-large-v3 | Same API key as LLM, <0.3s transcription, handles accents |
| **Voice Output** | gTTS | ≥2.5 | Google TTS, natural voice, no additional API key |
| **Charts** | Plotly | ≥5.20 | Interactive hover, zoom, pan — 7 chart types auto-selected |
| **Analytics** | Pandas | ≥2.2 | DataFrame construction from SQL/MQL results, column type detection |
| **Secrets** | python-dotenv | ≥1.0 | `.env` file loading, never hardcode credentials |

---

## Project Structure

```
bi-sql-agent/
├── app.py                      # Streamlit entry point — wires all modules together
│
├── agent/
│   ├── graph.py                # LangGraph state machine — SQL + MongoDB branches, router node
│   ├── nodes.py                # SQL nodes: generate_sql, execute_sql, synthesize
│   └── mongo_nodes.py          # MongoDB nodes: sample_schema, generate_mql, execute_mql
│
├── db/
│   ├── connector.py            # Universal connector — URI builder for all DB types + Atlas
│   ├── loader.py               # Demo DB loader + SQLite upload handler + schema summary
│   └── setup_dbs.py            # One-time script: downloads Chinook + Northwind
│
├── rag/
│   ├── schema_embedder.py      # Builds FAISS vector index from table descriptions
│   └── retriever.py            # Retrieves top-k relevant schemas per question
│
├── voice/
│   ├── stt.py                  # Speech-to-text via Groq Whisper API
│   └── tts.py                  # Text-to-speech via gTTS → temp .mp3 → st.audio()
│
├── charts/
│   └── auto_chart.py           # 7-type Plotly chart auto-selector from DataFrame shape
│
├── utils/
│   └── state.py                # Centralised Streamlit session state — AppState dataclass
│
├── .streamlit/
│   └── config.toml             # Dark theme, primary colour, font
│
├── .env.example                # Credential template — copy to .env and fill in key
├── .gitignore                  # Excludes .env, venv, FAISS index, temp DB files
├── requirements.txt
└── README.md
```

---

## Connecting Different Databases

| Database | What you need | Notes |
|----------|--------------|-------|
| SQLite file | Upload `.db` or `.sqlite` in sidebar | Zero config |
| CSV file | Upload `.csv` in sidebar | Auto-converts to SQL table |
| PostgreSQL | Host, port, database, user, password | `pip install psycopg2-binary` |
| MySQL / MariaDB | Host, port, database, user, password | `pip install pymysql` |
| SQL Server | Host, port, database, user, password | `pip install pyodbc` + ODBC Driver 17 |
| MongoDB self-hosted | Host, port, database, user, password | `pip install pymongo` |
| MongoDB Atlas | Cluster hostname, database, user, password | Select "MongoDB Atlas (cloud)" in sidebar |

**Saving credentials** — add to `.streamlit/secrets.toml` (never committed):
```toml
[connections.my_postgres]
type = "postgresql"
host = "localhost"
port = "5432"
database = "mydb"
user = "analyst"
password = "secret"
```

---

## Deploying to Streamlit Cloud (Free)

```bash
# 1. Push to GitHub (make sure .env is NOT committed — check .gitignore)
git add .
git commit -m "feat: initial deploy"
git push origin main

# 2. Go to share.streamlit.io
# 3. Connect your GitHub repo
# 4. Add secret:  GROQ_API_KEY = gsk_your_key_here
# 5. Deploy — you get a public URL instantly
```

Note: Streamlit Cloud supports SQLite uploads and CSV uploads out of the box.
For PostgreSQL/MongoDB, your database must be publicly reachable (Atlas works perfectly).

---

## Extending the Project

**Add a new SQL database type** — add one entry to `DB_TYPES` in `db/connector.py`. Nothing else changes.

**Add a new chart type** — add a new `_chart_fn()` in `charts/auto_chart.py` and a detection condition in `auto_chart()`.

**Improve MongoDB collection selection** — update `sample_mongo_schema_node()` in `agent/mongo_nodes.py` with better semantic matching.

**Add streaming responses** — replace `graph.invoke()` with `graph.stream()` in `agent/graph.py` and pipe tokens to `st.write_stream()`.

---

## License

MIT — use it, fork it, build on it.

---

<div align="center">
Built with LangGraph · Groq · Streamlit · FAISS · Plotly · pymongo
</div>
