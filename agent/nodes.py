"""
agent/nodes.py
──────────────
The three node functions that make up the LangGraph brain.

In LangGraph, a "node" is just a Python function that:
  - Takes the current state (a dict) as input
  - Does some work (LLM call, DB query, etc.)
  - Returns a dict of state updates

The graph engine merges the returned dict into the state and
decides which node to call next based on the conditional edges.

NODES IN OUR GRAPH:
    generate_sql_node  → asks Llama 3 to write SQL
    execute_sql_node   → runs the SQL against the database
    synthesize_node    → turns raw data rows into a human sentence

The self-correction loop is in graph.py (the conditional edge between
execute and generate).
"""

import os
import re
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag.retriever import get_relevant_schema


# ── LLM initialisation ────────────────────────────────────────────────────────
# We create the LLM client once at module load time.
# temperature=0 is CRITICAL for SQL generation — you want deterministic output.
# A temperature of 0.7 would make the LLM "creative" with column names. Bad.
def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Did you create a .env file from .env.example?"
        )
    return ChatGroq(
        model="llama-3.1-8b-instant",  # Fast 8B model — good for SQL generation
        temperature=0,
        api_key=api_key,
    )


# ── SQL GENERATION PROMPT ─────────────────────────────────────────────────────
# The prompt is the most important part of the SQL generation node.
# Every word here was chosen carefully. Let's break it down:
#
# 1. "You are an expert SQL analyst" — role prompting improves accuracy
# 2. "ONLY the raw SQL query" — prevents markdown fences (```sql ... ```)
#    which would cause db.run() to crash with a syntax error
# 3. "no explanations, no markdown" — reinforces constraint #2
# 4. "Use only tables and columns from the schema" — prevents hallucination
# 5. The {error_context} slot — this is what makes the self-correction work.
#    On first attempt it's empty. On retry, it contains the error message.

def _get_dialect(db) -> str:
    """Detect the SQL dialect from the active connection so the LLM writes
    dialect-correct syntax (SQLite LIMIT vs MSSQL TOP, date functions, etc.)"""
    try:
        return db._engine.dialect.name  # "sqlite", "postgresql", "mysql", "mssql"
    except Exception:
        return "sqlite"


SQL_PROMPT = ChatPromptTemplate.from_template("""
You are an expert SQL analyst. Your job is to write precise, correct {dialect} queries.

DATABASE SCHEMA (only these tables and columns exist):
{schema}

RULES — follow all of them:
1. Return ONLY the raw SQL query. No markdown, no backticks, no explanation.
2. Use ONLY the table and column names shown in the schema above. Do not invent names.
3. Use {dialect} syntax specifically — correct date functions, quoting, and LIMIT/TOP for {dialect}.
4. For string comparisons, use LIKE with % wildcards for flexibility.
5. Always use table aliases in JOINs (e.g. c.CustomerId, not just CustomerId).
6. If the question requires aggregation, include a GROUP BY clause.
7. Limit results to 50 rows maximum unless the user asks for all data.

{error_context}

USER QUESTION: {question}

SQL QUERY:""")

# ── SYNTHESIS PROMPT ──────────────────────────────────────────────────────────
# This turns raw database rows into a sentence a VP of Marketing can understand.
# It's also what gets spoken aloud by gTTS — so it must sound natural.

SYNTHESIS_PROMPT = ChatPromptTemplate.from_template("""
You are a friendly data analyst presenting results to a business executive.
Convert the following SQL query result into ONE clear, conversational sentence.

Question that was asked: {question}
SQL that was run: {sql}
Result data: {result}

Rules:
- ONE sentence maximum (this will be read aloud)
- Include the most important numbers or names from the result
- If the result is empty, say "No data was found for that query."
- Do not mention SQL, databases, or technical terms

Your one-sentence summary:""")


# ── NODE 1: Generate SQL ───────────────────────────────────────────────────────

def generate_sql_node(state: dict) -> dict:
    """
    Asks Llama 3 to write a SQL query for the user's question.

    On first attempt: error_context is empty, fresh query.
    On retry: error_context contains the previous error message,
              so the LLM knows what to fix.

    Returns state update:
        {"sql": "SELECT ...", "error": None, "retry_count": n}
    """
    llm = _get_llm()

    # Build the error context message for retry attempts
    if state.get("error"):
        error_context = (
            f"IMPORTANT — Your previous SQL query failed with this error:\n"
            f"  {state['error']}\n"
            f"Fix this specific error in your new query."
        )
    else:
        error_context = ""

    # Build and run the chain: prompt | llm | string_parser
    chain = SQL_PROMPT | llm | StrOutputParser()

    raw_sql = chain.invoke({
        "schema":        state["schema"],
        "question":      state["question"],
        "error_context": error_context,
        "dialect":       _get_dialect(state["db"]),
    })

    # Clean up the output — strip whitespace and any accidental markdown fences
    # Even with strict prompting, LLMs occasionally wrap output in ```sql ... ```
    cleaned_sql = _clean_sql(raw_sql)

    return {
        "sql":         cleaned_sql,
        "error":       None,           # Reset error — ready for execution
        "retry_count": state.get("retry_count", 0),
    }


def _clean_sql(raw: str) -> str:
    """
    Strips markdown code fences and extra whitespace from LLM SQL output.

    Examples of what this handles:
        ```sql\nSELECT ...\n```  →  SELECT ...
        `SELECT ...`              →  SELECT ...
        SELECT ...;               →  SELECT ...;  (semicolons are fine)
    """
    # Remove ```sql ... ``` or ``` ... ``` fences
    raw = re.sub(r"```(?:sql)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"```", "", raw)
    # Remove single backtick wrapping
    raw = re.sub(r"^`|`$", "", raw.strip())
    return raw.strip()


# ── NODE 2: Execute SQL ────────────────────────────────────────────────────────

def execute_sql_node(state: dict) -> dict:
    """
    Runs the generated SQL against the active database.

    Success path: populates state["result"] with rows, clears state["error"]
    Failure path: populates state["error"] with the error message.
                  The graph's conditional edge will route back to generate_sql_node.

    SAFETY: We only allow SELECT statements.
            This prevents the LLM from accidentally (or maliciously) writing
            DROP TABLE, DELETE, or INSERT queries.
    """
    sql = state["sql"]
    db  = state["db"]  # SQLDatabase object, passed in from app.py

    # Safety check: only allow read operations
    first_word = sql.strip().split()[0].upper() if sql.strip() else ""
    if first_word not in ("SELECT", "WITH"):
        return {
            "error":  f"Only SELECT queries are allowed. Got: {first_word}",
            "result": [],
        }

    try:
        # db.run() executes the SQL and returns the result as a string
        # We need the raw rows, so we use the underlying engine directly
        result_str = db.run(sql)

        # Parse the string result back into a list of dicts for Pandas
        # db.run() returns something like: [(1, 'AC/DC', 839.0), (2, ...)]
        result_rows = _parse_db_result(result_str, db, sql)

        return {
            "result": result_rows,
            "error":  None,
        }

    except Exception as e:
        # Capture the full error message — this gets fed back to the LLM
        # on the next attempt. The more specific the error, the better the fix.
        error_msg = str(e)
        return {
            "error":  error_msg,
            "result": [],
        }


def _parse_db_result(result_str: str, db, sql: str) -> list[dict]:
    """
    Converts the string output of db.run() into a list of dicts.

    WHY: db.run() returns a string representation of tuples. We need actual
         Python dicts to build a Pandas DataFrame later.

    We use SQLAlchemy directly to get proper typed results.
    """
    try:
        # Access the underlying SQLAlchemy engine from the LangChain wrapper
        with db._engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = result.fetchmany(50)  # cap at 50 rows
            return [dict(zip(columns, row)) for row in rows]
    except Exception:
        # If direct parsing fails, return the string in a single-column dict
        # This ensures we always have something to display
        return [{"result": result_str}]


# ── NODE 3: Synthesise ─────────────────────────────────────────────────────────

def synthesize_node(state: dict) -> dict:
    """
    Converts raw SQL results into a natural-language summary sentence.

    This is what gets:
      1. Displayed as the "AI Summary" in the chat bubble
      2. Spoken aloud by gTTS
      3. Stored in chat history as the assistant's response

    We truncate the result to 20 rows before sending to the LLM.
    No need to send all 50 rows for a summary sentence.
    """
    llm = _get_llm()
    chain = SYNTHESIS_PROMPT | llm | StrOutputParser()

    # Truncate for the summary — LLM doesn't need all rows to write one sentence
    result_preview = state["result"][:20] if state["result"] else []

    summary = chain.invoke({
        "question": state["question"],
        "sql":      state["sql"],
        "result":   str(result_preview),
    })

    return {"summary": summary.strip()}
