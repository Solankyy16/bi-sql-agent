"""
agent/mongo_nodes.py
─────────────────────
MongoDB-specific node functions for the LangGraph brain.

WHY SEPARATE FROM nodes.py?
    SQL and MQL (MongoDB Query Language) are completely different.
    SQL is declarative text. MQL is a Python list of dicts (aggregation pipeline).
    Keeping them in separate files means:
      - nodes.py stays clean and SQL-focused
      - mongo_nodes.py can evolve independently
      - graph.py just picks the right set of nodes at runtime

THE MQL FLOW:
    1. sample_mongo_schema_node  — samples 3 docs per collection → builds schema context
    2. generate_mql_node         — Llama 3 writes a pymongo aggregation pipeline as JSON
    3. execute_mql_node          — runs pipeline via pymongo, returns list of dicts
    4. synthesize_node (reused)  — same natural language synthesis as SQL path

WHY AGGREGATION PIPELINES?
    Simple find() queries can answer "show me 5 documents" but not
    "which category has the highest total revenue?" — that needs $group, $sort, $limit.
    Aggregation pipelines are MongoDB's equivalent of SQL's GROUP BY + ORDER BY.
    Teaching the LLM to write them is the right abstraction for BI queries.
"""

import os
import re
import json
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ── LLM (same model as SQL path) ─────────────────────────────────────────────

def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found.")
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=api_key)


# ── MQL GENERATION PROMPT ─────────────────────────────────────────────────────
# Key differences from the SQL prompt:
# 1. Output must be valid JSON (a list) — we parse it with json.loads()
# 2. We use "collection" language, not "table" language
# 3. We show sample documents as schema (MongoDB has no fixed schema)
# 4. Aggregation pipeline is the primary tool — $match, $group, $sort, $limit

MQL_PROMPT = ChatPromptTemplate.from_template("""
You are an expert MongoDB developer. Write a pymongo aggregation pipeline.

DATABASE: {database_name}

COLLECTION SCHEMAS (field names and sample values from real documents):
{schema}

RULES — follow all of them:
1. Return ONLY a valid JSON array — the aggregation pipeline stages.
   Example: [{{"$group": {{"_id": "$category", "total": {{"$sum": "$price"}}}}}}, {{"$sort": {{"total": -1}}}}, {{"$limit": 10}}]
2. Do NOT include collection.aggregate() wrapper — just the pipeline array.
3. No markdown, no backticks, no explanation — pure JSON only.
4. Use ONLY fields shown in the schema above.
5. Always include {{"$limit": 50}} as the last stage unless asked for all data.
6. For counts: use {{"$group": {{"_id": "$field", "count": {{"$sum": 1}}}}}}
7. For sums: use {{"$group": {{"_id": "$field", "total": {{"$sum": "$numeric_field"}}}}}}
8. For string matching: use {{"$match": {{"field": {{"$regex": "pattern", "$options": "i"}}}}}}

{error_context}

QUESTION: {question}

COLLECTION TO QUERY: {collection_name}

PIPELINE (JSON array only):""")


# ── NODE 0: Sample schema ──────────────────────────────────────────────────────

def sample_mongo_schema_node(state: dict) -> dict:
    """
    Samples documents from MongoDB collections to build schema context.

    WHY SAMPLING INSTEAD OF FIXED SCHEMA?
        MongoDB is schema-less — collections can have different fields per document.
        The best way to give the LLM "schema" is to show it real documents.
        We sample 3 docs per collection, extract all unique field names and
        example values, and format it as readable context.

    Also picks the most relevant collection for the question using a simple
    keyword match — avoids sending all collections' schemas.
    """
    mongo_db   = state["mongo_db"]
    question   = state["question"]

    try:
        collection_names = mongo_db.list_collection_names()
    except Exception as e:
        return {"error": f"Could not list collections: {e}", "schema": "", "collection_name": ""}

    if not collection_names:
        return {"error": "No collections found in this database.", "schema": "", "collection_name": ""}

    # ── Pick the most relevant collection ─────────────────────────────────────
    # Simple heuristic: find collection whose name appears in the question,
    # or whose field names most overlap with question words.
    question_lower = question.lower()
    best_collection = collection_names[0]  # default to first

    for name in collection_names:
        if name.lower() in question_lower or question_lower in name.lower():
            best_collection = name
            break

    # ── Sample documents and extract field structure ───────────────────────────
    schema_parts = []
    for cname in collection_names[:5]:  # cap at 5 collections to avoid token overflow
        try:
            sample_docs = list(mongo_db[cname].find({}, {"_id": 0}).limit(3))
            if not sample_docs:
                schema_parts.append(f"Collection '{cname}': (empty)")
                continue

            # Extract all unique fields across sample docs
            all_fields = {}
            for doc in sample_docs:
                for k, v in doc.items():
                    if k not in all_fields:
                        # Show field name + example value type + example value
                        ex = str(v)[:40] + "…" if len(str(v)) > 40 else str(v)
                        all_fields[k] = f"{type(v).__name__} (e.g. {ex})"

            field_lines = "\n  ".join(f"{k}: {vt}" for k, vt in list(all_fields.items())[:20])
            schema_parts.append(f"Collection '{cname}':\n  {field_lines}")
        except Exception as e:
            schema_parts.append(f"Collection '{cname}': (error sampling: {e})")

    schema_str = "\n\n".join(schema_parts)

    return {
        "schema":          schema_str,
        "collection_name": best_collection,
        "error":           None,
    }


# ── NODE 1: Generate MQL ───────────────────────────────────────────────────────

def generate_mql_node(state: dict) -> dict:
    """
    Asks Llama 3 to write a MongoDB aggregation pipeline for the question.

    The output is a JSON array string which execute_mql_node will parse
    and run via pymongo.
    """
    # If schema hasn't been sampled yet (first call), run sampling inline
    if not state.get("schema"):
        schema_result = sample_mongo_schema_node(state)
        state = {**state, **schema_result}
        if state.get("error"):
            return state

    llm = _get_llm()

    if state.get("error"):
        error_context = (
            f"IMPORTANT — Your previous pipeline failed:\n"
            f"  {state['error']}\n"
            f"Fix this specific error in your new pipeline."
        )
    else:
        error_context = ""

    chain = MQL_PROMPT | llm | StrOutputParser()

    # Get the database name safely
    try:
        db_name = state["mongo_db"].name
    except Exception:
        db_name = "database"

    raw_pipeline = chain.invoke({
        "database_name":   db_name,
        "schema":          state.get("schema", ""),
        "collection_name": state.get("collection_name", ""),
        "question":        state["question"],
        "error_context":   error_context,
    })

    cleaned_pipeline = _clean_pipeline(raw_pipeline)

    return {
        "sql":         cleaned_pipeline,   # reuse "sql" key — holds MQL pipeline string
        "error":       None,
        "retry_count": state.get("retry_count", 0),
    }


def _clean_pipeline(raw: str) -> str:
    """
    Strips markdown fences from the LLM's pipeline output.
    Returns a clean JSON array string.
    """
    raw = re.sub(r"```(?:json|javascript|js|python)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"```", "", raw)
    raw = raw.strip()
    # Extract just the array portion if the LLM added prose around it
    match = re.search(r"(\[.*\])", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw


# ── NODE 2: Execute MQL ────────────────────────────────────────────────────────

def execute_mql_node(state: dict) -> dict:
    """
    Parses the pipeline JSON and runs it against the MongoDB collection.

    SAFETY:
        We parse the pipeline as JSON first — if it's not valid JSON,
        we catch the error and route back to generate_mql_node.
        We also cap results at 50 documents to prevent memory issues.
    """
    pipeline_str = state["sql"]   # MQL pipeline stored in the "sql" key
    mongo_db     = state["mongo_db"]
    collection_name = state.get("collection_name", "")

    if not collection_name:
        return {"error": "No collection selected. Rephrase your question to mention a collection.", "result": []}

    # ── Parse the pipeline JSON ────────────────────────────────────────────────
    try:
        pipeline = json.loads(pipeline_str)
        if not isinstance(pipeline, list):
            return {"error": f"Pipeline must be a JSON array. Got: {type(pipeline).__name__}", "result": []}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON pipeline: {e}\nPipeline was: {pipeline_str[:200]}", "result": []}

    # ── Execute ────────────────────────────────────────────────────────────────
    try:
        collection = mongo_db[collection_name]

        # Ensure there's a $limit somewhere — safety cap
        has_limit = any("$limit" in str(stage) for stage in pipeline)
        if not has_limit:
            pipeline.append({"$limit": 50})

        cursor = collection.aggregate(pipeline)
        docs   = list(cursor)[:50]   # hard cap

        # Convert ObjectId and other non-serializable types to strings
        result_rows = []
        for doc in docs:
            clean_doc = {}
            for k, v in doc.items():
                if k == "_id":
                    continue   # skip internal MongoDB ID field
                try:
                    json.dumps(v)   # test if serializable
                    clean_doc[k] = v
                except (TypeError, ValueError):
                    clean_doc[k] = str(v)
            result_rows.append(clean_doc)

        return {"result": result_rows, "error": None}

    except Exception as e:
        return {"error": str(e), "result": []}
