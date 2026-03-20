"""
agent/graph.py — LangGraph state machine with SQL + MongoDB branches
"""
from typing import TypedDict, Optional, Any
from langgraph.graph import StateGraph, END

from agent.nodes       import generate_sql_node, execute_sql_node, synthesize_node
from agent.mongo_nodes import generate_mql_node, execute_mql_node, sample_mongo_schema_node

MAX_RETRIES = 3


class AgentState(TypedDict):
    # ── Common ────────────────────────────────────────────────────────────────
    question:        str
    schema:          str          # SQL schema string OR MongoDB sampled schema
    sql:             str          # SQL query string OR MQL pipeline JSON string
    result:          list[dict]
    error:           Optional[str]
    retry_count:     int
    summary:         str

    # ── SQL path ──────────────────────────────────────────────────────────────
    db:              Any          # LangChain SQLDatabase (None for MongoDB)

    # ── MongoDB path ──────────────────────────────────────────────────────────
    is_nosql:        bool         # True → MongoDB branch, False → SQL branch
    mongo_db:        Any          # pymongo.database.Database (None for SQL)
    collection_name: str          # which collection to query (set by schema node)


# ── Entry router ──────────────────────────────────────────────────────────────

def route_entry(state: AgentState) -> str:
    """First decision: SQL path or MongoDB path?"""
    return "generate_mql" if state.get("is_nosql") else "generate_sql"


# ── Post-execution router (same logic for both paths) ─────────────────────────

def route_after_sql(state: AgentState) -> str:
    if state.get("error"):
        if state.get("retry_count", 0) < MAX_RETRIES:
            return "generate_sql"
        return "synthesize"
    return "synthesize"


def route_after_mql(state: AgentState) -> str:
    if state.get("error"):
        if state.get("retry_count", 0) < MAX_RETRIES:
            return "generate_mql"
        return "synthesize"
    return "synthesize"


# ── Graph builder ─────────────────────────────────────────────────────────────

def _build_graph() -> Any:
    builder = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    # SQL path
    builder.add_node("generate_sql", generate_sql_node)
    builder.add_node("execute_sql",  execute_sql_node)

    # MongoDB path
    builder.add_node("sample_schema", sample_mongo_schema_node)
    builder.add_node("generate_mql",  generate_mql_node)
    builder.add_node("execute_mql",   execute_mql_node)

    # Shared final node
    builder.add_node("synthesize", synthesize_node)

    # ── Entry point — conditional on is_nosql ─────────────────────────────────
    # LangGraph requires a single set_entry_point, so we use a pass-through
    # router node that immediately branches
    builder.add_node("router", lambda s: s)   # no-op — just for routing
    builder.set_entry_point("router")
    builder.add_conditional_edges(
        "router",
        route_entry,
        {"generate_sql": "generate_sql", "generate_mql": "sample_schema"},
    )

    # ── SQL path edges ────────────────────────────────────────────────────────
    builder.add_edge("generate_sql", "execute_sql")
    builder.add_conditional_edges(
        "execute_sql", route_after_sql,
        {"generate_sql": "generate_sql", "synthesize": "synthesize"},
    )

    # ── MongoDB path edges ────────────────────────────────────────────────────
    builder.add_edge("sample_schema", "generate_mql")
    builder.add_edge("generate_mql",  "execute_mql")
    builder.add_conditional_edges(
        "execute_mql", route_after_mql,
        {"generate_mql": "generate_mql", "synthesize": "synthesize"},
    )

    # ── Terminal ──────────────────────────────────────────────────────────────
    builder.add_edge("synthesize", END)

    return builder.compile()


graph = _build_graph()


# ── Public API ────────────────────────────────────────────────────────────────

def run_agent(
    question: str,
    db:       Any  = None,
    schema:   str  = "",
    mongo_db: Any  = None,
) -> dict:
    """
    Single entry point for both SQL and MongoDB queries.

    For SQL:     pass db=<SQLDatabase>,  mongo_db=None
    For MongoDB: pass mongo_db=<pymongo.Database>, db=None
    """
    is_nosql = mongo_db is not None

    initial_state: AgentState = {
        "question":        question,
        "schema":          schema,
        "sql":             "",
        "result":          [],
        "error":           None,
        "retry_count":     0,
        "summary":         "",
        "db":              db,
        "is_nosql":        is_nosql,
        "mongo_db":        mongo_db,
        "collection_name": "",
    }

    final_state = graph.invoke(initial_state)

    return {
        "sql":     final_state["sql"],
        "result":  final_state["result"],
        "summary": final_state["summary"],
        "error":   final_state.get("error"),
        "retries": final_state.get("retry_count", 0),
        "is_nosql": is_nosql,
        "collection": final_state.get("collection_name", ""),
    }
