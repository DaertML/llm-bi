"""
Agent Service
=============
Exposes an OpenAI-compatible /v1/chat/completions endpoint consumed by
Open WebUI.  Each incoming chat message goes through:
 
  1. Catalog lookup  – fetch schema from PostgreSQL data-catalog
  2. Text-to-SQL     – ask Ollama to generate a DuckDB SQL query
  3. Query execution – run query via DuckDB against MinIO (S3)
  4. Answer          – ask Ollama to narrate the results naturally
  5. Stream back     – return as an SSE stream to Open WebUI
"""
from __future__ import annotations
 
import json
import logging
import os
import re
import time
import uuid
from typing import Any, AsyncIterator
 
import duckdb
import psycopg2
import psycopg2.extras
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from minio import Minio
from ollama import Client as OllamaClient
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
 
# ─── Config ──────────────────────────────────────────────────────────────────
 
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
 
    minio_endpoint: str   = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool    = False
 
    postgres_dsn: str = "postgresql://admin:admin@postgres:5432/datacatalog"
 
    ollama_host: str  = "http://host.docker.internal:11434"
    ollama_model: str = "qwen3:8b"
 
settings = Settings()
 
# ─── Logging ─────────────────────────────────────────────────────────────────
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AGENT] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)
 
# ─── Pydantic models for OpenAI-compatible API ───────────────────────────────
 
class Message(BaseModel):
    role: str
    content: str
 
 
class ChatRequest(BaseModel):
    model: str = settings.ollama_model
    messages: list[Message]
    stream: bool = True
    temperature: float = 0.2
    max_tokens: int = 2048
 
 
# ─── Clients ─────────────────────────────────────────────────────────────────
 
CATALOG_DDL = """
CREATE TABLE IF NOT EXISTS catalog_tables (
    id              SERIAL PRIMARY KEY,
    bucket          TEXT NOT NULL,
    object_key      TEXT NOT NULL,
    table_name      TEXT NOT NULL UNIQUE,
    minio_uri       TEXT NOT NULL,
    column_schema   JSONB NOT NULL DEFAULT '[]',
    row_count       BIGINT,
    file_size_bytes BIGINT,
    etag            TEXT,
    last_indexed_at TIMESTAMPTZ DEFAULT NOW(),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (bucket, object_key)
);
CREATE INDEX IF NOT EXISTS idx_catalog_table_name ON catalog_tables (table_name);
CREATE INDEX IF NOT EXISTS idx_catalog_bucket     ON catalog_tables (bucket);
"""
 
 
def get_pg_conn():
    return psycopg2.connect(settings.postgres_dsn)
 
 
def ensure_schema(retries: int = 15, delay: float = 3.0) -> None:
    """Wait for Postgres to be ready and ensure catalog table exists."""
    for attempt in range(1, retries + 1):
        try:
            conn = get_pg_conn()
            with conn.cursor() as cur:
                cur.execute(CATALOG_DDL)
            conn.commit()
            conn.close()
            log.info("Catalog schema ready.")
            return
        except Exception as e:
            log.warning("Waiting for Postgres (attempt %d/%d): %s", attempt, retries, e)
            time.sleep(delay)
    raise RuntimeError("Could not connect to Postgres after multiple retries.")
 
 
def build_minio() -> Minio:
    return Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )
 
 
ollama_client = OllamaClient(host=settings.ollama_host)
minio_client  = build_minio()
 
# ─── Data catalog helpers ─────────────────────────────────────────────────────
 
def fetch_catalog() -> list[dict]:
    """Return all catalog entries as a list of dicts."""
    conn = get_pg_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT table_name, bucket, object_key, minio_uri,
                       column_schema, row_count
                FROM catalog_tables
                ORDER BY table_name
                """
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
 
 
def build_catalog_description(catalog: list[dict]) -> str:
    """Render the catalog as a human/LLM-readable schema description."""
    if not catalog:
        return "No datasets have been indexed yet."
 
    lines = ["Available datasets (DuckDB-queryable via read_csv_auto / read_parquet):\n"]
    for entry in catalog:
        lines.append(f"Table alias : {entry['table_name']}")
        lines.append(f"MinIO URI   : {entry['minio_uri']}")
        lines.append(f"Rows        : {entry['row_count']}")
        columns: list[dict] = entry["column_schema"]
        if isinstance(columns, str):
            columns = json.loads(columns)
        # Pre-quote every column name so the LLM copies them verbatim into SQL
        col_desc = ", ".join(
            f'\"{c["name"]}\" ({c["dtype"]})' for c in columns
        )
        lines.append(f"Columns (use exactly as shown, with double-quotes): {col_desc}")
        lines.append("")
    return "\n".join(lines)
 
 
# ─── DuckDB S3 execution ──────────────────────────────────────────────────────
 
def execute_query(sql: str) -> tuple[list[str], list[tuple]]:
    """
    Run SQL in a fresh in-memory DuckDB session configured to reach
    MinIO over the S3 protocol.  Returns (column_names, rows).
    """
    endpoint_url = (
        f"{'https' if settings.minio_secure else 'http'}://{settings.minio_endpoint}"
    )
 
    con = duckdb.connect(database=":memory:")
    try:
        # Configure DuckDB S3 extension to point at MinIO
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute(f"SET s3_endpoint='{settings.minio_endpoint}';")
        con.execute(f"SET s3_access_key_id='{settings.minio_access_key}';")
        con.execute(f"SET s3_secret_access_key='{settings.minio_secret_key}';")
        con.execute(f"SET s3_use_ssl={'true' if settings.minio_secure else 'false'};")
        con.execute("SET s3_url_style='path';")
 
        rel     = con.execute(sql)
        columns = [d[0] for d in rel.description]
        rows    = rel.fetchall()
        return columns, rows
    finally:
        con.close()
 
 
# ─── Text-to-SQL pipeline ─────────────────────────────────────────────────────
 
SYSTEM_SQL = """You are an expert DuckDB SQL analyst. Given a schema catalog and a user question,
produce a single correct DuckDB SQL query.
 
CRITICAL COLUMN QUOTING RULES — follow exactly:
- ALWAYS wrap every column name in double-quotes: "AÑO", "CAMPAÑA_INFORMATIVA", "order_id"
- This is mandatory for ALL columns, especially those with accents, special characters,
  spaces, or uppercase letters (e.g. Ñ, É, Á, Ü, spaces, dots, slashes)
- The catalog shows the exact column names — copy them character-for-character inside double-quotes
- Wrong:  WHERE CAMPAÑA_INFORMATIVA = 'x'
- Correct: WHERE "CAMPAÑA_INFORMATIVA" = 'x'
 
TABLE REFERENCE RULES:
- CSV files:     SELECT ... FROM read_csv_auto('s3://bucket/file.csv')
- Parquet files: SELECT ... FROM read_parquet('s3://bucket/file.parquet')
- Always use the full s3:// URI from the catalog
 
OUTPUT RULES:
- Return ONLY the raw SQL — no markdown fences, no explanation, no preamble
- The query must be valid DuckDB SQL
"""
 
SYSTEM_NARRATE = """\
You are a helpful data assistant.  You will receive a user question, the SQL
query that was run, and the raw query results.  Summarise the results in clear,
concise natural language, including specific numbers and insights where helpful.
Do not show the SQL or raw data unless the user asks.
"""
 
 
def text_to_sql(user_question: str, catalog_description: str) -> str:
    prompt = (
        f"SCHEMA CATALOG:\n{catalog_description}\n\n"
        f"USER QUESTION: {user_question}\n\n"
        "Provide only the SQL query:"
    )
    response = ollama_client.chat(
        model=settings.ollama_model,
        messages=[
            {"role": "system", "content": SYSTEM_SQL},
            {"role": "user",   "content": prompt},
        ],
    )
    sql = response["message"]["content"].strip()
    # Strip accidental markdown fences
    sql = re.sub(r"^```[a-z]*\n?", "", sql, flags=re.I)
    sql = re.sub(r"\n?```$", "", sql)
    return sql.strip()
 
 
def narrate_results(
    user_question: str,
    sql: str,
    columns: list[str],
    rows: list[tuple],
) -> str:
    # Format results as a simple text table
    if not rows:
        results_text = "The query returned no rows."
    else:
        header = " | ".join(columns)
        separator = "-" * len(header)
        data_rows = "\n".join(" | ".join(str(v) for v in row) for row in rows[:50])
        results_text = f"{header}\n{separator}\n{data_rows}"
        if len(rows) > 50:
            results_text += f"\n… ({len(rows) - 50} more rows)"
 
    prompt = (
        f"USER QUESTION: {user_question}\n\n"
        f"SQL EXECUTED:\n{sql}\n\n"
        f"QUERY RESULTS:\n{results_text}\n\n"
        "Please answer the user's question based on the results."
    )
    response = ollama_client.chat(
        model=settings.ollama_model,
        messages=[
            {"role": "system", "content": SYSTEM_NARRATE},
            {"role": "user",   "content": prompt},
        ],
    )
    return response["message"]["content"].strip()
 
 
# ─── Pipeline orchestrator ────────────────────────────────────────────────────
 
def is_data_question(message: str) -> bool:
    """Heuristic: does the message look like a data/analytics question?"""
    keywords = [
        "how many", "total", "average", "sum", "count", "list", "show",
        "find", "which", "who", "what", "top", "bottom", "most", "least",
        "between", "compare", "revenue", "sales", "employee", "product",
        "salary", "stock", "inventory", "data", "table", "query", "sql",
        "highest", "lowest", "group", "where", "filter",
    ]
    lower = message.lower()
    return any(kw in lower for kw in keywords)
 
 
async def run_pipeline(user_message: str) -> AsyncIterator[str]:
    """
    Core agent pipeline.  Yields text chunks for SSE streaming.
    """
 
    def _sse_chunk(text: str, finish: str | None = None) -> str:
        chunk_id  = f"chatcmpl-{uuid.uuid4().hex}"
        delta     = {"role": "assistant", "content": text}
        choice    = {"index": 0, "delta": delta, "finish_reason": finish}
        payload   = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": settings.ollama_model,
            "choices": [choice],
        }
        return f"data: {json.dumps(payload)}\n\n"
 
    yield _sse_chunk("🔍 Checking the data catalog…\n")
 
    try:
        catalog = fetch_catalog()
    except Exception as e:
        yield _sse_chunk(f"❌ Could not reach the data catalog: {e}", finish="stop")
        yield "data: [DONE]\n\n"
        return
 
    if not catalog:
        yield _sse_chunk(
            "No datasets are indexed yet.  "
            "Please upload CSV or Parquet files to MinIO and wait for the indexer.",
            finish="stop",
        )
        yield "data: [DONE]\n\n"
        return
 
    catalog_description = build_catalog_description(catalog)
 
    if not is_data_question(user_message):
        # Pass through to Ollama as a regular question, context-enriched
        yield _sse_chunk("💬 Answering your question…\n")
        response = ollama_client.chat(
            model=settings.ollama_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant.  "
                        "The following datasets are available if the user asks:\n"
                        + catalog_description
                    ),
                },
                {"role": "user", "content": user_message},
            ],
        )
        answer = response["message"]["content"]
        yield _sse_chunk(answer, finish="stop")
        yield "data: [DONE]\n\n"
        return
 
    # ── Step 1: Text → SQL ────────────────────────────────────────────────
    yield _sse_chunk("🧠 Generating SQL query…\n")
    try:
        sql = text_to_sql(user_message, catalog_description)
        log.info("Generated SQL:\n%s", sql)
    except Exception as e:
        yield _sse_chunk(f"❌ Failed to generate SQL: {e}", finish="stop")
        yield "data: [DONE]\n\n"
        return
 
    yield _sse_chunk(f"\n```sql\n{sql}\n```\n\n")
 
    # ── Step 2: Execute query ──────────────────────────────────────────────
    yield _sse_chunk("⚡ Running query against MinIO via DuckDB…\n")
    try:
        columns, rows = execute_query(sql)
        log.info("Query returned %d rows", len(rows))
    except Exception as e:
        err_msg = str(e)
        log.error("Query execution failed: %s", err_msg)
        yield _sse_chunk(
            f"❌ Query execution failed:\n```\n{err_msg}\n```\n"
            "Trying to recover with a corrected query…\n"
        )
        # Self-correction: ask the LLM to fix the SQL
        try:
            fix_prompt = (
                f"The following DuckDB SQL query failed:\n{sql}\n\n"
                f"Error: {err_msg}\n\n"
                "IMPORTANT: The error is very likely a column quoting issue. "  
                "In DuckDB, ALL column names must be wrapped in double-quotes, "  
                "especially names with accents or special chars (Ñ, É, spaces). "  
                "Example: WHERE \"AÑO\" = 2020  NOT  WHERE AÑO = 2020\n\n"
                f"SCHEMA (columns are shown pre-quoted — copy them exactly):\n{catalog_description}\n\n"
                "Return ONLY the corrected SQL query:"
            )
            fixed_sql = ollama_client.chat(
                model=settings.ollama_model,
                messages=[
                    {"role": "system", "content": SYSTEM_SQL},
                    {"role": "user",   "content": fix_prompt},
                ],
            )["message"]["content"].strip()
            fixed_sql = re.sub(r"^```[a-z]*\n?", "", fixed_sql, flags=re.I)
            fixed_sql = re.sub(r"\n?```$", "", fixed_sql).strip()
            log.info("Corrected SQL:\n%s", fixed_sql)
            yield _sse_chunk(f"Corrected query:\n```sql\n{fixed_sql}\n```\n\n")
            columns, rows = execute_query(fixed_sql)
            sql = fixed_sql
        except Exception as e2:
            yield _sse_chunk(f"❌ Could not recover: {e2}", finish="stop")
            yield "data: [DONE]\n\n"
            return
 
    # ── Step 3: Narrate ────────────────────────────────────────────────────
    yield _sse_chunk("📝 Generating answer…\n\n")
    try:
        answer = narrate_results(user_message, sql, columns, rows)
    except Exception as e:
        answer = f"Query succeeded but narration failed: {e}"
 
    yield _sse_chunk(answer, finish="stop")
    yield "data: [DONE]\n\n"
 
 
# ─── FastAPI app ──────────────────────────────────────────────────────────────
 
app = FastAPI(title="LLM Data Query Agent")
 
 
@app.get("/v1/models")
async def list_models():
    """Open WebUI calls this to discover available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": settings.ollama_model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }
 
 
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    # Extract the latest user message
    user_msg = ""
    for m in reversed(request.messages):
        if m.role == "user":
            user_msg = m.content
            break
 
    log.info("Received: %s", user_msg[:120])
 
    if request.stream:
        return StreamingResponse(
            run_pipeline(user_msg),
            media_type="text/event-stream",
        )
    else:
        # Non-streaming: collect all chunks
        full_text = ""
        async for chunk in run_pipeline(user_msg):
            if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]":
                try:
                    payload = json.loads(chunk[6:])
                    full_text += payload["choices"][0]["delta"].get("content", "")
                except Exception:
                    pass
 
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": settings.ollama_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full_text},
                    "finish_reason": "stop",
                }
            ],
        }
 
 
@app.get("/health")
async def health():
    return {"status": "ok"}
 
 
# ─── Entry point ─────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    ensure_schema()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
