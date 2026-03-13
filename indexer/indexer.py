"""
Indexer Service
===============
Periodically scans MinIO buckets, discovers CSV/Parquet files,
infers their schemas using DuckDB, and writes metadata into the
PostgreSQL data catalog.  Acts as a lightweight data-catalog daemon.
"""
from __future__ import annotations
 
import io
import json
import logging
import os
import re
import time
from typing import Any
 
import duckdb
import psycopg2
import psycopg2.extras
from minio import Minio
from minio.error import S3Error
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
 
# ─── Configuration ───────────────────────────────────────────────────────────
 
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
 
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False
 
    postgres_dsn: str = "postgresql://admin:admin@postgres:5432/datacatalog"
 
    scan_interval_seconds: int = 30
    supported_extensions: list[str] = [".csv", ".parquet", ".json"]
 
    @field_validator("supported_extensions", mode="before")
    @classmethod
    def parse_ext_list(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [e.strip() for e in v.split(",")]
        return v
 
 
settings = Settings()
 
# ─── Logging ─────────────────────────────────────────────────────────────────
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [INDEXER] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)
 
# ─── Helpers ─────────────────────────────────────────────────────────────────
 
def slugify(text: str) -> str:
    """Turn an arbitrary string into a safe SQL identifier."""
    text = os.path.splitext(text)[0]          # drop extension
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    text = text.strip("_").lower()
    if text and text[0].isdigit():
        text = "t_" + text
    return text or "table"
 
 
def make_table_name(bucket: str, key: str) -> str:
    parts = [slugify(bucket)] + [slugify(p) for p in key.split("/") if p]
    return "_".join(parts)
 
 
# ─── MinIO client ────────────────────────────────────────────────────────────
 
def build_minio() -> Minio:
    return Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )
 
 
def list_all_objects(client: Minio) -> list[dict]:
    """Return a flat list of {bucket, key, size, etag} for supported files."""
    results = []
    try:
        buckets = client.list_buckets()
    except S3Error as e:
        log.error("Cannot list buckets: %s", e)
        return results
 
    for bucket in buckets:
        try:
            objects = client.list_objects(bucket.name, recursive=True)
            for obj in objects:
                ext = os.path.splitext(obj.object_name)[1].lower()
                if ext in settings.supported_extensions:
                    results.append(
                        {
                            "bucket": bucket.name,
                            "key": obj.object_name,
                            "size": obj.size,
                            "etag": obj.etag,
                        }
                    )
        except S3Error as e:
            log.error("Error listing bucket %s: %s", bucket.name, e)
 
    return results
 
 
# ─── DuckDB schema inference ──────────────────────────────────────────────────
 
def infer_schema(client: Minio, bucket: str, key: str) -> dict:
    """
    Download the file into memory and use DuckDB to infer column types,
    row count, and sample values.
    Returns {"columns": [...], "row_count": int}
    """
    ext = os.path.splitext(key)[1].lower()
 
    try:
        response = client.get_object(bucket, key)
        data = response.read()
    finally:
        try:
            response.close()
            response.release_conn()
        except Exception:
            pass
 
    con = duckdb.connect(database=":memory:")
 
    try:
        if ext == ".csv":
            con.execute(
                "CREATE TABLE _tmp AS SELECT * FROM read_csv_auto(?, sample_size=1000)",
                [io.BytesIO(data)],
            )
        elif ext == ".parquet":
            con.execute(
                "CREATE TABLE _tmp AS SELECT * FROM read_parquet(?)",
                [io.BytesIO(data)],
            )
        elif ext == ".json":
            con.execute(
                "CREATE TABLE _tmp AS SELECT * FROM read_json_auto(?)",
                [io.BytesIO(data)],
            )
        else:
            return {"columns": [], "row_count": 0}
 
        # Column metadata
        desc = con.execute("DESCRIBE _tmp").fetchall()
        # desc rows: (col_name, col_type, null, key, default, extra)
        columns = []
        for row in desc:
            col_name, col_type = row[0], row[1]
            samples = con.execute(
                f'SELECT DISTINCT "{col_name}" FROM _tmp WHERE "{col_name}" IS NOT NULL LIMIT 3'
            ).fetchall()
            columns.append(
                {
                    "name": col_name,
                    "dtype": col_type,
                    "samples": [str(s[0]) for s in samples],
                }
            )
 
        row_count = con.execute("SELECT COUNT(*) FROM _tmp").fetchone()[0]
        return {"columns": columns, "row_count": row_count}
 
    except Exception as e:
        log.warning("Schema inference failed for s3://%s/%s: %s", bucket, key, e)
        return {"columns": [], "row_count": 0}
    finally:
        con.close()
 
 
# ─── Postgres catalog operations ──────────────────────────────────────────────
 
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
 
 
def ensure_schema(retries: int = 20, delay: float = 3.0) -> psycopg2.extensions.connection:
    """Wait for Postgres, create catalog table if missing, return connection."""
    for attempt in range(1, retries + 1):
        try:
            conn = get_pg_conn()
            with conn.cursor() as cur:
                cur.execute(CATALOG_DDL)
            conn.commit()
            log.info("Catalog schema ready.")
            return conn
        except Exception as e:
            log.warning("Waiting for Postgres (attempt %d/%d): %s", attempt, retries, e)
            time.sleep(delay)
    raise RuntimeError("Could not connect to Postgres after multiple retries.")
 
 
def upsert_catalog_entry(
    conn,
    bucket: str,
    key: str,
    table_name: str,
    minio_uri: str,
    schema: dict,
    file_size: int,
    etag: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO catalog_tables
                (bucket, object_key, table_name, minio_uri, column_schema,
                 row_count, file_size_bytes, etag, last_indexed_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (bucket, object_key)
            DO UPDATE SET
                table_name      = EXCLUDED.table_name,
                minio_uri       = EXCLUDED.minio_uri,
                column_schema   = EXCLUDED.column_schema,
                row_count       = EXCLUDED.row_count,
                file_size_bytes = EXCLUDED.file_size_bytes,
                etag            = EXCLUDED.etag,
                last_indexed_at = NOW()
            """,
            (
                bucket,
                key,
                table_name,
                minio_uri,
                json.dumps(schema["columns"]),
                schema["row_count"],
                file_size,
                etag,
            ),
        )
    conn.commit()
 
 
def get_known_etags(conn) -> dict[tuple, str]:
    """Returns {(bucket, key): etag} for all indexed objects."""
    with conn.cursor() as cur:
        cur.execute("SELECT bucket, object_key, etag FROM catalog_tables")
        return {(r[0], r[1]): r[2] for r in cur.fetchall()}
 
 
def delete_stale_entries(conn, active_keys: set[tuple]) -> None:
    """Remove catalog entries for files that no longer exist in MinIO."""
    with conn.cursor() as cur:
        cur.execute("SELECT bucket, object_key FROM catalog_tables")
        existing = {(r[0], r[1]) for r in cur.fetchall()}
    stale = existing - active_keys
    if stale:
        with conn.cursor() as cur:
            for bucket, key in stale:
                cur.execute(
                    "DELETE FROM catalog_tables WHERE bucket=%s AND object_key=%s",
                    (bucket, key),
                )
                log.info("Removed stale entry: s3://%s/%s", bucket, key)
        conn.commit()
 
 
# ─── Main scan loop ───────────────────────────────────────────────────────────
 
def scan_once(minio_client: Minio, pg_conn) -> None:
    objects = list_all_objects(minio_client)
    known_etags = get_known_etags(pg_conn)
    active_keys: set[tuple] = set()
 
    for obj in objects:
        bucket, key, size, etag = obj["bucket"], obj["key"], obj["size"], obj["etag"]
        active_keys.add((bucket, key))
 
        # Skip if unchanged
        if known_etags.get((bucket, key)) == etag:
            continue
 
        log.info("Indexing s3://%s/%s (etag=%s)", bucket, key, etag)
        schema = infer_schema(minio_client, bucket, key)
        table_name = make_table_name(bucket, key)
        minio_uri = f"s3://{bucket}/{key}"
 
        upsert_catalog_entry(
            pg_conn, bucket, key, table_name, minio_uri, schema, size, etag
        )
        log.info(
            "  → table_name='%s' | %d columns | %d rows",
            table_name,
            len(schema["columns"]),
            schema["row_count"],
        )
 
    delete_stale_entries(pg_conn, active_keys)
 
 
def main() -> None:
    log.info("Indexer starting — interval=%ds", settings.scan_interval_seconds)
 
    minio_client = build_minio()
    pg_conn = ensure_schema()
 
    while True:
        try:
            scan_once(minio_client, pg_conn)
        except psycopg2.OperationalError:
            log.warning("PG connection lost, reconnecting…")
            pg_conn = get_pg_conn()
        except Exception as e:
            log.error("Scan error: %s", e, exc_info=True)
 
        time.sleep(settings.scan_interval_seconds)
 
 
if __name__ == "__main__":
    main()
