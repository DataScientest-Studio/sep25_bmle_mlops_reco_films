# ============================================================
# RUN_INGESTION_WITH_MONITORING.PY (VERSION SIMPLIFIÉE)
# ============================================================
# - Lance ingest_movielens()
# - Mesure durée
# - Capture SUCCESS / FAILED
# - Log :
#     * lignes injectées (delta)
#     * total en base
# - Tables monitorées :
#     raw_movies
#     raw_ratings
# ============================================================

from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone
import os

from sqlalchemy import create_engine, text

from src.ingestion.ingestion_movielens import ingest_movielens


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
PG_URL = os.getenv(
    "PG_URL",
    "postgresql+psycopg2://movie:movie@localhost:5432/movie_reco",
)


SCHEMA = os.getenv("PG_SCHEMA", "raw")

engine = create_engine(PG_URL)


# ------------------------------------------------------------
# CREATE TABLE IF NOT EXISTS
# ------------------------------------------------------------
def ensure_ingestion_log_table():
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.ingestion_run_log (
        run_id SERIAL PRIMARY KEY,
        started_at TIMESTAMP NOT NULL,
        finished_at TIMESTAMP,
        status TEXT NOT NULL,
        duration_sec FLOAT,

        rows_movies_injected BIGINT,
        rows_ratings_injected BIGINT,

        rows_movies_total BIGINT,
        rows_ratings_total BIGINT,

        error_message TEXT
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


# ------------------------------------------------------------
# COUNT TABLE
# ------------------------------------------------------------
def count_table(table_name: str) -> int:
    query = f"""
    SELECT COUNT(*)
    FROM information_schema.tables
    WHERE table_schema = :schema
      AND table_name = :table
    """

    with engine.begin() as conn:
        exists = conn.execute(
            text(query),
            {"schema": SCHEMA, "table": table_name}
        ).scalar()

        if not exists:
            return 0

        result = conn.execute(
            text(f"SELECT COUNT(*) FROM {SCHEMA}.{table_name}")
        ).scalar()

        return int(result or 0)


def get_counts():
    return {
        "movies": count_table("raw_movies"),
        "ratings": count_table("raw_ratings"),
    }


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    ensure_ingestion_log_table()

    started_at = datetime.now(timezone.utc)
    start_time = time.time()

    status = "SUCCESS"
    error_message = None

    # Comptage avant ingestion
    rows_before = get_counts()

    try:
        ingest_movielens()

    except Exception as e:
        status = "FAILED"
        error_message = str(e) + "\n" + traceback.format_exc()

    duration_sec = float(time.time() - start_time)
    finished_at = datetime.now(timezone.utc)

    # Comptage après ingestion
    rows_after = get_counts()

    rows_movies_injected = rows_after["movies"] - rows_before["movies"]
    rows_ratings_injected = rows_after["ratings"] - rows_before["ratings"]

    rows_movies_total = rows_after["movies"]
    rows_ratings_total = rows_after["ratings"]

    insert_query = f"""
    INSERT INTO {SCHEMA}.ingestion_run_log
        (started_at, finished_at, status, duration_sec,
         rows_movies_injected, rows_ratings_injected,
         rows_movies_total, rows_ratings_total,
         error_message)
    VALUES
        (:started_at, :finished_at, :status, :duration_sec,
         :rows_movies_injected, :rows_ratings_injected,
         :rows_movies_total, :rows_ratings_total,
         :error_message);
    """

    with engine.begin() as conn:
        conn.execute(
            text(insert_query),
            {
                "started_at": started_at,
                "finished_at": finished_at,
                "status": status,
                "duration_sec": duration_sec,
                "rows_movies_injected": rows_movies_injected,
                "rows_ratings_injected": rows_ratings_injected,
                "rows_movies_total": rows_movies_total,
                "rows_ratings_total": rows_ratings_total,
                "error_message": error_message,
            },
        )

    if status == "FAILED":
        raise RuntimeError("Ingestion failed. See ingestion_run_log table.")


if __name__ == "__main__":
    main()