# ============================================================
# RUN_TRAINING_WITH_MONITORING.PY
# ============================================================
# - Lance train_model2
# - Mesure durée
# - Capture statut SUCCESS / FAILED
# - Historise dans raw.training_run_log
# ============================================================

from __future__ import annotations

import time
import traceback
from datetime import datetime
import os

from sqlalchemy import create_engine, text

from src.models.train_model2 import train_item_based_cf


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
def ensure_training_log_table():
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.training_run_log (
        run_id SERIAL PRIMARY KEY,
        started_at TIMESTAMP NOT NULL,
        finished_at TIMESTAMP,
        status TEXT NOT NULL,
        duration_sec FLOAT,
        error_message TEXT
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    ensure_training_log_table()

    started_at = datetime.utcnow()
    start_time = time.time()

    status = "SUCCESS"
    error_message = None

    try:
        # Paramètres par défaut identiques CLI
        train_item_based_cf(
            k_neighbors=20,
            min_ratings=50
        )

    except Exception as e:
        status = "FAILED"
        error_message = str(e) + "\n" + traceback.format_exc()

    duration_sec = float(time.time() - start_time)
    finished_at = datetime.utcnow()

    insert_query = f"""
    INSERT INTO {SCHEMA}.training_run_log
        (started_at, finished_at, status, duration_sec, error_message)
    VALUES
        (:started_at, :finished_at, :status, :duration_sec, :error_message);
    """

    with engine.begin() as conn:
        conn.execute(
            text(insert_query),
            {
                "started_at": started_at,
                "finished_at": finished_at,
                "status": status,
                "duration_sec": duration_sec,
                "error_message": error_message,
            },
        )

    if status == "FAILED":
        raise RuntimeError("Training failed. See training_run_log table.")


if __name__ == "__main__":
    main()