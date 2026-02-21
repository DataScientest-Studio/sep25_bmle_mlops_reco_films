# ============================================================
# QUALITY_CHECKS.PY
# ============================================================
# OBJECTIF
# ------------------------------------------------------------
# Vérifier la qualité des données après ingestion.
#
# Responsabilités :
# - exécuter des checks SQL
# - historiser résultats dans raw.quality_checks_log
# - retourner un résumé structuré (API / cron)
#
# IMPORTANT :
# - Ne stoppe PAS le process dans la fonction principale
# - Le bloc CLI gère le code retour système
# ============================================================

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Any

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
PG_URL = os.getenv(
    "PG_URL",
    "postgresql+psycopg2://movie:movie@127.0.0.1:5432/movie_reco"
)

SCHEMA = os.getenv("PG_SCHEMA", "raw")

engine = create_engine(PG_URL)


# ------------------------------------------------------------
# CREATION TABLE LOG SI ABSENTE
# ------------------------------------------------------------
def ensure_quality_table() -> None:
    """
    Crée la table raw.quality_checks_log si absente.

    Clé primaire :
        (run_timestamp, check_name)

    Chaque run possède un timestamp unique.
    """

    create_table = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.quality_checks_log (
        run_timestamp TIMESTAMP NOT NULL,
        check_name VARCHAR(200) NOT NULL,
        error_count INTEGER,
        status VARCHAR(10),
        PRIMARY KEY (run_timestamp, check_name)
    );
    """

    with engine.begin() as conn:
        conn.execute(text(create_table))


# ------------------------------------------------------------
# EXECUTION D'UN CHECK
# ------------------------------------------------------------
def execute_check(
    conn,
    run_ts: datetime,
    name: str,
    query: str
) -> Dict[str, Any]:
    """
    Exécute un check SQL.
    Stocke le résultat en base.
    Retourne un dict structuré.
    """

    value = conn.execute(text(query)).scalar()
    value = int(value or 0)

    status = "OK" if value == 0 else "FAIL"

    conn.execute(
        text(f"""
            INSERT INTO {SCHEMA}.quality_checks_log
                (run_timestamp, check_name, error_count, status)
            VALUES
                (:run_ts, :name, :value, :status)
        """),
        {
            "run_ts": run_ts,
            "name": name,
            "value": value,
            "status": status,
        }
    )

    return {
        "check_name": name,
        "error_count": value,
        "status": status
    }


# ------------------------------------------------------------
# FONCTION PRINCIPALE
# ------------------------------------------------------------
def run_quality_checks() -> Dict[str, Any]:
    """
    Lance tous les checks qualité.

    Retourne :
    {
        "status": "SUCCESS" | "FAILED",
        "checks": [ {check_name, error_count, status}, ... ]
    }
    """

    ensure_quality_table()

    run_ts = datetime.utcnow()
    results: List[Dict[str, Any]] = []

    ratings_table = f"{SCHEMA}.raw_ratings"
    movies_table = f"{SCHEMA}.raw_movies"

    try:
        with engine.begin() as conn:

            # ------------------------------------------------
            # CHECK 1 — Ratings valides [0,5]
            # ------------------------------------------------
            results.append(execute_check(
                conn, run_ts,
                "Ratings dans [0,5]",
                f"""
                SELECT COUNT(*)
                FROM {ratings_table}
                WHERE rating < 0
                   OR rating > 5
                   OR rating IS NULL
                """
            ))

            # ------------------------------------------------
            # CHECK 2 — userId non NULL
            # ------------------------------------------------
            results.append(execute_check(
                conn, run_ts,
                "userId non NULL",
                f"""
                SELECT COUNT(*)
                FROM {ratings_table}
                WHERE "userId" IS NULL
                """
            ))

            # ------------------------------------------------
            # CHECK 3 — movieId non NULL
            # ------------------------------------------------
            results.append(execute_check(
                conn, run_ts,
                "movieId non NULL",
                f"""
                SELECT COUNT(*)
                FROM {ratings_table}
                WHERE "movieId" IS NULL
                """
            ))

            # ------------------------------------------------
            # CHECK 4 — movieId référencé existe
            # ------------------------------------------------
            results.append(execute_check(
                conn, run_ts,
                "movieId existe dans movies",
                f"""
                SELECT COUNT(*)
                FROM {ratings_table} r
                LEFT JOIN {movies_table} m
                       ON r."movieId" = m."movieId"
                WHERE m."movieId" IS NULL
                """
            ))

            # ------------------------------------------------
            # CHECK 5 — Pas de doublons exacts
            # ------------------------------------------------
            results.append(execute_check(
                conn, run_ts,
                "Pas de doublons (userId, movieId, timestamp)",
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT "userId", "movieId", "timestamp"
                    FROM {ratings_table}
                    GROUP BY "userId", "movieId", "timestamp"
                    HAVING COUNT(*) > 1
                ) t
                """
            ))

    except SQLAlchemyError as e:
        return {
            "status": "FAILED",
            "checks": results,
            "error_type": type(e).__name__,
            "error_message": str(e)
        }

    # --------------------------------------------------------
    # STATUT GLOBAL
    # --------------------------------------------------------
    failed = [c for c in results if c["status"] == "FAIL"]

    global_status = "SUCCESS" if len(failed) == 0 else "FAILED"

    return {
        "status": global_status,
        "checks": results
    }


# ------------------------------------------------------------
# EXECUTION CLI (CRON)
# ------------------------------------------------------------
if __name__ == "__main__":

    result = run_quality_checks()

    print("\n--- QUALITY CHECK REPORT ---")
    print(f"Global status : {result['status']}")

    for c in result["checks"]:
        print(f"{c['check_name']} → {c['status']} ({c['error_count']})")

    if result["status"] == "FAILED":
        raise SystemExit(1)
    else:
        raise SystemExit(0)