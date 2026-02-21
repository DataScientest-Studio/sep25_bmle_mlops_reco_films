# ============================================================
# FICHIER : src/monitoring/data_monitoring.py
# ============================================================
# OBJECTIF (PROD / GROS VOLUMES) :
# - Calculer et historiser des métriques de monitoring (volumétrie + drift data + KPI métier) 
# - Stockage en PostgreSQL dans raw.monitoring_metrics (Grafana datasource)
# - Compatible CRON (quotidien) et déclenchement manuel via API
# - Idempotent (UPSERT) : relançable sans doublons
#
# Optimisé :
# - Filtrage epoch (pas de to_timestamp)
# - Index exploitables
# - UPSERT idempotent
# - Drift mensuel / annuel uniquement
#
#
# LISTE COMPLÈTE DES MÉTRIQUES CALCULÉES
# Core (daily / monthly / yearly)
# - nb_notes
# - nb_users
# - nb_movies
# - avg_rating
# - nb_new_users
# - nb_new_movies
# - nb_new_ratings
#
# Distribution notes (monthly / yearly)
# - pct_rating_0_5
# - pct_rating_1
# - pct_rating_1_5
# …
# - pct_rating_5
#
# Genres (monthly / yearly)
# - avg_genres_per_movie
# - pct_genre_Action
# - pct_genre_Drama
# …
# 
# Drift (monthly / yearly)
# - psi_rating_prev_month
# - psi_genre_prev_month
# - psi_rating_prev_year
# - psi_genre_prev_year
#
# DRIFT SCORE (PSI) :
# - PSI (Population Stability Index) entre distribution "courante" et "référence"
# - Ici : référence = période précédente (mois-1, année-1)
#
#   PSI = Σ_i (p_i - q_i) * ln(p_i / q_i)
#   où :
#     p_i = proportion dans la période courante, bin i
#     q_i = proportion dans la période référence, bin i
#
# - PSI ≈ 0 : stable
# - PSI > 0.1 : drift modéré (règle empirique)
# - PSI > 0.25 : drift fort (règle empirique)
#
# IMPORTANT pour éviter ln(0) / divisions par 0 :
# - on applique un epsilon sur p_i et q_i : p_i = max(p_i, eps), q_i = max(q_i, eps)
#
# OPTIMISATION :
# - Pas de fenêtre ROW_NUMBER() sur toute la table (coûteux)
# - "Nouveaux users / films / user-film" calculés via MIN(timestamp) agrégé
# - Index SQL auto-créés pour accélérer les GROUP BY / filtres temporels
# ============================================================

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from sqlalchemy import create_engine, text

PG_URL = os.getenv(
    "PG_URL",
    "postgresql+psycopg2://movie:movie@127.0.0.1:5432/movie_reco"
)

SCHEMA = os.getenv("PG_SCHEMA", "raw")
engine = create_engine(PG_URL)


# ============================================================
# TABLE MONITORING
# ============================================================

def ensure_monitoring_table():

    query = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.monitoring_metrics (
        metric_date   date NOT NULL,
        period_type   varchar(20) NOT NULL,
        period_value  varchar(20) NOT NULL,
        metric_name   varchar(150) NOT NULL,
        metric_value  double precision,
        created_at    timestamp default now(),
        PRIMARY KEY (period_type, period_value, metric_name)
    );
    """

    with engine.begin() as conn:
        conn.execute(text(query))


# ============================================================
# UPSERT
# ============================================================

def upsert_metric(period_type, period_value, metric_name, metric_value):

    query = f"""
    INSERT INTO {SCHEMA}.monitoring_metrics
        (metric_date, period_type, period_value, metric_name, metric_value)
    VALUES
        (CURRENT_DATE, :pt, :pv, :mn, :mv)
    ON CONFLICT (period_type, period_value, metric_name)
    DO UPDATE SET
        metric_value = EXCLUDED.metric_value,
        created_at = now();
    """

    with engine.begin() as conn:
        conn.execute(
            text(query),
            {"pt": period_type, "pv": period_value,
             "mn": metric_name,
             "mv": float(metric_value) if metric_value is not None else None}
        )


# ============================================================
# BORNES EPOCH
# ============================================================

@dataclass(frozen=True)
class Bounds:
    start: int
    end: int


def to_epoch(dt):
    return int(dt.timestamp())


def day_bounds(d):
    start = datetime(d.year, d.month, d.day)
    return Bounds(to_epoch(start), to_epoch(start + timedelta(days=1)))


def month_bounds(ym):
    y, m = map(int, ym.split("-"))
    start = datetime(y, m, 1)
    end = datetime(y + (m == 12), 1 if m == 12 else m + 1, 1)
    return Bounds(to_epoch(start), to_epoch(end))


def year_bounds(y):
    start = datetime(y, 1, 1)
    return Bounds(to_epoch(start), to_epoch(datetime(y + 1, 1, 1)))


# ============================================================
# CORE METRICS + NOUVEAUTÉS
# ============================================================

def compute_core(period_type, period_value, bounds):

    params = {"start": bounds.start, "end": bounds.end}

    # Activité
    q_activity = f"""
    SELECT
        COUNT(*) nb_notes,
        COUNT(DISTINCT "userId") nb_users,
        COUNT(DISTINCT "movieId") nb_movies,
        AVG(rating) avg_rating
    FROM {SCHEMA}.raw_ratings
    WHERE "timestamp" >= :start
      AND "timestamp" <  :end;
    """

    # Nouveaux users
    q_new_users = f"""
    SELECT COUNT(*) FROM (
        SELECT "userId", MIN("timestamp") first_ts
        FROM {SCHEMA}.raw_ratings
        GROUP BY "userId"
    ) s
    WHERE first_ts >= :start
      AND first_ts <  :end;
    """

    # Nouveaux films
    q_new_movies = f"""
    SELECT COUNT(*) FROM (
        SELECT "movieId", MIN("timestamp") first_ts
        FROM {SCHEMA}.raw_ratings
        GROUP BY "movieId"
    ) s
    WHERE first_ts >= :start
      AND first_ts <  :end;
    """

    # Nouvelles interactions
    q_new_ratings = f"""
    SELECT COUNT(*) FROM (
        SELECT "userId","movieId", MIN("timestamp") first_ts
        FROM {SCHEMA}.raw_ratings
        GROUP BY "userId","movieId"
    ) s
    WHERE first_ts >= :start
      AND first_ts <  :end;
    """

    with engine.begin() as conn:
        a = conn.execute(text(q_activity), params).mappings().first()
        nu = conn.execute(text(q_new_users), params).scalar()
        nm = conn.execute(text(q_new_movies), params).scalar()
        nr = conn.execute(text(q_new_ratings), params).scalar()

    for k, v in a.items():
        upsert_metric(period_type, period_value, k, v)

    upsert_metric(period_type, period_value, "nb_new_users", nu)
    upsert_metric(period_type, period_value, "nb_new_movies", nm)
    upsert_metric(period_type, period_value, "nb_new_ratings", nr)


# ============================================================
# DISTRIBUTION NOTES
# ============================================================

def compute_rating_distribution(period_type, period_value, bounds):

    params = {"start": bounds.start, "end": bounds.end}

    query = f"""
    WITH dist AS (
        SELECT rating, COUNT(*)::float cnt
        FROM {SCHEMA}.raw_ratings
        WHERE "timestamp" >= :start
          AND "timestamp" <  :end
        GROUP BY rating
    ),
    tot AS (SELECT SUM(cnt) total FROM dist)
    SELECT rating, (cnt/NULLIF(total,0))*100 pct
    FROM dist, tot;
    """

    with engine.begin() as conn:
        rows = conn.execute(text(query), params).mappings().all()

    for r in rows:
        label = str(r["rating"]).replace(".", "_")
        upsert_metric(period_type, period_value,
                      f"pct_rating_{label}", r["pct"])


# ============================================================
# GENRES
# ============================================================

def compute_genres(period_type, period_value, bounds):

    params = {"start": bounds.start, "end": bounds.end}

    # Moyenne nb genres par film distinct
    q_avg = f"""
    WITH movies_noted AS (
        SELECT DISTINCT "movieId"
        FROM {SCHEMA}.raw_ratings
        WHERE "timestamp" >= :start
          AND "timestamp" <  :end
    )
    SELECT AVG(array_length(string_to_array(genres,'|'),1))
    FROM movies_noted mn
    JOIN {SCHEMA}.raw_movies m
      ON mn."movieId"=m."movieId"
    WHERE genres IS NOT NULL
      AND genres<>'(no genres listed)';
    """

    # Distribution genres
    q_dist = f"""
    WITH base AS (
        SELECT unnest(string_to_array(genres,'|')) genre
        FROM {SCHEMA}.raw_ratings r
        JOIN {SCHEMA}.raw_movies m
          ON r."movieId"=m."movieId"
        WHERE r."timestamp" >= :start
          AND r."timestamp" <  :end
          AND genres IS NOT NULL
          AND genres<>'(no genres listed)'
    ),
    dist AS (
        SELECT genre, COUNT(*)::float cnt
        FROM base
        GROUP BY genre
    ),
    tot AS (SELECT SUM(cnt) total FROM dist)
    SELECT genre, (cnt/NULLIF(total,0))*100 pct
    FROM dist, tot;
    """

    with engine.begin() as conn:
        avg_val = conn.execute(text(q_avg), params).scalar()
        rows = conn.execute(text(q_dist), params).mappings().all()

    upsert_metric(period_type, period_value,
                  "avg_genres_per_movie", avg_val)

    for r in rows:
        name = r["genre"].replace(" ", "_")
        upsert_metric(period_type, period_value,
                      f"pct_genre_{name}", r["pct"])


# ============================================================
# PSI (NOTES & GENRES)
# ============================================================

def compute_psi(period_type, period_value,
                cur_bounds, ref_bounds,
                metric_name, field, join_genres=False):

    eps = 1e-6

    if not join_genres:
        base_query = f"""
        SELECT {field}, COUNT(*)::float cnt
        FROM {SCHEMA}.raw_ratings
        WHERE "timestamp">=:start AND "timestamp"<:end
        GROUP BY {field}
        """
    else:
        base_query = f"""
        SELECT unnest(string_to_array(genres,'|')) {field},
               COUNT(*)::float cnt
        FROM {SCHEMA}.raw_ratings r
        JOIN {SCHEMA}.raw_movies m
          ON r."movieId"=m."movieId"
        WHERE r."timestamp">=:start AND r."timestamp"<:end
          AND genres IS NOT NULL
          AND genres<>'(no genres listed)'
        GROUP BY {field}
        """

    query = f"""
    WITH cur AS ({base_query.replace(':start',':cs').replace(':end',':ce')}),
         ref AS ({base_query.replace(':start',':rs').replace(':end',':re')}),
         cats AS (
            SELECT COALESCE(c.{field},r.{field}) val,
                   COALESCE(c.cnt,0) cur_cnt,
                   COALESCE(r.cnt,0) ref_cnt
            FROM cur c FULL OUTER JOIN ref r
              ON c.{field}=r.{field}
         ),
         totals AS (
            SELECT SUM(cur_cnt) cur_total,
                   SUM(ref_cnt) ref_total
            FROM cats
         ),
         probs AS (
            SELECT
                GREATEST(cur_cnt/NULLIF(cur_total,0),:eps) p,
                GREATEST(ref_cnt/NULLIF(ref_total,0),:eps) q
            FROM cats, totals
         )
    SELECT SUM((p-q)*LN(p/q)) FROM probs;
    """

    params = {
        "cs": cur_bounds.start,
        "ce": cur_bounds.end,
        "rs": ref_bounds.start,
        "re": ref_bounds.end,
        "eps": eps
    }

    with engine.begin() as conn:
        psi = conn.execute(text(query), params).scalar()

    upsert_metric(period_type, period_value, metric_name, psi)


# ============================================================
# ORCHESTRATEUR
# ============================================================

def run_monitoring(reference_date=None):

    ensure_monitoring_table()

    today = reference_date or date.today()
    yesterday = today - timedelta(days=1)

    results = {"date": str(yesterday)}

    # DAILY
    d_bounds = day_bounds(yesterday)
    compute_core("daily", str(yesterday), d_bounds)
    results["daily"] = "done"

    # MONTHLY
    ym = yesterday.strftime("%Y-%m")
    m_bounds = month_bounds(ym)
    compute_core("monthly", ym, m_bounds)
    compute_rating_distribution("monthly", ym, m_bounds)
    compute_genres("monthly", ym, m_bounds)

    prev_month = month_bounds(
        (yesterday.replace(day=1) - timedelta(days=1)).strftime("%Y-%m")
    )

    compute_psi("monthly", ym,
                m_bounds, prev_month,
                "psi_rating_prev_month",
                "rating")

    compute_psi("monthly", ym,
                m_bounds, prev_month,
                "psi_genre_prev_month",
                "genre", join_genres=True)

    results["monthly"] = "done"

    # YEARLY
    y = int(yesterday.strftime("%Y"))
    y_bounds = year_bounds(y)
    compute_core("yearly", str(y), y_bounds)
    compute_rating_distribution("yearly", str(y), y_bounds)
    compute_genres("yearly", str(y), y_bounds)

    prev_year = year_bounds(y - 1)

    compute_psi("yearly", str(y),
                y_bounds, prev_year,
                "psi_rating_prev_year",
                "rating")

    compute_psi("yearly", str(y),
                y_bounds, prev_year,
                "psi_genre_prev_year",
                "genre", join_genres=True)

    results["yearly"] = "done"

    results["status"] = "SUCCESS"
    return results


if __name__ == "__main__":
    print(run_monitoring())