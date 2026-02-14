# ============================================================
# TRAIN_MODEL2.PY
# ============================================================
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import subprocess
from datetime import datetime
import yaml
import mlflow
from sqlalchemy import create_engine, text
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import argparse


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
PG_URL = os.getenv(
    "PG_URL",
    "postgresql+psycopg2://movie:movie@127.0.0.1:5432/movie_reco"
)
SCHEMA = os.getenv("PG_SCHEMA", "raw")

EXPERIMENT_NAME = "reco-films/itemcf-v2"
REGISTERED_MODEL_NAME = "reco-films-itemcf-v2"


# ------------------------------------------------------------
# UTILITAIRES MLFLOW (versioning code & data)
# ------------------------------------------------------------
def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True
        ).strip()
    except Exception:
        return "unknown"


def get_dvc_md5(dvc_path: str = "data/raw.dvc") -> str:
    try:
        with open(dvc_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        return str(y["outs"][0].get("md5", "unknown"))
    except Exception:
        return "unknown"


# ------------------------------------------------------------
# POPULARITÉ BAYÉSIENNE (fallback cold-start)
# ------------------------------------------------------------
def compute_bayesian_popularity(ratings: pd.DataFrame) -> pd.DataFrame:
    stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"]).reset_index()

    C = stats["count"].mean()
    M = stats["mean"].mean()

    stats["bayes_score"] = (C * M + stats["count"] * stats["mean"]) / (C + stats["count"])

    return (
        stats.rename(columns={"count": "n_ratings", "mean": "mean_rating"})[
            ["movieId", "n_ratings", "mean_rating", "bayes_score"]
        ]
    )


# ENTRAÎNEMENT ITEM-BASED COLLABORATIVE FILTERING
# ------------------------------------------------------------
def train_item_based_cf(k_neighbors: int, min_ratings: int) -> None:

    # --------------------------------------------------------
    # INITIALISATION MLFLOW
    # --------------------------------------------------------
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = (
    f"train_itemcf_v2__k{k_neighbors}"
    f"__{datetime.now().strftime('%Y%m%d_%H%M')}"
)

    with mlflow.start_run(run_name=run_name):

        # ----------------------------------------------------
        # LOG PARAMÈTRES
        # ----------------------------------------------------
        mlflow.log_param("k_neighbors", k_neighbors)
        mlflow.log_param("min_ratings_per_movie", min_ratings)

        mlflow.set_tag("git_commit", get_git_commit())
        mlflow.set_tag("code_version", get_git_commit())
        mlflow.set_tag("data_version", get_dvc_md5())
        # Tag mis à jour pour refléter l'usage de la vue
        mlflow.set_tag("data_schema", f"{SCHEMA}.current_ratings(userId,movieId,rating)")
        mlflow.set_tag("pg_schema", SCHEMA)

        engine = create_engine(PG_URL)

        # --- 1) Chargement des données DEPUIS LA VUE SQL ---
        # On utilise current_ratings qui gère déjà le DISTINCT ON (userId, movieId)
        ratings = pd.read_sql(
            f"""
            SELECT "userId", "movieId", rating, "timestamp"
            FROM {SCHEMA}.current_ratings
            WHERE rating IS NOT NULL
            """,
            con=engine,
        )

        ratings["userId"] = ratings["userId"].astype(int)
        ratings["movieId"] = ratings["movieId"].astype(int)
        ratings["rating"] = ratings["rating"].astype(float)

        # LOG CONTEXTE DATA
        mlflow.set_tag("train_rows", int(len(ratings)))
        mlflow.set_tag("n_users", int(ratings["userId"].nunique()))
        mlflow.set_tag("n_movies_raw", int(ratings["movieId"].nunique()))

        # --- 2) Filtrage des films trop peu notés ---
        movie_counts = ratings["movieId"].value_counts()
        keep_movies = movie_counts[movie_counts >= min_ratings].index
        train_ratings = ratings[ratings["movieId"].isin(keep_movies)]

        mlflow.set_tag("n_movies_filtered", int(train_ratings["movieId"].nunique()))

        # --- 3) Construction matrice creuse ---
        user_ids = train_ratings["userId"].unique()
        movie_ids = train_ratings["movieId"].unique()

        user_to_idx = {u: i for i, u in enumerate(user_ids)}
        movie_to_idx = {m: j for j, m in enumerate(movie_ids)}

        rows = train_ratings["userId"].map(user_to_idx).to_numpy()
        cols = train_ratings["movieId"].map(movie_to_idx).to_numpy()
        vals = train_ratings["rating"].to_numpy()

        X_ui = csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(movie_ids)))
        X_iu = X_ui.T.tocsr()

        # --- 4) K plus proches voisins ---
        nn = NearestNeighbors(
            n_neighbors=k_neighbors + 1,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        nn.fit(X_iu)

        distances, indices = nn.kneighbors(X_iu)

        neighbors = []
        for i in range(len(movie_ids)):
            src_movie = int(movie_ids[i])
            for r in range(1, k_neighbors + 1):
                j = indices[i, r]
                sim = 1.0 - distances[i, r]
                neighbors.append((src_movie, int(movie_ids[j]), float(sim)))

        item_neighbors = pd.DataFrame(
            neighbors,
            columns=["movieId", "neighborMovieId", "similarity"],
        )

    
        # --- 5) Popularité globale ---
        movie_popularity = compute_bayesian_popularity(ratings)

        # ----------------------------------------------------
        # LOG ARTEFACTS
        # ----------------------------------------------------
        os.makedirs("mlflow_artifacts", exist_ok=True)

        neighbors_path = "mlflow_artifacts/item_neighbors.csv"
        popularity_path = "mlflow_artifacts/movie_popularity.csv"

        item_neighbors.to_csv(neighbors_path, index=False)
        movie_popularity.to_csv(popularity_path, index=False)

        mlflow.log_artifact(neighbors_path)
        mlflow.log_artifact(popularity_path)

        # --- 6) Sauvegarde en base ---
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.item_neighbors CASCADE"))
            conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.movie_popularity CASCADE"))

            item_neighbors.to_sql(
                "item_neighbors",
                con=conn,
                schema=SCHEMA,
                index=False,
                if_exists="replace",
            )

            movie_popularity.to_sql(
                "movie_popularity",
                con=conn,
                schema=SCHEMA,
                index=False,
                if_exists="replace",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-neighbors", type=int, default=50)
    parser.add_argument("--min-ratings", type=int, default=50)

    args = parser.parse_args()

    train_item_based_cf(
        k_neighbors=args.n_neighbors,
        min_ratings=args.min_ratings,
    )