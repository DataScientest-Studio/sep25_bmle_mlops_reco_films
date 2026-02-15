# ============================================================
# TRAIN_MODEL2.PY
# ============================================================
from __future__ import annotations

import os
import sys
import shutil
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
# 0. GIT FIX FOR WINDOWS & MLFLOW
# ------------------------------------------------------------
def configure_git_environment():
    # If git is already in PATH, do nothing
    if shutil.which("git"):
        return

    # Common installation paths for Git on Windows
    possible_paths = [
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files\Git\bin\git.exe",
        r"C:\Users\{}\AppData\Local\Programs\Git\cmd\git.exe".format(os.getenv("USERNAME")),
    ]
    
    found_git = False
    for path in possible_paths:
        if os.path.exists(path):
            os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = path
            found_git = True
            print(f"[OK] Git executable found and set: {path}")
            break
            
    if not found_git:
        os.environ["GIT_PYTHON_REFRESH"] = "quiet"
        print("[WARNING] Git not found in standard paths. MLflow git tracking might be disabled.")

configure_git_environment()


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
# URL Database (Application)
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
    # FORCE CONNECTION TO DOCKER MLFLOW SERVER
    # Use environment variable if set, otherwise default to localhost:5000
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    
    print(f"[INFO] Connecting to MLflow at: {mlflow_uri}")
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
        mlflow.set_tag("data_schema", f"{SCHEMA}.current_ratings(userId,movieId,rating)")
        mlflow.set_tag("pg_schema", SCHEMA)

        engine = create_engine(PG_URL)

        # --- 1) Chargement des données OPTIMISÉ (Chunking) ---
        print("[INFO] Chargement des donnees depuis SQL (mode chunked)...")
        
        query = f"""
            SELECT "userId", "movieId", rating
            FROM {SCHEMA}.current_ratings
            WHERE rating IS NOT NULL
        """
        
        chunk_iterator = pd.read_sql(
            query,
            con=engine,
            chunksize=1_000_000
        )
        
        chunks = []
        for i, chunk in enumerate(chunk_iterator):
            # Optimisation immédiate des types (int32/float32)
            chunk["userId"] = chunk["userId"].astype("int32")
            chunk["movieId"] = chunk["movieId"].astype("int32")
            chunk["rating"] = chunk["rating"].astype("float32")
            chunks.append(chunk)
            print(f"   ... chunk {i+1} charge")

        ratings = pd.concat(chunks, ignore_index=True)
        print(f"[OK] Chargement termine. Shape: {ratings.shape}")

        # LOG CONTEXTE DATA
        mlflow.set_tag("train_rows", int(len(ratings)))
        mlflow.set_tag("n_users", int(ratings["userId"].nunique()))
        mlflow.set_tag("n_movies_raw", int(ratings["movieId"].nunique()))

        # --- 2) Filtrage des films trop peu notés ---
        print("[INFO] Filtrage des films...")
        movie_counts = ratings["movieId"].value_counts()
        keep_movies = movie_counts[movie_counts >= min_ratings].index
        
        train_ratings = ratings[ratings["movieId"].isin(keep_movies)].copy()
        mlflow.set_tag("n_movies_filtered", int(train_ratings["movieId"].nunique()))

        # --- 3) Construction matrice creuse ---
        print("[INFO] Construction de la matrice creuse...")
        user_ids = train_ratings["userId"].unique()
        movie_ids = train_ratings["movieId"].unique()

        user_to_idx = {u: i for i, u in enumerate(user_ids)}
        movie_to_idx = {m: j for j, m in enumerate(movie_ids)}

        rows = train_ratings["userId"].map(user_to_idx).to_numpy()
        cols = train_ratings["movieId"].map(movie_to_idx).to_numpy()
        vals = train_ratings["rating"].to_numpy()

        # Nettoyage RAM
        del ratings
        del chunks
        import gc
        gc.collect()

        X_ui = csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(movie_ids)))
        X_iu = X_ui.T.tocsr()

        # --- 4) K plus proches voisins ---
        print("[INFO] Entrainement NearestNeighbors...")
        nn = NearestNeighbors(
            n_neighbors=k_neighbors + 1,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        nn.fit(X_iu)

        distances, indices = nn.kneighbors(X_iu)

        neighbors = []
        print("[INFO] Calcul des similarites...")
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
        movie_popularity = compute_bayesian_popularity(train_ratings)

        # ----------------------------------------------------
        # LOG ARTEFACTS
        # ----------------------------------------------------
        # Artifacts will be uploaded via HTTP to the Docker server
        os.makedirs("mlflow_artifacts", exist_ok=True)

        neighbors_path = "mlflow_artifacts/item_neighbors.csv"
        popularity_path = "mlflow_artifacts/movie_popularity.csv"

        item_neighbors.to_csv(neighbors_path, index=False)
        movie_popularity.to_csv(popularity_path, index=False)

        print(f"[INFO] Uploading artifacts to {mlflow_uri}...")
        mlflow.log_artifact(neighbors_path)
        mlflow.log_artifact(popularity_path)

        # --- 6) Sauvegarde en base ---
        print("[INFO] Sauvegarde en base de donnees...")
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.item_neighbors CASCADE"))
            conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.movie_popularity CASCADE"))

            chunk_size_sql = 100000
            item_neighbors.to_sql(
                "item_neighbors",
                con=conn,
                schema=SCHEMA,
                index=False,
                if_exists="replace",
                chunksize=chunk_size_sql
            )

            movie_popularity.to_sql(
                "movie_popularity",
                con=conn,
                schema=SCHEMA,
                index=False,
                if_exists="replace",
            )
        print("[OK] Termine avec succes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-neighbors", type=int, default=50)
    parser.add_argument("--min-ratings", type=int, default=50)

    args = parser.parse_args()

    train_item_based_cf(
        k_neighbors=args.n_neighbors,
        min_ratings=args.min_ratings,
    )