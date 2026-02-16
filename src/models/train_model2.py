# ============================================================
# TRAIN_MODEL2.PY - VERSION FINALE (ORDRE IMPORTS CORRIGÉ)
# ============================================================
from __future__ import annotations

import os
# FIX CRITIQUE : Force l'utilisation de distutils standard.
# Doit être placé juste après 'import os' mais AVANT 'import mlflow'
os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

import sys
import shutil
import numpy as np
import pandas as pd
import subprocess
from datetime import datetime
import yaml
import mlflow
import gc
from sqlalchemy import create_engine, text
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import argparse
import mlflow.pyfunc

# Import local sécurisé
try:
    from src.models.mlflow_model import ItemCFPyFunc
except ImportError:
    sys.path.append(os.getcwd())
    from src.models.mlflow_model import ItemCFPyFunc

# ------------------------------------------------------------
# CONFIGURATION GIT & CHEMINS
# ------------------------------------------------------------
def configure_git_environment():
    if shutil.which("git"): return
    possible_paths = [
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files\Git\bin\git.exe",
        r"C:\Users\{}\AppData\Local\Programs\Git\cmd\git.exe".format(os.getenv("USERNAME")),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = path
            break

configure_git_environment()

PG_URL = os.getenv("PG_URL", "postgresql+psycopg2://movie:movie@127.0.0.1:5432/movie_reco")
SCHEMA = os.getenv("PG_SCHEMA", "raw")
EXPERIMENT_NAME = "reco-films/itemcf-v2"
REGISTERED_MODEL_NAME = "reco-films-itemcf-v2"

# ------------------------------------------------------------
# UTILITAIRES
# ------------------------------------------------------------
def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except: return "unknown"

def compute_bayesian_popularity(ratings: pd.DataFrame) -> pd.DataFrame:
    stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"]).reset_index()
    C = stats["count"].mean()
    M = stats["mean"].mean()
    stats["bayes_score"] = (C * M + stats["count"] * stats["mean"]) / (C + stats["count"])
    return stats.rename(columns={"count": "n_ratings", "mean": "mean_rating"})[["movieId", "n_ratings", "mean_rating", "bayes_score"]]

def recall_at_10(recos, truth):
    if not truth: return 0
    return len(set(recos[:10]) & set(truth)) / len(set(truth))

def ndcg_at_10(recos, truth):
    if not truth: return 0
    dcg = 0.0
    truth_set = set(truth)
    for i, m in enumerate(recos[:10]):
        if m in truth_set: dcg += 1 / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(truth), 10)))
    return dcg / idcg if idcg > 0 else 0

# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------
def train_item_based_cf(k_neighbors: int, min_ratings: int) -> None:
    
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    print(f"[INFO] Connecting to MLflow at: {mlflow_uri}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = f"train_itemcf_v2__k{k_neighbors}__{datetime.now().strftime('%Y%m%d_%H%M')}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("k_neighbors", k_neighbors)
        mlflow.log_param("min_ratings", min_ratings)
        mlflow.set_tag("git_commit", get_git_commit())
        
        engine = create_engine(PG_URL)

        # 1. LOAD
        print("[INFO] Loading data...")
        query = f'SELECT "userId", "movieId", rating FROM {SCHEMA}.current_ratings WHERE rating IS NOT NULL'
        chunks = []
        for chunk in pd.read_sql(query, con=engine, chunksize=500_000):
            chunk["userId"] = chunk["userId"].astype("int32")
            chunk["movieId"] = chunk["movieId"].astype("int32")
            chunk["rating"] = chunk["rating"].astype("float32")
            chunks.append(chunk)
        ratings = pd.concat(chunks, ignore_index=True)
        print(f"[OK] Rows: {len(ratings)}")

        # 2. POPULARITY & SPLIT
        print("[INFO] Computing popularity...")
        movie_popularity = compute_bayesian_popularity(ratings)
        
        keep_movies = ratings["movieId"].value_counts()[lambda x: x >= min_ratings].index
        ratings_filtered = ratings[ratings["movieId"].isin(keep_movies)].copy()
        
        train, test = train_test_split(ratings_filtered, test_size=0.2, random_state=42)
        train_movies_per_user = train.groupby("userId")["movieId"].apply(set).to_dict()

        del ratings, chunks, ratings_filtered
        gc.collect()

        # 3. KNN
        print("[INFO] Training KNN...")
        user_ids = train["userId"].unique()
        movie_ids = train["movieId"].unique()
        
        u_map = {u: i for i, u in enumerate(user_ids)}
        m_map = {m: j for j, m in enumerate(movie_ids)}
        inv_m_map = {j: m for m, j in m_map.items()}

        rows = train["userId"].map(u_map).values
        cols = train["movieId"].map(m_map).values
        vals = train["rating"].values
        
        X_iu = csr_matrix((vals, (cols, rows)), shape=(len(movie_ids), len(user_ids)))
        
        nn = NearestNeighbors(n_neighbors=k_neighbors+1, metric="cosine", algorithm="brute", n_jobs=-1)
        nn.fit(X_iu)

        # 4. NEIGHBORS
        print("[INFO] Generating Neighbors...")
        dists, idxs = nn.kneighbors(X_iu)
        neighbors_list = []
        neighbors_dict = {}

        for i in range(len(movie_ids)):
            src_m = inv_m_map[i]
            m_neighbors = []
            for r in range(1, k_neighbors+1):
                nb_m = inv_m_map[idxs[i, r]]
                sim = 1.0 - dists[i, r]
                neighbors_list.append((src_m, nb_m, float(sim)))
                m_neighbors.append((nb_m, float(sim)))
            neighbors_dict[src_m] = m_neighbors
            
        item_neighbors_df = pd.DataFrame(neighbors_list, columns=["movieId", "neighborMovieId", "similarity"])

        # 5. EVALUATION
        print("[INFO] Evaluating...")
        sample_users = test["userId"].unique()[:1000]
        recalls, ndcgs = [], []
        
        for u in sample_users:
            hist = train_movies_per_user.get(u, set())
            truth = test[test["userId"]==u]["movieId"].tolist()
            if not hist or not truth: continue
            
            scores = {}
            for m in hist:
                for (rec, sim) in neighbors_dict.get(m, []):
                    if rec not in hist: scores[rec] = scores.get(rec, 0) + sim
            
            best = [x[0] for x in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]]
            recalls.append(recall_at_10(best, truth))
            ndcgs.append(ndcg_at_10(best, truth))

        mlflow.log_metric("recall_10", np.mean(recalls) if recalls else 0)
        mlflow.log_metric("ndcg_10", np.mean(ndcgs) if ndcgs else 0)

        # 6. LOGGING
        os.makedirs("mlflow_artifacts", exist_ok=True)
        item_neighbors_df.to_csv("mlflow_artifacts/item_neighbors.csv", index=False)
        movie_popularity.to_csv("mlflow_artifacts/movie_popularity.csv", index=False)
        
        mlflow.log_artifact("mlflow_artifacts/item_neighbors.csv")
        mlflow.log_artifact("mlflow_artifacts/movie_popularity.csv")

        # 7. MODEL LOGGING (SANS CODE_PATH pour éviter l'erreur)
        print("[INFO] Logging PyFunc Model...")
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=ItemCFPyFunc(n_reco=10, min_user_ratings=5),
            artifacts={
                "item_neighbors": "mlflow_artifacts/item_neighbors.csv",
                "movie_popularity": "mlflow_artifacts/movie_popularity.csv",
            },
            # PAS DE code_path ICI
            registered_model_name=REGISTERED_MODEL_NAME
        )
        
        # 8. ALIAS PRODUCTION
        print("[INFO] Setting Alias 'production'...")
        client = mlflow.tracking.MlflowClient()
        # On attend un peu pour être sûr que le modèle est bien créé
        latest_versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["None"])
        if latest_versions:
            latest_version = latest_versions[0].version
            client.set_registered_model_alias(REGISTERED_MODEL_NAME, "production", latest_version)
            print(f"[SUCCESS] Model v{latest_version} aliased as 'production'")

        # 9. SAVE TO SQL
        print("[INFO] Saving to SQL for Production API...")
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.item_neighbors"))
            conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.movie_popularity"))
            item_neighbors_df.to_sql("item_neighbors", conn, schema=SCHEMA, index=False, if_exists="replace", chunksize=10000)
            movie_popularity.to_sql("movie_popularity", conn, schema=SCHEMA, index=False, if_exists="replace")
            
        print("[OK] Training Full Pipeline Success.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--min-ratings", type=int, default=50)
    args = parser.parse_args()
    train_item_based_cf(args.n_neighbors, args.min_ratings)