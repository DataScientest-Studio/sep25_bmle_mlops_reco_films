# ==========================
# TRAIN_MODEL2_OPTIMIZED.PY
# ==========================
from __future__ import annotations

import os
os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

import sys
import shutil
import numpy as np
import pandas as pd
import subprocess
from datetime import datetime
import mlflow
import gc
import yaml  # Nécessaire pour lire le fichier .dvc
from sqlalchemy import create_engine, text
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import argparse
import mlflow.pyfunc
from collections import defaultdict
try:
    from src.models.mlflow_model import ItemCFPyFunc
except ImportError:
    sys.path.append(os.getcwd())
    from src.models.mlflow_model import ItemCFPyFunc

# ------------------------------------------------------------
# CONFIGURATION
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

# Fonction pour lire le hash DVC sans faire planter le script
def get_dvc_hash(dvc_path: str) -> str:
    try:
        with open(dvc_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            # On récupère le hash MD5
            return str(data["outs"][0].get("md5", "unknown"))
    except Exception:
        # Si fichier pas trouvé ou erreur, on renvoie "unknown" sans crasher
        return "unknown"

def compute_bayesian_popularity(ratings: pd.DataFrame) -> pd.DataFrame:
    stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"]).reset_index()
    C = stats["count"].mean()
    M = stats["mean"].mean()
    stats["bayes_score"] = ((C * M + stats["count"] * stats["mean"]) / (C + stats["count"])).astype("float32")
    
    result = stats.rename(columns={"count": "n_ratings", "mean": "mean_rating"})
    # Nettoyage types
    result["n_ratings"] = result["n_ratings"].astype("int32")
    result["mean_rating"] = result["mean_rating"].astype("float32")
    return result[["movieId", "n_ratings", "mean_rating", "bayes_score"]]

def recall_at_10(recos, truth_set):
    if not truth_set: return 0.0
    return len(set(recos[:10]) & truth_set) / len(truth_set)

def precision_at_10(recos, truth_set):
    if not recos:
        return 0.0
    return len(set(recos[:10]) & truth_set) / 10

def ndcg_at_10(recos, truth_set):
    if not truth_set: return 0.0
    dcg = 0.0
    for i, m in enumerate(recos[:10]):
        if m in truth_set: dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(truth_set), 10)))
    return dcg / idcg if idcg > 0 else 0.0

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


        # On suppose que le fichier s'appelle 'data/training_set.parquet.dvc'
        # S'il ne le trouve pas, il loggera "unknown" mais continuera l'entraînement.
        
        dvc_hash = get_dvc_hash("data/training_set.parquet.dvc")
        mlflow.set_tag("dvc_dataset_hash", dvc_hash)
        print(f"[INFO] DVC Hash logged: {dvc_hash}")
        # ------------------------------------------
        
        # 1. LOAD FROM PARQUET
        parquet_path = "data/training_set.parquet"
        print(f"[INFO] Loading data from {parquet_path}...")
        
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"❌ Le fichier {parquet_path} n'existe pas.")
            
        # Chargement sélectif des colonnes + typage immédiat pour économiser la RAM
        ratings = pd.read_parquet(
            parquet_path, 
            columns=["userId", "movieId", "rating"]
        )
        ratings["rating"] = ratings["rating"].astype("float32")
        ratings["userId"] = ratings["userId"].astype("int32")
        ratings["movieId"] = ratings["movieId"].astype("int32")
        
        print(f"[OK] Rows loaded: {len(ratings)}")

        # 2. POPULARITY & SPLIT
        print("[INFO] Computing popularity...")
        movie_popularity = compute_bayesian_popularity(ratings)
        
        # Filtrage
        counts = ratings["movieId"].value_counts()
        keep_movies = counts[counts >= min_ratings].index
        ratings = ratings[ratings["movieId"].isin(keep_movies)].copy()
        
        # Split
        train, test = train_test_split(ratings, test_size=0.2, random_state=42)
        
        # --- FIX MEMOIRE : On ne construit PAS l'historique global ici ---
        
        # Nettoyage immédiat
        del ratings, counts, keep_movies
        gc.collect()

        # 3. SPARSE MATRIX
        print("[INFO] Building Sparse Matrix...")
        
        # Utilisation de Categorical pour mapper ID -> Index
        train["user_idx"] = train["userId"].astype("category")
        train["movie_idx"] = train["movieId"].astype("category")
        
        # Récupération des mappings
        user_ids_map = train["user_idx"].cat.categories # Index -> Real UserID
        movie_ids_map = train["movie_idx"].cat.categories # Index -> Real MovieID
        
        # Création matrice
        rows = train["movie_idx"].cat.codes.values # Items en lignes
        cols = train["user_idx"].cat.codes.values  # Users en colonnes
        vals = train["rating"].values
        
        n_items = len(movie_ids_map)
        n_users = len(user_ids_map)
        
        X_iu = csr_matrix((vals, (rows, cols)), shape=(n_items, n_users), dtype=np.float32)
        
        # On garde 'train' pour l'évaluation plus tard, mais on supprime les colonnes inutiles
        train = train[["userId", "movieId"]] 
        del rows, cols, vals
        gc.collect()

        # 4. KNN TRAINING
        print("[INFO] Training KNN...")
        nn = NearestNeighbors(n_neighbors=k_neighbors+1, metric="cosine", algorithm="brute", n_jobs=-1)
        nn.fit(X_iu)

        # 5. GENERATING NEIGHBORS
        print("[INFO] Generating Neighbors (in batches to save RAM)...")
        batch_size = 2000
        dists_list, idxs_list = [], []

        # On boucle sur la matrice par paquets
        for i in range(0, X_iu.shape[0], batch_size):
            d_batch, i_batch = nn.kneighbors(X_iu[i : i + batch_size])
            dists_list.append(d_batch)
            idxs_list.append(i_batch)

        # On rassemble les résultats à la fin
        dists = np.vstack(dists_list)
        idxs = np.vstack(idxs_list)
        
        # Nettoyage: On n'a plus besoin du modèle KNN ni de la grosse matrice
        del nn, X_iu
        gc.collect()
        
        # -- Vectorisation de la création du DataFrame --
        neighbor_indices = idxs[:, 1:].flatten() 
        neighbor_dists = dists[:, 1:].flatten()
        
        source_indices = np.repeat(np.arange(n_items), k_neighbors)
        
        source_real_ids = movie_ids_map[source_indices]
        neighbor_real_ids = movie_ids_map[neighbor_indices]
        similarities = (1.0 - neighbor_dists).astype("float32")
        
        item_neighbors_df = pd.DataFrame({
            "movieId": source_real_ids,
            "neighborMovieId": neighbor_real_ids,
            "similarity": similarities
        })
        
        # Nettoyage tableaux numpy intermédiaires
        del dists, idxs, neighbor_indices, neighbor_dists, source_indices
        gc.collect()

        # 6. EVALUATION
        print("[INFO] Evaluating...")
        
        # Conversion DataFrame -> Dict pour accès rapide O(1)
        np_neighbors = item_neighbors_df.values 
        temp_dict = defaultdict(list)
        for row in np_neighbors:
             temp_dict[int(row[0])].append((int(row[1]), float(row[2])))
        neighbors_dict = dict(temp_dict)
        del temp_dict, np_neighbors
        
        # Sélection d'un échantillon
        sample_users = test["userId"].unique()[:1000]
        recalls, ndcgs, precisions = [], [], []
        
        # --- Construction de l'historique pour l'échantillon ---
        print(f"[INFO] Building partial history for {len(sample_users)} users...")
        
        # Filtre train pour ne garder que l'historique des users testés
        partial_train = train[train["userId"].isin(sample_users)]
        train_movies_per_user = partial_train.groupby("userId")["movieId"].apply(set).to_dict()
        
        # Filtre test aussi
        partial_test = test[test["userId"].isin(sample_users)]
        test_truth_dict = partial_test.groupby("userId")["movieId"].apply(set).to_dict()

        del train, test, partial_train, partial_test
        gc.collect()

        for u in sample_users:
            hist = train_movies_per_user.get(u, set())
            truth_set = test_truth_dict.get(u, set())
            
            if not hist or not truth_set: continue
            
            scores = {}
            for m in hist:
                for (rec, sim) in neighbors_dict.get(m, []):
                    if rec not in hist: # Pas déjà vu
                        scores[rec] = scores.get(rec, 0) + sim
            
            # Top 10
            best = [x[0] for x in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]]
            
            recalls.append(recall_at_10(best, truth_set))
            ndcgs.append(ndcg_at_10(best, truth_set))
            precisions.append(precision_at_10(best, truth_set))

        mlflow.log_metric("recall_10", np.mean(recalls) if recalls else 0)
        mlflow.log_metric("ndcg_10", np.mean(ndcgs) if ndcgs else 0)
        mlflow.log_metric("precision_10", np.mean(precisions) if precisions else 0)
        
        # Nettoyage avant sauvegarde finale
        del train_movies_per_user, test_truth_dict, neighbors_dict
        gc.collect()

        # 7. LOGGING ARTIFACTS
        os.makedirs("mlflow_artifacts", exist_ok=True)
        item_neighbors_df.to_parquet("mlflow_artifacts/item_neighbors.parquet", index=False)
        movie_popularity.to_parquet("mlflow_artifacts/movie_popularity.parquet", index=False)
        
        mlflow.log_artifact("mlflow_artifacts/item_neighbors.parquet")
        mlflow.log_artifact("mlflow_artifacts/movie_popularity.parquet")

        # 8. MODEL LOGGING
        print("[INFO] Logging PyFunc Model...")
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=ItemCFPyFunc(n_reco=10, min_user_ratings=5),
            artifacts={
                "item_neighbors": "mlflow_artifacts/item_neighbors.parquet",
                "movie_popularity": "mlflow_artifacts/movie_popularity.parquet",
            },
            registered_model_name=REGISTERED_MODEL_NAME
        )
    
        # 9. SAVE TO SQL
        print("[INFO] Saving to SQL...")
        engine = create_engine(PG_URL)
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.item_neighbors"))
            conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.movie_popularity"))
            item_neighbors_df.to_sql("item_neighbors", conn, schema=SCHEMA, index=False, if_exists="replace", method='multi', chunksize=1000)
            movie_popularity.to_sql("movie_popularity", conn, schema=SCHEMA, index=False, if_exists="replace", method='multi', chunksize=1000)
            
        print("[OK] Training Full Pipeline Success.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--min-ratings", type=int, default=50)
    args = parser.parse_args()
    train_item_based_cf(args.n_neighbors, args.min_ratings)