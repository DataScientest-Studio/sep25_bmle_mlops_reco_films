from pathlib import Path
import time
import json
import pickle
import argparse

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from src.models.evaluation import evaluate_precision_at_k

import mlflow

from src.config.mlflow_config import (
    get_mlflow_tracking_uri,
    get_mlflow_artifact_root,
    get_experiment_name,
)

# Racine du projet
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"

MAX_USERS = 10 # nombre max d'utilisateurs pour l'évaluation offline

def train_model(movie_matrix: pd.DataFrame, n_neighbors: int, algorithm: str, metric: str) -> NearestNeighbors:
    """
    Entraîne un modèle kNN item-item à partir de movie_matrix.
    """
    if "movieId" not in movie_matrix.columns:
        raise ValueError("movie_matrix doit contenir une colonne 'movieId'")

    X = movie_matrix.drop("movieId", axis=1)

    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        metric=metric
    )
    nbrs.fit(X)
    return nbrs


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    return pd.read_csv(path)


def save_model(model, path: Path):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def compute_sparsity(X: pd.DataFrame) -> float:
    # sparsity = % de zéros
    total = X.size
    zeros = (X.values == 0).sum()
    return zeros / total


def avg_intra_neighbors_similarity(X: pd.DataFrame, model: NearestNeighbors, sample_size: int = 200) -> float:
    """
    Proxy "qualité": similarité cosinus moyenne entre chaque item et ses voisins.
    """
    if X.shape[0] == 0:
        return 0.0

    n = X.shape[0]
    idx = list(range(min(sample_size, n)))

    X_sample = X.iloc[idx]  # DataFrame, PAS numpy

    distances, _ = model.kneighbors(
        X_sample,
        n_neighbors=model.n_neighbors,
    )

    similarities = 1.0 - distances
    return float(similarities.mean())

def parse_args():
    parser = argparse.ArgumentParser(description="Train kNN item-item recommender")
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--algorithm", type=str, default="brute")
    parser.add_argument("--metric", type=str, default="cosine")
    return parser.parse_args()

if __name__ == "__main__":
    # -----------------------------
    # Hyperparams (à rendre CLI plus tard)
    # -----------------------------

    args = parse_args()
    n_neighbors = args.n_neighbors
    algorithm = args.algorithm
    metric = args.metric

    movie_matrix_path = DATA_DIR / "movie_matrix.csv"
    movie_matrix = load_features(movie_matrix_path)

    # Features matrix
    X = movie_matrix.drop("movieId", axis=1)

    # -----------------------------
    # MLflow setup
    # -----------------------------
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())

    # Important : artifact root géré surtout par tracking server,
    # mais on le garde en env/config pour cohérence.
    experiment_name = get_experiment_name()
    mlflow.set_experiment(experiment_name)

    run_tags = {
        "project": "reco-films",
        "model_family": "knn-item-item",
        "data_source": "movielens",
    }

    RATINGS_PATH = ROOT / "data" / "raw" / "ratings.csv"
    ratings = pd.read_csv(RATINGS_PATH)

    # split simple user-wise
    ratings = ratings.sample(frac=1.0, random_state=42)
    train_size = int(0.8 * len(ratings))
    ratings_train = ratings.iloc[:train_size]
    ratings_test = ratings.iloc[train_size:]

    run_name = f"knn_k{n_neighbors}" # donne un nom lisible à la run dans MLflow UI

    with mlflow.start_run(run_name=run_name,tags=run_tags) as run:
        # ------------------------------------------------------------------
        # 1. PARAMÈTRES (contexte expérimental)
        # ------------------------------------------------------------------
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_param("algorithm", algorithm)
        mlflow.log_param("metric", metric)
        mlflow.log_param("features_path", str(movie_matrix_path))

        # Paramètres d’évaluation
        mlflow.log_param("eval_k", 10)
        mlflow.log_param("rating_threshold", 4.0)
        mlflow.log_param("eval_max_users", MAX_USERS)

        # Dataset stats (plutôt params que metrics car constantes)
        mlflow.log_param("n_movies", int(movie_matrix.shape[0]))
        mlflow.log_param("n_features", int(X.shape[1]))

        # ------------------------------------------------------------------
        # 2. ENTRAÎNEMENT DU MODÈLE
        # ------------------------------------------------------------------
        t0 = time.time()
        model = train_model(
            movie_matrix,
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            metric=metric,
        )
        fit_time = time.time() - t0
        mlflow.log_metric("fit_time_sec", fit_time)

        # ------------------------------------------------------------------
        # 3. MÉTRIQUES TECHNIQUES (diagnostic modèle)
        # ------------------------------------------------------------------
        sparsity = compute_sparsity(X)
        mlflow.log_metric("sparsity", float(sparsity))

        mean_nonzero = float((X.values != 0).sum(axis=1).mean())
        mlflow.log_metric("mean_nonzero_per_item", mean_nonzero)

        avg_sim = avg_intra_neighbors_similarity(X, model, sample_size=200)
        mlflow.log_metric("avg_intra_neighbors_similarity", avg_sim)
        

        # ------------------------------------------------------------------
        # 4. MÉTRIQUE MÉTIER (recommandation)
        # ------------------------------------------------------------------
        precision_10 = evaluate_precision_at_k(
            model=model,
            ratings_train=ratings_train,
            ratings_test=ratings_test,
            movie_matrix=movie_matrix,
            k_reco=10,
            max_users=MAX_USERS,
        )
        mlflow.log_metric("precision_at_10", precision_10)

        # ------------------------------------------------------------------
        # 5. ARTEFACTS (en dernier)
        # ------------------------------------------------------------------
        save_model(model, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")

        run_info = {
            "run_id": run.info.run_id,
            "experiment_name": experiment_name,
            "model_path": str(MODEL_PATH),
            "movie_matrix_path": str(movie_matrix_path),
        }

        tmp = ROOT / "reports" / "run_info.json"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(run_info, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(tmp), artifact_path="metadata")

    print("✅ Training terminé + MLflow Tracking OK")
    print(f"Modèle sauvegardé dans : {MODEL_PATH}")