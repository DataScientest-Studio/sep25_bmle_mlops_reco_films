from pathlib import Path
import time
import pickle
import argparse

import pandas as pd
from sklearn.neighbors import NearestNeighbors

import mlflow
from src.models.evaluation import evaluate_precision_at_k


# -----------------------------
# MLflow config (simple, local)
# -----------------------------
MLFLOW_TRACKING_URI = "sqlite:///data/db/mlflow.sqlite"
MLFLOW_EXPERIMENT_NAME = "reco-films-knn"

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"

MAX_USERS = 10  # volontairement faible pour l'offline eval


def train_model(movie_matrix: pd.DataFrame, n_neighbors: int, algorithm: str, metric: str):
    if "movieId" not in movie_matrix.columns:
        raise ValueError("movie_matrix doit contenir une colonne movieId")

    X = movie_matrix.drop("movieId", axis=1)

    model = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        metric=metric,
    )
    model.fit(X)
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--algorithm", type=str, default="brute")
    parser.add_argument("--metric", type=str, default="cosine")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # -----------------------------
    # Load data
    # -----------------------------
    movie_matrix = pd.read_csv(DATA_DIR / "movie_matrix.csv")
    ratings = pd.read_csv(ROOT / "data" / "raw" / "ratings.csv")

    ratings = ratings.sample(frac=1.0, random_state=42)
    split = int(0.8 * len(ratings))
    ratings_train = ratings.iloc[:split]
    ratings_test = ratings.iloc[split:]

    # -----------------------------
    # MLflow init
    # -----------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    run_name = f"knn_k{args.n_neighbors}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("n_neighbors", args.n_neighbors)
        mlflow.log_param("algorithm", args.algorithm)
        mlflow.log_param("metric", args.metric)
        mlflow.log_param("eval_k", 10)
        mlflow.log_param("max_users", MAX_USERS)

        # -----------------------------
        # Train
        # -----------------------------
        t0 = time.time()
        model = train_model(
            movie_matrix,
            n_neighbors=args.n_neighbors,
            algorithm=args.algorithm,
            metric=args.metric,
        )
        mlflow.log_metric("fit_time_sec", time.time() - t0)

        # -----------------------------
        # Evaluation
        # -----------------------------
        precision_10 = evaluate_precision_at_k(
            model=model,
            ratings_train=ratings_train,
            ratings_test=ratings_test,
            movie_matrix=movie_matrix,
            k_reco=10,
            max_users=MAX_USERS,
        )
        mlflow.log_metric("precision_at_10", precision_10)

        # -----------------------------
        # Save model
        # -----------------------------
        MODEL_DIR.mkdir(exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")

    print("✅ Training terminé")