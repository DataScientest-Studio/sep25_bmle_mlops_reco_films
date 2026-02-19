# ============================================================
# SRC/MODELS/PREDICT_MODEL2.PY
# ============================================================
from __future__ import annotations

import os
import argparse
import json
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from sqlalchemy import create_engine

# Nom du modèle dans MLflow
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "reco-films-itemcf-v2")

# Config BDD
PG_URL = os.getenv(
    "PG_URL",
    "postgresql+psycopg2://movie:movie@127.0.0.1:5432/movie_reco",
)
SCHEMA = os.getenv("PG_SCHEMA", "raw")

# ── Singletons (initialisés une seule fois) ──────────────────
_engine = None
_model_cache = None


def get_engine():
    """Retourne un engine SQLAlchemy partagé (créé une seule fois)."""
    global _engine
    if _engine is None:
        _engine = create_engine(PG_URL, pool_size=5, max_overflow=10)
    return _engine


def load_production_model():
    """Charge le modèle MLflow @production une seule fois, puis le met en cache."""
    global _model_cache
    if _model_cache is None:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        model_uri = f"models:/{REGISTERED_MODEL_NAME}@production"
        print(f"[INFO] Chargement du modèle depuis : {model_uri}")
        _model_cache = mlflow.pyfunc.load_model(model_uri)
    return _model_cache


def fetch_user_ratings(user_id: int) -> pd.DataFrame:
    """Récupère l'historique des notes de l'utilisateur."""
    query = f"""
        SELECT "movieId", rating
        FROM {SCHEMA}.current_ratings
        WHERE "userId" = %(user_id)s
          AND rating IS NOT NULL
    """
    df = pd.read_sql(query, con=get_engine(), params={"user_id": user_id})
    if df.empty:
        return pd.DataFrame(columns=["movieId", "rating"])

    df["movieId"] = df["movieId"].astype(int)
    df["rating"] = df["rating"].astype(float)
    return df[["movieId", "rating"]]


def predict(user_id: int, top_n: int = 10) -> dict:
    """
    Génère les recommandations.
    Retourne : { "userId": int, "recommendations": [{'movieId': int, 'score': float}, ...] }
    """
    # 1. Charger modèle (depuis le cache après le 1er appel)
    try:
        model = load_production_model()
    except Exception as e:
        print(f"[ERREUR] Impossible de charger le modèle : {e}")
        return {"userId": user_id, "recommendations": [], "error": str(e)}

    # 2. Charger historique user (engine partagé)
    user_ratings = fetch_user_ratings(user_id)

    if user_ratings.empty:
        return {"userId": user_id, "recommendations": []}

    # 3. Prédiction
    reco_df = model.predict(user_ratings)

    # 4. Nettoyage & Formatage
    if not isinstance(reco_df, pd.DataFrame) or "movieId" not in reco_df.columns:
        return {"userId": user_id, "recommendations": []}

    reco_df["movieId"] = reco_df["movieId"].astype(int)
    reco_df["score"] = reco_df["score"].astype(float) if "score" in reco_df.columns else 0.0

    # Filtrer films déjà vus
    seen_movies = set(user_ratings["movieId"].tolist())
    reco_df = reco_df[~reco_df["movieId"].isin(seen_movies)]

    # Trier et couper
    reco_df = reco_df.sort_values("score", ascending=False).head(top_n)

    recos = reco_df[["movieId", "score"]].to_dict(orient="records")
    return {"userId": user_id, "recommendations": recos}


def get_production_model_metadata():
    """
    Récupère les métadonnées et la config du modèle actuellement en production via MLflow.
    """
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()

    try:
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "production")
        run = client.get_run(mv.run_id)

        config = {
            "k_neighbors": run.data.params.get("k_neighbors"),
            "min_ratings": run.data.params.get("min_ratings"),
            "distance_metric": "cosine",
            "algorithm": "brute"
        }

        metadata = {
            "model_name": REGISTERED_MODEL_NAME,
            "model_version": mv.version,
            "run_id": mv.run_id,
            "creation_timestamp": pd.to_datetime(mv.creation_timestamp, unit="ms").isoformat(),
            "git_commit": run.data.tags.get("git_commit", "unknown"),
            "dvc_dataset_hash": run.data.tags.get("dvc_dataset_hash", "unknown"),
            "tags": dict(run.data.tags),
            "metrics": {
                "recall_at_10": round(float(run.data.metrics.get("recall_10", 0)), 4),
                "ndcg_at_10": round(float(run.data.metrics.get("ndcg_10", 0)), 4)
            }
        }

        return {"config": config, "metadata": metadata}

    except Exception as e:
        print(f"[WARNING] Impossible de récupérer les métadonnées MLflow: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    out = predict(user_id=args.user_id, top_n=args.top_n)

    if args.json:
        print(json.dumps(out))
    else:
        print(out)