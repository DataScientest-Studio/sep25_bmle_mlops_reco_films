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


def load_production_model():
    """Charge le modèle MLflow marqué par l'alias @production."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@production"
    print(f"[INFO] Chargement du modèle depuis : {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)


def fetch_user_ratings(engine, user_id: int) -> pd.DataFrame:
    """Récupère l'historique des notes de l'utilisateur."""
    query = f"""
        SELECT "movieId", rating
        FROM {SCHEMA}.current_ratings
        WHERE "userId" = %(user_id)s
          AND rating IS NOT NULL
    """
    df = pd.read_sql(query, con=engine, params={"user_id": user_id})
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
    # 1. Charger modèle
    try:
        model = load_production_model()
    except Exception as e:
        print(f"[ERREUR] Impossible de charger le modèle : {e}")
        return {"userId": user_id, "recommendations": [], "error": str(e)}

    # 2. Charger historique user
    engine = create_engine(PG_URL)
    user_ratings = fetch_user_ratings(engine, user_id)

    if user_ratings.empty:
        # Pas d'historique = pas de reco personnalisée (ou fallback populaire à implémenter)
        return {"userId": user_id, "recommendations": []}

    # 3. Prédiction
    reco_df = model.predict(user_ratings)

    # 4. Nettoyage & Formatage
    if not isinstance(reco_df, pd.DataFrame) or "movieId" not in reco_df.columns:
        return {"userId": user_id, "recommendations": []}

    # Typage
    reco_df["movieId"] = reco_df["movieId"].astype(int)
    reco_df["score"] = reco_df["score"].astype(float) if "score" in reco_df.columns else 0.0

    # Filtrer films déjà vus
    seen_movies = set(user_ratings["movieId"].tolist())
    reco_df = reco_df[~reco_df["movieId"].isin(seen_movies)]

    # Trier et couper
    reco_df = reco_df.sort_values("score", ascending=False).head(top_n)

    # Conversion en dictionnaire pur pour l'API
    recos = reco_df[["movieId", "score"]].to_dict(orient="records")

    return {"userId": user_id, "recommendations": recos}


def get_production_model_metadata():
    """
    Récupère les métadonnées et la config du modèle actuellement en production via MLflow.
    Utilisé par l'API pour l'observabilité.
    """
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()
    
    try:
        # 1. Récupérer la version du modèle avec l'alias 'production'
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "production")
        
        # 2. Récupérer les infos du Run associé
        run = client.get_run(mv.run_id)
        
        # 3. Extraction de la Config (Hyperparamètres)
        config = {
            "k_neighbors": run.data.params.get("k_neighbors"),
            "min_ratings": run.data.params.get("min_ratings"),
            "distance_metric": "cosine", 
            "algorithm": "brute"
        }

        # 4. Extraction des Métadonnées (Infos du run, métriques, version)
        metadata = {
            "model_name": REGISTERED_MODEL_NAME,
            "model_version": mv.version,
            "run_id": mv.run_id,
            "creation_timestamp": pd.to_datetime(mv.creation_timestamp, unit="ms").isoformat(),
            "git_commit": run.data.tags.get("git_commit", "unknown"),
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