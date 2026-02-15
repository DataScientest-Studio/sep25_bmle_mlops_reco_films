from __future__ import annotations

import os
import argparse
import json
import pandas as pd
import mlflow
from sqlalchemy import create_engine
from mlflow.tracking import MlflowClient

# Même nom que dans train_model2.py
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "reco-films-itemcf-v2")

PG_URL = os.getenv(
    "PG_URL",
    "postgresql+psycopg2://movie:movie@127.0.0.1:5432/movie_reco",
)
SCHEMA = os.getenv("PG_SCHEMA", "raw")


def load_production_model():
    # MLflow tracking server (ton container mlflow_server)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    # Tu as créé l'alias "production"
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@production"
    return mlflow.pyfunc.load_model(model_uri)


def fetch_user_ratings(engine, user_id: int) -> pd.DataFrame:
    # On récupère l'historique du user (ce que le PyFunc attend)
    query = f"""
        SELECT "movieId", rating
        FROM {SCHEMA}.raw_ratings
        WHERE "userId" = %(user_id)s
          AND rating IS NOT NULL
        ORDER BY "timestamp" ASC
    """
    df = pd.read_sql(query, con=engine, params={"user_id": user_id})
    if df.empty:
        return df
    df["movieId"] = df["movieId"].astype(int)
    df["rating"] = df["rating"].astype(float)
    return df[["movieId", "rating"]]


def predict(user_id: int, top_n: int = 10) -> dict:
    # 1) Charge le modèle production
    model = load_production_model()
    client = MlflowClient()
    alias_info = client.get_model_version_by_alias(
        REGISTERED_MODEL_NAME,
        "production"
    )

    print(f"[INFO] Chargement version production : {alias_info.version}")

    # 2) Lit les ratings user
    engine = create_engine(PG_URL)
    user_ratings = fetch_user_ratings(engine, user_id)

    # 3) Appelle le PyFunc
    # Le PyFunc renvoie un DataFrame avec colonnes: movieId, score
    reco_df = model.predict(user_ratings)

    # 4) Top-N côté client (au cas où le modèle renvoie déjà 10)
    if isinstance(reco_df, pd.DataFrame) and "movieId" in reco_df.columns:
        recos = reco_df["movieId"].astype(int).tolist()
    else:
        recos = []

    # 🔹 Films déjà vus par l'utilisateur
    seen_movies = set(user_ratings["movieId"].tolist())

    # 🔹 On enlève les films déjà notés
    recos = [m for m in recos if m not in seen_movies]

    # 🔹 On applique le top_n
    recos = recos[:top_n]


    return {"userId": user_id, "recommendations": recos}


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