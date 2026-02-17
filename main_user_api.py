# ============================================================
# MAIN_USER_API.PY
# ============================================================
import os
import sys
os.environ["PYTHONIOENCODING"] = "utf-8"

import subprocess
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine

# --- IMPORTS LOCAUX ---
try:
    from src.models.predict_model2 import predict as recommend_for_user
except ImportError:
    sys.exit(1)

from src.ingestion.ingestion_movielens import ingest_movielens

# --- CONFIG ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
PG_URL = os.getenv("PG_URL", "postgresql+psycopg2://movie:movie@127.0.0.1:5432/movie_reco")

# --- CACHE TITRES ---
def load_titles_from_sql():
    try:
        engine = create_engine(PG_URL)
        df = pd.read_sql('SELECT "movieId", "title" FROM raw.raw_movies', engine)
        if df.empty: return {}
        return dict(zip(df["movieId"], df["title"]))
    except: return {}

TITLE_MAP = load_titles_from_sql()
app = FastAPI(title="Movie Recommendation API")

@app.get("/")
def home():
    return {"status": "online", "cache": len(TITLE_MAP)}

# ------------------------------------------------------------
# 1. ROUTE DATA (Ordre : Download -> DVC)
# ------------------------------------------------------------
@app.post("/data")
def update_data_pipeline():
    report = {}
    
    # ÉTAPE A : INGESTION (Téléchargement + SQL)
    try:
        logger.info("🚀 Lancement Ingestion (Download + SQL)...")
        ingest_movielens() # Cela télécharge les fichiers frais dans data/raw
        report["ingestion"] = "Success (Downloaded & Inserted)"
    except Exception as e:
        logger.error(f"Ingestion Error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion Failed: {str(e)}")

    # ÉTAPE B : VERSIONNING (DVC)
    # Maintenant que data/raw est plein, on peut l'ajouter
    try:
        logger.info("📦 DVC Add & Push...")
        subprocess.run(["dvc", "add", "data/raw"], check=True)
        subprocess.run(["dvc", "push", "data/raw.dvc"], check=True)
        report["dvc"] = "Versioned & Pushed"
    except Exception as e:
        logger.error(f"DVC Error: {e}")
        # On ne bloque pas tout si DVC plante, car SQL est à jour
        report["dvc"] = f"Failed: {str(e)}"

    # Reload Cache
    global TITLE_MAP
    TITLE_MAP = load_titles_from_sql()
    
    return report

# ------------------------------------------------------------
# 2. ROUTE TRAINING (Parquet)
# ------------------------------------------------------------
@app.post("/training")
def training():
    python_exec = sys.executable 
    try:
        logger.info("📸 Création du snapshot Parquet...")
        subprocess.run([python_exec, "src/ingestion/create_snapshot.py"], check=True)
        
        logger.info("📦 DVC Add (Training Set)...")
        subprocess.run(["dvc", "add", "data/training_set.parquet"], check=True)
        subprocess.run(["dvc", "push", "data/training_set.parquet.dvc"], check=True)

        logger.info("🏃‍♂️ Entraînement...")
        subprocess.run([python_exec, "-m", "src.models.train_model2", "--n-neighbors", "20"], check=True)
        
        return {"status": "success", "message": "Training Complete"}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------------------------
# 3. ROUTE PREDICT
# ------------------------------------------------------------
@app.get("/recommend")
def recommend(user_id: int, top_n: int = 5):
    if not TITLE_MAP:
        raise HTTPException(status_code=503, detail="Cache vide. Lancez /data.")

    try:
        result = recommend_for_user(user_id=user_id, top_n=top_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    enriched = []
    for item in result.get("recommendations", []):
        mid = item.get("movieId")
        enriched.append({
            "movie_id": mid,
            "title": TITLE_MAP.get(mid, f"Unknown ({mid})"),
            "score": round(item.get("score", 0.0), 3)
        })

    return {"user_id": user_id, "recommendations": enriched}