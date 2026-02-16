# ============================================================
# MAIN_USER_API.PY - VERSION API JSON (POUR STREAMLIT)
# ============================================================
import os
import sys
os.environ["PYTHONIOENCODING"] = "utf-8"
import subprocess
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine

# --- IMPORT DU PREDICT ---
try:
    from src.models.predict_model2 import predict as recommend_for_user
except ImportError:
    try:
        from src.models.predict_model2 import recommend_for_user
    except ImportError as e:
        print(f"❌ ERREUR CRITIQUE D'IMPORT : {e}")
        sys.exit(1)

from src.ingestion.ingestion_movielens import ingest_movielens

# ------------------------------------------------------------
# CONFIG LOGGING
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# CONFIG & CACHE TITRES
# ------------------------------------------------------------
PG_URL = os.getenv("PG_URL", "postgresql+psycopg2://movie:movie@127.0.0.1:5432/movie_reco")

def load_titles_from_sql():
    """Charge le mapping ID -> Titre en mémoire."""
    try:
        logger.info(f"🔌 Connexion à la BDD pour charger les titres...")
        engine = create_engine(PG_URL)
        query = "SELECT \"movieId\", \"title\" FROM raw.raw_movies"
        movies_df = pd.read_sql(query, engine)
        
        if movies_df.empty:
            return {}
        
        return dict(zip(movies_df["movieId"], movies_df["title"]))

    except Exception as e:
        logger.error(f"❌ Impossible de charger les titres : {e}")
        return {}

TITLE_MAP = load_titles_from_sql()
logger.info(f"✅ {len(TITLE_MAP)} films chargés en mémoire.")

app = FastAPI(title="Movie Recommendation API")

# ------------------------------------------------------------
# ROOT
# ------------------------------------------------------------
@app.get("/")
def home():
    """Simple health check."""
    return {
        "status": "online",
        "cache_size": len(TITLE_MAP),
        "message": "Bienvenue sur l'API de Recommandation. Utilisez /recommend pour les prédictions."
    }

# ------------------------------------------------------------
# 1. AUTOMATISATION DATA (/data)
# ------------------------------------------------------------
@app.post("/data")
def update_data_pipeline():
    report = {}
    try:
        logger.info("📡 DVC PULL...")
        subprocess.run(["dvc", "pull", "data/raw.dvc"], check=True, capture_output=True, text=True)
        report["dvc"] = "OK"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DVC Error: {str(e)}")

    try:
        logger.info("💾 Ingestion SQL...")
        ingest_movielens()
        report["ingestion"] = "OK"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL Error: {str(e)}")

    global TITLE_MAP
    TITLE_MAP = load_titles_from_sql()
    report["cache"] = f"Reloaded ({len(TITLE_MAP)} items)"
    
    return report

# ------------------------------------------------------------
# 2. TRAINING (/training)
# ------------------------------------------------------------
@app.post("/training")
def training():
    python_exec = sys.executable 
    try:
        # Snapshot
        subprocess.run([python_exec, "src/ingestion/create_snapshot.py"], check=True)
        # DVC Add
        subprocess.run(["dvc", "add", "data/training_set.csv"], check=True)
        # Train
        subprocess.run([python_exec, "-m", "src.models.train_model2", "--n-neighbors", "20"], check=True)
        
        return {"status": "success", "message": "Pipeline complet terminé."}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------------------------
# 3. PREDICTION (/recommend) -> RETOURNE DU JSON
# ------------------------------------------------------------
@app.get("/recommend")
def recommend(user_id: int, top_n: int = 5):
    """
    Retourne un JSON exploitable par Streamlit.
    Format:
    {
      "user_id": 123,
      "recommendations": [
         {"movie_id": 1, "title": "Toy Story", "score": 0.95},
         ...
      ]
    }
    """
    # Vérif cache
    if not TITLE_MAP:
        raise HTTPException(status_code=503, detail="Cache vide. Lancez /data d'abord.")

    # Appel Modèle
    try:
        result = recommend_for_user(user_id=user_id, top_n=top_n)
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    raw_recos = result.get("recommendations", [])
    
    # Enrichissement avec les Titres pour Streamlit
    enriched_recos = []
    for item in raw_recos:
        m_id = item.get("movieId")
        score = item.get("score", 0.0)
        
        enriched_recos.append({
            "movie_id": m_id,
            "title": TITLE_MAP.get(m_id, f"Unknown Movie ({m_id})"),
            "score": round(score, 3) # Arrondi pour être propre
        })

    return {
        "user_id": user_id,
        "count": len(enriched_recos),
        "recommendations": enriched_recos
    }