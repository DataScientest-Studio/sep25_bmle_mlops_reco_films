# =============================================================================
# MAIN_USER_API.PY - VERSION PRODUCTION (ETENDUE)
# =============================================================================
import os
import sys
import subprocess
import logging
import pandas as pd
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text
from prometheus_fastapi_instrumentator import Instrumentator

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.append("/app")
sys.path.append(os.path.join(os.getcwd(), "src"))

# --- CONFIGURATION MLFLOW ---
if "MLFLOW_TRACKING_URI" not in os.environ:
    os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"

# --- IMPORTS LOCAUX ---
try:
    from src.models.predict_model2 import predict as recommend_for_user, get_production_model_metadata, load_production_model
    from src.ingestion.ingestion_movielens import ingest_movielens
    from src.ingestion.init_db import init_database
    print("✅ Imports src réussis")
except ImportError:
    from models.predict_model2 import predict as recommend_for_user, get_production_model_metadata, load_production_model
    from ingestion.ingestion_movielens import ingest_movielens
    from init_db import init_database

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PG_URL = os.getenv("PG_URL", "postgresql+psycopg2://movie:movie@postgres:5432/movie_reco")

# --- ENGINE SINGLETON ---
_engine = create_engine(PG_URL, pool_size=5, max_overflow=10)

def get_engine():
    return _engine


def load_titles_from_sql():
    try:
        with get_engine().connect() as conn:
            exists = conn.execute(text(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'raw' AND table_name = 'raw_movies')"
            )).scalar()
        if not exists:
            return {}
        df = pd.read_sql('SELECT "movieId", "title" FROM raw.raw_movies', get_engine())
        return dict(zip(df["movieId"], df["title"]))
    except Exception:
        return {}


TITLE_MAP = load_titles_from_sql()


# --- LIFESPAN : warm-up du modèle au démarrage ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Chargement du modèle en mémoire...")
    try:
        load_production_model()
        logger.info("✅ Modèle chargé et mis en cache")
    except Exception as e:
        logger.warning(f"⚠️ Modèle non disponible au démarrage : {e}")
    yield


app = FastAPI(title="Movie Recommendation API", lifespan=lifespan)

# -----------------------------------------------------------------------------
# 1. ROUTES DE BASE & SANTÉ
# -----------------------------------------------------------------------------

@app.get("/")
def home():
    return {"status": "online", "cache_size": len(TITLE_MAP)}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/ready")
def readiness_check():
    status = {"database": "unknown", "model": "unknown"}
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        status["database"] = "connected"
    except Exception:
        status["database"] = "error"
    try:
        meta = get_production_model_metadata()
        status["model"] = "ready" if meta else "not_found"
    except Exception:
        status["model"] = "error"
    return {"status": "ready" if status["database"] == "connected" else "not_ready", "checks": status}


Instrumentator().instrument(app).expose(app)

# -----------------------------------------------------------------------------
# 2. CORE BUSINESS (RECOMMANDATIONS)
# -----------------------------------------------------------------------------

@app.get("/recommend")
def recommend(user_id: int, top_n: int = 5):
    if not TITLE_MAP:
        raise HTTPException(status_code=503, detail="Cache vide")
    try:
        result = recommend_for_user(user_id=user_id, top_n=top_n)
        enriched = [
            {"movie_id": item["movieId"], "title": TITLE_MAP.get(item["movieId"], "Unknown"), "score": round(item["score"], 3)}
            for item in result.get("recommendations", [])
        ]
        return {"user_id": user_id, "recommendations": enriched}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# 3. CATALOGUE & DÉTAILS
# -----------------------------------------------------------------------------

@app.get("/movies/popular")
def get_popular_movies(limit: int = 10):
    try:
        query = f'SELECT "movieId", "mean_rating", "n_ratings", "bayes_score" FROM raw.movie_popularity ORDER BY "bayes_score" DESC LIMIT {limit}'
        df = pd.read_sql(query, get_engine())
        results = []
        for _, row in df.iterrows():
            mid = int(row["movieId"])
            results.append({
                "movie_id": mid,
                "title": TITLE_MAP.get(mid, f"Unknown ({mid})"),
                "stats": {"score": round(row["bayes_score"], 3), "mean_rating": round(row["mean_rating"], 2), "count": int(row["n_ratings"])}
            })
        return results
    except Exception as e:
        logger.error(f"Erreur films populaires: {e}")
        raise HTTPException(status_code=500, detail="Erreur récupération films populaires.")


@app.get("/movies/{movie_id}")
def get_movie_details(movie_id: int):
    try:
        query = text(
            'SELECT m."movieId", m.title, m.genres, p.mean_rating, p.n_ratings, p.bayes_score '
            'FROM raw.raw_movies m LEFT JOIN raw.movie_popularity p ON m."movieId" = p."movieId" '
            'WHERE m."movieId" = :mid'
        )
        with get_engine().connect() as conn:
            result = conn.execute(query, {"mid": movie_id}).mappings().one_or_none()
        if not result:
            raise HTTPException(status_code=404, detail="Film introuvable.")
        return {
            "movie_id": result["movieId"], "title": result["title"], "genres": result["genres"],
            "stats": {"score": round(result["bayes_score"] or 0, 3), "mean_rating": round(result["mean_rating"] or 0, 2), "count": int(result["n_ratings"] or 0)}
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur détails film: {e}")
        raise HTTPException(status_code=500, detail="Erreur détails film.")

# -----------------------------------------------------------------------------
# 4. OBSERVABILITÉ MODÈLE
# -----------------------------------------------------------------------------

@app.get("/model/metadata")
def get_model_metadata():
    info = get_production_model_metadata()
    return info["metadata"] if info else {"status": "unknown"}


@app.get("/model/config")
def get_model_config():
    info = get_production_model_metadata()
    return info["config"] if info else {"status": "unknown"}

# -----------------------------------------------------------------------------
# 5. PIPELINES (ÉCRITURE)
# -----------------------------------------------------------------------------

@app.post("/data")
def update_data_pipeline():
    report = {}
    try:
        logger.info("🏗️ 1/3 - Initialisation DB + Ingestion Massive...")
        ingest_movielens()
        report["ingestion"] = "OK"

        logger.info("📦 2/3 - Versioning DVC...")
        if not os.path.exists(".git"):
            subprocess.run(["git", "init"], check=False, capture_output=True)
            subprocess.run(["git", "config", "user.email", "bot@mlops.com"], check=False)
            subprocess.run(["git", "config", "user.name", "Bot"], check=False)
        subprocess.run(["dvc", "add", "data/raw"], check=False, capture_output=True)
        report["dvc"] = "Completed (Local)"

        logger.info("🔄 3/3 - Rechargement du Cache...")
        global TITLE_MAP
        TITLE_MAP = load_titles_from_sql()
        report["cache"] = f"Reloaded {len(TITLE_MAP)} items"

        return report
    except Exception as e:
        logger.error(f"❌ Erreur pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training")
def training():
    try:
        subprocess.run([sys.executable, "src/ingestion/create_snapshot.py"], check=True)
        subprocess.run(["dvc", "add", "data/training_set.parquet"], check=False)
        subprocess.run([sys.executable, "-m", "src.models.train_model2", "--n-neighbors", "20"], check=True)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))