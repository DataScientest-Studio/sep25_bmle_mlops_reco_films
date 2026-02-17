# =============================================================================
# MAIN_USER_API.PY
# =============================================================================
import os
import sys
# Ensure UTF-8 encoding for standard output
os.environ["PYTHONIOENCODING"] = "utf-8"

import subprocess
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text
from prometheus_fastapi_instrumentator import Instrumentator

# --- LOCAL MODULES ---
try:
    from src.models.predict_model2 import predict as recommend_for_user, get_production_model_metadata
except ImportError:
    # Exit if core model dependencies are missing
    sys.exit(1)

from src.ingestion.ingestion_movielens import ingest_movielens

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database connection string
PG_URL = os.getenv("PG_URL", "postgresql+psycopg2://movie:movie@127.0.0.1:5432/movie_reco")

# --- CACHE MANAGEMENT ---
def load_titles_from_sql():
    """
    Loads movie titles into memory to avoid SQL joins during inference.
    Returns:
        dict: A mapping of movieId -> title.
    """
    try:
        engine = create_engine(PG_URL)
        df = pd.read_sql('SELECT "movieId", "title" FROM raw.raw_movies', engine)
        if df.empty: return {}
        return dict(zip(df["movieId"], df["title"]))
    except Exception:
        return {}

# Initialize global cache
TITLE_MAP = load_titles_from_sql()

# --- APP INITIALIZATION ---
app = FastAPI(title="Movie Recommendation API")


# -----------------------------------------------------------------------------
# 1. SYSTEM & OPERATIONAL ROUTES
# -----------------------------------------------------------------------------

@app.get("/")
def home():
    """Root endpoint to check basic connectivity and cache status."""
    return {"status": "online", "cache_size": len(TITLE_MAP)}


@app.get("/health")
def health_check():
    """Liveness probe: Checks if the API process is running."""
    return {"status": "ok"}


@app.get("/ready")
def readiness_check():
    """
    Readiness probe: Checks critical dependencies.
    1. Database connectivity (Postgres)
    2. Model availability (MLflow access and production alias)
    """
    status = {"database": "unknown", "model": "unknown"}
    is_ready = True

    # 1. Check Database
    try:
        engine = create_engine(PG_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        status["database"] = "connected"
    except Exception as e:
        status["database"] = f"error: {str(e)}"
        is_ready = False

    # 2. Check Model
    try:
        meta = get_production_model_metadata()
        if meta:
            status["model"] = "ready"
        else:
            status["model"] = "not_found"
            is_ready = False
    except Exception as e:
        status["model"] = f"error: {str(e)}"
        is_ready = False

    if not is_ready:
        raise HTTPException(status_code=503, detail=status)
    
    return {"status": "ready", "checks": status}

# --- METRICS (PROMETHEUS) ---
# Placed here to appear after /ready in Swagger UI
Instrumentator().instrument(app).expose(app)


# -----------------------------------------------------------------------------
# 2. CORE BUSINESS LOGIC (RECOMMENDATIONS)
# -----------------------------------------------------------------------------

@app.get("/recommend")
def recommend(user_id: int, top_n: int = 5):
    """
    Main inference endpoint.
    Returns a list of recommended movies for a specific user.
    """
    if not TITLE_MAP:
        raise HTTPException(status_code=503, detail="Cache is empty. Please run /data to initialize.")

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


# -----------------------------------------------------------------------------
# 3. MOVIE CATALOG & DETAILS
# -----------------------------------------------------------------------------

@app.get("/movies/popular")
def get_popular_movies(limit: int = 10):
    """
    Returns popular movies based on Bayesian Score.
    Useful for 'Cold Start' scenarios (users with no history).
    """
    try:
        engine = create_engine(PG_URL)
        # Query the popularity table created during training
        query = f"""
            SELECT "movieId", "mean_rating", "n_ratings", "bayes_score"
            FROM raw.movie_popularity
            ORDER BY "bayes_score" DESC
            LIMIT {limit}
        """
        df = pd.read_sql(query, engine)
        
        results = []
        for _, row in df.iterrows():
            mid = int(row["movieId"])
            results.append({
                "movie_id": mid,
                "title": TITLE_MAP.get(mid, f"Unknown ({mid})"),
                "stats": {
                    "score": round(row["bayes_score"], 3),
                    "mean_rating": round(row["mean_rating"], 2),
                    "count": int(row["n_ratings"])
                }
            })
        return results
    except Exception as e:
        logger.error(f"Error fetching popular movies: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving popular movies.")


@app.get("/movies/{movie_id}")
def get_movie_details(movie_id: int):
    """
    Returns details for a specific movie (Title, Genres, Stats).
    """
    try:
        engine = create_engine(PG_URL)
        # Join between static info (raw_movies) and stats (movie_popularity)
        query = text("""
            SELECT m."movieId", m.title, m.genres, 
                   p.mean_rating, p.n_ratings, p.bayes_score
            FROM raw.raw_movies m
            LEFT JOIN raw.movie_popularity p ON m."movieId" = p."movieId"
            WHERE m."movieId" = :mid
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query, {"mid": movie_id}).mappings().one_or_none()
            
        if not result:
            raise HTTPException(status_code=404, detail=f"Movie ID {movie_id} not found.")
            
        return {
            "movie_id": result["movieId"],
            "title": result["title"],
            "genres": result["genres"],
            "stats": {
                # Handle nulls if movie has no ratings yet
                "score": round(result["bayes_score"], 3) if result["bayes_score"] else 0,
                "mean_rating": round(result["mean_rating"], 2) if result["mean_rating"] else 0,
                "count": int(result["n_ratings"]) if result["n_ratings"] else 0
            }
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error fetching movie {movie_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error.")


# -----------------------------------------------------------------------------
# 4. MODEL OBSERVABILITY
# -----------------------------------------------------------------------------

@app.get("/model/metadata")
def get_model_metadata():
    """Returns versioning and performance info for the production model."""
    info = get_production_model_metadata()
    if info:
        return info["metadata"]
    else:
        return {"status": "unknown", "detail": "Unable to contact MLflow or no production alias defined."}


@app.get("/model/config")
def get_model_config():
    """Returns hyperparameters used to train the production model."""
    info = get_production_model_metadata()
    if info:
        return info["config"]
    else:
        return {"status": "unknown", "detail": "Unable to retrieve configuration."}


# -----------------------------------------------------------------------------
# 5. PIPELINE & ADMINISTRATION (WRITE OPERATIONS)
# -----------------------------------------------------------------------------

@app.post("/data")
def update_data_pipeline():
    """
    Trigger the Data Pipeline:
    1. Ingestion (Download + SQL Insert)
    2. Versioning (DVC Add & Push)
    3. Cache Reload
    """
    report = {}
    
    # Step A: Ingestion
    try:
        logger.info("🚀 Starting Ingestion (Download + SQL)...")
        ingest_movielens() 
        report["ingestion"] = "Success (Downloaded & Inserted)"
    except Exception as e:
        logger.error(f"Ingestion Error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion Failed: {str(e)}")

    # Step B: Versioning (DVC)
    try:
        logger.info("📦 DVC Add & Push...")
        subprocess.run(["dvc", "add", "data/raw"], check=True)
        subprocess.run(["dvc", "push", "data/raw.dvc"], check=True)
        report["dvc"] = "Versioned & Pushed"
    except Exception as e:
        logger.error(f"DVC Error: {e}")
        report["dvc"] = f"Failed: {str(e)}"

    # Step C: Reload Cache
    global TITLE_MAP
    TITLE_MAP = load_titles_from_sql()
    
    return report


@app.post("/training")
def training():
    """
    Trigger the Training Pipeline:
    1. Create Parquet Snapshot
    2. DVC Versioning (Training Set)
    3. Train Model
    """
    python_exec = sys.executable 
    try:
        logger.info("📸 Creating Parquet snapshot...")
        subprocess.run([python_exec, "src/ingestion/create_snapshot.py"], check=True)
        
        logger.info("📦 DVC Add (Training Set)...")
        subprocess.run(["dvc", "add", "data/training_set.parquet"], check=True)
        subprocess.run(["dvc", "push", "data/training_set.parquet.dvc"], check=True)

        logger.info("🏃‍♂️ Training Model...")
        subprocess.run([python_exec, "-m", "src.models.train_model2", "--n-neighbors", "20"], check=True)
        
        return {"status": "success", "message": "Training Complete"}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=str(e))