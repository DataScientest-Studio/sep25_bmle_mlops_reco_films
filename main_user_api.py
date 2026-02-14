# ============================================================
# MAIN_USER_API.PY
# ============================================================
import os
import sys
import subprocess
import logging
import pandas as pd
from collections import defaultdict
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import create_engine

# Import de ton modèle
from src.models.predict_model2 import recommend_for_user
# Import de ton script d'ingestion
from src.ingestion.ingestion_movielens import ingest_movielens

# ------------------------------------------------------------
# CONFIG LOGGING
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# CONFIG & CONNEXION BDD
# ------------------------------------------------------------
PG_URL = os.getenv("PG_URL", "postgresql+psycopg2://movie:movie@127.0.0.1:5432/movie_reco")

def load_titles_from_sql():
    """
    Charge la table movies depuis PostgreSQL pour créer le mapping ID -> Titre.
    """
    try:
        logger.info(f"🔌 Connexion à la BDD pour charger les titres...")
        engine = create_engine(PG_URL)
        # On lit la table 'raw_movies' dans le schéma 'raw'
        query = "SELECT \"movieId\", \"title\" FROM raw.raw_movies"
        movies_df = pd.read_sql(query, engine)
        
        if movies_df.empty:
            logger.warning("⚠️ Table 'raw_movies' vide.")
            return {}
        
        return dict(zip(movies_df["movieId"], movies_df["title"]))

    except Exception as e:
        logger.error(f"❌ Impossible de charger les titres (BDD peut-être vide) : {e}")
        return {}

# ------------------------------------------------------------
# CHARGEMENT AU DÉMARRAGE
# ------------------------------------------------------------
TITLE_MAP = load_titles_from_sql()
logger.info(f"✅ {len(TITLE_MAP)} films chargés en mémoire.")

# ------------------------------------------------------------
# INITIALISATION FASTAPI
# ------------------------------------------------------------
app = FastAPI(title="Movie Recommendation API")

# ------------------------------------------------------------
# PAGE D’ACCUEIL
# ------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Movie Reco API</title>
            <style>body{font-family: Arial; padding: 40px; max-width: 800px; margin: auto; line-height: 1.6;}</style>
        </head>
        <body>
            <h1>🎬 API de Recommandation & Pipeline MLOps</h1>
            <p>Statut du cache : <b>""" + str(len(TITLE_MAP)) + """ films chargés</b></p>
            <hr>
            <ul>
                <li><a href="/docs">📄 Documentation Technique (Swagger UI)</a></li>
                <li>GET <b>/recommend?user_id=1</b> : Obtenir des prédictions</li>
                <li>POST <b>/data</b> : <b>Pipeline Ingestion</b> (DVC Pull Raw + SQL Append)</li>
                <li>POST <b>/training</b> : <b>Pipeline Training</b> (Export SQL + Train + DVC Add)</li>
            </ul>
        </body>
    </html>
    """

# ------------------------------------------------------------
# 1. AUTOMATISATION DATA PIPELINE (/data)
# ------------------------------------------------------------
@app.post("/data")
def update_data_pipeline():
    """
    1. Pull uniquement les données RAW (data/raw.dvc).
    2. Lance l'ingestion (Append) dans PostgreSQL.
    3. Met à jour le mapping TITLE_MAP en mémoire.
    """
    report = {}
    
    # A. DVC PULL SÉLECTIF (On ne prend que le RAW, pas le snapshot training)
    try:
        logger.info("📡 DVC PULL sur data/raw.dvc...")
        # On cible uniquement raw.dvc pour être rapide et précis
        subprocess.run(["dvc", "pull", "data/raw.dvc"], check=True, capture_output=True, text=True)
        report["dvc_pull"] = "Succès (data/raw synchronisé)"
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erreur DVC Pull : {e.stderr}")
        raise HTTPException(status_code=500, detail=f"Erreur DVC: {e.stderr}")

    # B. INGESTION SQL
    try:
        logger.info("💾 Lancement de l'ingestion vers PostgreSQL...")
        ingest_movielens() 
        report["ingestion"] = "Succès - Nouvelles données ajoutées."
        
    except Exception as e:
        logger.error(f"❌ Erreur Ingestion : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur Ingestion SQL: {str(e)}")

    # C. RECHARGEMENT DU MAPPING TITRES
    global TITLE_MAP
    TITLE_MAP = load_titles_from_sql()
    report["reload_cache"] = f"Mapping mis à jour : {len(TITLE_MAP)} films en mémoire."

    return JSONResponse(content=report, status_code=200)

# ------------------------------------------------------------
# 2. TRAINING PIPELINE (/training)
# ------------------------------------------------------------
@app.post("/training")
def training():
    """
    Lance le script de training qui s'occupe de :
    - Créer le snapshot CSV depuis SQL
    - Entraîner le modèle
    - Faire le 'dvc add' sur le nouveau snapshot
    """
    try:
        logger.info("🏋️‍♂️ Lancement du training pipeline...")
        completed = subprocess.run(
            [sys.executable, "src/models/train_model2.py"],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("✅ Training et Snapshot DVC terminés.")
        
        return {
            "status": "success",
            "message": "Modèle réentraîné et snapshot versionné",
            "logs": completed.stdout
        }

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erreur Training : {e.stderr}")
        raise HTTPException(status_code=500, detail=f"Erreur Training: {e.stderr}")

# ------------------------------------------------------------
# 3. PREDICTION (/recommend)
# ------------------------------------------------------------
@app.get("/recommend", response_class=HTMLResponse)
def recommend(user_id: int, top_n: int = 5):
    try:
        if not TITLE_MAP:
            return "<html><body><h2>⚠️ Base de données vide. Veuillez lancer /data d'abord.</h2></body></html>"

        result = recommend_for_user(user_id=user_id, n_reco=top_n)
    except Exception as e:
        logger.error(f"Erreur prediction : {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if not result.get("recommendations"):
        return f"<html><body><p>Aucune recommandation pour l'utilisateur {user_id} (inconnu ou pas assez de notes).</p></body></html>"

    # Formatage des résultats
    grouped = defaultdict(list)
    for rec in result["recommendations"]:
        reco_title = TITLE_MAP.get(rec["movieId"], f"Film {rec['movieId']}")
        score = round(rec["score"], 2)
        for exp in rec["explanations"]:
            src_title = TITLE_MAP.get(exp["because_movieId"], f"Film {exp['because_movieId']}")
            grouped[src_title].append((reco_title, score))

    html = f"""
    <html>
        <body style="font-family: Arial; padding: 20px;">
        <h2>🎬 Recommandations pour l'utilisateur {user_id}</h2>
        <hr>
    """
    for src_title, movies_list in grouped.items():
        html += f"<h3>Parce que vous avez aimé <b>{src_title}</b> :</h3><ul>"
        for title, score in movies_list:
            html += f"<li>{title} <small>(Score de confiance: {score})</small></li>"
        html += "</ul>"
    
    html += f'<br><a href="/">⬅️ Retour</a></body></html>'
    return html