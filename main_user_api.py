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

# Import de ton modèle (pour la prédiction)
from src.models.predict_model2 import recommend_for_user
# Import de ton script d'ingestion (pour la mise à jour data)
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
    
    # A. DVC PULL SÉLECTIF
    try:
        logger.info("📡 DVC PULL sur data/raw.dvc...")
        # On passe explicitement l'environnement pour git/dvc
        subprocess.run(["dvc", "pull", "data/raw.dvc"], check=True, capture_output=True, text=True, env=os.environ)
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
    Lance le pipeline complet :
    1. Création du snapshot (create_snapshot.py) -> génère data/training_set.csv
    2. Versionning DVC (dvc add)
    3. Entraînement (train_model2.py)
    """
    logs = []
    # Force l'utilisation du même interpréteur Python
    python_exec = sys.executable 
    # Copie de l'environnement actuel (important pour GIT, MLFLOW, etc.)
    current_env = os.environ.copy()

    try:
        # ETAPE 1 : Création du Snapshot CSV
        logger.info("📸 Lancement de create_snapshot.py...")
        # Note: create_snapshot est un script standalone, on peut l'appeler par chemin ou via module si présent
        # Ici on garde le chemin relatif
        snap_process = subprocess.run(
            [python_exec, "src/ingestion/create_snapshot.py"],
            check=True,
            capture_output=True,
            text=True,
            env=current_env
        )
        logs.append(f"--- SNAPSHOT ---\n{snap_process.stdout}")

        # ETAPE 2 : DVC Add (pour versionner le CSV généré)
        logger.info("📦 Versionning DVC (dvc add data/training_set.csv)...")
        dvc_process = subprocess.run(
            ["dvc", "add", "data/training_set.csv"],
            check=True,
            capture_output=True,
            text=True,
            env=current_env
        )
        logs.append(f"--- DVC ADD ---\n{dvc_process.stdout}")

        # ETAPE 3 : Entraînement du modèle
        logger.info("🏋️‍♂️ Lancement du training (train_model2.py)...")
        
        # FIX: On lance via "-m src.models.train_model2" pour gérer les imports correctement
        # On ajoute les arguments par défaut (20 voisins, 50 notes min)
        train_cmd = [
            python_exec, "-m", "src.models.train_model2",
            "--n-neighbors", "20",
            "--min-ratings", "50"
        ]
        
        train_process = subprocess.run(
            train_cmd,
            check=True,
            capture_output=True,
            text=True,
            env=current_env,
            cwd=os.getcwd() # S'assurer qu'on est à la racine
        )
        logs.append(f"--- TRAINING ---\n{train_process.stdout}")
        
        logger.info("✅ Pipeline complet terminé.")
        
        return {
            "status": "success",
            "message": "Snapshot créé, versionné et Modèle réentraîné.",
            "logs": "\n".join(logs)
        }

    except subprocess.CalledProcessError as e:
        # On capture la sortie d'erreur (stderr) du processus qui a échoué
        error_msg = e.stderr if e.stderr else str(e)
        logger.error(f"❌ Erreur Pipeline : {error_msg}")
        # On retourne aussi stdout pour le debug
        full_error = f"ERROR: {error_msg}\nSTDOUT: {e.stdout}"
        raise HTTPException(status_code=500, detail=full_error)

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