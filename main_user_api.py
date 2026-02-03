# ============================================================
# MAIN_USER_API.PY
# ============================================================
# API de recommandation utilisateur avec :
# - bouton Recommandation (predict)
# - bouton Réentraîner (training)
# - saisie du user_id
# - affichage lisible humainement
# - aucune sortie JSON brute
# - aucune interface Swagger
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path
import pandas as pd
import subprocess
import sys
from collections import defaultdict

from src.models.predict_model2 import recommend_for_user


# ------------------------------------------------------------
# CONFIG — chargement des titres de films
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MOVIES_PATH = ROOT / "data" / "processed" / "movies_clean.csv"

movies = pd.read_csv(MOVIES_PATH)
TITLE_MAP = dict(zip(movies["movieId"], movies["title"]))


# ------------------------------------------------------------
# INITIALISATION FASTAPI (DOCS DÉSACTIVÉES)
# ------------------------------------------------------------
app = FastAPI(
    title="Movie Recommendation API",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


# ------------------------------------------------------------
# PAGE D’ACCUEIL — DEUX BOUTONS
# ------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Recommandation de films</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f7f7f7;
                    margin: 0;
                    padding: 0;
                }
                .container {
                    max-width: 700px;
                    margin: 50px auto;
                    background: white;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                h1 {
                    margin-top: 0;
                }
                .box {
                    margin-bottom: 30px;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 6px;
                }
                label {
                    font-weight: bold;
                }
                input {
                    padding: 8px;
                    width: 100%;
                    margin-top: 5px;
                    margin-bottom: 15px;
                }
                button {
                    padding: 10px 20px;
                    font-size: 14px;
                    cursor: pointer;
                    border: none;
                    border-radius: 4px;
                }
                .btn-predict {
                    background-color: #2c7be5;
                    color: white;
                }
                .btn-train {
                    background-color: #28a745;
                    color: white;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎬 Système de recommandation de films</h1>

                <div class="box">
                    <h2>Recommandation personnalisée</h2>
                    <form action="/recommend" method="get">
                        <label>Utilisateur (user_id)</label>
                        <input type="number" name="user_id" required>

                        <label>Nombre de recommandations</label>
                        <input type="number" name="top_n" value="5">

                        <button class="btn-predict" type="submit">
                            Obtenir des recommandations
                        </button>
                    </form>
                </div>

                <div class="box">
                    <h2>Réentraîner le modèle</h2>
                    <form action="/training" method="post">
                        <button class="btn-train" type="submit">
                            Lancer le training
                        </button>
                    </form>
                </div>
            </div>
        </body>
    </html>
    """


# ------------------------------------------------------------
# RECOMMANDATION — AFFICHAGE LISIBLE
# ------------------------------------------------------------
@app.get("/recommend", response_class=HTMLResponse)
def recommend(user_id: int, top_n: int = 5):
    try:
        result = recommend_for_user(user_id=user_id, n_reco=top_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not result.get("recommendations"):
        return "<p>Aucune recommandation disponible.</p>"

    grouped = defaultdict(list)

    for rec in result["recommendations"]:
        reco_title = TITLE_MAP.get(rec["movieId"], "Titre inconnu")
        score = round(rec["score"], 2)

        for exp in rec["explanations"]:
            src_title = TITLE_MAP.get(exp["because_movieId"], "Titre inconnu")
            grouped[src_title].append((reco_title, score))

    html = """
    <html>
        <head>
            <title>Résultats</title>
            <style>
                body { font-family: Arial; background:#f7f7f7; }
                .container {
                    max-width: 700px;
                    margin: 40px auto;
                    background: white;
                    padding: 30px;
                    border-radius: 8px;
                }
            </style>
        </head>
        <body>
            <div class="container">
    """

    html += f"<h2>Recommandations pour l’utilisateur {user_id}</h2>"

    for src_title, movies in grouped.items():
        html += f"<h3>Parce que vous avez aimé {src_title} :</h3><ul>"
        for title, score in movies:
            html += f"<li>{title} — score {score}</li>"
        html += "</ul>"

    html += """
                <br>
                <a href="/">↩ Changer d’utilisateur</a>
            </div>
        </body>
    </html>
    """

    return html


# ------------------------------------------------------------
# TRAINING — LANCEMENT DU SCRIPT OFFLINE
# ------------------------------------------------------------
@app.post("/training", response_class=HTMLResponse)
def training():
    try:
        completed = subprocess.run(
            [sys.executable, "src/models/train_model2.py"],
            check=True,
            capture_output=True,
            text=True,
        )

        return f"""
        <html><body style="font-family: Arial; margin:40px">
            <h2>✅ Training terminé avec succès</h2>
            <pre>{completed.stdout}</pre>
            <a href="/">↩ Retour</a>
        </body></html>
        """

    except subprocess.CalledProcessError as e:
        return f"""
        <html><body style="font-family: Arial; margin:40px">
            <h2>❌ Erreur pendant le training</h2>
            <pre>{e.stderr}</pre>
            <a href="/">↩ Retour</a>
        </body></html>
        """
