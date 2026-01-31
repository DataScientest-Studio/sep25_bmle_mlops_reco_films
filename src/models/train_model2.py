# ============================================================
# TRAIN_MODEL.PY
# ============================================================
# OBJECTIF
# ------------------------------------------------------------
# Construire les artefacts nécessaires à un système de
# recommandation collaboratif item-based :
#
# - Identifier, pour chaque film, un sous-ensemble limité
#   de films similaires (voisinage local).
# - Calculer une mesure de popularité robuste pour gérer
#   les situations de cold-start utilisateur.
#
# Le calcul est effectué OFFLINE afin de ne pas recalculer
# les similarités à chaque demande de recommandation.
#
# SOURCE DES DONNÉES
# ------------------------------------------------------------
# - Les interactions utilisateur–film (ratings) sont lues
#   depuis PostgreSQL (schéma raw).
# - Ces tables sont produites en amont par ingestion_movielens.py
#
# SORTIES
# ------------------------------------------------------------
# Les résultats sont persistés dans PostgreSQL :
#
# - raw.item_neighbors :
#     (movieId, neighborMovieId, similarity)
#     → voisins les plus proches pour chaque film
#
# - raw.movie_popularity :
#     (movieId, n_ratings, mean_rating, bayes_score)
#     → score global robuste pour fallback
#
# PRINCIPES
# ------------------------------------------------------------
# - Le modèle repose uniquement sur les interactions
#   utilisateur–film (ratings).
# - Les matrices de notations sont très creuses.
# - On ne compare jamais un film à tous les autres :
#   seules les similarités les plus fortes (K voisins)
#   sont conservées.
# ============================================================

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
PG_URL = os.getenv(
    "PG_URL",
    "postgresql+psycopg2://movie:movie@127.0.0.1:5432/movie_reco"
)
SCHEMA = os.getenv("PG_SCHEMA", "raw")

K_NEIGHBORS = 50
MIN_RATINGS_PER_MOVIE = 50


# ------------------------------------------------------------
# POPULARITÉ BAYÉSIENNE (fallback cold-start)
# ------------------------------------------------------------
def compute_bayesian_popularity(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule un score de popularité robuste pour chaque film.

    Le score combine :
    - la moyenne des notes du film
    - le nombre de notes reçues

    Objectif :
    éviter de survaloriser les films avec peu de notes.
    """
    stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"]).reset_index()

    C = stats["count"].mean()
    M = stats["mean"].mean()

    stats["bayes_score"] = (C * M + stats["count"] * stats["mean"]) / (C + stats["count"])

    return (
        stats.rename(columns={"count": "n_ratings", "mean": "mean_rating"})[
            ["movieId", "n_ratings", "mean_rating", "bayes_score"]
        ]
    )

# ------------------------------------------------------------
# ENTRAÎNEMENT ITEM-BASED COLLABORATIVE FILTERING
# ------------------------------------------------------------
def train_item_based_cf() -> None:
    """
    Étapes principales :

    1) Charger les interactions utilisateur–film depuis PostgreSQL.
    2) Filtrer les films trop peu notés.
    3) Construire une matrice de notations creuse.
    4) Représenter chaque film par son vecteur de notes utilisateurs.
    5) Identifier les K films les plus similaires (cosine).
    6) Calculer une popularité globale robuste.
    7) Sauvegarder les résultats pour l’inférence.
    """

    engine = create_engine(PG_URL)

    # --- 1) Chargement des données ---
    ratings = pd.read_sql(
        f"""
        SELECT "userId", "movieId", rating
        FROM {SCHEMA}.raw_ratings
        WHERE rating IS NOT NULL
        """,
        con=engine,
    )
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    # --- 2) Filtrage des films trop peu notés ---
    movie_counts = ratings["movieId"].value_counts()
    keep_movies = movie_counts[movie_counts >= MIN_RATINGS_PER_MOVIE].index
    ratings = ratings[ratings["movieId"].isin(keep_movies)]

    # --- 3) Construction de la matrice creuse ---
    user_ids = ratings["userId"].unique()
    movie_ids = ratings["movieId"].unique()

    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    movie_to_idx = {m: j for j, m in enumerate(movie_ids)}

    rows = ratings["userId"].map(user_to_idx).to_numpy()
    cols = ratings["movieId"].map(movie_to_idx).to_numpy()
    vals = ratings["rating"].to_numpy()

    X_ui = csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(movie_ids)))
    X_iu = X_ui.T.tocsr()  # films × utilisateurs

    # --- 4) Recherche des K plus proches voisins ---
    nn = NearestNeighbors(
        n_neighbors=K_NEIGHBORS + 1,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    nn.fit(X_iu)

    distances, indices = nn.kneighbors(X_iu)

    neighbors = []
    for i in range(len(movie_ids)):
        src_movie = int(movie_ids[i])
        for r in range(1, K_NEIGHBORS + 1):
            j = indices[i, r]
            sim = 1.0 - distances[i, r]
            neighbors.append((src_movie, int(movie_ids[j]), float(sim)))

    item_neighbors = pd.DataFrame(
        neighbors,
        columns=["movieId", "neighborMovieId", "similarity"],
    )

    # --- 5) Popularité globale ---
    movie_popularity = compute_bayesian_popularity(ratings)

    # --- 6) Sauvegarde dans PostgreSQL ---
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS item_neighbors"))
        conn.execute(text("DROP TABLE IF EXISTS movie_popularity"))

        item_neighbors.to_sql(
            "item_neighbors",
            con=conn,
            schema=SCHEMA,
            index=False,
        )

        movie_popularity.to_sql(
            "movie_popularity",
            con=conn,
            schema=SCHEMA,
            index=False,
        )

        conn.execute(
            text(
                f"""
                CREATE INDEX IF NOT EXISTS idx_item_neighbors_movie
                ON {SCHEMA}.item_neighbors(movieId)
                """
            )
        )

if __name__ == "__main__":
    train_item_based_cf()
