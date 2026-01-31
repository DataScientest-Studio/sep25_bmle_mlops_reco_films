# ============================================================
# PREDICT_MODEL.PY
# ============================================================
# OBJECTIF
# ------------------------------------------------------------
# Générer des recommandations personnalisées pour un utilisateur
# à partir d’un modèle collaboratif item-based pré-entraîné.
#
# Le processus repose sur :
# - l’historique de notation de l’utilisateur
# - les voisinages film–film calculés offline
#
# Aucune similarité n’est recalculée ici.
#
# SOURCE DES DONNÉES
# ------------------------------------------------------------
# - raw.raw_ratings       : historique utilisateur
# - raw.item_neighbors   : similarités item-item
# - raw.movie_popularity : fallback cold-start
#
# SORTIE
# ------------------------------------------------------------
# Une liste de films recommandés, triés par score décroissant.
#
# PRINCIPES
# ------------------------------------------------------------
# - Les recommandations sont locales :
#   on exploite uniquement les films similaires à ceux déjà notés.
# - Les films déjà vus sont exclus.
# - En cas d’historique insuffisant, une stratégie globale
#   basée sur la popularité est utilisée.
# ============================================================

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
PG_URL = os.getenv(
    "PG_URL",
    "postgresql+psycopg2://movie:movie@127.0.0.1:5432/movie_reco"
)
SCHEMA = os.getenv("PG_SCHEMA", "raw")

# ------------------------------------------------------------
# FONCTION DE RECOMMANDATION
# ------------------------------------------------------------
def recommend_for_user(
    user_id: int,
    n_reco: int = 10,
    min_user_ratings: int = 5,
    positive_threshold: float = 4.0,
) -> pd.DataFrame:
    """
    Étapes principales :
    1) Charger l’historique de l’utilisateur.
    2) Vérifier si l’historique est suffisant.
    3) Identifier les films appréciés.
    4) Générer des candidats via les films similaires.
    5) Calculer un score pondéré pour chaque candidat.
    6) Exclure les films déjà vus et trier les résultats.
    """

    engine = create_engine(PG_URL)

    # --- 1) Historique utilisateur ---
    user_ratings = pd.read_sql(
        f"""
        SELECT "movieId", rating
        FROM {SCHEMA}.raw_ratings
        WHERE "userId" = %(user_id)s
        """,
        con=engine,
        params={"user_id": user_id},
    )

    # --- 2) Cold-start utilisateur ---
    if user_ratings.empty or len(user_ratings) < min_user_ratings:
        return pd.read_sql(
            f"""
            SELECT movieId, bayes_score AS score
            FROM {SCHEMA}.movie_popularity
            ORDER BY bayes_score DESC
            LIMIT %(n)s
            """,
            con=engine,
            params={"n": n_reco},
        )

    # --- 3) Sélection des films appréciés ---
    seed = user_ratings[user_ratings["rating"] >= positive_threshold]
    if seed.empty:
        seed = user_ratings

    seen = set(user_ratings["movieId"])
    seed_movies = tuple(seed["movieId"].tolist())

    # --- 4) Récupération des voisins ---
    neighbors = pd.read_sql(
        f"""
        SELECT movieId AS srcMovieId, neighborMovieId, similarity
        FROM {SCHEMA}.item_neighbors
        WHERE movieId IN %(movies)s
        """,
        con=engine,
        params={"movies": seed_movies},
    )

    if neighbors.empty:
        return pd.read_sql(
            f"""
            SELECT movieId, bayes_score AS score
            FROM {SCHEMA}.movie_popularity
            ORDER BY bayes_score DESC
            LIMIT %(n)s
            """,
            con=engine,
            params={"n": n_reco},
        )

    # --- 5) Association des notes utilisateur ---
    neighbors = neighbors.merge(
        user_ratings.rename(
            columns={"movieId": "srcMovieId", "rating": "srcRating"}
        ),
        on="srcMovieId",
    )

    # --- 6) Calcul du score pondéré ---
    scores = (
        neighbors.groupby("neighborMovieId")
        .apply(
            lambda g: np.dot(g["similarity"], g["srcRating"])
            / (np.abs(g["similarity"]).sum() + 1e-12)
        )
        .reset_index(name="score")
    )

    # --- 7) Reclassement final ---
    scores = scores[~scores["neighborMovieId"].isin(seen)]
    scores = scores.sort_values("score", ascending=False).head(n_reco)

    return scores.rename(columns={"neighborMovieId": "movieId"})


if __name__ == "__main__":
    print(recommend_for_user(user_id=1))
