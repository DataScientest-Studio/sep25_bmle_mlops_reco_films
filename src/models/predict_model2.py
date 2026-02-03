# ============================================================
# PREDICT_MODEL2.PY
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
# Pour chaque film recommandé :
# - un score (note prédite implicite)
# - une explication (films sources ayant contribué)
#
# PRINCIPES
# ------------------------------------------------------------
# - Recommandations locales (item-based CF)
# - Films déjà vus exclus
# - Modèle explicable par construction
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
# FONCTION UTILITAIRE — EXPLICATION D’UNE RECOMMANDATION
# ------------------------------------------------------------
def explain_group(g: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """
    Décompose le score d’un film recommandé en contributions
    provenant des films déjà notés par l’utilisateur.

    Chaque ligne correspond à un film source expliquant la reco.
    """
    denom = np.abs(g["similarity"]).sum() + 1e-12

    g = g.assign(
        contribution=(g["similarity"] * g["srcRating"]) / denom
    )

    return (
        g.sort_values("contribution", ascending=False)
         .head(top_k)
         [["srcMovieId", "srcRating", "similarity", "contribution"]]
    )


# ------------------------------------------------------------
# FONCTION PRINCIPALE DE RECOMMANDATION
# ------------------------------------------------------------
def recommend_for_user(
    user_id: int,
    n_reco: int = 10,
    min_user_ratings: int = 5,
    positive_threshold: float = 4.0,
) -> dict:
    """
    Génère des recommandations personnalisées pour un utilisateur.

    Étapes principales :
    1) Charger l’historique utilisateur
    2) Gérer le cold-start si nécessaire
    3) Sélectionner les films appréciés
    4) Récupérer les voisins film–film
    5) Calculer un score pondéré
    6) Générer une explication par film recommandé
    """

    engine = create_engine(PG_URL)

    # --------------------------------------------------------
    # 1) Historique utilisateur
    # --------------------------------------------------------
    user_ratings = pd.read_sql(
        f"""
        SELECT "movieId", rating
        FROM {SCHEMA}.raw_ratings
        WHERE "userId" = %(user_id)s
        """,
        con=engine,
        params={"user_id": user_id},
    )

    # --------------------------------------------------------
    # 2) Cold-start utilisateur
    # --------------------------------------------------------
    if user_ratings.empty or len(user_ratings) < min_user_ratings:
        popular = pd.read_sql(
            f"""
            SELECT movieId, bayes_score AS score
            FROM {SCHEMA}.movie_popularity
            ORDER BY bayes_score DESC
            LIMIT %(n)s
            """,
            con=engine,
            params={"n": n_reco},
        )

        return {
            "user_id": user_id,
            "strategy": "popularity",
            "recommendations": popular.to_dict(orient="records"),
        }

    # --------------------------------------------------------
    # 3) Sélection des films appréciés
    # --------------------------------------------------------
    seed = user_ratings[user_ratings["rating"] >= positive_threshold]
    if seed.empty:
        seed = user_ratings

    seen_movies = set(user_ratings["movieId"])
    seed_movies = tuple(seed["movieId"].tolist())

    # --------------------------------------------------------
    # 4) Récupération des voisins film–film
    # --------------------------------------------------------
    neighbors = pd.read_sql(
        f"""
        SELECT
            "movieId"         AS "srcMovieId",
            "neighborMovieId" AS "neighborMovieId",
            similarity
        FROM {SCHEMA}.item_neighbors
        WHERE "movieId" IN %(movies)s
        """,
        con=engine,
        params={"movies": seed_movies},
    )

    if neighbors.empty:
        return {
            "user_id": user_id,
            "strategy": "popularity_fallback",
            "recommendations": [],
        }

    # --------------------------------------------------------
    # 5) Association des notes utilisateur
    # --------------------------------------------------------
    neighbors = neighbors.merge(
        user_ratings.rename(
            columns={"movieId": "srcMovieId", "rating": "srcRating"}
        ),
        on="srcMovieId",
    )

    # --------------------------------------------------------
    # 6) Calcul du score par film candidat
    # --------------------------------------------------------
    scores = (
        neighbors.groupby("neighborMovieId")
        .apply(
            lambda g: np.dot(g["similarity"], g["srcRating"])
            / (np.abs(g["similarity"]).sum() + 1e-12)
        )
        .reset_index(name="score")
    )

    scores = scores[~scores["neighborMovieId"].isin(seen_movies)]

    # --------------------------------------------------------
    # 7) Explications
    # --------------------------------------------------------
    explanations = (
        neighbors
        .groupby("neighborMovieId")
        .apply(explain_group)
        .reset_index()
    )

    # --------------------------------------------------------
    # 8) Fusion score + explications
    # --------------------------------------------------------
    final = (
        scores
        .merge(explanations, on="neighborMovieId", how="left")
        .sort_values("score", ascending=False)
        .head(n_reco)
    )

    # --------------------------------------------------------
    # 9) Sortie JSON-friendly
    # --------------------------------------------------------
    recommendations = []

    for movie_id, group in final.groupby("neighborMovieId"):
        recommendations.append({
            "movieId": int(movie_id),
            "score": float(group["score"].iloc[0]),
            "explanations": [
                {
                    "because_movieId": int(row["srcMovieId"]),
                    "rating": float(row["srcRating"]),
                    "similarity": float(row["similarity"]),
                    "contribution": float(row["contribution"]),
                }
                for _, row in group.iterrows()
            ],
        })

    return {
        "user_id": user_id,
        "strategy": "item_based_cf",
        "recommendations": recommendations,
    }


# ------------------------------------------------------------
# MODE SCRIPT (CLI) — TEST LOCAL
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Test de recommandation utilisateur (item-based CF)"
    )
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=10)

    args = parser.parse_args()

    result = recommend_for_user(
        user_id=args.user_id,
        n_reco=args.top_n,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
