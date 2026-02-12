# ============================================================
# TRAIN_MODEL2.PY
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
import subprocess
from datetime import datetime
import yaml
import mlflow
from sqlalchemy import create_engine, text
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import argparse


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
PG_URL = os.getenv(
    "PG_URL",
    "postgresql+psycopg2://movie:movie@127.0.0.1:5432/movie_reco"
)
SCHEMA = os.getenv("PG_SCHEMA", "raw")

#K_NEIGHBORS = 50
#MIN_RATINGS_PER_MOVIE = 50

EXPERIMENT_NAME = "reco-films/itemcf-v2"
REGISTERED_MODEL_NAME = "reco-films-itemcf-v2"


# ------------------------------------------------------------
# UTILITAIRES MLFLOW (versioning code & data)
# ------------------------------------------------------------
def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True
        ).strip()
    except Exception:
        return "unknown"


def get_dvc_md5(dvc_path: str = "data/raw.dvc") -> str:
    try:
        with open(dvc_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        return str(y["outs"][0].get("md5", "unknown"))
    except Exception:
        return "unknown"


# ------------------------------------------------------------
# POPULARITÉ BAYÉSIENNE (fallback cold-start)
# ------------------------------------------------------------
def compute_bayesian_popularity(ratings: pd.DataFrame) -> pd.DataFrame:
    stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"]).reset_index()

    C = stats["count"].mean()
    M = stats["mean"].mean()

    stats["bayes_score"] = (C * M + stats["count"] * stats["mean"]) / (C + stats["count"])

    return (
        stats.rename(columns={"count": "n_ratings", "mean": "mean_rating"})[
            ["movieId", "n_ratings", "mean_rating", "bayes_score"]
        ]
    )

def recall_at_10(recommended, relevant):
    recommended_set = set(recommended[:10])
    relevant_set = set(relevant)
    if len(relevant_set) == 0:
        return 0
    return len(recommended_set & relevant_set) / len(relevant_set)

def ndcg_at_10(recommended, relevant):
    dcg = 0.0
    for i, movie in enumerate(recommended[:10]):
        if movie in relevant:
            dcg += 1 / np.log2(i + 2)

    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), 10)))
    if idcg == 0:
        return 0
    return dcg / idcg

# ENTRAÎNEMENT ITEM-BASED COLLABORATIVE FILTERING
# ------------------------------------------------------------
def train_item_based_cf(k_neighbors: int, min_ratings: int) -> None:

    # --------------------------------------------------------
    # INITIALISATION MLFLOW
    # --------------------------------------------------------
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = (
    f"train_itemcf_v2__k{k_neighbors}"
    f"__{datetime.now().strftime('%Y%m%d_%H%M')}"
)

    with mlflow.start_run(run_name=run_name):

        # ----------------------------------------------------
        # LOG PARAMÈTRES
        # ----------------------------------------------------
        mlflow.log_param("k_neighbors", k_neighbors)
        mlflow.log_param("min_ratings_per_movie", min_ratings)

        mlflow.set_tag("git_commit", get_git_commit())
        mlflow.set_tag("code_version", get_git_commit())
        mlflow.set_tag("data_version", get_dvc_md5())
        mlflow.set_tag("data_schema", "raw.raw_ratings(userId,movieId,rating)")
        mlflow.set_tag("pg_schema", SCHEMA)

        engine = create_engine(PG_URL)

        # --- 1) Chargement des données ---
        ratings = pd.read_sql(
            f"""
            SELECT "userId", "movieId", rating, "timestamp"
            FROM {SCHEMA}.raw_ratings
            WHERE rating IS NOT NULL
            """,
            con=engine,
        )

        ratings["userId"] = ratings["userId"].astype(int)
        ratings["movieId"] = ratings["movieId"].astype(int)
        ratings["rating"] = ratings["rating"].astype(float)

# ------------------------------------------------------------
# SPLIT TEMPOREL 80/20 PAR UTILISATEUR
# ------------------------------------------------------------
        ratings = ratings.sort_values(["userId", "timestamp"])

        train_list = []
        test_list = []

        for user_id, user_data in ratings.groupby("userId"):
            split_index = int(len(user_data) * 0.8)
            train_list.append(user_data.iloc[:split_index])
            test_list.append(user_data.iloc[split_index:])

        train_ratings = pd.concat(train_list)
        test_ratings = pd.concat(test_list)
        train_movies_per_user = (
        train_ratings.groupby("userId")["movieId"].apply(list).to_dict()
)


        # LOG CONTEXTE DATA
        mlflow.set_tag("train_rows", int(len(ratings)))
        mlflow.set_tag("n_users", int(ratings["userId"].nunique()))
        mlflow.set_tag("n_movies_raw", int(ratings["movieId"].nunique()))

        # --- 2) Filtrage des films trop peu notés ---
        movie_counts = train_ratings["movieId"].value_counts()
        keep_movies = movie_counts[movie_counts >= min_ratings].index
        train_ratings = train_ratings[train_ratings["movieId"].isin(keep_movies)]

        mlflow.set_tag("n_movies_filtered", int(train_ratings["movieId"].nunique()))

        # --- 3) Construction matrice creuse ---
        user_ids = train_ratings["userId"].unique()
        movie_ids = train_ratings["movieId"].unique()

        user_to_idx = {u: i for i, u in enumerate(user_ids)}
        movie_to_idx = {m: j for j, m in enumerate(movie_ids)}

        rows = train_ratings["userId"].map(user_to_idx).to_numpy()
        cols = train_ratings["movieId"].map(movie_to_idx).to_numpy()
        vals = train_ratings["rating"].to_numpy()

        X_ui = csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(movie_ids)))
        X_iu = X_ui.T.tocsr()

        # --- 4) K plus proches voisins ---
        nn = NearestNeighbors(
            n_neighbors=k_neighbors + 1,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        nn.fit(X_iu)

        distances, indices = nn.kneighbors(X_iu)

        neighbors = []
        for i in range(len(movie_ids)):
            src_movie = int(movie_ids[i])
            for r in range(1, k_neighbors + 1):
                j = indices[i, r]
                sim = 1.0 - distances[i, r]
                neighbors.append((src_movie, int(movie_ids[j]), float(sim)))

        item_neighbors = pd.DataFrame(
            neighbors,
            columns=["movieId", "neighborMovieId", "similarity"],
        )

    
        # --- 5) Popularité globale ---
        movie_popularity = compute_bayesian_popularity(ratings)

        # ------------------------------------------------------------
        # ÉVALUATION
        # ------------------------------------------------------------
#        recalls = []
#        ndcgs = []
#
#        recommended_movies = movie_popularity.sort_values(
#            "bayes_score", ascending=False
#        )["movieId"].tolist()

#        for user_id, user_data in test_ratings.groupby("userId"):
#            relevant_movies = user_data["movieId"].tolist()
#
#            recalls.append(recall_at_10(recommended_movies, relevant_movies))
#            ndcgs.append(ndcg_at_10(recommended_movies, relevant_movies))

#        mean_recall = np.mean(recalls)
#        mean_ndcg = np.mean(ndcgs)

#        mlflow.log_metric("recall_10", mean_recall)
#        mlflow.log_metric("ndcg_10", mean_ndcg)


        # ------------------------------------------------------------
        # ÉVALUATION (utilise réellement les voisins KNN)
        # ------------------------------------------------------------

        neighbors_dict = (
            item_neighbors.groupby("movieId")[["neighborMovieId", "similarity"]]
            .apply(lambda df: list(df.itertuples(index=False, name=None)))
            .to_dict()
)


        recalls = []
        ndcgs = []

        for user_id, user_data in test_ratings.groupby("userId"):

            user_train_movies = train_movies_per_user.get(user_id, [])

            if not user_train_movies:
                continue

            scores = {}

            for movie in user_train_movies:
                if movie in neighbors_dict:
                    for neighbor_movie, sim in neighbors_dict[movie]:
                        scores[neighbor_movie] = scores.get(neighbor_movie, 0) + sim

            seen = set(user_train_movies)
            scores = {m: s for m, s in scores.items() if m not in seen}

            if not scores:
                continue

            recommended_movies = sorted(
                scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            recommended_movies = [m for m, _ in recommended_movies][:10]

            relevant_movies = user_data["movieId"].tolist()

            recalls.append(recall_at_10(recommended_movies, relevant_movies))
            ndcgs.append(ndcg_at_10(recommended_movies, relevant_movies))

        mean_recall = np.mean(recalls)
        mean_ndcg = np.mean(ndcgs)

        mlflow.log_metric("recall_10", mean_recall)
        mlflow.log_metric("ndcg_10", mean_ndcg)

        # ----------------------------------------------------
        # LOG ARTEFACTS
        # ----------------------------------------------------
        os.makedirs("mlflow_artifacts", exist_ok=True)

        neighbors_path = "mlflow_artifacts/item_neighbors.csv"
        popularity_path = "mlflow_artifacts/movie_popularity.csv"

        item_neighbors.to_csv(neighbors_path, index=False)
        movie_popularity.to_csv(popularity_path, index=False)

        mlflow.log_artifact(neighbors_path)
        mlflow.log_artifact(popularity_path)

        # --- 6) Sauvegarde en base ---
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS item_neighbors"))
            conn.execute(text("DROP TABLE IF EXISTS movie_popularity"))

            item_neighbors.to_sql(
                "item_neighbors",
                con=conn,
                schema=SCHEMA,
                index=False,
                if_exists="replace",
            )

            movie_popularity.to_sql(
                "movie_popularity",
                con=conn,
                schema=SCHEMA,
                index=False,
                if_exists="replace",
            )




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-neighbors", type=int, default=50)
    parser.add_argument("--min-ratings", type=int, default=50)

    args = parser.parse_args()

    train_item_based_cf(
        k_neighbors=args.n_neighbors,
        min_ratings=args.min_ratings,
    )
