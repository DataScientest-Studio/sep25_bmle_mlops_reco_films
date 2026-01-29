# src/data/build_features.py

from __future__ import annotations

from pathlib import Path
import sqlite3

import pandas as pd


def build_user_movie_matrices_from_db(
    db_path: Path,
    out_dir: Path,
    ratings_table: str = "raw_ratings",
    movies_table: str = "raw_movies",
    logger=None,
) -> None:
    """
    Objectif (Content-based) :
    - Construire une matrice "films" : movieId + genres (one-hot)
    - Construire une matrice "users" : userId + préférences de genres (moyenne des genres vus)

    Entrées (dans SQLite) :
    - ratings_table : userId, movieId, rating, timestamp
    - movies_table  : movieId, title, genres

    Sorties (CSV) dans out_dir :
    - movie_matrix.csv : movieId + colonnes genres 0/1
    - user_matrix.csv  : userId + colonnes genres (moyenne des genres vus)
    """

    # --- 0) Préparer le dossier de sortie ---
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Petite fonction de log "safe"
    def _log(msg: str) -> None:
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    _log(f"Reading SQLite DB: {db_path}")
    _log(f"Tables: ratings={ratings_table} | movies={movies_table}")

    # --- 1) Lire les tables depuis SQLite ---
    # On utilise sqlite3 (standard library) + pandas.read_sql_query (pratique).
    with sqlite3.connect(str(db_path)) as conn:
        ratings = pd.read_sql_query(f"SELECT userId, movieId, rating, timestamp FROM {ratings_table};", conn)
        movies = pd.read_sql_query(f"SELECT movieId, title, genres FROM {movies_table};", conn)

    # --- 2) Sanity checks minimalistes ---
    # Si tu as un bug de colonnes, tu le vois tout de suite ici.
    needed_ratings = ["userId", "movieId", "rating"]
    needed_movies = ["movieId", "title", "genres"]

    for col in needed_ratings:
        if col not in ratings.columns:
            raise ValueError(f"Missing column in {ratings_table}: {col}")

    for col in needed_movies:
        if col not in movies.columns:
            raise ValueError(f"Missing column in {movies_table}: {col}")

    # On nettoie les types au passage (utile si SQLite a fait des surprises)
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    movies["movieId"] = movies["movieId"].astype(int)
    movies["title"] = movies["title"].astype(str)
    movies["genres"] = movies["genres"].fillna("").astype(str)

    # --- 3) Construire la "movie_matrix" : movieId + one-hot genres ---
    # Exemple genres: "Adventure|Animation|Children|Comedy|Fantasy"
    # str.get_dummies sep="|" crée directement des colonnes binaires 0/1.
    genres_oh = movies["genres"].replace({"(no genres listed)": ""}).str.get_dummies(sep="|")

    # Optionnel mais pratique : préfixer les colonnes pour éviter collisions / clarté
    genres_oh = genres_oh.add_prefix("genre__")

    # On garde movieId (clé) + colonnes genres
    movie_matrix = pd.concat([movies[["movieId", "title"]], genres_oh], axis=1)

    # --- 4) Construire la "user_matrix" (profil utilisateur moyen par genre) ---
    # Étape A : joindre ratings avec movie_matrix pour récupérer les genres des films notés
    # -> on obtient : userId, movieId, rating, timestamp, title, genre__Action, genre__Comedy, ...
    movie_ratings = ratings.merge(movie_matrix, on="movieId", how="inner")

    # Étape B : on ne garde que userId + colonnes genres
    # (On jette rating/timestamp/title car notre "profil" = moyenne des genres vus)
    # NB: Ça reproduit ton code CSV, mais version DB.
    cols_to_drop = ["movieId", "timestamp", "title", "rating"]
    cols_to_drop = [c for c in cols_to_drop if c in movie_ratings.columns]  # safe si timestamp absent
    movie_ratings = movie_ratings.drop(columns=cols_to_drop)

    # Étape C : moyenne par userId
    # -> si un user a vu 10 films dont 7 "Action", il aura une valeur élevée en genre__Action.
    user_matrix = movie_ratings.groupby("userId").mean(numeric_only=True)

    # --- 5) Sauvegarder les sorties ---
    # IMPORTANT :
    # - movie_matrix : on enlève "title" si tu veux strictement un "movieId + features"
    # - user_matrix : index=userId, donc index=True est OK (tu auras userId en 1ère colonne du CSV)
    movie_features_only = movie_matrix.drop(columns=["title"])

    movie_out = out_dir / "movie_matrix.csv"
    user_out = out_dir / "user_matrix.csv"

    movie_features_only.to_csv(movie_out, index=False)
    user_matrix.to_csv(user_out)  # index = userId

    _log(f"Saved: {movie_out}")
    _log(f"Saved: {user_out}")

    # --- 6) Mini rapport (super utile pour debug) ---
    _log(f"ratings rows: {len(ratings):,} | movies rows: {len(movies):,}")
    _log(f"movie_matrix shape: {movie_features_only.shape} (rows, cols)")
    _log(f"user_matrix shape: {user_matrix.shape} (users, genre-features)")
