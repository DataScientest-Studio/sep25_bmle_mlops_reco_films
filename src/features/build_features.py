# ------------------------------------------------------------
# FICHIER : src/features/build_features.py
# ------------------------------------------------------------
# OBJECTIF :
# - Lire les CSV propres produits par make_dataset.py
# - Construire des features "content-based" simples :
#   1) movie_matrix.csv : movieId + one-hot des genres
#   2) user_matrix.csv  : userId + moyenne des genres vus
#
# IMPORTANT :
# - Ce script ne parle pas à la base PostgreSQL
# - Il travaille sur des fichiers data/processed/
# - Ces fichiers peuvent être versionnés avec DVC
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
import pandas as pd


# ------------------------------------------------------------
# 1) CONFIG SIMPLE (pas d'optionnel)
# ------------------------------------------------------------

# Dossier où make_dataset.py a écrit ses fichiers
PROCESSED_DIR = Path("data/processed")

# Fichiers d'entrée
MOVIES_CSV = PROCESSED_DIR / "movies_clean.csv"
RATINGS_CSV = PROCESSED_DIR / "ratings_clean.csv"

# Fichiers de sortie
MOVIE_MATRIX_OUT = PROCESSED_DIR / "movie_matrix.csv"
USER_MATRIX_OUT = PROCESSED_DIR / "user_matrix.csv"


def main() -> None:
    # --------------------------------------------------------
    # 2) Vérifier que les fichiers existent
    # --------------------------------------------------------
    if not MOVIES_CSV.exists():
        raise FileNotFoundError(f"Fichier manquant : {MOVIES_CSV} (lance make_dataset.py avant)")

    if not RATINGS_CSV.exists():
        raise FileNotFoundError(f"Fichier manquant : {RATINGS_CSV} (lance make_dataset.py avant)")

    # --------------------------------------------------------
    # 3) Charger les CSV
    # --------------------------------------------------------
    movies = pd.read_csv(MOVIES_CSV)
    ratings = pd.read_csv(RATINGS_CSV)

    # --------------------------------------------------------
    # 4) Sanity checks : colonnes indispensables
    # --------------------------------------------------------
    # Si une colonne manque, on préfère planter immédiatement avec un message clair.
    for col in ["movieId", "title", "genres"]:
        if col not in movies.columns:
            raise ValueError(f"Colonne manquante dans movies_clean.csv : {col}")

    for col in ["userId", "movieId", "rating"]:
        if col not in ratings.columns:
            raise ValueError(f"Colonne manquante dans ratings_clean.csv : {col}")

    # --------------------------------------------------------
    # 5) Construire movie_matrix (one-hot genres)
    # --------------------------------------------------------
    # Exemple genres MovieLens :
    # "Adventure|Animation|Children|Comedy|Fantasy"
    #
    # get_dummies(sep="|") crée une colonne par genre.
    # Chaque colonne vaut 0/1 (absent / présent).
    genres_oh = (
        movies["genres"]
        .fillna("")
        .replace({"(no genres listed)": ""})
        .astype(str)
        .str.get_dummies(sep="|")
    )

    # Pour rendre clair que ces colonnes sont des genres
    genres_oh = genres_oh.add_prefix("genre__")

    # movie_matrix = movieId + colonnes de genres
    movie_matrix = pd.concat([movies[["movieId"]], genres_oh], axis=1)

    # --------------------------------------------------------
    # 6) Construire user_matrix (profil utilisateur)
    # --------------------------------------------------------
    # On associe chaque rating à ses genres via un merge sur movieId
    movie_ratings = ratings.merge(movie_matrix, on="movieId", how="inner")

    # On garde uniquement userId + colonnes genres
    # (on jette rating car on fait un profil "vu", pas pondéré)
    cols_to_drop = ["movieId", "rating"]
    cols_to_drop = [c for c in cols_to_drop if c in movie_ratings.columns]
    movie_ratings = movie_ratings.drop(columns=cols_to_drop)

    # Maintenant movie_ratings ressemble à :
    # userId, genre__Action, genre__Comedy, genre__Drama, ...
    #
    # groupby("userId").mean() :
    # - moyenne des colonnes genres pour chaque utilisateur
    # - donc plus un user voit des films Action, plus genre__Action sera élevé
    user_matrix = movie_ratings.groupby("userId").mean(numeric_only=True)

    # --------------------------------------------------------
    # 7) Sauvegarder les résultats
    # --------------------------------------------------------
    movie_matrix.to_csv(MOVIE_MATRIX_OUT, index=False)
    user_matrix.to_csv(USER_MATRIX_OUT)  # index = userId

    # --------------------------------------------------------
    # 8) Logs simples
    # --------------------------------------------------------
    print("✅ build_features terminé")
    print(f"   - {MOVIE_MATRIX_OUT} -> {movie_matrix.shape[0]} lignes, {movie_matrix.shape[1]} colonnes")
    print(f"   - {USER_MATRIX_OUT}  -> {user_matrix.shape[0]} users, {user_matrix.shape[1]} features")


if __name__ == "__main__":
    main()
