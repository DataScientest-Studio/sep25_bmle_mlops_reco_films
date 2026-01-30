# ------------------------------------------------------------
# FICHIER : src/data/make_dataset.py
# ------------------------------------------------------------
# OBJECTIF (très simple) :
# - Lire des tables "raw" depuis PostgreSQL (serveur SQL local)
# - Sortir des CSV "propres" dans data/processed/
#
# Pourquoi faire ça ?
# - PostgreSQL = stockage opérationnel (raw)
# - data/processed = artefacts de pipeline versionnables avec DVC
#
# IMPORTANT :
# - Ce script ne touche pas à un fichier .db
# - Il parle au serveur PostgreSQL via une URL PG_URL
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine


# ------------------------------------------------------------
# 1) CONFIG SIMPLE (pas d'optionnel, pas d'arguments)
# ------------------------------------------------------------

# URL de connexion PostgreSQL.
# On force 127.0.0.1 (IPv4) car sous Windows, "localhost" peut pointer sur IPv6 (::1)
PG_URL = "postgresql+psycopg2://movie:movie@127.0.0.1:5432/movie_reco"

# Schéma où sont les tables raw.* dans PostgreSQL
SCHEMA = "raw"

# Dossier de sortie (sur ton disque, en local)
OUT_DIR = Path("data/processed")


def main() -> None:
    # --------------------------------------------------------
    # 2) Préparer le dossier de sortie
    # --------------------------------------------------------
    # mkdir(parents=True) = crée aussi les dossiers parents si besoin
    # exist_ok=True = ne plante pas si le dossier existe déjà
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 3) Se connecter à PostgreSQL
    # --------------------------------------------------------
    # create_engine crée un "moteur" SQL : c'est l'objet qui sait se connecter
    engine = create_engine(PG_URL)

    # --------------------------------------------------------
    # 4) Lire les tables raw depuis Postgres
    # --------------------------------------------------------
    # On lit seulement les colonnes utiles.
    #
    # ⚠️ Détail important PostgreSQL :
    # - Les colonnes "userId" / "movieId" / "timestamp" contiennent des majuscules.
    # - Dans PostgreSQL, si un nom contient des majuscules, on doit le mettre entre guillemets.
    #
    # Donc on écrit : "userId" et non userId.
    movies = pd.read_sql(
        f'SELECT "movieId", title, genres FROM {SCHEMA}.raw_movies;',
        con=engine
    )

    ratings = pd.read_sql(
        f'SELECT "userId", "movieId", rating, "timestamp" FROM {SCHEMA}.raw_ratings;',
        con=engine
    )

    # --------------------------------------------------------
    # 5) Nettoyage minimal (= make_dataset)
    # --------------------------------------------------------
    # L'idée ici est de produire des CSV propres, stables, réutilisables.

    # a) Nettoyage movies
    # - movieId doit être int
    # - title = string
    # - genres = string (remplacer NaN)
    movies["movieId"] = movies["movieId"].astype(int)
    movies["title"] = movies["title"].astype(str)
    movies["genres"] = movies["genres"].fillna("").astype(str)

    # b) Nettoyage ratings
    # - userId / movieId int
    # - rating float
    # - timestamp int (ou on le garde tel quel)
    # - filtrer ratings invalides (sécurité)
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    # Ici on supprime les ratings hors [0,5] (normalement déjà OK grâce aux checks)
    ratings = ratings[(ratings["rating"] >= 0) & (ratings["rating"] <= 5)]

    # --------------------------------------------------------
    # 6) Sauvegarder en CSV dans data/processed/
    # --------------------------------------------------------
    movies_out = OUT_DIR / "movies_clean.csv"
    ratings_out = OUT_DIR / "ratings_clean.csv"

    # index=False : on ne veut pas l’index pandas dans le CSV
    movies.to_csv(movies_out, index=False)
    ratings.to_csv(ratings_out, index=False)

    # --------------------------------------------------------
    # 7) Logs simples (pour comprendre ce qui a été produit)
    # --------------------------------------------------------
    print("✅ make_dataset terminé")
    print(f"   - {movies_out}  -> {movies.shape[0]} lignes, {movies.shape[1]} colonnes")
    print(f"   - {ratings_out} -> {ratings.shape[0]} lignes, {ratings.shape[1]} colonnes")


# ------------------------------------------------------------
# 8) Point d'entrée du script
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
