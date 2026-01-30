# ------------------------------------------------------------
# SCRIPT : check_db.py
# ------------------------------------------------------------
# Objectif :
# Vérifier la qualité des données STOCKÉES DANS POSTGRESQL
# (serveur SQL local), AVANT d'autoriser la suite du pipeline.
#
# Ce script :
# - se connecte à PostgreSQL
# - exécute plusieurs requêtes SQL de contrôle
# - affiche ✅ ou ❌ pour chaque règle
# - ARRÊTE le pipeline si un problème est détecté
#
# IMPORTANT :
# - Ce script NE LIT PAS de fichier .db
# - Il interroge un SERVEUR PostgreSQL via SQL
# ------------------------------------------------------------

from __future__ import annotations

# os : permet de lire les variables d'environnement (PG_URL, PG_SCHEMA)
import os

# sys : permet de quitter le programme avec un code (0 = OK, 1 = erreur)
import sys

# create_engine : objet SQLAlchemy pour se connecter à PostgreSQL
# text : permet d'écrire des requêtes SQL "propres"
from sqlalchemy import create_engine, text

# SQLAlchemyError : permet d'attraper proprement les erreurs SQL
from sqlalchemy.exc import SQLAlchemyError


# ------------------------------------------------------------
# PARAMÈTRES PAR DÉFAUT
# ------------------------------------------------------------
# On évite "localhost" sur Windows (IPv6 ::1 peut poser problème)
DEFAULT_PG_URL = "postgresql+psycopg2://movie:movie@127.0.0.1:5432/movie_reco"

# Schéma SQL dans lequel sont stockées les tables
# (équivalent d'un "dossier" côté base de données)
DEFAULT_SCHEMA = "raw"


# ------------------------------------------------------------
# FONCTION run_check
# ------------------------------------------------------------
def run_check(conn, name: str, query: str) -> bool:
    """
    Exécute UNE règle de qualité.

    Paramètres :
    - conn  : connexion active à PostgreSQL
    - name  : nom lisible du check (pour l'affichage)
    - query : requête SQL qui retourne un COUNT(*)

    Principe :
    - La requête DOIT retourner un nombre
    - 0  → aucun problème → ✅
    - >0 → problèmes détectés → ❌
    """

    # On exécute la requête SQL
    # scalar() = on récupère la première valeur (COUNT)
    value = conn.execute(text(query)).scalar()

    # Si aucun problème détecté
    if value == 0:
        print(f"✅ {name}")
        return True

    # Sinon, on affiche le nombre d'erreurs
    else:
        print(f"❌ {name} → {value} problème(s) détecté(s)")
        return False


# ------------------------------------------------------------
# FONCTION PRINCIPALE
# ------------------------------------------------------------
def main() -> None:
    print("🔍 Démarrage des checks qualité sur PostgreSQL\n")

    # --------------------------------------------------------
    # 1) RÉCUPÉRATION DES PARAMÈTRES
    # --------------------------------------------------------
    # Si une variable d'environnement existe, on l'utilise.
    # Sinon, on prend la valeur par défaut.

    pg_url = os.getenv("PG_URL", DEFAULT_PG_URL)
    schema = os.getenv("PG_SCHEMA", DEFAULT_SCHEMA)

    # Construction des noms de tables COMPLETS
    # (schéma + nom de table)
    ratings_table = f"{schema}.raw_ratings"
    movies_table = f"{schema}.raw_movies"

    # Liste des résultats des checks (True / False)
    checks = []

    try:
        # ----------------------------------------------------
        # 2) CONNEXION À POSTGRESQL
        # ----------------------------------------------------
        # create_engine ne se connecte PAS encore.
        # Il prépare juste la connexion.
        engine = create_engine(pg_url)

        # engine.begin() :
        # - ouvre une connexion
        # - démarre une transaction
        # - ferme proprement à la fin du bloc
        with engine.begin() as conn:

            # ------------------------------------------------
            # CHECK 1 : ratings valides (entre 0 et 5)
            # ------------------------------------------------
            checks.append(run_check(
                conn,
                "Ratings dans [0,5]",
                f"""
                SELECT COUNT(*)
                FROM {ratings_table}
                WHERE rating < 0
                   OR rating > 5
                   OR rating IS NULL
                """
            ))

            # ------------------------------------------------
            # CHECK 2 : userId non NULL
            # ------------------------------------------------
            # Les guillemets sont nécessaires car les colonnes
            # ont été créées avec des majuscules (userId).
            checks.append(run_check(
                conn,
                "userId non NULL",
                f"""
                SELECT COUNT(*)
                FROM {ratings_table}
                WHERE "userId" IS NULL
                """
            ))

            # ------------------------------------------------
            # CHECK 3 : movieId non NULL
            # ------------------------------------------------
            checks.append(run_check(
                conn,
                "movieId non NULL",
                f"""
                SELECT COUNT(*)
                FROM {ratings_table}
                WHERE "movieId" IS NULL
                """
            ))

            # ------------------------------------------------
            # CHECK 4 : intégrité référentielle
            # ------------------------------------------------
            # Vérifie que chaque movieId présent dans ratings
            # existe aussi dans la table movies.
            checks.append(run_check(
                conn,
                "ratings.movieId existe dans movies",
                f"""
                SELECT COUNT(*)
                FROM {ratings_table} r
                LEFT JOIN {movies_table} m
                       ON r."movieId" = m."movieId"
                WHERE m."movieId" IS NULL
                """
            ))

            # ------------------------------------------------
            # CHECK 5 : doublons exacts
            # ------------------------------------------------
            # On cherche des triplets identiques :
            # (userId, movieId, timestamp)
            checks.append(run_check(
                conn,
                "Pas de doublons (userId, movieId, timestamp)",
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT "userId", "movieId", "timestamp", COUNT(*) AS c
                    FROM {ratings_table}
                    GROUP BY "userId", "movieId", "timestamp"
                    HAVING COUNT(*) > 1
                ) t
                """
            ))

    # --------------------------------------------------------
    # GESTION DES ERREURS SQL
    # --------------------------------------------------------
    except SQLAlchemyError as e:
        print("⛔ Erreur SQL ou connexion PostgreSQL impossible :")
        print(str(e))
        sys.exit(2)

    # --------------------------------------------------------
    # 3) RÉSUMÉ FINAL
    # --------------------------------------------------------
    print("\n📊 Résumé :")

    # Si TOUS les checks sont vrais → OK
    if all(checks):
        print("🎉 Tous les checks sont OK")
        sys.exit(0)

    # Sinon → on bloque le pipeline
    else:
        print("⛔ Échec des checks qualité — pipeline stoppé")
        sys.exit(1)


# ------------------------------------------------------------
# POINT D’ENTRÉE DU SCRIPT
# ------------------------------------------------------------
if __name__ == "__main__":
    main()

