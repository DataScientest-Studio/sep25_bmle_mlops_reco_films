# src/ingestion/ingestion_movielens.py
# ------------------------------------------------------------
# OBJECTIF
# ------------------------------------------------------------
# Ce script remplace l'ingestion SQLite (fichier .db sur disque)
# par une ingestion DIRECTE dans PostgreSQL (serveur SQL local).
#
# Pourquoi ?
# - SQLite = base "embarquée" : accès via un fichier local (.sqlite)
# - PostgreSQL = base "serveur" : accès via localhost:5432 (port réseau)
#
# Résultat :
# - Une "vraie" base qui tourne comme un service.
# - Ton pipeline n'est plus dépendant d'un chemin local vers un .db.
#
# ------------------------------------------------------------
# PRÉREQUIS
# ------------------------------------------------------------
# 1) Lancer PostgreSQL en local (Docker recommandé)
#    docker compose up -d
#
# 2) Installer dépendances Python
#    pip install pandas sqlalchemy psycopg2-binary
#
# 3) Avoir les CSV dans data/raw/
#    movies.csv, ratings.csv, tags.csv, links.csv, genome-scores.csv, genome-tags.csv
#
# 4) Variable d'environnement possible (optionnel)
#    PG_URL=postgresql+psycopg2://movie:movie@localhost:5432/movie_reco
#    PG_SCHEMA=raw
#
# ------------------------------------------------------------
# TABLES CRÉÉES DANS POSTGRES
# ------------------------------------------------------------
# Schéma (namespace) : raw (par défaut)
#
# - raw.raw_movies
# - raw.raw_ratings
# - raw.raw_tags
# - raw.raw_links
# - raw.raw_genome_scores
# - raw.raw_genome_tags
#
# + table de suivi :
# - raw.ingestion_metadata
#
# ingestion_metadata contient :
# - table_name : nom logique (movies, ratings, ...)
# - row_count : nombre de lignes ingérées
# - column_count : nombre de colonnes ingérées
# - ingestion_date : date/heure de la dernière ingestion
# ------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


# Mapping "nom logique" -> "nom de fichier"
# Le nom logique sert à construire le nom de table raw_<name>.
CSV_FILES = {
    "movies": "movies.csv",
    "ratings": "ratings.csv",
    "tags": "tags.csv",
    "links": "links.csv",
    "genome_scores": "genome-scores.csv",
    "genome_tags": "genome-tags.csv",
}


def _get_postgres_engine(pg_url: str):
    """
    Crée un 'engine' SQLAlchemy.

    Un engine est un objet qui sait :
    - ouvrir des connexions à la base (pool de connexions)
    - exécuter des requêtes SQL
    - être utilisé par pandas.to_sql()

    Ici on utilise PostgreSQL via psycopg2.
    """
    return create_engine(pg_url)


def _ensure_schema_and_metadata_table(engine, schema: str) -> None:
    """
    - Crée le schéma si nécessaire (équivalent d'un dossier/namespace SQL)
    - Crée la table ingestion_metadata si elle n'existe pas

    Note :
    - Dans PostgreSQL, un "schema" (ex: raw) organise les tables.
      Ça évite de polluer 'public' et c'est plus propre en projet.
    """
    with engine.begin() as conn:
        # CREATE SCHEMA IF NOT EXISTS raw;
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema};"))

        # Crée la table de métadonnées si nécessaire.
        # BIGINT pour row_count : ça évite les soucis si la table grossit.
        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {schema}.ingestion_metadata (
                    table_name TEXT PRIMARY KEY,
                    row_count  BIGINT,
                    column_count INTEGER,
                    ingestion_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
        )


def _upsert_metadata(engine, schema: str, table_name_logic: str, row_count: int, column_count: int) -> None:
    """
    Met à jour ou insère une ligne dans ingestion_metadata.

    - SQLite utilisait : INSERT OR REPLACE
    - PostgreSQL utilise : INSERT ... ON CONFLICT ... DO UPDATE

    Ici, "table_name_logic" correspond à tes clés :
    movies / ratings / tags / etc.
    """
    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                INSERT INTO {schema}.ingestion_metadata(table_name, row_count, column_count, ingestion_date)
                VALUES (:table_name, :row_count, :column_count, CURRENT_TIMESTAMP)
                ON CONFLICT (table_name)
                DO UPDATE SET
                    row_count = EXCLUDED.row_count,
                    column_count = EXCLUDED.column_count,
                    ingestion_date = EXCLUDED.ingestion_date;
                """
            ),
            {
                "table_name": table_name_logic,
                "row_count": int(row_count),
                "column_count": int(column_count),
            },
        )


def _print_metadata(engine, schema: str) -> None:
    """
    Affiche un tableau de suivi ingestion_metadata.

    C'est l'équivalent de ton:
    meta = pd.read_sql("SELECT * FROM ingestion_metadata ...", conn)
    print(meta.to_string(...))
    """
    meta = pd.read_sql(
        f"SELECT * FROM {schema}.ingestion_metadata ORDER BY table_name;",
        con=engine,
    )
    print("\n[METADATA] ingestion_metadata")
    print(meta.to_string(index=False))


def ingest_movielens(
    raw_dir: str = "data/raw",
    pg_url: str | None = None,
    schema: str | None = None,
    if_exists: str = "replace",
) -> None:
    """
    Ingestion MovieLens -> PostgreSQL

    Paramètres :
    - raw_dir : dossier où se trouvent les CSV (data/raw)
    - pg_url  : string de connexion PostgreSQL (si None, on prend env PG_URL ou valeur par défaut)
    - schema  : schéma SQL cible (si None, on prend env PG_SCHEMA ou "raw")
    - if_exists : comportement si la table existe déjà :
        - "replace" : drop & recreate (simple pour dev / re-run)
        - "append"  : ajoute des lignes (utile si ingestion incrémentale)
        - "fail"    : refuse si table existe
    """
    raw_dir_path = Path(raw_dir)

    # 1) Vérifier la présence des fichiers CSV attendus
    missing = [fn for fn in CSV_FILES.values() if not (raw_dir_path / fn).exists()]
    if missing:
        raise FileNotFoundError(
            f"Fichiers manquants dans {raw_dir_path.resolve()} : {missing}\n"
            "👉 Vérifie que DVC a bien récupéré data/raw/ (dvc pull)."
        )

    # 2) Construire les paramètres de connexion
    #    On privilégie les variables d'environnement si disponibles.
    if pg_url is None:
        pg_url = os.getenv(
            "PG_URL",
            "postgresql+psycopg2://movie:movie@127.0.0.1:5432/movie_reco"
        )

    if schema is None:
        schema = os.getenv("PG_SCHEMA", "raw")

    # 3) Créer l'engine SQLAlchemy
    engine = _get_postgres_engine(pg_url)

    # 4) Préparer l'environnement SQL (schema + metadata)
    _ensure_schema_and_metadata_table(engine, schema)

    # 5) Boucle d'ingestion : lire CSV -> écrire table Postgres
    for logical_name, filename in CSV_FILES.items():
        csv_path = raw_dir_path / filename

        # Lecture CSV
        # low_memory=False évite des inférences de types "bizarres" par morceaux
        df = pd.read_csv(csv_path, low_memory=False)

        # Nom de table final (comme ton SQLite): raw_<name>
        table_name = f"raw_{logical_name}"

        # Écriture dans PostgreSQL
        #
        # df.to_sql(...) va :
        # - créer la table si elle n'existe pas
        # - pousser les données en INSERT
        # - gérer "replace" en drop + create + insert
        #
        # method="multi" + chunksize accélère en envoyant des INSERT groupés.
        df.to_sql(
            name=table_name,
            con=engine,
            schema=schema,
            if_exists=if_exists,
            index=False,
            method="multi",
            chunksize=5000,
        )

        print(
            f"[INGEST] {filename} -> {schema}.{table_name} "
            f"({df.shape[0]} lignes, {df.shape[1]} colonnes)"
        )

        # 6) Mettre à jour ingestion_metadata
        _upsert_metadata(
            engine=engine,
            schema=schema,
            table_name_logic=logical_name,
            row_count=df.shape[0],
            column_count=df.shape[1],
        )

    # 7) Afficher metadata (résumé final)
    _print_metadata(engine, schema)


if __name__ == "__main__":
    # Lancement par défaut (compatible avec ton usage actuel)
    ingest_movielens()
