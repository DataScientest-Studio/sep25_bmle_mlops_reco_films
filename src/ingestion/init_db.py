# src/init_db.py
import os
from sqlalchemy import create_engine, text

PG_URL = os.getenv("PG_URL", "postgresql+psycopg2://movie:movie@127.0.0.1:5432/movie_reco")

def init_database():
    engine = create_engine(PG_URL)
    with engine.begin() as conn:
        # 1. On nettoie tout
        conn.execute(text("DROP SCHEMA IF EXISTS raw CASCADE;"))
        conn.execute(text("CREATE SCHEMA raw;"))

        print("🏗️  Création des tables (Mode Historique - Append Only)...")

        # Table RATINGS : On stocke TOUT l'historique.
        # Notez l'absence de PRIMARY KEY sur (userId, movieId).
        # On ajoute une colonne 'ingested_at' pour savoir QUAND on a reçu la donnée.
        conn.execute(text("""
            CREATE TABLE raw.raw_ratings (
                "userId" BIGINT,
                "movieId" BIGINT,
                "rating" FLOAT,
                "timestamp" BIGINT,
                "ingested_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

        # Table MOVIES
        conn.execute(text("""
            CREATE TABLE raw.raw_movies (
                "movieId" BIGINT,
                "title" TEXT,
                "genres" TEXT,
                "ingested_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        
        # (On fait pareil pour tags, links, etc. si besoin)

        print("👀 Création des VUES (Ce que l'API va utiliser)...")

        # VUE MAGIQUE : current_ratings
        # Cette vue ne garde que la note la plus récente pour chaque couple User/Movie.
        # L'API lira "raw.current_ratings" et aura toujours la donnée fraîche.
        conn.execute(text("""
            CREATE VIEW raw.current_ratings AS
            SELECT DISTINCT ON ("userId", "movieId") *
            FROM raw.raw_ratings
            ORDER BY "userId", "movieId", "ingested_at" DESC;
        """))

        conn.execute(text("""
            CREATE VIEW raw.current_movies AS
            SELECT DISTINCT ON ("movieId") *
            FROM raw.raw_movies
            ORDER BY "movieId", "ingested_at" DESC;
        """))

    print("✅ Base prête : Historique complet + Vue actuelle.")

if __name__ == "__main__":
    init_database()