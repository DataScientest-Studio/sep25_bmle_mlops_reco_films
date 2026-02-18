# src/init_db.py
import os
from sqlalchemy import create_engine, text

PG_URL = os.getenv("PG_URL", "postgresql+psycopg2://movie:movie@127.0.0.1:5432/movie_reco")

def init_database():
    # AUTOCOMMIT est vital ici pour forcer la création du schéma immédiatement
    engine = create_engine(PG_URL, isolation_level="AUTOCOMMIT")
    
    with engine.connect() as conn:
        print("🏗️  Vérification de la structure de la base de données...")
        
        # 1. Création du schéma 'raw'
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS raw;"))

        # 2. Création des tables
        tables = {
            "raw_ratings": '"userId" BIGINT, "movieId" BIGINT, rating FLOAT, timestamp BIGINT',
            "raw_movies": '"movieId" BIGINT, title TEXT, genres TEXT',
            "raw_tags": '"userId" BIGINT, "movieId" BIGINT, tag TEXT, timestamp BIGINT',
            "raw_links": '"movieId" BIGINT, "imdbId" BIGINT, "tmdbId" FLOAT',
            "raw_genome_scores": '"movieId" BIGINT, "tagId" BIGINT, relevance FLOAT',
            "raw_genome_tags": '"tagId" BIGINT, tag TEXT'
        }

        for table_name, columns in tables.items():
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS raw.{table_name} (
                    {columns},
                    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))

        print("👀 Mise à jour des VUES...")
        conn.execute(text("""
            CREATE OR REPLACE VIEW raw.current_ratings AS
            SELECT DISTINCT ON ("userId", "movieId") *
            FROM raw.raw_ratings
            ORDER BY "userId", "movieId", ingested_at DESC;
        """))

        conn.execute(text("""
            CREATE OR REPLACE VIEW raw.current_movies AS
            SELECT DISTINCT ON ("movieId") *
            FROM raw.raw_movies
            ORDER BY "movieId", ingested_at DESC;
        """))

    print("✅ Base de données prête.")

if __name__ == "__main__":
    init_database()