# src/ingestion/create_snapshot.py
import os
from sqlalchemy import create_engine, text

# Configuration
PG_URL = os.getenv("PG_URL", "postgresql+psycopg2://movie:movie@127.0.0.1:5432/movie_reco")
SNAPSHOT_PATH = os.path.abspath("data/training_set.csv")

def create_snapshot():
    # S'assurer que le dossier existe
    os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)
    
    engine = create_engine(PG_URL)
    
    print(f"📸 Extraction du snapshot (20M+ lignes) via PostgreSQL COPY...")

    copy_sql = """
    COPY (
        SELECT * FROM raw.current_ratings
    ) TO STDOUT WITH CSV HEADER;
    """

    try:
        # On récupère la connexion brute
        raw_conn = engine.raw_connection()
        try:
            # On accède au curseur psycopg2 sous-jacent
            with open(SNAPSHOT_PATH, 'w', encoding='utf-8') as f:
                cursor = raw_conn.cursor()
                cursor.copy_expert(copy_sql, f)
                cursor.close()
            
            print(f"✅ Snapshot créé avec succès : {SNAPSHOT_PATH}")
            size_mb = os.path.getsize(SNAPSHOT_PATH) / (1024*1024)
            print(f"ℹ️ Taille du fichier : {size_mb:.2f} MB")
            
        finally:
            # Important : toujours refermer la connexion brute manuellement
            raw_conn.close()

    except Exception as e:
        print(f"❌ Erreur lors de l'export : {e}")

if __name__ == "__main__":
    create_snapshot()