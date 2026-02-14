# src/ingestion/create_snapshot.py
import os
import sys
from sqlalchemy import create_engine

# Configuration
PG_URL = os.getenv("PG_URL", "postgresql+psycopg2://movie:movie@127.0.0.1:5432/movie_reco")
SNAPSHOT_PATH = os.path.abspath("data/training_set.csv")

def create_snapshot():
    # 1. S'assurer que le dossier existe
    os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)
    
    engine = create_engine(PG_URL)
    
    # MODIFICATION : Suppression de l'émoji 📸 pour éviter le crash Windows
    print(f"[START] Extraction du snapshot (20M+ lignes) via PostgreSQL COPY...")

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
            
            # MODIFICATION : Suppression des émojis ✅ et ℹ️
            print(f"[SUCCESS] Snapshot créé avec succès : {SNAPSHOT_PATH}")
            
            if os.path.exists(SNAPSHOT_PATH):
                size_mb = os.path.getsize(SNAPSHOT_PATH) / (1024*1024)
                print(f"[INFO] Taille du fichier : {size_mb:.2f} MB")
            
        finally:
            # Important : toujours refermer la connexion brute manuellement
            raw_conn.close()

    except Exception as e:
        # MODIFICATION : Suppression de l'émoji ❌
        print(f"[ERROR] Erreur lors de l'export : {e}")
        # On relève l'erreur pour que le script parent (API) sache qu'il y a eu un échec
        sys.exit(1)

if __name__ == "__main__":
    # Sécurité supplémentaire pour l'encodage sous Windows
    sys.stdout.reconfigure(encoding='utf-8')
    create_snapshot()