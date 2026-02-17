# src/ingestion/create_snapshot.py
import os
import sys
import pandas as pd
from sqlalchemy import create_engine

# Configuration
PG_URL = os.getenv("PG_URL", "postgresql+psycopg2://movie:movie@127.0.0.1:5432/movie_reco")
# ⚠️ Changement d'extension ici (.parquet)
SNAPSHOT_PATH = os.path.abspath("data/training_set.parquet")

def create_snapshot():
    os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)
    
    engine = create_engine(PG_URL)
    
    print(f"[START] Extraction du snapshot vers Parquet...")

    try:
        # On lit via SQL (Pandas va gérer le type mapping)
        # "Chunksize" est une option si tu manques de RAM, mais pour <30M lignes,
        # le chargement direct est souvent plus rapide et tient dans 8-16Go RAM.
        query = 'SELECT "userId", "movieId", "rating" FROM raw.current_ratings'
        
        # Astuce : On force les types numpy pour réduire la RAM immédiatemment
        # int32 suffit pour les IDs jusqu'à 2 milliards
        # float32 suffit pour les notes (0.5, 1.0...)
        df = pd.read_sql(query, engine)
        
        if df.empty:
            print("⚠️ Aucune donnée trouvée dans la base.")
            return

        # Optimisation types (Réduit la RAM de 50%)
        df["userId"] = df["userId"].astype("int32")
        df["movieId"] = df["movieId"].astype("int32")
        df["rating"] = df["rating"].astype("float32")

        # Écriture Parquet (compression snappy par défaut)
        df.to_parquet(SNAPSHOT_PATH, index=False)
        
        if os.path.exists(SNAPSHOT_PATH):
            size_mb = os.path.getsize(SNAPSHOT_PATH) / (1024*1024)
            print(f"[SUCCESS] Snapshot Parquet créé : {SNAPSHOT_PATH}")
            print(f"[INFO] Taille du fichier : {size_mb:.2f} MB")
            print(f"[INFO] Lignes : {len(df)}")

    except Exception as e:
        print(f"[ERROR] Erreur lors de l'export : {e}")
        sys.exit(1)

if __name__ == "__main__":
    create_snapshot()