import os
import sys
import pandas as pd
from sqlalchemy import create_engine

# Configuration
PG_URL = os.getenv("PG_URL", "postgresql+psycopg2://movie:movie@127.0.0.1:5432/movie_reco")
SNAPSHOT_PATH = os.path.abspath("data/training_set.parquet")

def create_snapshot():
    os.makedirs(os.path.dirname(SNAPSHOT_PATH), exist_ok=True)
    engine = create_engine(PG_URL)
    
    print(f"[START] Extraction du snapshot vers Parquet (Mode Chunk)...")

    try:
        # On ne sélectionne QUE ce qui est nécessaire
        query = 'SELECT "userId", "movieId", "rating" FROM raw.current_ratings'
        
        chunks = []
        total_rows = 0
        
        # On lit par paquets de 500,000 lignes
        for chunk in pd.read_sql(query, engine, chunksize=500000):
            # Optimisation immédiate des types AVANT d'accumuler en mémoire
            chunk["userId"] = chunk["userId"].astype("int32")
            chunk["movieId"] = chunk["movieId"].astype("int32")
            chunk["rating"] = chunk["rating"].astype("float32")
            
            chunks.append(chunk)
            total_rows += len(chunk)
            print(f"   -> Chunk traité : +{len(chunk)} lignes (Total: {total_rows})", end="\r")

        if total_rows == 0:
            print("\n⚠️ Aucune donnée trouvée dans la base.")
            return

        print(f"\n[INFO] Assemblage final ({len(chunks)} chunks)...")
        # On assemble tout le puzzle maintenant que les pièces sont légères
        full_df = pd.concat(chunks, ignore_index=True)
        
        print("[INFO] Écriture du fichier Parquet...")
        full_df.to_parquet(SNAPSHOT_PATH, index=False, compression='snappy')
        
        if os.path.exists(SNAPSHOT_PATH):
            size_mb = os.path.getsize(SNAPSHOT_PATH) / (1024*1024)
            print(f"[SUCCESS] Snapshot créé : {SNAPSHOT_PATH}")
            print(f"[INFO] Taille : {size_mb:.2f} MB | Lignes : {len(full_df)}")

    except Exception as e:
        print(f"\n[ERROR] Erreur lors de l'export : {e}")
        sys.exit(1)

if __name__ == "__main__":
    create_snapshot()