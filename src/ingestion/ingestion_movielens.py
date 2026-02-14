# src/ingestion/ingestion_movielens.py
import os
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text

# URL de connexion
PG_URL = os.getenv("PG_URL", "postgresql+psycopg2://movie:movie@127.0.0.1:5432/movie_reco")

# Fichiers à ingérer
FILES = {
    "movies.csv": "raw_movies",
    "ratings.csv": "raw_ratings",
    "tags.csv": "raw_tags",
    "links.csv": "raw_links",
    "genome-scores.csv": "raw_genome_scores",
    "genome-tags.csv": "raw_genome_tags"
}

def ingest_movielens():
    engine = create_engine(PG_URL)
    raw_dir = "data/raw"
    now = datetime.now()

    print(f"🚀 Démarrage ingestion : {now}")

    for filename, table_name in FILES.items():
        filepath = os.path.join(raw_dir, filename)
        if not os.path.exists(filepath):
            print(f"⚠️ Fichier introuvable : {filename}")
            continue
            
        print(f"📥 Lecture de {filename}...")
        
        # Compteur pour le suivi
        chunk_number = 0
        total_rows = 0
        
        # Lecture par morceaux (Chunking)
        chunk_size = 50000
        
        try:
            for chunk in pd.read_csv(filepath, chunksize=chunk_size):
                chunk_number += 1
                
                # 1. Ajout Timestamp
                chunk['ingested_at'] = now
                
                # 2. Insertion SQL
                chunk.to_sql(
                    table_name, 
                    engine, 
                    schema="raw", 
                    if_exists='append', 
                    index=False
                )
                
                total_rows += len(chunk)
                # FEEDBACK VISUEL : On affiche un point ou un message tous les chunks
                print(f"   ... Chunk {chunk_number} inséré ({total_rows} lignes au total)")
                
            print(f"✅ {table_name} : Terminé avec succès ({total_rows} lignes).")
            
        except Exception as e:
            print(f"❌ Erreur sur {filename} : {e}")

if __name__ == "__main__":
    ingest_movielens()