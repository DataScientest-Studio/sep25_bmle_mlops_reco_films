import os
import requests
import zipfile
import io
import shutil
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine

# --- CONFIGURATION 20M ---
# URL officielle du dataset 20M
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
RAW_DIR = "data/raw"

# Connexion SQL
PG_URL = os.getenv("PG_URL", "postgresql+psycopg2://movie:movie@127.0.0.1:5432/movie_reco")

# Mapping Fichiers 20M -> Tables SQL
FILES = {
    "movies.csv": "raw_movies",
    "ratings.csv": "raw_ratings",
    "tags.csv": "raw_tags",
    "links.csv": "raw_links",
    "genome-scores.csv": "raw_genome_scores",
    "genome-tags.csv": "raw_genome_tags"
}

def download_data():
    """Télécharge et extrait le dataset MovieLens 20M."""
    print(f"🚀 [Download] Récupération du dataset 20M depuis {MOVIELENS_URL}...")
    
    # 1. Nettoyage dossier existant
    if os.path.exists(RAW_DIR):
        shutil.rmtree(RAW_DIR)
    os.makedirs(RAW_DIR, exist_ok=True)

    # 2. Téléchargement
    try:
        r = requests.get(MOVIELENS_URL, stream=True)
        r.raise_for_status()
        
        # Téléchargement dans un fichier temporaire (car le fichier fait ~190Mo)
        zip_path = "data/ml-20m.zip"
        print("   ... Téléchargement en cours (patience)...")
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("📦 [Download] Extraction du ZIP...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall("data/temp")
        
        # 3. Déplacement des fichiers
        # --- CORRECTION ICI : Le dossier s'appelle 'ml-20m' ---
        extracted_folder = os.path.join("data/temp", "ml-20m") 
        
        if not os.path.exists(extracted_folder):
             # Fallback de sécurité : Si le nom change, on prend le premier dossier trouvé
             subfolders = [f.path for f in os.scandir("data/temp") if f.is_dir()]
             if subfolders:
                 extracted_folder = subfolders[0]
                 print(f"⚠️ Nom de dossier inattendu, utilisation de : {extracted_folder}")

        for file in os.listdir(extracted_folder):
            shutil.move(os.path.join(extracted_folder, file), os.path.join(RAW_DIR, file))
        
        # Nettoyage
        shutil.rmtree("data/temp")
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
        print(f"✅ [Download] Dataset 20M prêt dans {RAW_DIR}")
        
    except Exception as e:
        print(f"❌ Erreur Téléchargement : {e}")
        # Nettoyage en cas d'erreur
        if os.path.exists("data/temp"): shutil.rmtree("data/temp")
        raise e

def load_to_sql():
    """Charge les CSV vers PostgreSQL."""
    engine = create_engine(PG_URL)
    now = datetime.now()
    
    print(f"💾 [SQL] Démarrage insertion massive (20M lignes)...")

    for filename, table_name in FILES.items():
        filepath = os.path.join(RAW_DIR, filename)
        if not os.path.exists(filepath):
            continue 

        print(f"   📥 Traitement de {filename}...")
        
        try:
            # Chunk size optimisé pour 20M
            chunk_size = 100000 
            total_rows = 0
            
            for i, chunk in enumerate(pd.read_csv(filepath, chunksize=chunk_size)):
                chunk['ingested_at'] = now
                chunk.to_sql(table_name, engine, schema="raw", if_exists='append', index=False)
                
                total_rows += len(chunk)
                if i % 10 == 0:
                    print(f"      ... {total_rows} lignes insérées")
            
            print(f"   ✅ {table_name} terminé ({total_rows} lignes).")
        except Exception as e:
            print(f"   ❌ Erreur SQL sur {filename} : {e}")

def ingest_movielens():
    download_data()
    load_to_sql()

if __name__ == "__main__":
    ingest_movielens()