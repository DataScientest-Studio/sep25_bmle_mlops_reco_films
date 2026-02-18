import os
import sys
import requests
import zipfile
import shutil
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine

# FORCER LE CHEMIN POUR DOCKER
current_dir = os.path.dirname(os.path.abspath(__file__)) # src/ingestion
project_root = os.path.dirname(current_dir) # src
sys.path.append(project_root)

from ingestion.init_db import init_database

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
RAW_DIR = "data/raw"
PG_URL = os.getenv("PG_URL", "postgresql+psycopg2://movie:movie@127.0.0.1:5432/movie_reco")

FILES = {
    "movies.csv": "raw_movies",
    "ratings.csv": "raw_ratings",
    "tags.csv": "raw_tags",
    "links.csv": "raw_links",
    "genome-scores.csv": "raw_genome_scores",
    "genome-tags.csv": "raw_genome_tags"
}

def download_data():
    if os.path.exists(RAW_DIR): shutil.rmtree(RAW_DIR)
    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = "data/ml-20m.zip"
    r = requests.get(MOVIELENS_URL, stream=True)
    with open(zip_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall("data/temp")
    extracted_folder = os.path.join("data/temp", "ml-20m") 
    if not os.path.exists(extracted_folder):
         extracted_folder = [f.path for f in os.scandir("data/temp") if f.is_dir()][0]
    for file in os.listdir(extracted_folder):
        shutil.move(os.path.join(extracted_folder, file), os.path.join(RAW_DIR, file))
    shutil.rmtree("data/temp")
    if os.path.exists(zip_path): os.remove(zip_path)

def load_to_sql():
    engine = create_engine(PG_URL)
    now = datetime.now()
    for filename, table_name in FILES.items():
        filepath = os.path.join(RAW_DIR, filename)
        if not os.path.exists(filepath): continue 
        for chunk in pd.read_csv(filepath, chunksize=100000):
            chunk['ingested_at'] = now
            chunk.to_sql(table_name, engine, schema="raw", if_exists='append', index=False)

def ingest_movielens():
    init_database()
    download_data()
    load_to_sql()

if __name__ == "__main__":
    ingest_movielens()