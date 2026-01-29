from __future__ import annotations

import sqlite3
from pathlib import Path
import pandas as pd

CSV_FILES = {
    "movies": "movies.csv",
    "ratings": "ratings.csv",
    "tags": "tags.csv",
    "links": "links.csv",
    "genome_scores": "genome-scores.csv",
    "genome_tags": "genome-tags.csv",
}

def ingest_movielens(raw_dir="data/raw", db_path="data/db/staging.sqlite", if_exists="replace") -> None:
    raw_dir = Path(raw_dir)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    missing = [fn for fn in CSV_FILES.values() if not (raw_dir / fn).exists()]
    if missing:
        raise FileNotFoundError(f"Fichiers manquants dans {raw_dir.resolve()}: {missing}")

    with sqlite3.connect(db_path) as conn:
        for name, fn in CSV_FILES.items():
            df = pd.read_csv(raw_dir / fn, low_memory=False)
            df.to_sql(f"raw_{name}", conn, if_exists=if_exists, index=False)
            print(f"[INGEST] {fn} -> raw_{name} ({df.shape[0]} lignes, {df.shape[1]} colonnes)")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_metadata (
                table_name TEXT PRIMARY KEY,
                row_count INTEGER,
                column_count INTEGER,
                ingestion_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        for name in CSV_FILES.keys():
            conn.execute(f"""
                INSERT OR REPLACE INTO ingestion_metadata(table_name, row_count, column_count)
                SELECT
                    '{name}',
                    COUNT(*),
                    (SELECT COUNT(*) FROM pragma_table_info('raw_{name}'))
                FROM raw_{name}
            """)

        conn.commit()

        meta = pd.read_sql("SELECT * FROM ingestion_metadata ORDER BY table_name", conn)
        print(meta.to_string(index=False))

if __name__ == "__main__":
    ingest_movielens()
