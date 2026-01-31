from pathlib import Path
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle

# Racine du projet
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"


def train_model(movie_matrix: pd.DataFrame) -> NearestNeighbors:
    """
    Entraîne un modèle kNN item-item à partir de movie_matrix.
    """
    if "movieId" not in movie_matrix.columns:
        raise ValueError("movie_matrix doit contenir une colonne 'movieId'")

    X = movie_matrix.drop("movieId", axis=1)

    nbrs = NearestNeighbors(
        n_neighbors=20,
        algorithm="brute",
        metric="cosine"
    )
    nbrs.fit(X)

    return nbrs


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    return pd.read_csv(path)


def save_model(model, path: Path):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    movie_matrix_path = DATA_DIR / "movie_matrix.csv"

    movie_matrix = load_features(movie_matrix_path)
    model = train_model(movie_matrix)
    save_model(model, MODEL_PATH)

    print("✅ Training terminé")
    print(f"Modèle sauvegardé dans : {MODEL_PATH}")
    print(f"Nombre de films : {movie_matrix.shape[0]}")
