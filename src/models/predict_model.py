# src/models/predict_model.py

from pathlib import Path
import pandas as pd
import pickle
import json


# -----------------------------
# 1) Chemins du projet
# -----------------------------
# On remonte à la racine du repo :
# .../src/models/predict_model.py -> parents[2] = racine du projet
ROOT = Path(__file__).resolve().parents[2]

# Dossier où sont les fichiers produits par build_features
DATA_DIR = ROOT / "data" / "processed"

# Fichier de features (matrice film x utilisateurs/features)
MOVIE_MATRIX_PATH = DATA_DIR / "movie_matrix.csv"

# Optionnel : pour afficher des titres de films (movieId -> title)
MOVIES_PATH = DATA_DIR / "movies_clean.csv"

# Modèle entraîné (produit par src/models/train_model.py)
MODEL_PATH = ROOT / "models" / "model.pkl"


# -----------------------------
# 2) Fonctions utilitaires
# -----------------------------
def load_model(model_path: Path = MODEL_PATH):
    """
    Charge le modèle entraîné (NearestNeighbors picklé).
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable: {model_path}. "
            f"Lance d'abord: python src/models/train_model.py"
        )
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_movie_matrix(path: Path = MOVIE_MATRIX_PATH) -> pd.DataFrame:
    """
    Charge movie_matrix.csv (doit contenir une colonne 'movieId').
    """
    if not path.exists():
        raise FileNotFoundError(f"Introuvable: {path}")

    df = pd.read_csv(path)

    if "movieId" not in df.columns:
        raise ValueError("movie_matrix.csv doit contenir une colonne 'movieId'")

    return df


def load_title_map(movies_path: Path = MOVIES_PATH) -> dict[int, str]:
    """
    Charge un mapping movieId -> title si movies_clean.csv est présent et conforme.
    Si le fichier n'existe pas ou si les colonnes ne correspondent pas, on renvoie {}.
    """
    if not movies_path.exists():
        return {}

    movies = pd.read_csv(movies_path)

    # On vérifie les colonnes minimales
    if "movieId" not in movies.columns or "title" not in movies.columns:
        return {}

    # On sécurise les types (movieId en int, title en str)
    movie_ids = movies["movieId"].astype(int)
    titles = movies["title"].astype(str)

    return dict(zip(movie_ids, titles))


# -----------------------------
# 3) Fonction principale de prédiction
# -----------------------------
def recommend_similar_movies(movie_id: int, top_n: int = 10) -> dict:
    """
    Recommande des films similaires à 'movie_id' via kNN item-item.

    Important :
    - On utilise movie_matrix.csv (issus de build_features)
    - On charge models/model.pkl (modèle entraîné)
    - On renvoie un dict (JSON-friendly) => parfait pour l'API

    Retour :
    {
      "movieId": ...,
      "top_n": ...,
      "query_title": "...",            # si disponible
      "recommendations": [
        {"movieId": ..., "distance": ..., "title": "..."},
        ...
      ]
    }
    """
    # Charger les features et le modèle
    movie_matrix = load_movie_matrix(MOVIE_MATRIX_PATH)
    model = load_model(MODEL_PATH)

    # Charger les titres si possible (sinon dict vide)
    title_map = load_title_map(MOVIES_PATH)

    # On prépare X exactement comme au training :
    # training = movie_matrix.drop("movieId", axis=1)
    X = movie_matrix.drop("movieId", axis=1)

    # Retrouver l'index (ligne) correspondant au movieId demandé
    # (movie_matrix contient une colonne movieId)
    matches = movie_matrix.index[movie_matrix["movieId"] == movie_id].tolist()
    if not matches:
        raise ValueError(f"movieId {movie_id} introuvable dans {MOVIE_MATRIX_PATH}")

    row_idx = matches[0]

    # On récupère un vecteur de requête (1 ligne), sous forme DataFrame
    # (kneighbors attend un tableau 2D)
    query_vec = X.iloc[[row_idx]]

    # On demande top_n + 1 voisins pour inclure le film lui-même
    # (souvent distance 0 avec lui-même), puis on l'exclut.
    distances, indices = model.kneighbors(query_vec, n_neighbors=top_n + 1)

    # sklearn renvoie des arrays 2D (shape: 1 x k)
    distances = distances[0].tolist()
    indices = indices[0].tolist()

    # Construire la liste des recommandations
    recs = []
    for d, idx in zip(distances, indices):
        # idx est un index de ligne dans movie_matrix
        rec_movie_id = int(movie_matrix.loc[idx, "movieId"])

        # On retire le film de départ (le plus proche, distance ~0)
        if rec_movie_id == movie_id:
            continue

        item = {
            "movieId": rec_movie_id,
            "distance": float(d),
        }

        # Ajouter un titre si on a movies_clean.csv
        if title_map:
            item["title"] = title_map.get(rec_movie_id, None)

        recs.append(item)

    # On coupe à top_n (au cas où l'exclusion a décalé)
    recs = recs[:top_n]

    # Construire un résultat JSON-friendly
    result = {
        "movieId": int(movie_id),
        "top_n": int(top_n),
        "recommendations": recs,
    }

    # Ajouter le titre du film requête si on peut
    if title_map:
        result["query_title"] = title_map.get(int(movie_id), None)

    return result


# -----------------------------
# 4) Mode "script" (CLI) : affichage joli
# -----------------------------
if __name__ == "__main__":
    import argparse

    # On définit des arguments pour pouvoir faire :
    # python src/models/predict_model.py --movie_id 1 --top_n 10 --pretty
    parser = argparse.ArgumentParser(description="Movie recommender – item-item kNN")
    parser.add_argument(
        "--movie_id",
        type=int,
        required=True,
        help="movieId pour lequel recommander des films similaires",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Nombre de recommandations à retourner",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Affiche une recommandation par ligne (plus lisible). Sinon JSON indenté.",
    )

    args = parser.parse_args()

    # On appelle la fonction principale (celle qu'utilisera aussi l'API)
    result = recommend_similar_movies(movie_id=args.movie_id, top_n=args.top_n)

    # Affichage lisible (humain)
    if args.pretty:
        query_title = result.get("query_title") or "N/A"
        print(f"\n🎬 Film de référence : {query_title} (movieId={result['movieId']})")
        print("-" * 60)

        for i, rec in enumerate(result["recommendations"], start=1):
            title = rec.get("title", "N/A")
            # distance est un float -> on formate à 4 décimales
            print(
                f"{i:02d}. {title} | movieId={rec['movieId']} | distance={rec['distance']:.4f}"
            )

        print("")  # ligne vide à la fin

    # Affichage JSON (machine-friendly, copie-collable)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))

