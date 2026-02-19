from __future__ import annotations
import pandas as pd
import numpy as np
import mlflow.pyfunc

class ItemCFPyFunc(mlflow.pyfunc.PythonModel):
    def __init__(self, n_reco: int = 10, min_user_ratings: int = 5, positive_threshold: float = 4.0):
        self.n_reco = n_reco
        self.min_user_ratings = min_user_ratings
        self.positive_threshold = positive_threshold

        self.neighbors_dict = None
        self.popularity = None

    def load_context(self, context):
        # --- MODIFICATION PARQUET ---
        print(f"[MODEL LOAD] Loading artifacts from: {context.artifacts}")
        item_neighbors = pd.read_parquet(context.artifacts["item_neighbors"])
        movie_popularity = pd.read_parquet(context.artifacts["movie_popularity"])
        neighbors_dict = {}
        # itertuples fonctionne pareil sur un DF chargé depuis Parquet
        for row in item_neighbors.itertuples(index=False):
            neighbors_dict.setdefault(int(row.movieId), []).append((int(row.neighborMovieId), float(row.similarity)))
        self.neighbors_dict = neighbors_dict

        # popularity triée
        self.popularity = movie_popularity.sort_values("bayes_score", ascending=False).reset_index(drop=True)
        print("[MODEL LOAD] Loaded successfully.")

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        model_input attend un DataFrame avec colonnes:
        - movieId (int)
        - rating (float)
        """
        if model_input is None or model_input.empty or len(model_input) < self.min_user_ratings:
            top = self.popularity.head(self.n_reco)[["movieId", "bayes_score"]].copy()
            top = top.rename(columns={"bayes_score": "score"})
            return top

        user_ratings = model_input.copy()
        user_ratings["movieId"] = user_ratings["movieId"].astype(int)
        user_ratings["rating"] = user_ratings["rating"].astype(float)

        seen_movies = set(user_ratings["movieId"].tolist())

        seed = user_ratings[user_ratings["rating"] >= self.positive_threshold]
        if seed.empty:
            seed = user_ratings

        seed_movies = seed["movieId"].tolist()
        if not seed_movies:
            top = self.popularity.head(self.n_reco)[["movieId", "bayes_score"]].copy()
            top = top.rename(columns={"bayes_score": "score"})
            return top

        scores = {}
        for m in seed_movies:
            for neigh, sim in self.neighbors_dict.get(int(m), []):
                if neigh in seen_movies:
                    continue
                scores[neigh] = scores.get(neigh, 0.0) + sim

        if not scores:
            top = self.popularity.head(self.n_reco)[["movieId", "bayes_score"]].copy()
            top = top.rename(columns={"bayes_score": "score"})
            return top

        reco = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: self.n_reco]
        out = pd.DataFrame(reco, columns=["movieId", "score"])
        return out