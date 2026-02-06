import numpy as np
import pandas as pd


def precision_at_k(recommended_items, relevant_items, k: int) -> float:
    if k == 0:
        return 0.0
    recommended_k = recommended_items[:k]
    hits = len(set(recommended_k) & set(relevant_items))
    return hits / k


def evaluate_precision_at_k(
    model,
    ratings_train: pd.DataFrame,
    ratings_test: pd.DataFrame,
    movie_matrix: pd.DataFrame,
    k_reco: int = 10,
    rating_threshold: float = 4.0,
    max_users: int = 10,
) -> float:
    """
    Precision@K offline pour un modèle kNN item-item.
    On compare les recommandations aux films aimés dans le test
    (cachés pendant l'entraînement).
    """

    precisions = []

    movie_ids = movie_matrix["movieId"].values
    movie_id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    idx_to_movie = {i: mid for mid, i in movie_id_to_idx.items()}

    X = movie_matrix.drop("movieId", axis=1)

    # Pré-calcul des voisins (clé pour éviter les temps infinis)
    distances_all, indices_all = model.kneighbors(
        X,
        n_neighbors=model.n_neighbors,
    )

    for i, (user_id, user_test) in enumerate(ratings_test.groupby("userId")):
        if i >= max_users:
            break

        relevant_items = user_test[user_test["rating"] >= rating_threshold]["movieId"].tolist()
        if not relevant_items:
            continue

        seen_items = ratings_train[ratings_train["userId"] == user_id]["movieId"].tolist()
        if not seen_items:
            continue

        scores = {}

        for seen_movie in seen_items:
            if seen_movie not in movie_id_to_idx:
                continue

            idx = movie_id_to_idx[seen_movie]
            neighbors_idx = indices_all[idx]
            neighbors_dist = distances_all[idx]

            for n_idx, dist in zip(neighbors_idx, neighbors_dist):
                movie_j = idx_to_movie[n_idx]
                if movie_j in seen_items:
                    continue
                similarity = 1.0 - dist
                scores[movie_j] = scores.get(movie_j, 0.0) + similarity

        if not scores:
            continue

        recommended_items = [
            m for m, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k_reco]
        ]

        precisions.append(
            precision_at_k(recommended_items, relevant_items, k_reco)
        )

    if not precisions:
        return 0.0

    return float(np.mean(precisions))