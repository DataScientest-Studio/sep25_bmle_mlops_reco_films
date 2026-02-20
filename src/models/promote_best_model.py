import os
import mlflow
from mlflow.tracking import MlflowClient

REGISTERED_MODEL_NAME = "reco-films-itemcf-v2"
ALIAS_NAME = "production"

PRIMARY_METRIC = "ndcg_10"
PRECISION_METRIC = "precision_10"
RECALL_METRIC = "recall_10"


def compute_weighted_score(run):
    ndcg = run.data.metrics.get(PRIMARY_METRIC, 0)
    precision = run.data.metrics.get(PRECISION_METRIC, 0)
    recall = run.data.metrics.get(RECALL_METRIC, 0)

    score = (
        0.6 * ndcg +
        0.3 * precision +
        0.1 * recall
    )

    return score, ndcg, precision, recall


def promote_best_model():

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )

    client = MlflowClient()

    # Récupérer version en production (Champion)
    try:
        prod_version = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            ALIAS_NAME
        )
    except Exception:
        print("Aucun modèle actuellement en production.")
        prod_version = None

    # Récupérer dernière version enregistrée (Challenger)
    versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )

    if not versions:
        print("Aucune version trouvée.")
        return

    # Trier par numéro de version décroissant
    versions_sorted = sorted(
        versions,
        key=lambda v: int(v.version),
        reverse=True
    )

    challenger = versions_sorted[0]

    challenger_run = client.get_run(challenger.run_id)
    challenger_score, ndcg, precision, recall = compute_weighted_score(challenger_run)

    print("\n--- Challenger ---")
    print(f"Version {challenger.version}")
    print(f"NDCG={ndcg:.4f} | Precision={precision:.4f} | Recall={recall:.4f}")
    print(f"Score pondéré = {challenger_score:.4f}")

    # Si aucun champion → on promeut directement
    if prod_version is None:
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME,
            ALIAS_NAME,
            challenger.version
        )
        print("\nAucun modèle en production → Promotion automatique.")
        return

    # Score du Champion
    champion_run = client.get_run(prod_version.run_id)
    champion_score, ndcg_c, precision_c, recall_c = compute_weighted_score(champion_run)

    print("\n--- Champion actuel ---")
    print(f"Version {prod_version.version}")
    print(f"NDCG={ndcg_c:.4f} | Precision={precision_c:.4f} | Recall={recall_c:.4f}")
    print(f"Score pondéré = {champion_score:.4f}")

    # 🔹 Comparaison
    if challenger_score > champion_score:
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME,
            ALIAS_NAME,
            challenger.version
        )
        print("\nNouveau modèle supérieur → Promotion effectuée.")
    else:
        print("\nLe modèle actuel reste en production.")


if __name__ == "__main__":
    promote_best_model()