# ============================================================
# PROMOTE_BEST_MODEL.PY
# ============================================================
# Objectif :
# Comparer les versions enregistrées d’un modèle MLflow
# et définir automatiquement l’alias "production"
# vers la meilleure version selon ndcg_10.
# Compatible MLflow 3.x (alias-based system).
# ============================================================

import os
import mlflow
from mlflow.tracking import MlflowClient


REGISTERED_MODEL_NAME = "reco-films-itemcf-v2"
METRIC_NAME = "ndcg_10"
ALIAS_NAME = "production"


def promote_best_model():

    # Connexion au serveur MLflow
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )

    client = MlflowClient()

    # Récupérer toutes les versions du modèle
    versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )

    if not versions:
        print("Aucune version trouvée.")
        return

    best_version = None
    best_metric = -1

    print("\nComparaison des versions :\n")

    for v in versions:
        run_id = v.run_id
        run = client.get_run(run_id)

        metric_value = run.data.metrics.get(METRIC_NAME)

        if metric_value is None:
            continue

        print(f"Version {v.version} → {METRIC_NAME} = {metric_value}")

        if metric_value > best_metric:
            best_metric = metric_value
            best_version = v

    if best_version is None:
        print("Aucune métrique valide trouvée.")
        return

    print(f"\nMeilleure version : {best_version.version}")
    print(f"{METRIC_NAME} = {best_metric}")

    # Définir l’alias "production" vers la meilleure version
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=ALIAS_NAME,
        version=best_version.version
    )

    print(f"\nAlias '{ALIAS_NAME}' mis à jour vers la version {best_version.version}")
    print("Promotion terminée.")


if __name__ == "__main__":
    promote_best_model()