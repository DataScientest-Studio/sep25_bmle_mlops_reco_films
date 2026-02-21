# ============================================================
# RUN_DATA_MONITORING_PIPELINE.PY
# ============================================================
# OBJECTIF
# ------------------------------------------------------------
# Orchestrateur post-ingestion complet :
#
#   1) Data Quality Checks
#   2) Monitoring KPI + Drift
#
# Hypothèses :
# - run_monitoring() exécute toujours :
#       daily + monthly + yearly + drift
# - Aucun paramètre de granularité configurable
# - Idempotence assurée côté SQL (UPSERT)
#
# Compatible :
# - CRON (exit code)
# - API (retour JSON structuré)
# ============================================================

from __future__ import annotations

from datetime import date, datetime
from typing import Optional, Dict, Any

from src.monitoring.quality_checks import run_quality_checks
from src.monitoring.data_monitoring import run_monitoring


# ------------------------------------------------------------
# ORCHESTRATEUR PRINCIPAL
# ------------------------------------------------------------
def run_data_monitoring_pipeline(
    reference_date: Optional[date] = None,
) -> Dict[str, Any]:
    """
    Lance le monitoring data complet.

    Étapes :
        1) Vérification qualité des données
        2) Calcul KPI + drift data

    Paramètre :
        reference_date :
            - None  -> mode CRON (today -> calcule la veille)
            - date  -> mode replay (API DS)

    Retour :
        dict structuré (API-friendly)
    """

    started_at = datetime.utcnow()

    report: Dict[str, Any] = {
        "pipeline": "data_monitoring",
        "started_at": started_at.isoformat(),
        "reference_date": str(reference_date) if reference_date else None,
        "quality_status": "NOT_RUN",
        "monitoring_status": "NOT_RUN",
        "status": "UNKNOWN",
    }

    # =========================================================
    # 1️⃣ QUALITY CHECKS
    # =========================================================
    try:
        quality_result = run_quality_checks()

        report["quality_result"] = quality_result
        report["quality_status"] = quality_result["status"]

    except Exception as e:
        # Si exception technique pendant les checks
        report["quality_status"] = "FAILED"
        report["quality_error_type"] = type(e).__name__
        report["quality_error_message"] = str(e)[:2000]

    # =========================================================
    # 2️⃣ MONITORING KPI + DRIFT
    # =========================================================
    try:
        monitoring_result = run_monitoring(
            reference_date=reference_date
        )

        report["monitoring_result"] = monitoring_result
        report["monitoring_status"] = monitoring_result.get("status", "SUCCESS")

    except Exception as e:
        # Si exception technique pendant le monitoring
        report["monitoring_status"] = "FAILED"
        report["monitoring_error_type"] = type(e).__name__
        report["monitoring_error_message"] = str(e)[:2000]

    # =========================================================
    # 3️⃣ STATUT GLOBAL
    # =========================================================
    # Règle simple :
    # - Si quality FAIL -> global FAIL
    # - Si monitoring FAIL -> global FAIL
    # - Sinon SUCCESS
    # =========================================================
    global_status = "SUCCESS"

    if report["quality_status"] == "FAILED":
        global_status = "FAILED"

    if report["monitoring_status"] == "FAILED":
        global_status = "FAILED"

    report["status"] = global_status
    report["ended_at"] = datetime.utcnow().isoformat()

    return report


# ------------------------------------------------------------
# CLI (CRON)
# ------------------------------------------------------------
if __name__ == "__main__":

    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Run full data monitoring (quality + KPI + drift)"
    )

    # Mode replay optionnel
    parser.add_argument(
        "--reference-date",
        type=str,
        default=None,
        help="YYYY-MM-DD (replay mode). Default: today -> computes yesterday",
    )

    args = parser.parse_args()

    ref = None
    if args.reference_date:
        y, m, d = map(int, args.reference_date.split("-"))
        ref = date(y, m, d)

    result = run_data_monitoring_pipeline(
        reference_date=ref,
    )

    # Affichage structuré pour logs CRON
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Exit code compatible alerting
    raise SystemExit(0 if result["status"] == "SUCCESS" else 1)