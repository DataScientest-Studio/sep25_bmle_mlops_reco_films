# ============================================================
# FICHIER : src/monitoring/backfill_monitoring.py
# ============================================================
# OBJECTIF
# ------------------------------------------------------------
# Backfill historique des métriques de monitoring.
#
# Ce script permet de reconstituer l'historique complet des
# métriques comme si le job de monitoring avait été exécuté
# en production depuis l'origine des données.
#
# VERSION OPTIMISÉE
# ------------------------------------------------------------
# - Appelle directement les fonctions du moteur
#   (pas d’orchestrateur global)
# - Calcule uniquement le niveau requis :
#       yearly  → métriques annuelles
#       monthly → métriques mensuelles
#       daily   → métriques quotidiennes
# - Conserve la cohérence des PSI
# - Idempotent (UPSERT SQL côté moteur)
# - Compatible gros volumes (MovieLens 20M)
#
# UTILISATION
# ------------------------------------------------------------
# À exécuter ponctuellement (hors CRON) :
#
#     python backfill_monitoring.py
#
# Ne doit pas être intégré au pipeline journalier.
# ============================================================

from datetime import date, timedelta

from src.monitoring.data_monitoring import (
    ensure_monitoring_table,
    compute_core,
    compute_rating_distribution,
    compute_genres,
    compute_psi,
    day_bounds,
    month_bounds,
    year_bounds
)

# =========================================================
# CONFIGURATION DU PÉRIMÈTRE
# =========================================================
# Les bornes définissent la profondeur historique du backfill.
# Elles peuvent être adaptées selon la volumétrie souhaitée.

YEAR_START  = 1996
YEAR_END    = 2015

MONTH_START = date(2013, 1, 1)
MONTH_END   = date(2015, 3, 31)

# Daily : on démarre au 2 janvier car le moteur calcule J-1
DAY_START   = date(2015, 1, 2)
DAY_END     = date(2015, 3, 31)


# =========================================================
# BACKFILL ANNUEL
# =========================================================
# Calcule :
#   - Core metrics
#   - Distribution ratings
#   - Distribution genres
#   - PSI rating (année vs année-1)
#   - PSI genres (année vs année-1)
#
# Chaque année est comparée à l’année précédente réelle
# afin de préserver la cohérence temporelle du drift.

def backfill_yearly():

    print("=== BACKFILL YEARLY START ===")

    for year in range(YEAR_START, YEAR_END + 1):

        print(f"[YEAR] {year}")

        bounds = year_bounds(year)
        prev_bounds = year_bounds(year - 1)

        # --- Core metrics
        compute_core("yearly", str(year), bounds)

        # --- Distributions
        compute_rating_distribution("yearly", str(year), bounds)
        compute_genres("yearly", str(year), bounds)

        # --- Drift (PSI)
        compute_psi(
            "yearly",
            str(year),
            bounds,
            prev_bounds,
            "psi_rating_prev_year",
            "rating"
        )

        compute_psi(
            "yearly",
            str(year),
            bounds,
            prev_bounds,
            "psi_genre_prev_year",
            "genre",
            join_genres=True
        )

    print("=== BACKFILL YEARLY DONE ===\n")


# =========================================================
# BACKFILL MENSUEL
# =========================================================
# Calcule :
#   - Core metrics
#   - Distribution ratings
#   - Distribution genres
#   - PSI rating (mois vs mois-1)
#   - PSI genres (mois vs mois-1)
#
# La référence mensuelle est calculée dynamiquement
# à partir du mois précédent réel.

def backfill_monthly():

    print("=== BACKFILL MONTHLY START ===")

    current = MONTH_START

    while current <= MONTH_END:

        ym = current.strftime("%Y-%m")
        print(f"[MONTH] {ym}")

        bounds = month_bounds(ym)

        # Calcul du mois précédent réel
        prev_month_date = current.replace(day=1) - timedelta(days=1)
        prev_bounds = month_bounds(prev_month_date.strftime("%Y-%m"))

        # --- Core metrics
        compute_core("monthly", ym, bounds)

        # --- Distributions
        compute_rating_distribution("monthly", ym, bounds)
        compute_genres("monthly", ym, bounds)

        # --- Drift (PSI)
        compute_psi(
            "monthly",
            ym,
            bounds,
            prev_bounds,
            "psi_rating_prev_month",
            "rating"
        )

        compute_psi(
            "monthly",
            ym,
            bounds,
            prev_bounds,
            "psi_genre_prev_month",
            "genre",
            join_genres=True
        )

        # Passage au mois suivant
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)

    print("=== BACKFILL MONTHLY DONE ===\n")


# =========================================================
# BACKFILL DAILY
# =========================================================
# Calcule uniquement les métriques core journalières.
# Aucun PSI quotidien (non défini dans le moteur).
#
# On calcule J-1 pour reproduire la logique du job prod.

def backfill_daily():

    print("=== BACKFILL DAILY START ===")

    current = DAY_START

    while current <= DAY_END:

        target_day = current - timedelta(days=1)
        print(f"[DAY] {target_day}")

        bounds = day_bounds(target_day)

        compute_core("daily", str(target_day), bounds)

        current += timedelta(days=1)

    print("=== BACKFILL DAILY DONE ===\n")


# =========================================================
# MAIN
# =========================================================
# Ordre d'exécution :
#   1. Création table si nécessaire
#   2. Backfill annuel
#   3. Backfill mensuel
#   4. Backfill journalier
#
# L’ordre garantit que les périodes de référence
# existent déjà au moment du calcul des PSI.

if __name__ == "__main__":

    print("====================================")
    print("   HISTORICAL MONITORING BACKFILL   ")
    print("====================================\n")

    ensure_monitoring_table()

    backfill_yearly()
    backfill_monthly()
    backfill_daily()

    print("====================================")
    print("         BACKFILL COMPLETED         ")
    print("====================================")