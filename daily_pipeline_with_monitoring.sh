#!/bin/bash
set -e
set -o pipefail
export PYTHONUTF8=1

# ==============================================================================
# DAILY PIPELINE WITH MONITORING
# ==============================================================================
# Orchestration complète :
#   0. Git Update
#   1. Ingestion + Monitoring volumétrique
#   1bis. Data Quality + KPI + Drift
#   2. Snapshot + Training + Monitoring training (durée, success)
#   3. Promotion modèle
#   4. Git Push final
# ==============================================================================


# --- CONFIGURATION ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$SCRIPT_DIR"
LOG_FILE="$PROJECT_DIR/pipeline_with_monitoring.log"

cd "$PROJECT_DIR"

# Fonction pour logger
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
log "🌿 Branche détectée : $GIT_BRANCH"
log "🚀 Démarrage du Pipeline MLOps..."

# ==============================================================================
# 0. GIT UPDATE
# ==============================================================================
log "🔄 0. Git Pull..."
if git pull origin "$GIT_BRANCH"; then
    log "✅ Code à jour."
    dvc pull >> "$LOG_FILE" 2>&1 || true 
else
    log "❌ Erreur Git Pull. Arrêt."
    exit 1
fi
   

# ==============================================================================
# 1. INGESTION (Source -> Local)
# + MONITORING VOLUMÉTRIE
# ==============================================================================
log "📡 1. Ingestion (Téléchargement + SQL + Monitoring)..."
# On lance l'ingestion qui va TELECHARGER les fichiers

if python -m src.monitoring.run_ingestion_with_monitoring >> "$LOG_FILE" 2>&1; then

    # --- SAFETY CHECK : Est-ce que le dossier est vide ? ---
    if [ -z "$(ls -A data/raw)" ]; then
       log "❌ CRITIQUE : Ingestion terminée mais data/raw est vide ! Arrêt."
       exit 1
    fi
    log "✅ Ingestion et Téléchargement terminés."


    # --- VERSIONNING DVC (Local -> Remote) ---
    log "📦 Versionning DVC (Raw Data)..."
    dvc add data/raw >> "$LOG_FILE" 2>&1
    dvc push data/raw.dvc >> "$LOG_FILE" 2>&1
    
    # --- GIT COMMIT ---
    if git diff --name-only | grep -q "data/raw.dvc"; then
        log "📝 Mise à jour des données détectée. Commit..."
        git add data/raw.dvc
        git commit -m "data: fresh ingestion $(date '+%Y-%m-%d')"
    else
        log "ℹ️ Données identiques à la version précédente."
    fi

else
    log "❌ Ingestion FAILED."
    exit 1
fi



# ==============================================================================
# 1bis. DATA MONITORING (Quality + KPI + Drift)
# ==============================================================================
log "📊 1bis. Data Monitoring..."

if python -m src.monitoring.run_data_monitoring_pipeline >> "$LOG_FILE" 2>&1; then
    log "✅ Data Monitoring terminé."
else
    log "⚠️ Data Monitoring FAILED (non bloquant)."
fi


# ==============================================================================
# 2. TRAINING (Parquet)
# ==============================================================================
log "📸 2. Snapshot & Training..."

# Création du Snapshot
if python src/ingestion/create_snapshot.py >> "$LOG_FILE" 2>&1; then
    
    # Versionning du Parquet
    if [ -f "data/training_set.parquet" ]; then
        dvc add data/training_set.parquet >> "$LOG_FILE" 2>&1
        dvc push data/training_set.parquet.dvc >> "$LOG_FILE" 2>&1

        # Nettoyage vieux CSV s'ils existent
        git rm data/training_set.csv.dvc 2>/dev/null || true

        # Git Commit Model Data
        if git diff --name-only | grep -q "data/training_set.parquet.dvc"; then
            git add data/training_set.parquet.dvc
            git commit -m "model: update training set $(date '+%Y-%m-%d')"
        fi
    fi
else
    log "❌ Erreur Snapshot."
    exit 1
fi

# Entraînement
log "🏋️‍♂️ Lancement Entraînement..."
if python -m src.monitoring.run_training_with_monitoring >> "$LOG_FILE" 2>&1; then
    log "✅ Entraînement terminé."
else
    log "❌ Erreur Training."
    exit 1
fi


# ==============================================================================
# 3. PROMOTION & PUSH
# ==============================================================================
log "🏆 3. Promotion..."
if python -m src.models.promote_best_model >> "$LOG_FILE" 2>&1; then
    log "✅ Promotion terminée."
else
    log "❌ Promotion FAILED."
    exit 1
fi

log "☁️ 4. Git Push Final..."
if git push origin "$GIT_BRANCH" >> "$LOG_FILE" 2>&1; then
    log "✅ Git Push terminé."
else
    log "❌ Git Push FAILED."
    exit 1
fi



log "🎯 Pipeline terminé avec succès."


