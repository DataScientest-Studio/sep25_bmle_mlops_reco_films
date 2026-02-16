#!/bin/bash

# --- CONFIGURATION ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$SCRIPT_DIR"
API_URL="http://127.0.0.1:8000"
LOG_FILE="$PROJECT_DIR/pipeline.log"
GIT_BRANCH="master"

# Fonction pour logger
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# On s'assure d'être dans le bon dossier
cd "$PROJECT_DIR"

log "🚀 Démarrage du Pipeline MLOps..."

# ==============================================================================
# 0. MISE À JOUR DU CODE (GIT PULL)
# ==============================================================================
log "🔄 0. Récupération de la dernière version du code..."
# On pull d'abord pour être sûr d'avoir les derniers scripts python de l'équipe
if git pull origin "$GIT_BRANCH"; then
    log "✅ Code à jour."
else
    log "❌ Erreur lors du Git Pull. Arrêt du pipeline."
    # On arrête tout, car lancer un training sur un code en conflit est dangereux
    exit 1
fi

# ==============================================================================
# 1. DATA PIPELINE
# ==============================================================================
log "📡 1. Lancement Ingestion & Versionning (/data)..."

HTTP_CODE_DATA=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_URL/data")

if [ "$HTTP_CODE_DATA" -eq 200 ]; then
    log "✅ Ingestion API & DVC Push OK"
    
    # --- COMMIT LOCAL : DATA ---
    if git diff --name-only | grep -q "data/raw.dvc"; then
        log "📦 Nouveaux fichiers raw détectés. Commit local..."
        git add data/raw.dvc
        git commit -m "data: daily update raw dataset $(date '+%Y-%m-%d')"
        log "✅ Commit Data effectué."
    else
        log "ℹ️ Pas de changement Data."
    fi
else
    log "❌ Erreur Ingestion (Code HTTP: $HTTP_CODE_DATA)"
    exit 1
fi

# ==============================================================================
# 2. TRAINING PIPELINE
# ==============================================================================
log "🏋️‍♂️ 2. Lancement Training (/training)..."

HTTP_CODE_TRAIN=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_URL/training")

if [ "$HTTP_CODE_TRAIN" -eq 200 ]; then
    log "✅ Training API & DVC Push OK"

    # --- COMMIT LOCAL : MODEL ---
    if git diff --name-only | grep -q "data/training_set.csv.dvc"; then
        log "📦 Nouveau Training Set détecté. Commit local..."
        git add data/training_set.csv.dvc
        git commit -m "model: update training set snapshot $(date '+%Y-%m-%d')"
        log "✅ Commit Model effectué."
    else
        log "ℹ️ Pas de changement Model."
    fi
else
    log "❌ Erreur Training (Code HTTP: $HTTP_CODE_TRAIN)"
    exit 1
fi

# ==============================================================================
# 3. PROMOTION DU MODÈLE
# ==============================================================================
log "🏆 3. Promotion du meilleur modèle..."
python -m src.models.promote_best_model >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    log "✅ Promotion terminée."
else
    log "❌ Erreur Promotion."
    # On n'exit pas forcément ici, on veut peut-être quand même push les data
fi

# ==============================================================================
# 4. SYNCHRONISATION FINALE (GIT PUSH)
# ==============================================================================
log "☁️ 4. Envoi des modifications vers GitHub (Push)..."

# On push tout ce qui a été commité (Data et/ou Model) en une seule fois
if git push origin "$GIT_BRANCH"; then
    log "✅ Git Push réussi. Pipeline terminé avec succès."
else
    log "❌ Erreur lors du Git Push. Vérifie tes accès."
    exit 1
fi