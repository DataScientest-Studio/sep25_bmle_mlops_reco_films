#!/bin/bash

# --- AUTOMATISATION DES CHEMINS ---
# Cette commande récupère le dossier où se trouve le script .sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$SCRIPT_DIR"

API_URL="http://127.0.0.1:8000"
LOG_FILE="$PROJECT_DIR/pipeline.log"

# Fonction pour logger
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# On s'assure d'être dans le bon dossier pour les imports Python
cd "$PROJECT_DIR"

log "🚀 Démarrage du Pipeline MLOps (Mode Relatif)..."

# 1. DATA PIPELINE
log "📡 1. Lancement Ingestion (/data)..."
curl -s -X POST "$API_URL/data" >> "$LOG_FILE" 2>&1
if [ $? -eq 0 ]; then log "✅ Ingestion OK"; else log "❌ Erreur Ingestion"; exit 1; fi

# 2. TRAINING PIPELINE
log "🏋️‍♂️ 2. Lancement Training (/training)..."
curl -s -X POST "$API_URL/training" >> "$LOG_FILE" 2>&1
if [ $? -eq 0 ]; then log "✅ Training OK"; else log "❌ Erreur Training"; exit 1; fi

# 3. PROMOTION DU MODÈLE
log "🏆 3. Promotion du meilleur modèle..."
# L'utilisation de python -m nécessite d'être à la racine du projet (déjà fait avec cd)
python -m src.models.promote_best_model >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    log "✅ Pipeline complet terminé avec succès."
else
    log "❌ Erreur Promotion."
fi