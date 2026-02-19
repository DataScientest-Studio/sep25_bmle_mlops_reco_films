# project_prez.py
# ============================================================
# Système de recommandation de films — Streamlit (Soutenance + Demo)
# ============================================================

from __future__ import annotations
import os
from pathlib import Path
import time
import requests  # Nécessaire pour l'API
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================
# Paths
# =========================
# NOTE : On adapte le ROOT pour qu'il soit robuste peu importe où on lance le script
# Si lancé depuis la racine (cas Docker), on ajuste.
try:
    # On tente de garder votre logique actuelle
    ROOT = Path(__file__).resolve().parents[2]
except IndexError:
    # Fallback si la structure de dossier est différente dans Docker
    ROOT = Path(__file__).resolve().parent

# =========================
# Figures Visualization (PNG)
# =========================
MOVIELENS_IMG = ROOT / "src" / "streamlit" / "movielens.png"
DATA_IMG = ROOT / "src" / "streamlit" / "pipeline_data_IMG.png"
viz1_IMG = ROOT / "Reports" / "figures" / "visualize_Figure_1.png"
viz2_IMG = ROOT / "Reports" / "figures" / "visualize_Figure_2.png"
SQL1_IMG = ROOT / "Reports" / "figures" / "SQL1.png"
archi_IMG = ROOT / "Reports" / "figures" / "architecture.png"

DEFAULT_FIG_DIRS = [
    ROOT / "Reports" / "figures",
    ROOT / "reports" / "figures",
    ROOT / "assets",
    ROOT / "Assets",
]

# =========================
# Config & constants (MODIFIÉ POUR DOCKER)
# =========================
st.set_page_config(
    page_title="Système de recommandation de films (Soutenance)",
    page_icon="🎬",
    layout="wide",
)

APP_TITLE = "🎬 Création d'un système de recommandation de films"

# --- MODIFICATIONS ICI ---
# L'API URL est utilisée par le container Python (backend-to-backend)
# Par défaut localhost pour dev local, mais surchargé par Docker
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# MLFlow URL interne (pour les check health requests python)
MLFLOW_INTERNAL_URL = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# MLFlow URL externe (pour le lien cliquable par l'utilisateur dans son navigateur)
# L'utilisateur ne peut pas accéder au réseau docker interne, il passe par localhost
MLFLOW_EXTERNAL_URL = "http://127.0.0.1:5000" 
# -------------------------

# =========================
# UI Helpers
# =========================

def slide_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
    st.markdown("---")

def key_takeaways(title, items: list[str]) -> None:
    st.markdown(f"### ✅ {title}")
    for it in items:
        st.markdown(f"- **{it}**")

def find_first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None

def list_pngs_in_known_dirs() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for d in DEFAULT_FIG_DIRS:
        if d.exists() and d.is_dir():
            for p in d.glob("*.png"):
                found[p.name] = p
    return found

def show_png_if_exists(filename_contains: str, png_map: dict[str, Path], caption: str | None = None) -> bool:
    needle = filename_contains.lower()
    for name, path in png_map.items():
        if needle in name.lower():
            st.image(str(path), caption=caption, use_container_width=True)
            return True
    return False

def check_api_health():
    """Vérifie si l'API est en ligne"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=1)
        if response.status_code == 200:
            return True
    except:
        return False
    return False

# =========================
# PARTIE 1 : PRÉSENTATION
# =========================
def show_presentation_mode():
    
    png_map = list_pngs_in_known_dirs()

    st.sidebar.markdown("---")
    st.sidebar.header("🧭 Navigation Slides") 
    SECTIONS = [
        "Contexte & objectifs",
        "Architecture générale",
        "Modèle & métriques d’évaluation",
        "Suivi des Expériences via MLflow",    
        "Monitoring & maintenance",
        "Conclusion & perspectives",
    ]
    section = st.sidebar.radio("Aller à :", SECTIONS, index=0)

    st.sidebar.markdown("---")
    st.sidebar.caption("""
                    Pierre Barbetti       
                    Raphaël Da Silva       
                    Martine Mateus       
                    Laurent Piacentile
                    """)

    if section == "Contexte & objectifs": 
        slide_header("Contexte & objectifs")
        
        st.markdown("## 🧪 **Cadre du projet**")
        
        st.markdown("""
        ### 🎯 Objectif : Déployer un système de recommandation de films 
        
        ### Cadre technique :
        - MLOps : Automatiser et monitorer le cycle de vie d'un projet ML
        - Disposer d'une application de recommandation de films en production
        
        ### Choix de conception :
        - Item-Based Collaborative Filtering : similarité entre films en fonction des comportements utilisateurs
        - Tables de recommandations générées offline, pas d'inférence en direct
        """)
            
        st.markdown("""
        ## 🎯 Focus sur les pratiques MLOps / performances de l'architecture
        
        - Architecture robuste de type microservices
        - Versioning des données et des modèles
        - Reproductibilité et traçabilité
        - Monitoring des métriques en production
        - Documentation claire
                    
        Aspects spécifiques du projet :  
        - résoudre la problématique de cold-start.
        """)

    elif section == "Architecture générale":
        slide_header("🧷 Architecture générale")
        st.subheader("Schéma de l'architecture MLOps conteneurisée")
        
        if archi_IMG.exists():
            st.image(str(archi_IMG), caption="Schéma MLOps", use_container_width=True)
        else:
            st.warning("Image introuvable: architecture.png")

    elif section == "Bases de données PostgreSQL":
        slide_header("Bases de données")    
        st.subheader("Architecture de la base de données PostgreSQL")
        
        col1, col2 = st.columns(2)
        with col1:
            if SQL1_IMG.exists():
                st.image(str(SQL1_IMG), caption="Schéma DB", use_container_width=True)
            else:
                st.warning("Image introuvable: SQL1.png")
        with col2:
            st.success("**Versioning des données**")

        st.write("---")
        st.subheader("📊 Exploration des données MovieLens")
        st.markdown("*https://grouplens.org/datasets/movielens/20m/*")
        
        col1, col2 = st.columns(2)
        with col1:
            if viz1_IMG.exists():
                st.image(str(viz1_IMG), caption="MovieLens — En chiffres", use_container_width=True)
        with col2:
            if viz2_IMG.exists():
                st.image(str(viz2_IMG), caption="MovieLens — En graphiques", use_container_width=True)

    elif section == "Modèle & métriques d’évaluation":
        slide_header(
            "🔎 Modèle & métriques d’évaluation",
            "Item-based CF + évaluation orientée ranking (Top-10)"
        )

        # ==========================================================
        # MODÈLE
        # ==========================================================
        st.subheader("🎯 Modèle : Item-Based Collaborative Filtering (ItemCF)")

        col1, col2 = st.columns(2)

        with col1:
            st.info("""
            ### 🔹 Principe
            - Chaque film est représenté par un **vecteur de notes utilisateurs** (user-item matrix).
            - On calcule la similarité **cosine** entre les films.
            - On conserve les **K voisins** les plus similaires par film (**offline**).
            - En recommandation (online), on agrège les voisins des films vus et on **rank** les candidats.

            👉 Modèle explicable, rapide en inférence, adapté aux systèmes Top-N.
            """)

        with col2:
            st.info("""
            ### 🔹 Scoring (ranking)
            - On part de l’historique utilisateur (films déjà vus).
            - On récupère les voisins item-item et on agrège un score.
            - On exclut les films déjà vus.
            - On retourne le **Top-10**.

            ✔️ Offline: calcul voisinage / index  
            ✔️ Online: scoring léger + tri  
            ✔️ Usage production : faible latence  
            """)

        st.markdown("---")

        # ==========================================================
        # COLD START
        # ==========================================================
        st.subheader("🧊 Gestion du Cold-Start")

        col1, col2 = st.columns(2)

        with col1:
            st.success("""
            ### 🔹 Nouveaux utilisateurs
            Fallback vers une recommandation **popularité bayésienne** :
            - robuste aux faibles volumes de notes
            - évite de survaloriser des films avec peu de ratings
            - garantit une recommandation même sans historique
            """)

        with col2:
            st.success("""
            ### 🔹 Robustesse produit
            - Le pipeline garantit toujours un Top-N
            - Le système gère explicitement les cas sans historique utilisateur
            - Le fallback vers la popularité assure une continuité de service
            """)

        st.markdown("---")

        # ==========================================================
        # MÉTRIQUES
        # ==========================================================
        st.subheader("📊 Métriques d’évaluation (Top-10 Ranking Metrics)")

        st.markdown("""
        Le système est optimisé pour la **recommandation Top-10** (ranking) et non la prédiction exacte d’une note.
        Les métriques évaluent la qualité du classement des films pertinents.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.info("""
            ### 🔹 Precision@10
            Proportion de recommandations pertinentes dans le Top-10.

            👉 Indique la “pureté” du Top-10 (qualité immédiate).
            """)
            st.latex(r"""
            Precision@10 =
            \frac{|\{pertinents\} \cap \{Top10\}|}{10}
            """)

        with col2:
            st.info("""
            ### 🔹 Recall@10
            Proportion des films pertinents retrouvés dans le Top-10.

            👉 Indique la couverture des préférences utilisateur.
            """)
            st.latex(r"""
            Recall@10 =
            \frac{|\{pertinents\} \cap \{Top10\}|}{|\{pertinents\}|}
            """)

        st.markdown("---")

        st.subheader("NDCG@10")

        col1, col2 = st.columns(2)

        with col1:
            st.info("""
            NDCG@10 valorise :
            - la pertinence
            - la position dans le ranking

            👉 Un film pertinent en rank 1 “vaut” plus qu’en rank 10.
            """)

        with col2:
            st.markdown("""
                **DCG@10** : somme des films pertinents pondérée par leur position dans le classement.

                👉 Plus un film pertinent apparaît haut dans le Top-10, plus sa contribution est importante.
                """)

            st.markdown("""
                **NDCG@10** : version normalisée du DCG.

                👉 Permet de comparer les modèles entre eux sur une échelle comprise entre 0 et 1.
                1 = classement parfait.
                """)

        st.markdown("---")

        key_takeaways("Pourquoi ces métriques ?", [
            "Le produit est un moteur de ranking (Top-10), pas une régression sur la note",
            "NDCG@10 = métrique qui permer de tenir compte de l’ordre des recommandations",
            "Precision@10 et Recall@10 complètent l’évaluation (qualité / couverture)",
        ])

    elif section == "Suivi des Expériences via MLflow":
        slide_header(
            "📊 Suivi des Expériences via MLflow",
            "Traçabilité, reproductibilité, gouvernance modèle (Registry + alias production)"
        )

        # ==========================================================
        # Logo + Intro
        # ==========================================================
        col_logo, col_txt = st.columns([1, 3])
        with col_logo:
            show_png_if_exists("MLflow-logo", png_map, caption=None)
        with col_txt:
            st.markdown("""
            MLflow est utilisé comme **système central de tracking & registry** :
            - suivi des runs (params, métriques, artefacts)
            - comparaison d’expériences
            - gouvernance modèle via **Model Registry**
            - promotion contrôlée en production via **alias `@production`**
            """)

        st.markdown("---")

        # ==========================================================
        # Objectifs
        # ==========================================================
        st.subheader("🎯 Objectifs MLOps couverts")

        col1, col2 = st.columns(2)
        with col1:
            st.info("""
            ✔️ Tracer chaque entraînement (runs)  
            ✔️ Logger hyperparamètres (ex: k_neighbors)  
            ✔️ Logger métriques **Top-10** : ndcg_10, recall_10, precision_10  
            ✔️ Sauvegarder le modèle (PyFunc) comme artefact  
            ✔️ Reproductibilité / auditabilité  
            """)
        with col2:
            st.info("""
            ✔️ Versioning du modèle dans le Registry  
            ✔️ Gouvernance : alias `production`  
            ✔️ Promotion automatique basée sur métriques  
            ✔️ Historique complet des versions  
            ✔️ Tag git_commit pour relier code ↔ run  
            """)

        st.markdown("---")

        # ==========================================================
        # Screenshot runs (liste)
        # ==========================================================
        st.subheader("🧪 Tracking des runs (params + métriques + artefacts)")
        displayed = show_png_if_exists(
            "mlflow_runs_metriques",
            png_map,
            caption="Liste des runs : comparaison des métriques et hyperparamètres (k_neighbors)."
        )
        if not displayed:
            st.warning("Image 'mlflow_runs_metriques.png' introuvable (place-la dans Reports/figures/).")

        st.markdown("---")

        # ==========================================================
        # Comparaison runs (visualization)
        # ==========================================================
        st.subheader("📈 Comparaison d’expériences (visualisations MLflow)")
        displayed = show_png_if_exists(
            "mlflow_run_comparaison",
            png_map,
            caption="Comparaison multi-runs : impact de k_neighbors sur recall_10 / ndcg_10 / precision_10."
        )
        if not displayed:
            st.warning("Image 'mlflow_run_comparaison.png' introuvable (place-la dans Reports/figures/).")

        st.markdown("---")

        # ==========================================================
        # Run detail (preuve de traçabilité)
        # ==========================================================
        st.subheader("🔍 Détail d’un run : métriques, paramètres, tags")
        displayed = show_png_if_exists(
            "mlflow_run_k10v8",
            png_map,
            caption="Détail d’un run : métriques + paramètres + tag git_commit + modèle enregistré."
        )
        if not displayed:
            st.warning("Image 'mlflow_run_k10v8.png' introuvable (place-la dans Reports/figures/).")

        st.markdown("""
        **Points clés :**
        - métriques Top-10 disponibles (ndcg_10, recall_10, precision_10)
        - paramètres explicites (k_neighbors, min_ratings)
        - tag **git_commit** : traçabilité code → run
        """)

        st.markdown("---")

        # ==========================================================
        # Registry + alias production
        # ==========================================================
        st.subheader("🏷️ Model Registry & Alias `@production` (contrat de déploiement)")
        col1, col2 = st.columns(2)
        with col1:
            st.success("""
            Le service de prédiction charge toujours :

            **models:/reco-films-itemcf-v2@production**

            👉 aucune référence directe à une version dans le code.
            """)
        with col2:
            st.success("""
            La promotion en production met à jour **uniquement l’alias** :
            - rollback simple
            - gouvernance centralisée
            - découplage code / modèle
            """)

        displayed = show_png_if_exists(
            "mlflow_registry_alias",
            png_map,
            caption="Model Registry : versions + alias @production sur la version active."
        )
        if not displayed:
            st.warning("Image 'mlflow_registry_alias.png' introuvable (place-la dans Reports/figures/).")

        st.markdown("---")

        # ==========================================================
        # Promotion automatique (nouvelle règle)
        # ==========================================================
        st.subheader("🚀 Promotion automatique : score pondéré (gouvernance modèle)")

        st.markdown("""
        La promotion n’est plus basée uniquement sur NDCG.
        Nous utilisons une règle simple, explicable et stable :

        """)
        st.latex(r"""
        Score = 0.6 \cdot NDCG@10 + 0.3 \cdot Precision@10 + 0.1 \cdot Recall@10
        """)

        st.info("""
        **Pourquoi ce choix ?**
        - NDCG@10 prioritaire : qualité du ranking (position)
        - Precision@10 : qualité brute du Top-10
        - Recall@10 : couverture des préférences utilisateur
        """)

        st.markdown("---")

        key_takeaways("Valeur ajoutée MLflow dans ce projet :", [
            "Traçabilité complète des expérimentations (params, métriques, artefacts)",
            "Reproductibilité + auditabilité (tag git_commit)",
            "Gouvernance modèle via Registry + alias @production",
            "Promotion contrôlée et réversible (rollback simple)",
        ])
        
    elif section == "Monitoring & maintenance":

            ML_cycle_IMG = ROOT / "src" / "streamlit" / "ML_cycle.png"
            grafana_icon_IMG = ROOT / "src" / "streamlit" / "grafana_icon.png"
            grafana_dashboards_IMG = ROOT / "src" / "streamlit" / "grafana_dashboards.png"
            grafana_pipelineHealth_IMG = ROOT / "src" / "streamlit" / "grafana_pipelineHealth.png"
            grafana_KPIannuels_IMG = ROOT / "src" / "streamlit" / "grafana_KPIannuels.png"
            grafana_KPImensuels_IMG = ROOT / "src" / "streamlit" / "grafana_KPImensuels.png"
            grafana_monitoringQuo_IMG = ROOT / "src" / "streamlit" / "grafana_MonitoringQuo.png"
            grafana_dataDrift_IMG = ROOT / "src" / "streamlit" / "grafana_dataDrift.png"
            grafana_dataQuality_IMG = ROOT / "src" / "streamlit" / "grafana_dataQuality.png"

            slide_header("📈 Monitoring & Maintenance")

            col1, col2 = st.columns([1.2, 1])
            with col1:
                st.info("""
                ## 🎯 Principe
                ### Le monitoring est **transversal** au cycle de vie du modèle.  
                Il intervient à chaque étape :  
                            - ingestion des nouvelles données,  
                            - data processig (surveillance data drift),  
                            - training du modèle,   
                            - performance et utilisation du produit,
                
                Il se place aussi au niveau des **infrastructures** et du suivi du **développement métier** (outil d'aide à la décision)
                """)

            with col2:
                if ML_cycle_IMG.exists():
                    st.image(str(ML_cycle_IMG), width=600)
                else:
                    st.error("❌ Image cycle ML introuvable")

            st.divider()

            st.markdown("""
                ## 🔍 Surveillance continue""")

            col1, col2 = st.columns([1.2, 1])
            with col1:
                st.markdown("""   
                ### 1️⃣ Données
                - Process d'ingestion (durée, statut, volumétrie)
                -  Qualité des données
                -  Data drift (notes, genres, PSI, nouvautés)
                -  KPI métiers

                ### 2️⃣ Modèle
                -  Durée d'entraînement
                -  métriques de performance
                -  gestion du cold-start
                -  back-testing après déploiement
                """)

            with col2:
                st.markdown("""
                ### 3️⃣ Produit (API)
                - Performances techniques
                - Utilisation
                - Cold-start & performance prédictions online

                ### 4️⃣ Infrastructure
                - CPU / Mémoire
                - Stockage
                - Disponibilité
                """)

            st.divider()

            st.markdown("## 📊 Déploiement du monitoring")
            col1, col2 = st.columns(2)

            with col1:
                st.success("""
                ### ✔️ Monitoring Data
                - Ingestion des données
                - KPI croissance & nouveauté
                - Data drift (PSI)
                """)

            with col2:
                st.success("""
                ### ✔️ Monitoring Modèle
                - Training logs
                - recall@K / ndcg@K
                - Promotion automatique
                """)

            st.markdown("## 🖥 Dashboards Grafana")

            col1, col2 = st.columns(2)
            with col1:
                if grafana_icon_IMG.exists():
                    st.image(str(grafana_icon_IMG), width=200)
                else:
                    st.error("❌ Image grafana_icon introuvable")

            with col2:
                if grafana_dashboards_IMG.exists():
                    st.image(str(grafana_dashboards_IMG), caption="Dashboards Ingestion, training et Data Grafana", width=900)
                else:
                    st.error("❌ Image grafana_dashboardsicon introuvable")

            col1, col2 = st.columns(2)
            with col1:
                if grafana_pipelineHealth_IMG.exists():
                    st.image(str(grafana_pipelineHealth_IMG), caption="Dashboard Grafana Pipeline Health - Vision sur l'ensemble des runs", width=600)
                else:
                    st.error("❌ Image grafana_pipelineHealth introuvable")

            with col2:
                if grafana_dataQuality_IMG.exists():
                    st.image(str(grafana_dataQuality_IMG), caption="Dashboard Grafana data Quality - Focus sur un run", width=600)
                else:
                    st.error("❌ Image grafana_dataQuality introuvable")

            col1, col2 = st.columns(2)
            with col1:
                if grafana_dataDrift_IMG.exists():
                    st.image(str(grafana_dataDrift_IMG), caption="Dashboard Grafana Drift - PSI notes et genres - Note moyenne", width=600)
                else:
                    st.error("❌ Image grafana_dataDrift introuvable")

            with col2:
                if grafana_KPIannuels_IMG.exists():
                    st.image(str(grafana_KPIannuels_IMG), caption="Dashboard Grafana KPIs annuels - Notes, new users & movies, note moyenne, PSI genre, %rRomance", width=600)
                else:
                    st.error("❌ Image grafana_KPIannuels introuvable")

            col1, col2 = st.columns(2)
            with col1:
                if grafana_KPImensuels_IMG.exists():
                    st.image(str(grafana_KPImensuels_IMG), caption="Dashboard Grafana KPIs mensuels - Nb notes, note moyenne - (new) users, movies", width=600)
                else:
                    st.error("❌ Image grafana_KPImensuels introuvable")

            with col2:
                if grafana_monitoringQuo_IMG.exists():
                    st.image(str(grafana_monitoringQuo_IMG), caption="Dashboard Grafana Monitoring quotidien - Notes, (new) users, new movies", width=600)
                else:
                    st.error("❌ Image grafana_monitoringQuo introuvable")

            st.divider()

            st.markdown("## 🚧 Monitoring Produit & Infrastructure — À mettre en place")

            col1, col2 = st.columns(2)

            with col1:
                st.info("""
                ### 🎬 Monitoring Produit (API de reco)

                🔹 Nombre de requêtes, taux d'erreur  
                🔹 Latence moyenne & p95  
                🔹 Nombre d'utilisateurs, taux de rebond, temps d'utilisation  
                🔹 Cold-start (nouveaux utilisateurs / nouveaux films)  
                🔹 Taux d'adoption / satisfaction des recommandations  

                👉 Objectif : mesurer l'usage réel et la performance en ligne
                """)

            with col2:
                st.info("""
                ### 🖥 Monitoring Infrastructure

                🔹 Charge CPU des containers  
                🔹 Utilisation mémoire  
                🔹 Espace disque (base PostgreSQL + artifacts MLflow)  
                🔹 Disponibilité des services  
                🔹 Temps de réponse base de données  

                👉 Objectif : garantir stabilité et scalabilité
                """)

            st.warning("""
            💡 Évolution prévue :
            - Intégration Prometheus + Grafana pour métriques techniques  
            - Mise en place d'alertes automatiques (latence, erreurs, drift critique, baisse de pertinence des prédictions)  
            - Dashboard unifié : Data + Modèle + Produit + Infra  
            """)

            st.divider()

            st.markdown("""
            ## 🔧 Stratégie de maintenance — Prochaines étapes

            Le pipeline est aujourd'hui monitoré (ingestion, drift, training, promotion automatique).  
            **La prochaine étape consiste à renforcer sa robustesse via la formulation de règles de gestion et de l'automatisation :**  

            - 🔄 Ré-entraînement conditionnel en cas de drift ou baisse de performance
            - 🚦 Validation automatique des métriques avant promotion modèle  
            - 🔁 Stratégie formalisée de rollback via l'alias MLflow `production`  
            - 🧪 Tests automatisés ingestion → snapshot → training (CI)  
            - 🚀 Déploiement API Docker automatisé (CD)
            """)

            st.warning("Objectif : passer d'un pipeline fonctionnel en phase test à un système MLOps sécurisé et industrialisable.")

            st.divider()

            st.markdown("""
                ### 🏗 Industrialisation — CI/CD 
                **Objectif : sécuriser le pipeline d'ingestion, le modèle ML et le déploiement API pour garantir un système fiable et industrialisable**  
                (empêcher qu'un code, des données ou un modèle dégradé atteigne la production).  
                👉 Passer d'un pipeline monitoré en phase de test  à un système sécurisé et automatisé en production.
                """)

            col1, col2 = st.columns(2)

            with col1:
                st.success("""
                ## ✔️ INITIÉ (Phase test)

                🔹 Pipeline d'ingestion monitoré (durée, statut, volumétrie)  
                🔹 Monitoring data drift & KPI métiers  
                🔹 Training monitoré (logs SQL + MLflow)  
                🔹 Promotion automatique du meilleur modèle  
                🔹 Containers Docker existants  
                🔹 Orchestration batch quotidienne  

                👉 Pipeline fonctionnel et monitoré
                """)

            with col2:
                st.info("""
                ## 🚧 À METTRE EN PLACE

                ### 🔄 CI (Avant merge)
                - Lint automatique (qualité code)
                - Tests unitaires ingestion / snapshot / training / prédiction
                - Seuil de validation du modèle
                - Blocage automatique si régression

                ### 🚀 CD (Après merge)
                - Build Docker automatisé via GitHub Actions
                - Déploiement automatique API
                - Promotion modèle conditionnelle
                - Rollback version précédente si dégradation

                👉 Passage à un système industrialisable
                """)

            st.divider()

            st.warning("""
            🔒 Prochaine étape clé :
            Coupler monitoring + CI/CD pour empêcher toute régression data, modèle ou API d'atteindre la production.
            """)

    elif section == "Conclusion & perspectives":
        slide_header("Conclusion & perspectives")
        st.markdown("""
          #### Ce projet a permis de concevoir un **système de recommandation de films**, structuré autour d'une approche MLOps.
                    
        ## 🔍 Rappel des objectifs MLOps visés
        - Faciliter la prise en main et le déploiement du produit  
        - Garantir la reproductibilité des entraînements  
        - Assurer la fiabilité et la stabilité à long terme  

        ---

        ## ✔️ Ce qui a été accompli

        - Mise en place d'un pipeline batch automatisé : ingestion → snapshot → training → promotion → déploiement  
        - Architecture append-only avec vues "current" garantissant traçabilité et historisation  
        - Monitoring transverse des process et des données via Grafana (qualité, KPI, drift)  
        - Suivi des performances modèle (recall@K, ndcg@K) via MLflow  
        - Promotion automatique du meilleur modèle  
        - Versioning des données (DVC), du code (Git), des modèles (MLflow)
        - Conteneurisation Docker de tous les services pour la reproductibilité  

        Le système ne repose pas sur une classification ou une régression classique,
        mais sur un **algorithme de recommandation collaborative**,  
        où la dérive vient principalement des **évolutions de comportements utilisateurs** (notations)
        et des problématiques de **cold-start**.

        ---

        ## 🚧 Ce qui reste à mettre en place

        - Validation automatique des métriques avant promotion  
        - Gestion robuste du cold-start et suivi de la couverture modèle    
        - Monitoring produit (usage API, latence, adoption des recommandations)  
        - Monitoring infrastructure (CPU, mémoire, disponibilité)  
        - Formalisation d'une stratégie de maintenance (automatisation de la génération et de la gestion des alertes, notamment retraining conditionnel )  
        - Industrialisation complète via CI/CD     
        """)

        st.warning("🚀 Le projet passe ainsi d'un pipeline fonctionnel en phase de test à une base solide pour un système de recommandation industrialisable.")

# =========================
# PARTIE 2 : DÉMONSTRATION (Live App)
# =========================
def show_demo_mode():
    st.markdown("## 🍿 Démonstration Live")
    
    # Vérification Healthcheck
    api_is_alive = check_api_health()
    if api_is_alive:
        st.sidebar.success(f"🟢 API Connectée")
    else:
        st.sidebar.error(f"🔴 API Déconnectée")
        st.error(f"Impossible de contacter l'API sur : {API_URL}")
        return

    # Tabs pour différentes fonctionnalités de démo
    tab1, tab2, tab3, tab4 = st.tabs([
        "👤 Recommandation Utilisateur", 
        "🔥 Films Populaires", 
        "🧠 Métadonnées Modèle",
        "🖥️ Statut Système"
    ])

    # --- TAB 1: RECO USER ---
    with tab1:
        st.subheader("Simuler un utilisateur")
        
        col_input, col_action = st.columns([1, 2])
        with col_input:
            user_id = st.number_input("User ID", min_value=1, value=1, step=1)
            top_n = st.slider("Nombre de films", min_value=1, max_value=10, value=5)
            
        btn_reco = st.button("✨ Générer les recommandations", type="primary")

        if btn_reco:
            with st.spinner(f"Calcul des recommandations pour User {user_id}..."):
                try:
                    response = requests.get(f"{API_URL}/recommend", params={"user_id": user_id, "top_n": top_n})
                    
                    if response.status_code == 200:
                        data = response.json()
                        recos = data.get("recommendations", [])
                        
                        if not recos:
                            st.warning("Aucune recommandation trouvée.")
                        else:
                            st.success(f"Top {len(recos)} pour l'utilisateur {user_id}")
                            df_reco = pd.DataFrame(recos)
                            calc_height = (35 * len(recos) + 38)
                            
                            st.dataframe(
                                df_reco,
                                column_config={
                                    "movie_id": st.column_config.NumberColumn("ID Film", format="%d"),
                                    "title": st.column_config.TextColumn("Titre du film"),
                                    "score": st.column_config.ProgressColumn(
                                        "Score de Pertinence",
                                        format="%.3f",
                                        min_value=0,
                                        max_value=df_reco["score"].max() + 0.5,
                                    ),
                                },
                                hide_index=True,
                                use_container_width=True,
                                height=calc_height
                            )
                            with st.expander("Voir la réponse JSON brute"):
                                st.json(data)
                    else:
                        st.error(f"Erreur API : {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Erreur de connexion : {str(e)}")

    # --- TAB 2: POPULAR ---
    with tab2:
        st.subheader("Scénario Cold Start")
        if st.button("Charger les populaires"):
            try:
                res = requests.get(f"{API_URL}/movies/popular", params={"limit": 20})
                if res.status_code == 200:
                    pop_movies = res.json()
                    flat_data = []
                    for m in pop_movies:
                        flat_data.append({
                            "Titre": m["title"],
                            "Score Bayésien": m["stats"]["score"],
                            "Note Moyenne": m["stats"]["mean_rating"],
                            "Nb Votes": m["stats"]["count"]
                        })
                    st.dataframe(pd.DataFrame(flat_data), use_container_width=True)
            except Exception as e:
                st.error(e)

    # --- TAB 3: INFO MODEL ---
    with tab3:
        st.subheader("📦 Observabilité du Modèle")
        c_refresh, c_link = st.columns([1, 4])
        with c_refresh:
            btn_refresh = st.button("🔄 Rafraîchir Métadonnées")
        with c_link:
             # Utilisation du lien EXTERNE pour le navigateur de l'utilisateur
             st.link_button("🚀 Ouvrir MLFlow UI", MLFLOW_EXTERNAL_URL)

# Remplace le bloc "if btn_refresh:" dans tab3 par ceci :

        if btn_refresh:
            try:
                res_config = requests.get(f"{API_URL}/model/config")
                res_meta = requests.get(f"{API_URL}/model/metadata")
                
                config_data = res_config.json()
                meta_data = res_meta.json()

                st.markdown("#### 🆔 Identité du Run MLflow")
                col1, col2, col3 = st.columns(3)
                run_id = meta_data.get("run_id", "N/A")
                col1.metric("Run ID", run_id)
                col2.metric("Version Modèle", meta_data.get("model_version", "Latest"))
                col3.metric("Status", "Production", delta="Active")
                
                st.divider()
                st.markdown("#### ⚙️ Hyperparamètres")
                if config_data and "detail" not in config_data:
                    st.table(pd.DataFrame(list(config_data.items()), columns=["Paramètre", "Valeur"]))
                else:
                    st.warning("Configuration non disponible.")

                # ✅ NOUVEAU BLOC : Métriques
                st.divider()
                st.markdown("#### 📊 Métriques du Modèle")
                metrics = meta_data.get("metrics", {})
                if metrics:
                    cols = st.columns(len(metrics))
                    for col, (k, v) in zip(cols, metrics.items()):
                        col.metric(k, f"{v:.4f}" if isinstance(v, float) else v)
                else:
                    st.info("Aucune métrique enregistrée pour ce run.")

                # ✅ NOUVEAU BLOC : Tags / Dataset Hash
                st.divider()
                st.markdown("#### 🏷️ Tags & Traçabilité Dataset")
                tags = meta_data.get("tags", {})
                dvc_hash = tags.get("dvc_dataset_hash", "N/A")
                git_commit = tags.get("git_commit", "N/A")

                col1, col2 = st.columns(2)
                col1.code(f"DVC Hash : {dvc_hash}", language="text")
                col2.code(f"Git Commit : {git_commit}", language="text")

                if tags:
                    with st.expander("Voir tous les tags"):
                        st.table(pd.DataFrame(list(tags.items()), columns=["Tag", "Valeur"]))

            except Exception as e:
                st.error(f"Impossible de récupérer les métadonnées : {e}")
        
        st.markdown("---")
        st.subheader("🛠️ Actions Rapide")
        if st.button("Ré-entraîner le modèle", type="secondary"):
            status_train = st.status("Démarrage du pipeline d'entraînement...", expanded=True)
            try:
                res_train = requests.post(f"{API_URL}/training", timeout=600)
                if res_train.status_code == 200:
                    status_train.update(label="Entraînement terminé !", state="complete", expanded=False)
                    st.success("Nouveau modèle entraîné !")
                    st.balloons()
                else:
                    status_train.update(label="Erreur", state="error")
                    st.error(res_train.text)
            except Exception as e:
                status_train.update(label="Erreur connexion", state="error")
                st.error(str(e))

    # --- TAB 4: SYSTEM STATUS ---
    with tab4:
        st.subheader("🖥️ Santé du Système")
        
        # Section Diagnostics
        if st.button("Lancer les Diagnostics", type="primary"):
            status_container = st.status("Analyse des composants...", expanded=True)
            
            # 1. API Latency
            t0 = time.time()
            try:
                requests.get(f"{API_URL}/health", timeout=2)
                latency = (time.time() - t0) * 1000
                api_ok = True
            except:
                latency = 0
                api_ok = False
            
            # 2. MLFlow (Internal Check)
            try:
                mf_res = requests.get(MLFLOW_INTERNAL_URL, timeout=1)
                mlflow_ok = (mf_res.status_code == 200)
            except:
                mlflow_ok = False
            
            # 3. Deep Check
            try:
                ready_res = requests.get(f"{API_URL}/ready", timeout=5)
                checks = ready_res.json().get("checks", {})
                db_status = checks.get("database", "error")
                model_status = checks.get("model", "error")
            except:
                db_status = "unreachable"
                model_status = "unreachable"

            status_container.update(label="Terminé !", state="complete", expanded=False)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("FastAPI", "En Ligne" if api_ok else "Hors Ligne", f"{latency:.0f} ms" if api_ok else None)
            col2.metric("PostgreSQL", "Connecté" if db_status == "connected" else "Erreur")
            col3.metric("MLFlow Server", "Accessible" if mlflow_ok else "Inaccessible")
            col4.metric("Modèle IA", "Chargé" if model_status == "ready" else "Erreur")
            
        # --- MODIFICATION ICI : SECTION SORTIE DU IF ---
        st.divider()
        st.markdown("#### 🛠️ Actions Rapides")
        
        # Le bouton est maintenant au premier niveau, pas besoin de cliquer sur diagnostics avant
        if st.button("Relancer Ingestion Données"):
                try:
                    with st.spinner("Pipeline ingestion en cours..."):
                        requests.post(f"{API_URL}/data")
                        st.toast("Pipeline ingestion lancé avec succès !", icon="🚀")
                        st.success("Ingestion terminée.")
                except:
                    st.error("Échec appel API")

# =========================
# MAIN
# =========================
st.title(APP_TITLE)
mode = st.sidebar.selectbox("Choisir le mode :", ["Présentation (Slides)", "Application Démo"])
if mode == "Présentation (Slides)":
    show_presentation_mode()
else:
    show_demo_mode()