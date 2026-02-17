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
import matplotlib.pyplot as plt # Import du Streamlit 1

# =========================
# Paths
# =========================
ROOT = Path(__file__).resolve().parents[2]

# =========================
# Figures Visualization (PNG) - Chemins (Issus du Streamlit 1)
# =========================
MOVIELENS_IMG = ROOT / "src" / "streamlit" / "movielens.png"
DATA_IMG = ROOT / "src" / "streamlit" / "pipeline_data_IMG.png"
viz1_IMG = ROOT / "Reports" / "figures" / "visualize_Figure_1.png"
viz2_IMG = ROOT / "Reports" / "figures" / "visualize_Figure_2.png"
SQL1_IMG = ROOT / "Reports" / "figures" / "SQL1.png"
archi_IMG = ROOT / "Reports" / "figures" / "architecture_MLOps.png"

# Figures (PNG) Helpers du Streamlit 1
DEFAULT_FIG_DIRS = [
    ROOT / "Reports" / "figures",
    ROOT / "reports" / "figures",
    ROOT / "assets",
    ROOT / "Assets",
]

# =========================
# Config & constants
# =========================
st.set_page_config(
    page_title="Système de recommandation de films (Soutenance)",
    page_icon="🎬",
    layout="wide",
)

APP_TITLE = "🎬 Création d'un système de recommandation de films"
API_URL = "http://127.0.0.1:8000"  # URL de ton API FastAPI
MLFLOW_UI_URL = "http://127.0.0.1:5000" # URL par défaut de MLFlow

# =========================
# UI Helpers (Mix des deux versions)
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

# Helpers spécifiques Streamlit 1 pour la gestion des images
def find_first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None

def list_pngs_in_known_dirs() -> dict[str, Path]:
    """
    Retourne un mapping {nom_fichier: chemin} pour les PNG trouvés.
    Permet d’afficher facilement des figures si elles existent localement.
    """
    found: dict[str, Path] = {}
    for d in DEFAULT_FIG_DIRS:
        if d.exists() and d.is_dir():
            for p in d.glob("*.png"):
                found[p.name] = p
    return found

def show_png_if_exists(filename_contains: str, png_map: dict[str, Path], caption: str | None = None) -> bool:
    """
    Affiche la 1ère image PNG dont le nom contient 'filename_contains' (case-insensitive).
    Retourne True si affichée, False sinon.
    """
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
# PARTIE 1 : PRÉSENTATION (Contenu du Streamlit 1)
# =========================
def show_presentation_mode():
    
    png_map = list_pngs_in_known_dirs()

    # =========================
    # Sidebar — navigation (Interne à la fonction pour le mode présentation)
    # =========================
    st.sidebar.markdown("---")
    st.sidebar.header("🧭 Navigation Slides") 
    SECTIONS = [
        "Contexte & objectifs",
        "Pipeline d'ingestion de données",
        "Bases de données PostgreSQL",
        "Modèle & métriques d’évaluation",
        "Suivi des Expériences via MLflow",    
        "API user & DS",
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

    # =========================
    # Sections     
    # =========================
    if section == "Contexte & objectifs": 
        slide_header(
            "Contexte & objectifs",
        )
        
        st.markdown("""
        ## 🧪 **Cadre du projet**""")
        col1, col2 = st.columns(2)
        with col1:
            if MOVIELENS_IMG.exists():
                st.image(
                    str(MOVIELENS_IMG),
                    use_container_width=True
                )
            else:
                st.error("❌ MOVIELENS.png introuvable")

        with col2:  
            st.markdown("""
        ### Objectif : construire un système de recommandation de films en production
        - Application de **collaborative filtering** et/ou **content based filtering**.
        - Finalité : disposer d'une application de recommandation de films pour les utilisateurs.
        - Aspects spécifiques du projet :  
                      - traiter la problématique du Data Drift,  
                      - monitorer le modèle (bonne vs mauvaise recommandation),  
                      - résoudre la problématique de cold-start pour les nouveaux utilisateurs et les nouveaux films.
        """)
            
                
        st.markdown("""
        ## 🎯 **Enjeux : projet dédié aux pratiques MLOps**
        ### Focus sur la performance de l'architecture construite autour du modèle :
        -  les microservices doivent fonctionner de manière fluide et intégrée
        -  les environnements doivent être reproductibles avec des flux de travail automatisés
        -  la surveillance doit être continue, avec une stratégie de maintenance efficace pour assurer la fiabilité à long terme du modèle
        -  la documentation doit être claire et complète pour faciliter la prise en main du projet par les équipes de développement et de data science
        """)

        st.subheader("Schéma d'implémentation de l'architecture MLOps")
        if archi_IMG.exists():
            st.image(
                str(archi_IMG),
                caption="Schéma d'implémentation de l'architecture MLOps",
                use_container_width=True
            )
        else:
            st.error("❌ architecture_MLOps.png introuvable")

    elif section == "Pipeline d'ingestion de données":
        slide_header(
            "🧷 Pipeline d'ingestion de données",        
        )
        st.subheader("Ingestion de nouvelles données")
        col1, col2 = st.columns(2)
        with col1:
            if DATA_IMG.exists():
                st.image(
                    str(DATA_IMG),
                    caption="Schéma de la base de données PostgreSQL",
                    use_container_width=True
                )
            else:
                st.error("❌ pipeline_data_IMG.png introuvable")
        with col2:
            st.success("""
                **Automatisation du processus d'ingestion de nouvelles données via un cronjob** - Insertion automatique de nouvelles données dans la base PostgreSQL
                - Versioning des données 
                - Processus de validation des données (checks qualité, alertes en cas de données manquantes ou incohérentes)
            """)

    elif section == "Bases de données PostgreSQL":
        slide_header(
            "Bases de données",        
        )    
        st.subheader("Architecture de la base de données PostgreSQL")
        col1, col2 = st.columns(2)
        with col1:
            if SQL1_IMG.exists():
                st.image(
                    str(SQL1_IMG),
                    caption="Schéma de la base de données PostgreSQL",
                    use_container_width=True
                )
            else:
                st.error("❌ SQL1_IMG.png introuvable")
        with col2:
            st.success("""
                **Versioning des données** - 
            """)

        st.write("---")
        st.subheader("📊 Exploration des données MovieLens")
        st.markdown("*https://grouplens.org/datasets/movielens/20m/** ")
        col1, col2 = st.columns(2)
        with col1:
            if viz1_IMG.exists():
                st.image(
                    str(viz1_IMG),
                    caption="MovieLens — En chiffres",
                    use_container_width=True
                )
                st.info("""
                **Entre 1995 et 2015  :  20 millions de notations   -  138 000 noteurs  -  27 000 films évalués.** Sur les dernières années : entre 120 et 220 votants par jour, 8 et 16 notes par session de notation, plus de 70 notes par utilisateur par an.
                """)
            else:
                st.error("❌ viz1_IMG.png introuvable")
        with col2:
            if viz2_IMG.exists():
                st.image(
                    str(viz2_IMG),
                    caption="MovieLens — En graphiques",
                    use_container_width=True
                )
            else:
                st.error("❌ viz2_IMG.png introuvable")

    elif section == "Modèle & métriques d’évaluation":
        slide_header(
            "🔎 Modèle & métriques d’évaluation",
            "Architecture algorithmique & logique d'évaluation ranking"
        )

        # ==========================================================
        # MODÈLE
        # ==========================================================
        st.subheader("🎯 Modèle : Item-Based Collaborative Filtering")

        col1, col2 = st.columns(2)

        with col1:
            st.info("""
            ### 🔹 Principe mathématique

            Chaque film est représenté par un **vecteur de notes utilisateurs**.

            La similarité entre deux films est calculée avec la **cosine similarity** :

            - Cosine ≈ 1 → films très similaires  
            - Cosine ≈ 0 → pas de similarité  
            - Cosine < 0 → préférences opposées  

            Le voisinage est **pré-calculé offline** (K plus proches voisins par film).
            """)

            st.latex(r"""
            sim(i,j) = \frac{v_i \cdot v_j}{||v_i|| \cdot ||v_j||}
            """)

        with col2:
            st.info("""
            ### 🔹 Logique de recommandation (online)

            1️⃣ Sélection des films bien notés par l’utilisateur  
            2️⃣ Récupération de leurs voisins similaires  
            3️⃣ Score pondéré par similarité × note utilisateur  
            4️⃣ Exclusion des films déjà vus  
            5️⃣ Classement Top-N

            ✔️ Inférence rapide  
            ✔️ Modèle explicable  
            ✔️ Adapté au ranking
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
            Fallback vers un **score de popularité bayésien** :

            - Moyenne pondérée
            - Correction pour faible nombre de votes
            - Évite le biais des films avec peu de notes
            """)

        with col2:
            st.success("""
            ### 🔹 Nouveaux films
            Un film est recommandé seulement s’il atteint :
            - un nombre minimum de ratings
            - un score suffisant

            👉 Garantit robustesse & qualité.
            """)

        st.markdown("---")

        # ==========================================================
        # MÉTRIQUES
        # ==========================================================
        st.subheader("📊 Métriques d’évaluation (Ranking Metrics)")

        st.markdown("""
        Le modèle est optimisé pour la **recommandation Top-N**,  
        et non pour la prédiction exacte des notes.

        L’objectif est de maximiser la qualité du classement.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.info("""
            ### 🔹 Precision@K
            Proportion de recommandations pertinentes parmi les K proposées.

            👉 Mesure la qualité immédiate du Top-K.
            """)

            st.latex(r"""
            Precision@K =
            \frac{|\{films\ pertinents\} \cap \{TopK\}|}{K}
            """)

        with col2:
            st.info("""
            ### 🔹 Recall@K
            Capacité à retrouver les films pertinents dans le Top-K.

            👉 Mesure la couverture des préférences utilisateur.
            """)

            st.latex(r"""
            Recall@K =
            \frac{|\{films\ pertinents\} \cap \{TopK\}|}
            {|\{films\ pertinents\}|}
            """)

        st.markdown("---")

        st.subheader("🏆 NDCG@K (métrique principale du projet)")

        col1, col2 = st.columns(2)

        with col1:
            st.info("""
            NDCG prend en compte :
            - la pertinence
            - la position dans le classement

            👉 Une recommandation pertinente en position 1 vaut plus
            qu’en position 10.
            """)

        with col2:
            st.latex(r"""
            DCG@K = \sum_{i=1}^{K}
            \frac{rel_i}{\log_2(i+1)}
            """)

            st.latex(r"""
            NDCG@K = \frac{DCG@K}{IDCG@K}
            """)

        st.markdown("---")

        key_takeaways("Pourquoi ces métriques ?", [
            "Projet orienté ranking et non régression",
            "Optimisation basée sur NDCG@10",
            "Alignement avec les standards des systèmes de recommandation industriels",
        ])

    elif section == "Suivi des Expériences via MLflow":
        slide_header(
            "📊 Suivi des Expériences via MLflow",
            "Traçabilité, reproductibilité et gouvernance modèle"
        )

        st.subheader("🎯 Objectifs MLOps")

        col1, col2 = st.columns(2)

        with col1:
            st.info("""
            ✔️ Tracer chaque entraînement  
            ✔️ Logger hyperparamètres  
            ✔️ Logger métriques (recall@10, ndcg@10)  
            ✔️ Sauvegarder artefacts (modèle PyFunc)  
            ✔️ Garantir reproductibilité
            """)

        with col2:
            st.info("""
            ✔️ Versioning des modèles  
            ✔️ Registry centralisé  
            ✔️ Promotion contrôlée en production  
            ✔️ Historique complet des runs  
            ✔️ Auditabilité (git commit)
            """)

        st.markdown("---")

        st.subheader("🔄 Cycle de vie du modèle")

        st.markdown("""
        1️⃣ Entraînement → `mlflow.start_run()`  
        2️⃣ Log des paramètres & métriques  
        3️⃣ Log du modèle via `mlflow.pyfunc.log_model()`  
        4️⃣ Enregistrement dans le **Model Registry** 5️⃣ Promotion automatique si métrique meilleure  
        6️⃣ Chargement via alias `production`
        """)

        st.markdown("---")

        st.subheader("🏷️ Model Registry & Alias Production")

        col1, col2 = st.columns(2)

        with col1:
            st.success("""
            Le modèle n’est jamais appelé par numéro de version.

            Il est chargé via :

            models:/reco-films-itemcf-v2@production
            """)

        with col2:
            st.success("""
            👉 Décorrélation totale entre :
            - code de serving
            - version du modèle

            La promotion modifie uniquement l’alias.
            """)

        st.markdown("---")

        st.subheader("📈 Métriques loggées automatiquement")

        st.markdown("""
        - recall_10  
        - ndcg_10  
        - paramètres (k_neighbors, min_ratings…)  
        - tags (git_commit)  
        - artefacts modèle  
        """)

        st.markdown("---")

        st.subheader("🚀 Promotion automatique")

        st.info("""
        Script `promote_best_model.py` :

        - Compare les versions enregistrées
        - Sélectionne la meilleure selon NDCG@10
        - Met à jour l’alias `production`
        """)

        st.markdown("---")

        key_takeaways("Valeur ajoutée MLflow dans ce projet :", [
            "Traçabilité complète des expérimentations",
            "Reproductibilité garantie",
            "Déploiement sécurisé via alias",
            "Approche alignée standards MLOps industriels",
        ])    

    elif section == "API user & DS":
        slide_header(
            "API user & DS")

        st.write("""
        # L’API est l’interface entre le modèle, la base de données et l’utilisateur.  
        Il n’est pas obligatoire, dans le cadre de ce projet, d’y intégrer une interface graphique.  
        En revanche, cette API devra intégrer une notion d’authentification des différents types d’utilisateurs/administrateurs 
        qui devront l’utiliser.  
        Cette partie doit détailler les différents endpoints que vous souhaitez intégrer à votre API, 
        la manière dont cette dernière fera appel à la base de données, au modèle, écrire dans les logs 
        et éventuellement modifier la base de données. 
        """)

    elif section ==  "Monitoring & maintenance":
        slide_header(
            "📈 Monitoring & maintenance",
        )

        st.write("""
            ## Stratégie de déploiement du monitoring et de la maintenance du modèle en production   
              
            ### Monitoring du processus d’ingestion de nouvelles données
                - statut de la dernière ingestion (succès/échec)
                - durée de la dernière ingestion
                - nombre de lignes chargées lors de la dernière ingestion
                - nombre total de notes en base (indicateur de croissance du dataset)
                 
            ### Vérification qualité des données
                - nombre de checks qualité réalisés 
                - nombre de checks qualité ayant échoué
                - statut du dernier run de vérification qualité (succès/échec)
              
            ### KPI & Monitoring drift data
                - nombre de notes mensuelles (indicateur de croissance du dataset)
                - note moyenne mensuelle (indicateur de dérive potentielle des notes)
              
            ### Monitoring du modèle en production
                - durée du dernier entraînement
                - statut du dernier entraînement (succès/échec)
                - precision@K, recall@K et ndcg@K du train du modèle en production
                - coverage users du modèle en production
                - nouveauté des recommandations (ex : proportion de films recommandés qui n’ont pas été vus par l’utilisateur)
    

        """)

    elif section == "Conclusion & perspectives":
        slide_header(
            "Conclusion & perspectives",
        )    

        st.subheader("Composantes clés de l'architecture MLOps : MVP vs Next steps")
        col1, col2 = st.columns(2)
        with col1:
            st.info("""
            **👉 Faciliter la prise en main du produit par les équipes de développement et de data science :** **https://github.com/DataScientest-Studio/sep25_bmle_mlops_reco_films.git** :  
                  -->  Documentation du projet via README pour expliquer les différentes composantes du projet, les instructions d’utilisation et de contribution, les bonnes pratiques à suivre, etc.  
                  -->  Code dans src commenté et structuré pour faciliter la compréhension et la contribution des équipes de développement et de data science  
                 
            Pipelines de traitement des données et d'entraînement ML basique automatisé  
                 
            Processus de déploiement des microservices (ex : Docker, Kubernetes, etc.) pour gérer les environnements de production  
                 
            API de serving (ex : FastAPI, Flask, etc.) pour exposer le modèle en production  
            """)

            st.info("""
            **👉 Garantir la reproductibilité des résultats :** Versioning du code (Git)   
                 
            Système de suivi des expériences organisé (ex : MLflow, Weights & Biases, etc.)
                pour tracer les métriques d'entraînement et d'inférence, les hyperparamètres, les artefacts (modèles, figures, etc.) et les métadonnées des expériences  

            Registre de modèles (ex : MLflow, Sagemaker, etc.) pour stocker et versionner les modèles entraînés  
                 
            Versioning des données (ex : DVC, Git LFS, etc.) pour la traçabilité des jeux de données utilisés pour l'entraînement et l'inférence
            """)

        with col2:
            st.success("""
            **👉 S'assurer de la fiabilité à long terme du système** via une stratégie de maintenance efficace :  
            CI/CD (ex : GitHub Actions, Jenkins, etc.) pour automatiser les tests et le déploiement des microservices  
                 
            Monitoring des performances  
            - Détection de data drift  
            - Système d'alerte  
                 
            Pipeline de ré-entraînement automatisé : périodique, basé sur les performances ou sur la détection de data drift  
                 
            Stratégie de rollback en cas de défaillance du modèle en production
            """)

            key_takeaways("Aspects spécifiques du projet :", [
                """Data Drift :** - mise à jour des données et actualisation du modèle quotidiennes,  
                - monitoring pour détecter les dérives  
                """,
                """Evaluation du modèle de recommandation :** precision@K, recall@K et ndcgs@K pour évaluer la qualité des recommandations  
                """, 
                """Cold-start :** - nouveaux utilisateurs : recommandation basée sur un score de popularité bayésien  
                - nouveaux films : recommandés dès lors qu’ils ont reçu un nombre minimum de notes  
                """]
                )

# =========================
# PARTIE 2 : DÉMONSTRATION (Live App)
# =========================
def show_demo_mode():
    st.markdown("## 🍿 Démonstration Live")
    
    # Vérification Healthcheck
    api_is_alive = check_api_health()
    if api_is_alive:
        st.sidebar.success(f"🟢 API Connectée : {API_URL}")
    else:
        st.sidebar.error(f"🔴 API Déconnectée ({API_URL})")
        st.error("L'API semble éteinte. Lancez `uvicorn main_user_api:app --reload`.")
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
            user_id = st.number_input("User ID", min_value=1, value=1, step=1, help="ID présent dans la base de données")
            # Modification demandée : Limite à 10
            top_n = st.slider("Nombre de films", min_value=1, max_value=10, value=5)
            
        btn_reco = st.button("✨ Générer les recommandations", type="primary")

        if btn_reco:
            with st.spinner(f"Calcul des recommandations pour User {user_id}..."):
                try:
                    # Appel API avec le paramètre top_n correct
                    response = requests.get(f"{API_URL}/recommend", params={"user_id": user_id, "top_n": top_n})
                    
                    if response.status_code == 200:
                        data = response.json()
                        recos = data.get("recommendations", [])
                        
                        if not recos:
                            st.warning("Aucune recommandation trouvée (ou utilisateur inconnu / sans historique).")
                        else:
                            st.success(f"Top {len(recos)} pour l'utilisateur {user_id}")
                            
                            # Création d'un DataFrame pour un affichage propre
                            df_reco = pd.DataFrame(recos)
                            
                            # Calcul de hauteur dynamique
                            calc_height = (35 * len(recos) + 38)
                            
                            st.dataframe(
                                df_reco,
                                column_config={
                                    "movie_id": st.column_config.NumberColumn("ID Film", format="%d"),
                                    "title": st.column_config.TextColumn("Titre du film"),
                                    "score": st.column_config.ProgressColumn(
                                        "Score de Pertinence",
                                        help="Somme cumulée des similarités",
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

    # --- TAB 2: POPULAR (COLD START) ---
    with tab2:
        st.subheader("Scénario Cold Start")
        st.caption("Recommandations génériques basées sur le score Bayésien pour les utilisateurs inconnus.")
        
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
                    
                    df_pop = pd.DataFrame(flat_data)
                    
                    st.dataframe(
                        df_pop,
                        column_config={
                            "Score Bayésien": st.column_config.ProgressColumn(
                                "Score Bayésien", format="%.3f", min_value=0, max_value=5
                            ),
                            "Note Moyenne": st.column_config.NumberColumn("Note Moyenne", format="%.2f ⭐")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
            except Exception as e:
                st.error(e)

    # --- TAB 3: INFO MODEL (BEAUTIFIED + MLFLOW LINK + RETRAIN) ---
    with tab3:
        st.subheader("📦 Observabilité du Modèle")
        
        # Ajout du bouton vers MLFlow
        c_refresh, c_link = st.columns([1, 4])
        with c_refresh:
            btn_refresh = st.button("🔄 Rafraîchir Métadonnées")
        with c_link:
             st.link_button("🚀 Ouvrir MLFlow UI", MLFLOW_UI_URL, help="Ouvre l'interface MLFlow dans un nouvel onglet")

        # --- Section Affichage Metadata ---
        if btn_refresh:
            try:
                # 1. Récupération Config
                res_config = requests.get(f"{API_URL}/model/config")
                config_data = res_config.json()
                
                # 2. Récupération Metadata
                res_meta = requests.get(f"{API_URL}/model/metadata")
                meta_data = res_meta.json()

                # --- Affichage Joli ---
                
                # Bloc 1: Identité du Run
                st.markdown("#### 🆔 Identité du Run MLflow")
                col1, col2, col3 = st.columns(3)
                
                run_id = meta_data.get("run_id", "N/A")
                version = meta_data.get("version", "Latest")
                status = "Production" 
                
                col1.metric("Run ID", run_id[:8] + "..." if len(run_id) > 8 else run_id)
                col2.metric("Version Modèle", version)
                col3.metric("Status", status, delta="Active", delta_color="normal")
                
                st.divider()

                # Bloc 2: Hyperparamètres (Config)
                st.markdown("#### ⚙️ Hyperparamètres")
                if config_data and "detail" not in config_data:
                    df_config = pd.DataFrame(list(config_data.items()), columns=["Paramètre", "Valeur"])
                    st.table(df_config)
                else:
                    st.warning("Configuration non disponible ou erreur.")

                # Bloc 3: Métadonnées Brutes
                with st.expander("🔍 Voir les JSON bruts (Debug)"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.caption("Config")
                        st.json(config_data)
                    with c2:
                        st.caption("Metadata")
                        st.json(meta_data)

            except Exception as e:
                st.error(f"Impossible de récupérer les métadonnées : {e}")
        else:
            st.info("Cliquez sur rafraîchir pour voir les données du modèle actuel.")
        
        # --- Section Actions (Ré-entraînement) Ajoutée ---
        st.markdown("---")
        st.subheader("🛠️ Actions Rapide")
        st.warning("Attention : Cette action déclenche un processus lourd côté serveur.")

        if st.button("Ré-entraîner le modèle", type="secondary"):
            
            # --- COLLECTE DES DONNÉES ---
            status_train = st.status("Démarrage du pipeline d'entraînement...", expanded=True)
            
            try:
                # 1. Snapshot & DVC
                status_train.write("📸 Snapshot des données & Versioning DVC...")
                # On simule un léger délai pour l'UX si l'API est trop rapide au début
                time.sleep(1) 
                
                status_train.write("⏳ Entraînement du modèle (KNN)...")
                
                # Appel API (Timeout long car entraînement)
                t0 = time.time()
                res_train = requests.post(f"{API_URL}/training", timeout=600)
                
                if res_train.status_code == 200:
                    duration = time.time() - t0
                    status_train.write(f"✅ Terminé en {duration:.1f} secondes.")
                    status_train.update(label="Entraînement terminé avec succès !", state="complete", expanded=False)
                    
                    st.success("Nouveau modèle entraîné, versionné et prêt en production !")
                    st.balloons()
                else:
                    status_train.update(label="Erreur lors de l'entraînement", state="error")
                    st.error(f"Erreur API : {res_train.text}")

            except Exception as e:
                status_train.update(label="Erreur de connexion", state="error")
                st.error(f"L'API n'a pas répondu ou a timed out : {e}")


    # --- TAB 4: SYSTEM STATUS (MODIFIÉ) ---
    with tab4:
        # Modification demandée : Retrait de "(Full Stack)"
        st.subheader("🖥️ Santé du Système")
        
        if st.button("Lancer les Diagnostics", type="primary"):
            
            # --- COLLECTE DES DONNÉES ---
            status_container = st.status("Analyse des composants en cours...", expanded=True)
            
            # 1. Test Latence API
            status_container.write("📡 Test connectivité API...")
            t0 = time.time()
            try:
                requests.get(f"{API_URL}/health", timeout=2)
                latency = (time.time() - t0) * 1000
                api_ok = True
            except:
                latency = 0
                api_ok = False
            
            # 2. Test MLFlow UI (Check Frontend)
            status_container.write("🧪 Test serveur MLFlow...")
            try:
                mf_res = requests.get(MLFLOW_UI_URL, timeout=1)
                mlflow_ok = (mf_res.status_code == 200)
            except:
                mlflow_ok = False
            
            # 3. Test API Deep Health (DB + Model)
            status_container.write("💾 Test Base de données & Modèle...")
            db_status = "Inconnu"
            model_status = "Inconnu"
            try:
                ready_res = requests.get(f"{API_URL}/ready", timeout=5)
                ready_data = ready_res.json()
                checks = ready_data.get("checks", {})
                db_status = checks.get("database", "error")
                model_status = checks.get("model", "error")
            except:
                db_status = "unreachable"
                model_status = "unreachable"

            status_container.update(label="Diagnostics terminés !", state="complete", expanded=False)

            # --- AFFICHAGE DASHBOARD ---
            
            st.markdown("### 🚦 Vue d'ensemble")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # COL 1 : API
            with col1:
                if api_ok:
                    st.metric("FastAPI", "En Ligne", f"{latency:.0f} ms")
                else:
                    st.metric("FastAPI", "Hors Ligne", "-1 ms", delta_color="inverse")
            
            # COL 2 : Postgres
            with col2:
                if db_status == "connected":
                    st.metric("PostgreSQL", "Connecté", "Ready")
                else:
                    st.metric("PostgreSQL", "Erreur", "Down", delta_color="inverse")

            # COL 3 : MLFlow UI
            with col3:
                if mlflow_ok:
                    st.metric("MLFlow UI", "Accessible", "HTTP 200")
                else:
                    st.metric("MLFlow UI", "Inaccessible", "Timeout", delta_color="inverse")
            
            # COL 4 : Model Inference
            with col4:
                if model_status == "ready":
                    st.metric("Modèle IA", "Chargé", "Production")
                else:
                    st.metric("Modèle IA", "Erreur", model_status, delta_color="inverse")
            
            st.divider()
            
            # --- DÉTAILS TECHNIQUES ---
            c_logs, c_actions = st.columns([2, 1])
            
            with c_logs:
                st.markdown("#### 📝 Logs détaillés")
                if db_status != "connected":
                    st.error(f"**Database Error:** L'API n'arrive pas à joindre PostgreSQL. ({db_status})")
                
                if model_status != "ready":
                    st.warning(f"**Model Warning:** Le modèle n'est pas correctement monté dans l'API. ({model_status})")
                
                if api_ok and db_status == "connected" and model_status == "ready":
                    st.success("Tous les systèmes sont nominaux.")

            with c_actions:
                 st.markdown("#### 🛠️ Actions Rapides")
                 if st.button("Relancer Ingestion Données"):
                     try:
                         requests.post(f"{API_URL}/data")
                         st.toast("Pipeline ingestion lancé !", icon="🚀")
                     except:
                         st.error("Échec appel API")


# =========================
# MAIN LAYOUT & ROUTER
# =========================
st.title(APP_TITLE)

# Sélecteur principal en haut de la sidebar
st.sidebar.header("🎯 Mode d'affichage")
mode = st.sidebar.selectbox("Choisir le mode :", ["Présentation (Slides)", "Application Démo"])

if mode == "Présentation (Slides)":
    show_presentation_mode()
else:
    show_demo_mode()