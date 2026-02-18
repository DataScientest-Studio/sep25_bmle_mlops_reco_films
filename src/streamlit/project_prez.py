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
archi_IMG = ROOT / "Reports" / "figures" / "architecture_MLOps.png"

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

    if section == "Contexte & objectifs": 
        slide_header("Contexte & objectifs")
        
        st.markdown("## 🧪 **Cadre du projet**")
        col1, col2 = st.columns(2)
        with col1:
            if MOVIELENS_IMG.exists():
                st.image(str(MOVIELENS_IMG), use_container_width=True)
            else:
                st.warning(f"Image introuvable: {MOVIELENS_IMG}")

        with col2:  
            st.markdown("""
        ### Objectif : construire un système de recommandation de films en production
        - Application de **collaborative filtering** et/ou **content based filtering**.
        - Finalité : disposer d'une application de recommandation de films pour les utilisateurs.
        - Aspects spécifiques du projet :  
                      - traiter la problématique du Data Drift,  
                      - monitorer le modèle (bonne vs mauvaise recommandation),  
                      - résoudre la problématique de cold-start.
        """)
            
        st.markdown("""
        ## 🎯 **Enjeux : projet dédié aux pratiques MLOps**
        ### Focus sur la performance de l'architecture construite autour du modèle :
        -  les microservices doivent fonctionner de manière fluide et intégrée
        -  les environnements doivent être reproductibles
        -  la surveillance doit être continue
        -  la documentation doit être claire
        """)

        st.subheader("Schéma d'implémentation de l'architecture MLOps")
        if archi_IMG.exists():
            st.image(str(archi_IMG), caption="Schéma MLOps", use_container_width=True)
        else:
            st.warning("Image introuvable: architecture_MLOps.png")

    elif section == "Pipeline d'ingestion de données":
        slide_header("🧷 Pipeline d'ingestion de données")
        st.subheader("Ingestion de nouvelles données")
        col1, col2 = st.columns(2)
        with col1:
            if DATA_IMG.exists():
                st.image(str(DATA_IMG), caption="Schéma Pipeline Data", use_container_width=True)
            else:
                st.warning("Image introuvable: pipeline_data_IMG.png")
        with col2:
            st.success("""
                **Automatisation du processus d'ingestion**
                - Insertion automatique via cronjob
                - Versioning des données 
                - Processus de validation (Data Quality)
            """)

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
        slide_header("🔎 Modèle & métriques d’évaluation", "Architecture algorithmique & logique d'évaluation ranking")
        st.subheader("🎯 Modèle : Item-Based Collaborative Filtering")

        col1, col2 = st.columns(2)
        with col1:
            st.info("""
            ### 🔹 Principe mathématique
            Chaque film est représenté par un **vecteur de notes utilisateurs**.
            La similarité est calculée avec la **cosine similarity**.
            Le voisinage est **pré-calculé offline**.
            """)
            st.latex(r"sim(i,j) = \frac{v_i \cdot v_j}{||v_i|| \cdot ||v_j||}")
        with col2:
            st.info("""
            ### 🔹 Logique de recommandation (online)
            1️⃣ Sélection des films aimés par l’utilisateur  
            2️⃣ Récupération des voisins similaires  
            3️⃣ Score pondéré par similarité  
            4️⃣ Exclusion déjà vus -> Classement Top-N
            """)

        st.markdown("---")
        st.subheader("🧊 Gestion du Cold-Start")
        col1, col2 = st.columns(2)
        with col1:
            st.success("### 🔹 Nouveaux utilisateurs\nFallback vers un **score de popularité bayésien**.")
        with col2:
            st.success("### 🔹 Nouveaux films\nRecommandé seulement si nombre min de ratings atteint.")

        st.markdown("---")
        st.subheader("📊 Métriques d’évaluation (Ranking Metrics)")
        col1, col2 = st.columns(2)
        with col1:
            st.info("### 🔹 Precision@K\nProportion de recommandations pertinentes parmi les K proposées.")
        with col2:
            st.info("### 🔹 Recall@K\nCapacité à retrouver les films pertinents dans le Top-K.")
        
        st.markdown("---")
        st.subheader("🏆 NDCG@K (métrique principale)")
        col1, col2 = st.columns(2)
        with col1:
            st.info("NDCG prend en compte la pertinence et la position.")
        with col2:
            st.latex(r"NDCG@K = \frac{DCG@K}{IDCG@K}")

    elif section == "Suivi des Expériences via MLflow":
        slide_header("📊 Suivi des Expériences via MLflow", "Traçabilité, reproductibilité et gouvernance modèle")
        st.subheader("🎯 Objectifs MLOps")
        col1, col2 = st.columns(2)
        with col1:
            st.info("✔️ Tracer entraînements, Logger hyperparamètres, Sauvegarder artefacts")
        with col2:
            st.info("✔️ Versioning modèles, Registry centralisé, Promotion production")

        st.markdown("---")
        st.subheader("🔄 Cycle de vie du modèle")
        st.markdown("1️⃣ Entraînement → 2️⃣ Log → 3️⃣ Registry → 4️⃣ Promotion alias `production`")

        st.markdown("---")
        st.subheader("🏷️ Model Registry & Alias Production")
        st.success("Chargement via : `models:/reco-films-itemcf-v2@production`")

    elif section == "API user & DS":
        slide_header("API user & DS")
        st.write("Interface entre le modèle, la DB et l'utilisateur via FastAPI.")

    elif section ==  "Monitoring & maintenance":
        slide_header("📈 Monitoring & maintenance")
        st.write("""
            **Monitoring ingestion** : succès/échec, volumétrie.  
            **Monitoring Data Quality** : checks validés ou non.  
            **Monitoring Drift** : évolution moyenne des notes.  
            **Monitoring Modèle** : métriques techniques (latency) et métier (coverage).
        """)

    elif section == "Conclusion & perspectives":
        slide_header("Conclusion & perspectives")    
        st.subheader("MVP vs Next steps")
        col1, col2 = st.columns(2)
        with col1:
            st.info("**👉 Faciliter la prise en main** : Documentation, Code propre, Pipelines auto.")
            st.info("**👉 Reproductibilité** : Git, MLflow, DVC.")
        with col2:
            st.success("**👉 Fiabilité** : CI/CD, Monitoring, Rollback strategy.")

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

        if btn_refresh:
            try:
                res_config = requests.get(f"{API_URL}/model/config")
                res_meta = requests.get(f"{API_URL}/model/metadata")
                
                config_data = res_config.json()
                meta_data = res_meta.json()

                st.markdown("#### 🆔 Identité du Run MLflow")
                col1, col2, col3 = st.columns(3)
                run_id = meta_data.get("run_id", "N/A")
                col1.metric("Run ID", run_id[:8] + "..." if len(run_id) > 8 else run_id)
                col2.metric("Version Modèle", meta_data.get("version", "Latest"))
                col3.metric("Status", "Production", delta="Active")
                
                st.divider()
                st.markdown("#### ⚙️ Hyperparamètres")
                if config_data and "detail" not in config_data:
                    st.table(pd.DataFrame(list(config_data.items()), columns=["Paramètre", "Valeur"]))
                else:
                    st.warning("Configuration non disponible.")

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