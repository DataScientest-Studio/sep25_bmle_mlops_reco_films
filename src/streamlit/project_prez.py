# project_prez.py
# ============================================================
# Système de reommandation de films — Streamlit (Soutenance)
# ============================================================

from __future__ import annotations

from operator import le
import os
from pathlib import Path
from tokenize import Comment
import numpy as np
import pandas as pd
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Paths
# =========================
ROOT = Path(__file__).resolve().parents[2]


# =========================
# Figures Visualization (PNG)
# =========================
MOVIELENS_IMG = ROOT / "src" / "streamlit" / "movielens.png"
DATA_IMG = ROOT / "src" / "streamlit" / "pipeline_data_IMG.png"
viz1_IMG = ROOT / "Reports" / "figures" / "visualize_Figure_1.png"
viz2_IMG = ROOT / "Reports" / "figures" / "visualize_Figure_2.png"

SQL1_IMG = ROOT / "Reports" / "figures" / "SQL1.png"
archi_IMG = ROOT / "Reports" / "figures" / "architecture_MLOps.png"

# =========================
# Config & constants
# =========================
st.set_page_config(
    page_title="Système de recommandation de films (Soutenance)",
    page_icon="🫀",
    layout="wide",
)

APP_TITLE = "🎬 Création d'un système de recommandation de films"
N_SAMPLES = 300




# Figures (PNG)
DEFAULT_FIG_DIRS = [
    ROOT / "Reports" / "figures",
    ROOT / "reports" / "figures",
    ROOT / "assets",
    ROOT / "Assets",
]




# =========================
# UI helpers (style "slides")
# =========================
def slide_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
    st.markdown("---")


def key_takeaways(title, items: list[str]) -> None: ### AFFICHE LES MESSAGES CLES SOUS FORME DE LISTE A PUCE 
    st.markdown(f"### ✅ {title}")
    for it in items:
        st.markdown(f"- **{it}**")


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




# =========================
# Header
# =========================
st.title(APP_TITLE)
png_map = list_pngs_in_known_dirs()

# =========================
# Sidebar — navigation
# =========================
st.sidebar.header("🧭 Navigation") ## DEFINITION DES CHAPITRES 
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
st.sidebar.header("⚙️ Soutenance Projet MLOps — sep25_bmle")
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
            **Automatisation du processus d'ingestion de nouvelles données via un cronjob**  
            - Insertion automatique de nouvelles données dans la base PostgreSQL
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
            **Versioning des données**  
            - 
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
            **Entre 1995 et 2015  :  20 millions de notations   -  138 000 noteurs  -  27 000 films évalués.**  
            Sur les dernières années : entre 120 et 220 votants par jour, 8 et 16 notes par session de notation, plus de 70 notes par utilisateur par an.
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
    )  
    # ==========================================================
    # MODÈLE
    # ==========================================================
    st.subheader("Modèle de recommandation de films basé sur le filtrage collaboratif item-based")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        ### 🔹 Principe
        - Chaque film est représenté par son vecteur de notes utilisateurs.
        - **La similarité entre films est calculée avec la cosine similarity** basée sur l’angle entre les vecteurs :
            - si l'angle est petit (cosine proche de 1), les films sont similaires
            - si l'angle est proche de 90° (cosine proche de 0), les films sont orthogonaux (pas de similarité)
            - si l'angle est obtu (cosine négatif), les goûts sont opposés
        - Pour chaque film, on conserve ses K plus proches voisins (calcul offline).
        - **En recommandation utilisateur** (online):
            1. On sélectionne les films bien notés par l’utilisateur.
            2. On récupère leurs voisins similaires.
            3. On calcule un score pondéré par la note et la similarité (plus il apparaît souvent parmi les voisins, plus le score est élevé).
            4. Les films déjà vus sont exclus.
        """)



    with col2:  
        st.info("""
        ### 🔹 Gestion du cold-start utilisateur
        Si l’utilisateur possède peu ou pas d’historique :
        - Recommandation basée sur un score de popularité bayésien.
        - Permet d’éviter de survaloriser les films avec peu de notes.

        ### 🔹 Caractéristiques
        - Entraînement offline (pré-calcul des voisinages).
        - Inférence rapide.
        - Modèle explicable (décomposition des contributions).
        - Approche orientée ranking (Top-N).
        """)

    # ==========================================================
    # MÉTRIQUES
    # ==========================================================
    st.markdown("---")
    st.subheader("Métriques d’évaluation")

    st.markdown("""
    Le modèle est optimisé pour la recommandation Top-N (pas pour la prédiction exacte des notes).
    Il est donc évalué sur sa capacité à bien classer les films pertinents dans les premières positions.
    **Les métriques sont orientées ranking**.
    """)

    st.markdown("### 🔹 Precision@K")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        Proportion de films pertinents dans les K recommandations.
        """)

    with col2:
        st.latex(r"""
        Precision@K =
        \frac{\text{Nombre de films pertinents recommandés}}
            {K}
        """)

    st.markdown("### 🔹 Recall@K")
    col1, col2 = st.columns(2)
    with col1:        
        st.info("""
        Proportion de films retrouvés dans les K recommandations parmi les films pertinents.
        """)

    with col2:
        st.latex(r"""
        Recall@K =
        \frac{\text{Nombre de films pertinents recommandés}}
            {\text{Nombre total de films pertinents}}
        """)


    

   

elif section == "Suivi des Expériences via MLflow":
    slide_header(
        "Suivi des Expériences via MLflow",        
    )     
    st.write("""
        ### Objectifs  
        Tracer efficacement les expériences d'entraînement  
        Versionner données et modèles  
        Créer des pipelines reproductibles  
             
        ### Outils utilisés
             
        ### Screenshots de l'interface MLflow / Démo
  
    """)




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
        **👉 Faciliter la prise en main du produit par les équipes de développement et de data science :**  
        **https://github.com/DataScientest-Studio/sep25_bmle_mlops_reco_films.git** :  
              -->  Documentation du projet via README pour expliquer les différentes composantes du projet, les instructions d’utilisation et de contribution, les bonnes pratiques à suivre, etc.  
              -->  Code dans src commenté et structuré pour faciliter la compréhension et la contribution des équipes de développement et de data science  
                
        Pipelines de traitement des données et d'entraînement ML basique automatisé  
                
        Processus de déploiement des microservices (ex : Docker, Kubernetes, etc.) pour gérer les environnements de production  
                
        API de serving (ex : FastAPI, Flask, etc.) pour exposer le modèle en production  
        """)

        st.info("""
        **👉 Garantir la reproductibilité des résultats :**  
        Versioning du code (Git)   
                
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
            """Data Drift :**  
            - mise à jour des données et actualisation du modèle quotidiennes,  
            - monitoring pour détecter les dérives  
            """,
            """Evaluation du modèle de recommandation :**  
                precision@K, recall@K et ndcgs@K pour évaluer la qualité des recommandations  
            """, 
            """Cold-start :**  
            - nouveaux utilisateurs : recommandation basée sur un score de popularité bayésien  
            - nouveaux films : recommandés dès lors qu’ils ont reçu un nombre minimum de notes  
            """]
            )






