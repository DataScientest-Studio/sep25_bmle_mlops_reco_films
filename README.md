# Reco Films MLOps

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)](#)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-4169E1?logo=postgresql&logoColor=white)](#)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)](#)
[![Grafana](https://img.shields.io/badge/Grafana-Monitoring-F46800?logo=grafana&logoColor=white)](#)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-13ADC7?logo=dvc&logoColor=white)](#)
[![Docker Compose](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](#)

</div>

> Systeme de recommandation de films industrialise MLOps : 
> ingestion MovieLens 20M, entrainement item-based CF, tracking MLflow, versioning DVC, serving FastAPI, interface Streamlit 
> + monitoring data, model & pipeline quotidien (observabilité ingestion des données + training)

---

## Vue d'ensemble

Ce repository contient une plateforme de recommandation orientee production avec une architecture microservices.

- Dataset: MovieLens 20M (ratings, movies, tags, etc.)
- Modele: Item-Based Collaborative Filtering (`kNN cosine` sur matrice sparse)
- Serving: API FastAPI + frontend Streamlit
- MLOps: 
    - MLflow (experiences + registry), 
    - DVC (versioning data), 
    - instrumentation Prometheus pour le monitoring API
    - monitoring Grafana : data (volumétrie, KPI, drift) 
                           pipeline ingestion + training (statut, erreurs, temps d'exécution)
- Objectif metier cle: gerer le cold-start avec fallback de popularite bayesienne

---

## Architecture

### Services Docker

| Service | Role | Port |
|---|---|---|
| `postgres` | Base applicative (`raw.*` tables, `current_*` views) | `5432` |
| `mlflow-db` | Base PostgreSQL du tracking MLflow | interne |
| `mlflow` | Tracking server + model registry | `5000` |
| `api` | FastAPI (recommandation, pipeline triggers, observabilite) | `8000` |
| `frontend` | App Streamlit (slides + demo live) | `8501` |
| `grafana` | Monitoring runs ingestion + training, drift et KPIs | `3000` |

### Flux end-to-end

```mermaid
flowchart LR
    A[MovieLens 20M] --> B[Ingestion + SQL raw schema]
    B --> C[Snapshot Parquet]
    C --> D[Training ItemCF]
    D --> E[MLflow Tracking + Registry]
    D --> F[SQL tables item_neighbors/movie_popularity]
    E --> G[Alias production]
    F --> H[FastAPI Serving]
    G --> H
    H --> I[Streamlit Demo]
    B --> M[Monitoring Metrics PostgreSQL]
    D --> M
    M --> N[Grafana Dashboards]
```

### Automatisation

![Architecture MLOps](reports/figures/Architecture_with_grafana.png)

---

## Pipelines

### 1) Pipeline ingestion data

- Telechargement MovieLens (`src/ingestion/ingestion_movielens.py`)
- Initialisation DB + schema `raw` (`src/ingestion/init_db.py`)
- Chargement SQL par chunks des CSV
- Versioning DVC des donnees brutes (`data/raw.dvc`)

### 2) Pipeline entrainement modele

- Creation du snapshot `data/training_set.parquet` (`src/ingestion/create_snapshot.py`)
- Entrainement item-item kNN (`src/models/train_model2.py`)
- Evaluation ranking (`recall@10`, `ndcg@10`)
- Log des artefacts (`item_neighbors.parquet`, `movie_popularity.parquet`)
- Enregistrement dans MLflow Registry (`reco-films-itemcf-v2`)
- Promotion de la meilleure version (`src/models/promote_best_model.py`)

### 3) Pipeline serving online

- Chargement du modele `models:/reco-films-itemcf-v2@production`
- Warm-up cache modele au demarrage API
- Recommandations personnalisees via `/recommend`
- Endpoints fallback/diagnostic (`/movies/popular`, `/ready`, `/model/*`)

### 4) Pipeline data monitoring
Calcul des indicateurs alimentant le monitoring data (`src/monitoring/run_training_with_monitoring.py`) :
- Qualité des données (`src/monitoring/quality_checks.py`)
- Data drift + KPI métiers (`src/monitoring/data_monitoring.py`)


---

## Strategie cold-start

Deux fallbacks concrets sont implementes:

- Nouveaux utilisateurs: fallback sur les films populaires via score bayesien (`raw.movie_popularity`)
- Nouveaux films: filtrage par seuil minimum de ratings avant entree dans le graphe de voisins

Cette strategie garde des recommandations robustes meme avec peu ou pas d'historique utilisateur.

---

## Pratiques MLOps implementees

- Architecture microservices avec Docker Compose
- Versioning donnees avec DVC
- Tracabilite modele avec MLflow (params, metrics, artefacts, aliases)
- Checks de qualite SQL (`src/ingestion/check_db.py`)
- Reproductibilite via scripts pipeline
- Observabilite API via `prometheus-fastapi-instrumentator`
- Dashboards Grafana initialisés en phase test
- Documentation/demo Streamlit pour la communication projet

---

## Quickstart

### Prerequis

- Docker + Docker Compose
- Optionnel hors Docker: Python 3.10

### Lancer la stack

```bash
docker compose up --build
```
### Pipeline Data (Via API)

<img width="754" height="416" alt="data pipeline" src="https://github.com/user-attachments/assets/3476b9a7-dd61-4274-bf52-b251a5aefb5f" />

### Pipeline Training (Via API)

<img width="834" height="454" alt="training pipeline" src="https://github.com/user-attachments/assets/566ea15c-072e-4603-b97f-172628c02087" />

### Détermination du Modèle de Prod

A lancer à la racine du projet:  
````
python -m src.models.promote_best_model
````

## Automatisation

### Pré-requis  
* Git, DVC, Python installés.
* Accès au dépôt Git
* Crédential pour le DVC

### Configuration du dvc  
````
dvc remote modify --local origin url 'https://dagshub.com/pierreB-boop/sep25_bmle_mlops_reco_films.dvc'  
dvc remote modify --local origin auth basic  
dvc remote modify --local origin user 'votre_username_dagshub'  
dvc remote modify --local origin password 'votre_token_dagshub'  
````

### Lancement du Pipeline complet avec enregistrement des indicateurs pour monitoring Grafana   
````
chmod +x daily_pipeline_with_monitoring.sh  
bash daily_pipeline_with_monitoring.sh
````

### Automatisation via cronjobs  
````
crontab -e
0 2 * * * /bin/bash /home/user/sep25_bmle_mlops_reco_films/daily_pipeline_with_monitoring.sh #A remplacer par votre chemin absolu
````

## Points d'acces

- Frontend Streamlit: `http://localhost:8501`
- FastAPI: `http://localhost:8000`
- Docs API: `http://localhost:8000/docs`
- MLflow UI: `http://localhost:5000`
- Grafana: `http://localhost:3000`

---

## Endpoints API principaux - initialisation suivi Prometheus

### Sante systeme

- `GET /health`: statut simple
- `GET /ready`: readiness DB + modele

### Recommandation et catalogue

- `GET /recommend?user_id=1&top_n=5`
- `GET /movies/popular?limit=10`
- `GET /movies/{movie_id}`

### Observabilite modele

- `GET /model/metadata`
- `GET /model/config`

### Triggers pipeline

- `POST /data`: ingestion + reload cache
- `POST /training`: snapshot + training + update modele

---

## Structure du projet

```text
.
|-- database/
    |-- monitoring_metrics.sql
|-- docker-compose.yml
|-- grafana/
    |-- dashboards/
    |-- provisioning/
|-- main_user_api.py
|-- daily_pipeline.sh
|-- daily_pipeline_with_monitoring.sh
|-- data/
|-- mlartifacts/
|-- reports/figures/
`-- src/
    |-- ingestion/
    |-- models/
    |-- monitoring/
        |-- backfill_monitoring.py
    |-- streamlit/
    `-- visualization/
```

---

## Execution manuelle (hors Docker)

```bash
# 1) Ingestion + collecte indicateurs monitoring run
python -m src.monitoring.run_ingestion_with_monitoring

# 1bis) Collecte des indicateurs de monitoring data
python -m src.monitoring.run_data_monitoring_pipeline 

# 2) Snapshot
python src/ingestion/create_snapshot.py

# 3) Training + indicateurs monitoring run
python -m src.monitoring.run_training_with_monitoring

# 4) Promotion
python -m src.models.promote_best_model

# 5) API
uvicorn main_user_api:app --host 0.0.0.0 --port 8000

# 6) Frontend
streamlit run src/streamlit/project_prez.py
```

---

## Restaurer le snapshot des métriques de monitoring (Grafana)

Permet d’alimenter **Grafana** sans recalcul historique long.  
Remplit la table `raw.monitoring_metrics`.

#### Option 1 — Restaurer le dump (recommandé)
```bash
docker exec -i movie_reco_postgres \
psql -U movie -d movie_reco < database/monitoring_metrics.sql
```

#### Option 2 — Reconstruction complète
Recalcule l’historique depuis les données brutes (⚠️ Opération plus longue - volumes importants) :
```bash
python src/monitoring/backfill_monitoring.py
```

---

## Notes

- `daily_pipeline.sh` orchestre Git + DVC + ingestion + training + promotion.
- Les artefacts MLflow sont persistants dans `./mlartifacts`.
- Le serving online evite la reinference lourde en temps reel grace a des artefacts precalcules.

- `daily_pipeline_with_monitoring.sh` orchestre Git + DVC + ingestion + training + promotion + calcul des indicateurs de monitoring  
    - Temps de calcul et statuts erreur Ingestion + training  
    - Data drift + KPI métiers  
---

## Equipe sep25 bootcamp MLE

- Pierre Barbetti
- Raphael Da Silva
- Martine Mateus
- Laurent Piacentile
