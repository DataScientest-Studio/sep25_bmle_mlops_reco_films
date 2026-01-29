Project Name
==============================

This project is a starting Pack for MLOps projects based on the subject "movie_recommandation". It's not perfect so feel free to make some modifications on it.

Project Organization
------------

    ├── .dvc               <- DVC metadata and cache
    ├── .dvcignore         <- DVC ignore rules
    ├── .github
    │   └── workflows      <- CI workflows
    ├── .gitignore         <- Git ignore rules
    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── data
    │   ├── db             <- Local database files
    │   │   └── staging.sqlite
    │   ├── raw            <- Raw MovieLens CSV files
    │   └── raw.dvc        <- DVC pointer for raw data
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    ├── notebooks          <- Jupyter notebooks
    ├── references         <- Data dictionaries, manuals, and explanatory materials
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures for reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        ├── config         <- Parameters used by training and prediction scripts
        │
        ├── data           <- Scripts to download or generate data
        │   ├── __init__.py
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   ├── __init__.py
        │   └── build_features.py
        │
        ├── ingestion      <- Data ingestion scripts
        │   └── ingestion_movielens.py
        │
        ├── models         <- Scripts to train models and make predictions
        │   ├── __init__.py
        │   ├── predict_model.py
        │   └── train_model.py
        │
        ├── quality        <- Data quality checks
        │   └── check_db.py
        │
        └── visualization  <- Visualization scripts
            ├── __init__.py
            └── visualize.py

--------

Data & DVC
------------
- Raw data lives in `data/raw/`
- The `data/raw.dvc` file tracks that folder with DVC
- Local databases are stored in `data/db/`

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
