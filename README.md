# Senegal Rental Price

[![CI](https://github.com/OWNER/REPOSITORY/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPOSITORY/actions/workflows/ci.yml)

Projet M2 DSIA de mise en production d'un modèle de prédiction des loyers au Sénégal.
L'objectif principal est la qualité de la chaîne MLOps : reproductibilité, typage, tests,
configuration, API, suivi d'expériences, conteneurs et CI.

## Démarrage rapide

Prérequis : Python 3.11 et Docker.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install ".[dev]"
senegal-rental-train
```

Comparer les deux modèles :

```bash
python -m senegal_rental_price.models.train model=ridge
python -m senegal_rental_price.models.train model=random_forest
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Hydra permet de modifier un paramètre sans changer le code :

```bash
python -m senegal_rental_price.models.train model=random_forest model.params.max_depth=8
```

## Exécution locale

```bash
uvicorn api.main:app --reload
streamlit run frontend/app.py
```

- Front : http://localhost:8501
- API : http://localhost:8000
- Documentation OpenAPI : http://localhost:8000/docs

## Démonstration Docker

Le modèle doit avoir été entraîné une fois avant le build.

```bash
docker compose -f docker/docker-compose.yml up --build
```

## Qualité

```bash
ruff check .
black --check .
mypy --strict src/
pytest --cov=src --cov-fail-under=70
```

## Architecture

Le notebook est réservé à l'exploration. La logique réutilisable vit dans le package `src/`.
Le pipeline scikit-learn sérialise prétraitement et modèle ensemble, ce qui empêche les écarts
entre entraînement et prédiction. FastAPI charge cet artefact une fois au démarrage. Le front
ne connaît pas le modèle : il appelle exclusivement l'API.

## Limites

- Les données sont synthétiques et ne reflètent pas toutes les dynamiques du marché réel.
- Seules cinq villes et quatre catégories de biens sont supportées.
- Il n'y a ni authentification, ni surveillance de dérive, ni réentraînement automatique.
- Le registre est un stockage MLflow local pour la démonstration ; une production réelle
  utiliserait un backend distant et un stockage d'artefacts persistant.

Avant de publier le dépôt, remplacer `OWNER/REPOSITORY` dans le badge CI par l'adresse GitHub
réelle du groupe.
