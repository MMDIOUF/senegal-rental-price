# Senegal Rental Price

[![CI](https://github.com/MMDIOUF/senegal-rental-price/actions/workflows/ci.yml/badge.svg)](https://github.com/MMDIOUF/senegal-rental-price/actions/workflows/ci.yml)
[![CD](https://github.com/MMDIOUF/senegal-rental-price/actions/workflows/cd.yml/badge.svg)](https://github.com/MMDIOUF/senegal-rental-price/actions/workflows/cd.yml)

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

Comparer les trois modèles sur le même split stratifié :

```bash
python -m senegal_rental_price.models.train model=ridge
python -m senegal_rental_price.models.train model=random_forest
python -m senegal_rental_price.models.train model=gradient_boosting
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Hydra permet de modifier un paramètre sans changer le code :

```bash
python -m senegal_rental_price.models.train model=gradient_boosting model.params.max_depth=2
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

## Bonus CD - livraison continue

Après une CI réussie sur `main`, le workflow `.github/workflows/cd.yml` reconstruit le
Gradient Boosting puis publie deux images dans GitHub Container Registry :

- `ghcr.io/mmdiouf/senegal-rental-api:latest`
- `ghcr.io/mmdiouf/senegal-rental-frontend:latest`

Chaque image reçoit également un tag immuable `sha-<commit>`, utilisable pour revenir à une
version précise. Le workflow peut aussi être lancé manuellement depuis GitHub Actions.

Déployer les images publiées :

```powershell
$env:IMAGE_TAG = "latest"
docker compose -f docker/docker-compose.release.yml pull
docker compose -f docker/docker-compose.release.yml up -d
docker compose -f docker/docker-compose.release.yml ps
```

Revenir à une version connue :

```powershell
$env:IMAGE_TAG = "sha-<identifiant_commit>"
docker compose -f docker/docker-compose.release.yml pull
docker compose -f docker/docker-compose.release.yml up -d
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
Le Gradient Boosting est retenu après comparaison de MAE, RMSE, R², MAPE et biais par ville.
Le pipeline scikit-learn sérialise prétraitement et modèle ensemble, ce qui empêche les écarts
entre entraînement et prédiction. FastAPI charge cet artefact une fois au démarrage. Le front
ne connaît pas le modèle : il appelle exclusivement l'API.

## Limites

- Les données sont synthétiques et ne reflètent pas toutes les dynamiques du marché réel.
- Seules cinq villes et quatre catégories de biens sont supportées.
- Il n'y a ni authentification, ni surveillance de dérive, ni réentraînement automatique.
- Le registre est un stockage MLflow local pour la démonstration ; une production réelle
  utiliserait un backend distant et un stockage d'artefacts persistant.
