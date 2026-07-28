"""Dependances partagees de l'API."""

from functools import lru_cache
from pathlib import Path

from senegal_rental_price.models.predict import ModelBundle, load_bundle


@lru_cache(maxsize=1)
def get_model() -> ModelBundle:
    """Charge le modele une seule fois par processus."""
    import os

    path = Path(os.getenv("MODEL_PATH", "models/random_forest.joblib"))
    return load_bundle(path)
