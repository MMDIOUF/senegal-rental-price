"""Chargement et utilisation du bundle de prediction."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import joblib
import pandas as pd

from senegal_rental_price.features.build_features import build_features


class Regressor(Protocol):
    """Contrat minimal d'un pipeline de regression."""

    def predict(self, data: pd.DataFrame) -> Any:
        """Retourne une prediction pour chaque ligne."""


@dataclass(frozen=True)
class ModelBundle:
    """Modele et metadonnees versionnees ensemble."""

    model: Regressor
    metadata: dict[str, Any]


def load_bundle(path: str | Path) -> ModelBundle:
    """Charge un artefact joblib et controle sa structure."""
    payload = cast(dict[str, Any], joblib.load(path))
    if "model" not in payload or "metadata" not in payload:
        raise ValueError("Artefact modele invalide")
    return ModelBundle(model=cast(Regressor, payload["model"]), metadata=payload["metadata"])


def predict_price(bundle: ModelBundle, features: dict[str, Any]) -> float:
    """Applique exactement les memes features qu'a l'entrainement."""
    frame = build_features(pd.DataFrame([features]))
    prediction = float(bundle.model.predict(frame)[0])
    return max(0.0, prediction)
