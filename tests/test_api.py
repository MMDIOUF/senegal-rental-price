from typing import Any

import pandas as pd
from fastapi.testclient import TestClient

from api.dependencies import get_model
from api.main import app
from senegal_rental_price.models.predict import ModelBundle


class FakeModel:
    def predict(self, _: pd.DataFrame) -> list[float]:
        return [487_600]


METADATA: dict[str, Any] = {
    "version": "test-1",
    "model_name": "fake",
    "trained_at": "2026-07-27T10:00:00+00:00",
    "metrics": {"mae": 10.0, "rmse": 12.0, "r2": 0.9},
    "mlflow_run_id": "test-run",
    "training_rows": 100,
}


def override_model() -> ModelBundle:
    return ModelBundle(FakeModel(), METADATA)


app.dependency_overrides[get_model] = override_model
client = TestClient(app)


def payload() -> dict[str, Any]:
    return {
        "ville": "Dakar",
        "quartier": "Mermoz",
        "type_bien": "Appartement",
        "surface_m2": 90,
        "nb_pieces": 4,
        "nb_chambres": 3,
        "meuble": False,
        "equipements": ["parking", "climatisation"],
    }


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info() -> None:
    response = client.get("/model/info")
    assert response.status_code == 200
    assert response.json()["model_name"] == "fake"


def test_predict_rounds_to_thousand() -> None:
    response = client.post("/predict", json=payload())
    assert response.status_code == 200
    assert response.json()["prix_loyer_mensuel_estime"] == 488_000


def test_invalid_surface_is_rejected_with_422() -> None:
    invalid = payload()
    invalid["surface_m2"] = 0
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422
    assert "surface_m2" in str(response.json())


def test_inconsistent_rooms_are_rejected() -> None:
    invalid = payload()
    invalid["nb_chambres"] = 5
    invalid["nb_pieces"] = 2
    assert client.post("/predict", json=invalid).status_code == 422
