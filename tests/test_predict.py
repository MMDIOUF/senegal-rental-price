from typing import Any

import pandas as pd
import pytest

from senegal_rental_price.models.predict import ModelBundle, predict_price


class FakeModel:
    def __init__(self, result: float) -> None:
        self.result = result
        self.seen_columns: list[str] = []

    def predict(self, data: pd.DataFrame) -> list[float]:
        self.seen_columns = list(data.columns)
        return [self.result]


def sample_features() -> dict[str, Any]:
    return {
        "ville": "Dakar",
        "quartier": "Yoff",
        "type_bien": "Appartement",
        "surface_m2": 70.0,
        "nb_pieces": 3,
        "nb_chambres": 2,
        "meuble": False,
        "equipements": "parking",
    }


def test_predict_price_builds_features() -> None:
    model = FakeModel(325_400)
    result = predict_price(ModelBundle(model, {"version": "test"}), sample_features())
    assert result == pytest.approx(325_400)
    assert "equip_parking" in model.seen_columns


def test_predict_price_never_returns_negative_price() -> None:
    result = predict_price(ModelBundle(FakeModel(-25), {}), sample_features())
    assert result == 0
