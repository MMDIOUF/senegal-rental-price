import pandas as pd
import pytest

from senegal_rental_price.data.preprocessing import clean_data, validate_columns


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ville": "Dakar",
                "quartier": None,
                "type_bien": "Appartement",
                "surface_m2": 80,
                "nb_pieces": 3,
                "nb_chambres": 2,
                "meuble": None,
                "equipements": None,
                "prix_loyer_mensuel": 450_000,
            }
        ]
    )


def test_clean_data_fills_optional_values() -> None:
    cleaned = clean_data(valid_frame())
    assert cleaned.loc[0, "quartier"] == "Non renseigne"
    assert cleaned.loc[0, "equipements"] == ""
    assert bool(cleaned.loc[0, "meuble"]) is False


def test_clean_data_drops_invalid_surface() -> None:
    frame = valid_frame()
    frame.loc[0, "surface_m2"] = -2
    assert clean_data(frame).empty


def test_validate_columns_reports_missing_column() -> None:
    with pytest.raises(ValueError, match="prix_loyer_mensuel"):
        validate_columns(valid_frame().drop(columns=["prix_loyer_mensuel"]))
