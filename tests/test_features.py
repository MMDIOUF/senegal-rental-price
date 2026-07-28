import pandas as pd

from senegal_rental_price.features.build_features import build_features


def test_build_features_encodes_equipment_and_furnished() -> None:
    frame = pd.DataFrame(
        [
            {
                "ville": "Dakar",
                "quartier": "Yoff",
                "type_bien": "Studio",
                "surface_m2": 35,
                "nb_pieces": 1,
                "nb_chambres": 0,
                "meuble": True,
                "equipements": "parking|climatisation",
            }
        ]
    )
    result = build_features(frame)
    assert result.loc[0, "equip_parking"] == 1
    assert result.loc[0, "equip_piscine"] == 0
    assert result.loc[0, "nb_equipements"] == 2
    assert result.loc[0, "meuble"] == 1
    assert "equipements" not in result
