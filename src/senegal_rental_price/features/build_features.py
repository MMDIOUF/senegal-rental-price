"""Features deterministes partagees entre entrainement et inference."""

import pandas as pd

EQUIPMENT_COLUMNS = [
    "climatisation",
    "parking",
    "gardiennage",
    "piscine",
    "groupe_electrogene",
]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Transforme la liste d'equipements en indicateurs explicites."""
    featured = frame.copy()
    equipment_text = featured["equipements"].fillna("").astype(str).str.lower()
    for equipment in EQUIPMENT_COLUMNS:
        featured[f"equip_{equipment}"] = equipment_text.str.contains(equipment, regex=False).astype(
            int
        )
    featured["nb_equipements"] = featured[[f"equip_{item}" for item in EQUIPMENT_COLUMNS]].sum(
        axis=1
    )
    featured["meuble"] = featured["meuble"].astype(int)
    return featured.drop(columns=["equipements"])
