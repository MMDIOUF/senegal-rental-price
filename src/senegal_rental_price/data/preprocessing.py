"""Validation et nettoyage des donnees tabulaires."""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "ville",
    "quartier",
    "type_bien",
    "surface_m2",
    "nb_pieces",
    "nb_chambres",
    "meuble",
    "equipements",
    "prix_loyer_mensuel",
}


def validate_columns(frame: pd.DataFrame) -> None:
    """Leve une erreur explicite si le schema minimal est incomplet."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {sorted(missing)}")


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les valeurs manquantes, doublons et valeurs impossibles."""
    validate_columns(frame)
    cleaned = frame.copy()
    cleaned["quartier"] = cleaned["quartier"].fillna("Non renseigne").astype(str)
    cleaned["equipements"] = cleaned["equipements"].fillna("").astype(str)
    cleaned["meuble"] = cleaned["meuble"].astype("boolean").fillna(False).astype(bool)
    numeric = ["surface_m2", "nb_pieces", "nb_chambres", "prix_loyer_mensuel"]
    for column in numeric:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.dropna(subset=numeric)
    cleaned = cleaned[
        (cleaned["surface_m2"] > 0)
        & (cleaned["nb_pieces"].between(1, 20))
        & (cleaned["nb_chambres"].between(0, 15))
        & (cleaned["prix_loyer_mensuel"] > 0)
    ]
    return cleaned.drop_duplicates().reset_index(drop=True)


def load_and_clean(raw_path: str | Path, processed_path: str | Path) -> pd.DataFrame:
    """Charge, nettoie et persiste les donnees traitees."""
    frame = clean_data(pd.read_csv(raw_path))
    destination = Path(processed_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame
