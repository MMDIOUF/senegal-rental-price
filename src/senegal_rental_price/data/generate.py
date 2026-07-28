"""Generation reproductible d'annonces locatives synthetiques."""

from pathlib import Path

import numpy as np
import pandas as pd

from senegal_rental_price.utils.logger import configure_logging, get_logger

LOGGER = get_logger(__name__)

CITY_BASE = {
    "Dakar": 180_000,
    "Thiès": 90_000,
    "Saint-Louis": 82_000,
    "Mbour": 95_000,
    "Saly": 140_000,
}
QUARTER_FACTORS = {
    "Almadies": 2.00,
    "Mermoz": 1.55,
    "Plateau": 1.75,
    "Yoff": 1.20,
    "Parcelles Assainies": 0.90,
    "Centre": 1.00,
}
EQUIPMENT_CHOICES = ["climatisation", "parking", "gardiennage", "piscine", "groupe_electrogene"]


def generate_dataset(rows: int = 1_200, seed: int = 42) -> pd.DataFrame:
    """Cree un jeu synthetique coherent sans reproduire de vraies annonces."""
    rng = np.random.default_rng(seed)
    cities = rng.choice(list(CITY_BASE), size=rows, p=[0.55, 0.14, 0.10, 0.11, 0.10])
    property_types = rng.choice(
        ["Appartement", "Maison", "Studio", "Villa"], size=rows, p=[0.48, 0.24, 0.16, 0.12]
    )
    surface = np.clip(rng.gamma(4.0, 24.0, rows), 18, 500).round(1)
    rooms = np.clip(np.rint(surface / 28 + rng.normal(0, 0.8, rows)), 1, 12).astype(int)
    bedrooms = np.maximum(0, rooms - rng.integers(1, 3, rows))
    furnished = rng.random(rows) < 0.27

    quarters: list[str] = []
    equipments: list[str] = []
    prices: list[int] = []
    type_factor = {"Appartement": 1.0, "Maison": 1.08, "Studio": 0.92, "Villa": 1.55}
    for index, city in enumerate(cities):
        quarter = (
            str(
                rng.choice(
                    ["Almadies", "Mermoz", "Plateau", "Yoff", "Parcelles Assainies"],
                    p=[0.12, 0.18, 0.10, 0.25, 0.35],
                )
            )
            if city == "Dakar"
            else "Centre"
        )
        selected = [equipment for equipment in EQUIPMENT_CHOICES if rng.random() < 0.22]
        equipment_bonus = len(selected) * 18_000 + (70_000 if "piscine" in selected else 0)
        expected = (
            CITY_BASE[str(city)]
            + surface[index] * (2_500 if city == "Dakar" else 1_450)
            + rooms[index] * 12_000
            + equipment_bonus
        )
        expected *= QUARTER_FACTORS[quarter] * type_factor[str(property_types[index])]
        if furnished[index]:
            expected *= 1.22
        noisy_price = expected * rng.lognormal(mean=0, sigma=0.13)
        quarters.append(quarter)
        equipments.append("|".join(selected))
        prices.append(int(round(noisy_price / 5_000) * 5_000))

    frame = pd.DataFrame(
        {
            "ville": cities,
            "quartier": quarters,
            "type_bien": property_types,
            "surface_m2": surface,
            "nb_pieces": rooms,
            "nb_chambres": bedrooms,
            "meuble": furnished,
            "equipements": equipments,
            "prix_loyer_mensuel": prices,
        }
    )
    missing_indices = rng.choice(frame.index, size=max(1, rows // 50), replace=False)
    frame.loc[missing_indices, "quartier"] = None
    return frame


def main() -> None:
    """Genere les donnees brutes attendues par la configuration par defaut."""
    configure_logging()
    output = Path("data/raw/locations_senegal.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_dataset().to_csv(output, index=False)
    LOGGER.info("Jeu synthetique genere: %s", output)


if __name__ == "__main__":
    main()
