"""Generation reproductible d'annonces locatives synthetiques."""

from pathlib import Path

import numpy as np
import pandas as pd

from senegal_rental_price.utils.logger import configure_logging, get_logger

LOGGER = get_logger(__name__)

CITY_MARKET = {
    "Dakar": {"base": 65_000, "price_m2": 4_250},
    "Thiès": {"base": 42_000, "price_m2": 2_050},
    "Saint-Louis": {"base": 40_000, "price_m2": 1_900},
    "Mbour": {"base": 44_000, "price_m2": 2_250},
    "Saly": {"base": 58_000, "price_m2": 3_050},
}
CITY_QUARTERS = {
    "Dakar": {
        "Almadies": 1.42,
        "Mermoz": 1.24,
        "Plateau": 1.32,
        "Yoff": 1.08,
        "Parcelles Assainies": 0.86,
    },
    "Thiès": {"Grand Standing": 1.18, "Randoulène": 1.00, "Médina Fall": 0.88},
    "Saint-Louis": {"Île": 1.25, "Sor": 0.96, "Hydrobase": 1.12},
    "Mbour": {"Zone résidentielle": 1.16, "Grand Mbour": 0.94, "Mbour centre": 1.02},
    "Saly": {"Saly Portudal": 1.23, "Saly centre": 1.00, "Niakh Niakhal": 0.91},
}
EQUIPMENT_CHOICES = ["climatisation", "parking", "gardiennage", "piscine", "groupe_electrogene"]


def generate_dataset(rows: int = 1_200, seed: int = 42) -> pd.DataFrame:
    """Cree un jeu synthetique coherent sans reproduire de vraies annonces."""
    rng = np.random.default_rng(seed)
    cities = rng.choice(list(CITY_MARKET), size=rows, p=[0.40, 0.16, 0.14, 0.16, 0.14])
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
        quarter_market = CITY_QUARTERS[str(city)]
        quarter = str(rng.choice(list(quarter_market)))
        selected = [equipment for equipment in EQUIPMENT_CHOICES if rng.random() < 0.24]
        equipment_bonus = len(selected) * 14_000 + (45_000 if "piscine" in selected else 0)
        city_market = CITY_MARKET[str(city)]
        expected = (
            city_market["base"]
            + surface[index] * city_market["price_m2"]
            + rooms[index] * 8_500
            + bedrooms[index] * 7_500
            + equipment_bonus
        )
        expected *= quarter_market[quarter] * type_factor[str(property_types[index])]
        if furnished[index]:
            expected *= 1.18
        noisy_price = expected * rng.lognormal(mean=0, sigma=0.08)
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
