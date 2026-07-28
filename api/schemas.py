"""Contrats Pydantic publics de l'API."""

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator


class Ville(StrEnum):
    """Villes couvertes par les donnees d'entrainement."""

    dakar = "Dakar"
    thies = "Thiès"
    saint_louis = "Saint-Louis"
    mbour = "Mbour"
    saly = "Saly"


class TypeBien(StrEnum):
    """Types de biens supportes."""

    appartement = "Appartement"
    maison = "Maison"
    studio = "Studio"
    villa = "Villa"


class RentalFeatures(BaseModel):
    """Caracteristiques validees d'un bien."""

    ville: Ville = Field(description="Ville du bien")
    quartier: str = Field(default="Non renseigne", min_length=2, max_length=80)
    type_bien: TypeBien
    surface_m2: float = Field(gt=0, le=2000, description="Surface habitable en m2")
    nb_pieces: int = Field(ge=1, le=20)
    nb_chambres: int = Field(ge=0, le=15)
    meuble: bool
    equipements: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("equipements")
    @classmethod
    def normalize_equipment(cls, values: list[str]) -> list[str]:
        """Nettoie et dedoublonne les equipements."""
        normalized = [value.strip().lower() for value in values if value.strip()]
        return list(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def rooms_are_consistent(self) -> "RentalFeatures":
        """Une chambre ne peut pas depasser le nombre total de pieces."""
        if self.nb_chambres > self.nb_pieces:
            raise ValueError("nb_chambres ne peut pas depasser nb_pieces")
        return self


class PredictionResponse(BaseModel):
    """Resultat d'une estimation."""

    prix_loyer_mensuel_estime: float
    devise: str = "FCFA"
    model_version: str


class HealthResponse(BaseModel):
    """Etat operationnel du service."""

    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Metadonnees du modele charge."""

    version: str
    model_name: str
    trained_at: str
    metrics: dict[str, float]
    mlflow_run_id: str
    training_rows: int
