"""Application FastAPI de prediction."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI

from api.dependencies import get_model
from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    RentalFeatures,
)
from senegal_rental_price.models.predict import ModelBundle, predict_price
from senegal_rental_price.utils.logger import configure_logging, get_logger

LOGGER = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Prepare le logging et charge l'artefact au demarrage."""
    configure_logging()
    get_model()
    LOGGER.info("API prete")
    yield


app = FastAPI(
    title="Senegal Rental Price API",
    version="1.0.0",
    description="Estimation pedagogique du loyer mensuel au Senegal.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Supervision"])
def health(model: Annotated[ModelBundle, Depends(get_model)]) -> HealthResponse:
    """Confirme que le service et son modele sont disponibles."""
    return HealthResponse(status="ok", model_loaded=model is not None)


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Modele"])
def model_info(model: Annotated[ModelBundle, Depends(get_model)]) -> ModelInfoResponse:
    """Expose les metadonnees et performances de l'artefact charge."""
    return ModelInfoResponse.model_validate(model.metadata)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(
    request: RentalFeatures, model: Annotated[ModelBundle, Depends(get_model)]
) -> PredictionResponse:
    """Valide un bien et estime son loyer mensuel."""
    payload = request.model_dump(mode="json")
    payload["equipements"] = "|".join(payload["equipements"])
    price = predict_price(model, payload)
    uncertainty = float(model.metadata.get("metrics", {}).get("p80_abs_error", 0.0))
    reference = model.metadata.get("reference", {})
    target_min = float(reference.get("target_min", 0.0))
    target_max = float(reference.get("target_max", float("inf")))
    in_range = target_min <= price <= target_max
    factors = [f"localisation : {request.ville.value} / {request.quartier}"]
    factors.append(f"surface : {request.surface_m2:.0f} m²")
    factors.append(f"type : {request.type_bien.value}")
    if request.meuble:
        factors.append("prime : bien meublé")
    if request.equipements:
        factors.append(f"confort : {len(request.equipements)} équipement(s)")
    return PredictionResponse(
        prix_loyer_mensuel_estime=round(price / 1_000) * 1_000,
        fourchette_basse=max(0, round((price - uncertainty) / 1_000) * 1_000),
        fourchette_haute=round((price + uncertainty) / 1_000) * 1_000,
        fiabilite="bonne" if in_range else "prudente",
        facteurs_principaux=factors,
        model_version=str(model.metadata["version"]),
    )
