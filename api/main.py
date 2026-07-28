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
    return PredictionResponse(
        prix_loyer_mensuel_estime=round(price / 1_000) * 1_000,
        model_version=str(model.metadata["version"]),
    )
