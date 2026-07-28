"""Entrainement configure avec Hydra et suivi avec MLflow."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hydra
import joblib
import mlflow
from omegaconf import DictConfig, OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from senegal_rental_price.data.preprocessing import load_and_clean
from senegal_rental_price.features.build_features import build_features
from senegal_rental_price.utils.logger import configure_logging, get_logger

LOGGER = get_logger(__name__)


def evaluation_metrics(y_true: Any, predictions: Any) -> dict[str, float]:
    """Calcule des metriques globales lisibles, dont le biais signe."""
    errors = predictions - y_true
    absolute_errors = abs(errors)
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(mean_squared_error(y_true, predictions) ** 0.5),
        "r2": float(r2_score(y_true, predictions)),
        "mape": float((absolute_errors / y_true.clip(lower=1)).mean()),
        "bias": float(errors.mean()),
        "p80_abs_error": float(absolute_errors.quantile(0.80)),
    }


def segment_metrics(frame: Any, y_true: Any, predictions: Any) -> dict[str, dict[str, float]]:
    """Mesure le biais et l'erreur par ville pour detecter un score trompeur."""
    evaluated = frame[["ville"]].copy()
    evaluated["actual"] = y_true.to_numpy()
    evaluated["prediction"] = predictions
    evaluated["error"] = evaluated["prediction"] - evaluated["actual"]
    evaluated["ape"] = abs(evaluated["error"]) / evaluated["actual"].clip(lower=1)
    result: dict[str, dict[str, float]] = {}
    for city, group in evaluated.groupby("ville"):
        result[str(city)] = {
            "count": float(len(group)),
            "mae": float(abs(group["error"]).mean()),
            "mape": float(group["ape"].mean()),
            "bias": float(group["error"].mean()),
        }
    return result


def reference_summary(frame: Any) -> dict[str, Any]:
    """Produit les reperes synthetiques affiches par l'interface."""
    city_medians = (
        frame.groupby("ville")["prix_loyer_mensuel"].median().round(0).astype(int).to_dict()
    )
    type_medians = (
        frame.groupby("type_bien")["prix_loyer_mensuel"].median().round(0).astype(int).to_dict()
    )
    return {
        "city_medians": {str(key): int(value) for key, value in city_medians.items()},
        "type_medians": {str(key): int(value) for key, value in type_medians.items()},
        "target_min": int(frame["prix_loyer_mensuel"].min()),
        "target_max": int(frame["prix_loyer_mensuel"].max()),
    }


def create_model(name: str, params: dict[str, Any]) -> Any:
    """Construit le regresseur demande par la configuration."""
    if name == "ridge":
        return Ridge(**params)
    if name == "random_forest":
        return RandomForestRegressor(**params)
    if name == "gradient_boosting":
        return GradientBoostingRegressor(**params)
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise RuntimeError("Installer l'extra xgboost pour utiliser ce modele") from error
        return XGBRegressor(**params)
    raise ValueError(f"Modele inconnu: {name}")


def train(config: DictConfig) -> dict[str, Any]:
    """Entraine, evalue, trace et serialise un pipeline complet."""
    raw_path = Path(config.data.raw_path)
    if not raw_path.exists():
        from senegal_rental_price.data.generate import generate_dataset

        raw_path.parent.mkdir(parents=True, exist_ok=True)
        generate_dataset(seed=int(config.seed)).to_csv(raw_path, index=False)
    frame = load_and_clean(raw_path, config.data.processed_path)
    target = str(config.data.target)
    x_data = build_features(frame.drop(columns=[target]))
    y_data = frame[target]
    categorical = ["ville", "quartier", "type_bien"]
    numeric = [column for column in x_data.columns if column not in categorical]
    preprocessing = ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )
    model_name = str(config.model.name)
    params = OmegaConf.to_container(config.model.params, resolve=True)
    if not isinstance(params, dict):
        raise TypeError("Les parametres du modele doivent former un dictionnaire")
    pipeline = Pipeline(
        [("preprocessing", preprocessing), ("model", create_model(model_name, params))]
    )
    strata = frame["ville"].astype(str) + "|" + frame["type_bien"].astype(str)
    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=float(config.test_size),
        random_state=int(config.seed),
        stratify=strata,
    )

    mlflow.set_tracking_uri(str(config.mlflow.tracking_uri))
    mlflow.set_experiment(str(config.mlflow.experiment_name))
    with mlflow.start_run(run_name=model_name) as run:
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics = evaluation_metrics(y_test, predictions)
        city_metrics = segment_metrics(x_test, y_test, predictions)
        mlflow.log_params({"model": model_name, **params})
        mlflow.log_metrics(metrics)
        metadata = {
            "version": "2.0.0",
            "model_name": model_name,
            "trained_at": datetime.now(UTC).isoformat(),
            "metrics": metrics,
            "mlflow_run_id": run.info.run_id,
            "training_rows": len(x_train),
            "test_rows": len(x_test),
            "segment_metrics": city_metrics,
            "reference": reference_summary(frame),
            "data_provenance": "synthetic_option_2",
        }
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = output_dir / f"{model_name}.joblib"
        joblib.dump({"model": pipeline, "metadata": metadata}, artifact_path)
        mlflow.log_artifact(str(artifact_path), artifact_path="model")
    LOGGER.info("Modele %s sauvegarde dans %s", model_name, artifact_path)
    return metadata


@hydra.main(  # type: ignore[untyped-decorator]
    version_base=None, config_path="../../../conf", config_name="config"
)
def hydra_entry(config: DictConfig) -> None:
    """Point d'entree Hydra."""
    configure_logging()
    train(config)


def main() -> None:
    """Point d'entree console."""
    hydra_entry()


if __name__ == "__main__":
    main()
