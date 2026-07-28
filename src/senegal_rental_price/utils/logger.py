"""Configuration centralisee du logging."""

import logging
import os


def configure_logging(level: str | None = None) -> None:
    """Configure une seule fois le logger racine."""
    selected_level = level.upper() if level is not None else os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, selected_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Retourne un logger nomme."""
    return logging.getLogger(name)
