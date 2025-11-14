"""
MLflow configuration utility module.

This module provides centralized MLflow setup functionality that reads
the tracking URI from the configuration and initializes MLflow accordingly.
"""

import os
from pathlib import Path
from typing import Optional

import mlflow
from omegaconf import DictConfig


def setup_mlflow(cfg: DictConfig, project_root: Optional[Path] = None) -> None:
    """
    Setup MLflow tracking URI from configuration.

    This function reads the MLflow tracking URI from the config and sets it
    before any MLflow operations. If tracking_uri is not set in config,
    it defaults to SQLite backend.

    Args:
        cfg: OmegaConf configuration object (should contain mlflow.tracking_uri)
        project_root: Optional project root path (defaults to detecting from file location)
    """
    if project_root is None:
        # Default to parent of src directory
        project_root = Path(__file__).resolve().parent.parent

    # Get tracking URI from config
    tracking_uri = None
    if hasattr(cfg, "mlflow") and hasattr(cfg.mlflow, "tracking_uri"):
        tracking_uri = str(cfg.mlflow.tracking_uri)
    elif "mlflow" in cfg and "tracking_uri" in cfg.mlflow:
        tracking_uri = str(cfg.mlflow.tracking_uri)

    # If tracking URI is relative (starts with sqlite:///), resolve it relative to project root
    if tracking_uri and tracking_uri.startswith("sqlite:///"):
        # Extract the database path (everything after sqlite:///)
        db_path = tracking_uri.replace("sqlite:///", "")
        if not os.path.isabs(db_path):
            # Resolve relative to project root
            db_path = str(project_root / db_path)
        tracking_uri = f"sqlite:///{db_path}"

    # Set tracking URI (or use default if not specified)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow tracking URI set to: {tracking_uri}")
    else:
        # Default to SQLite if not specified
        default_db = str(project_root / "mlflow.db")
        default_uri = f"sqlite:///{default_db}"
        mlflow.set_tracking_uri(default_uri)
        print(
            f"MLflow tracking URI not specified in config, using default: {default_uri}"
        )


def get_tracking_uri(cfg: DictConfig, project_root: Optional[Path] = None) -> str:
    """
    Get the MLflow tracking URI from configuration.

    Args:
        cfg: OmegaConf configuration object
        project_root: Optional project root path

    Returns:
        The tracking URI string
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    tracking_uri = None
    if hasattr(cfg, "mlflow") and hasattr(cfg.mlflow, "tracking_uri"):
        tracking_uri = str(cfg.mlflow.tracking_uri)
    elif "mlflow" in cfg and "tracking_uri" in cfg.mlflow:
        tracking_uri = str(cfg.mlflow.tracking_uri)

    if tracking_uri and tracking_uri.startswith("sqlite:///"):
        db_path = tracking_uri.replace("sqlite:///", "")
        if not os.path.isabs(db_path):
            db_path = str(project_root / db_path)
        tracking_uri = f"sqlite:///{db_path}"

    return tracking_uri or f"sqlite:///{project_root / 'mlflow.db'}"
