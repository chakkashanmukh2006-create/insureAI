"""
Model versioning, storage, and retrieval manager.

Handles saving/loading trained models to disk and managing
the model registry in the database for version tracking.
"""

import os
from typing import Any, Optional

import joblib
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.config.settings import settings
from app.models.training import ModelRegistry
from app.utils.logger import logger


class ModelManager:
    """Manages model versioning, storage, and retrieval.

    Provides static utilities for:
    - Generating sequential model version strings (v1, v2, …).
    - Saving / loading serialised model files via joblib.
    - Querying the ``ModelRegistry`` for the latest active model.
    """

    @staticmethod
    def get_next_version(db: Session, model_type: str) -> str:
        """Determine the next sequential version string for *model_type*.

        Queries the ``ModelRegistry`` for the latest entry of the given
        type and increments the version counter.

        Args:
            db: Active SQLAlchemy session.
            model_type: ``'lead'`` or ``'customer'``.

        Returns:
            Version string such as ``'v1'``, ``'v2'``, etc.
        """
        latest: Optional[ModelRegistry] = (
            db.query(ModelRegistry)
            .filter(ModelRegistry.model_type == model_type)
            .order_by(desc(ModelRegistry.id))
            .first()
        )

        if latest is None:
            return "v1"

        try:
            current_num = int(latest.model_version.replace('v', ''))
        except (ValueError, AttributeError):
            logger.warning(
                f"Could not parse version '{getattr(latest, 'model_version', None)}' "
                f"for model_type='{model_type}'. Defaulting to v1."
            )
            return "v1"

        return f"v{current_num + 1}"

    @staticmethod
    def save_model(model: Any, model_type: str, version: str) -> str:
        """Serialise a trained model to disk.

        Args:
            model: The trained scikit-learn / XGBoost model object.
            model_type: ``'lead'`` or ``'customer'``.
            version: Version string (e.g. ``'v1'``).

        Returns:
            Absolute path of the saved model file.
        """
        model_dir: str = settings.MODEL_STORAGE_PATH
        os.makedirs(model_dir, exist_ok=True)

        path = os.path.join(model_dir, f"{model_type}_model_{version}.joblib")
        joblib.dump(model, path)
        logger.info(f"Saved {model_type} model {version} to {path}")
        return path

    @staticmethod
    def load_model(path: str) -> Any:
        """Load a model from a specific file path.

        Args:
            path: Full path to the joblib file.

        Returns:
            De-serialised model object.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        model = joblib.load(path)
        logger.info(f"Loaded model from {path}")
        return model

    @staticmethod
    def load_latest_model(
        db: Session, model_type: str
    ) -> tuple[Any | None, ModelRegistry | None]:
        """Load the latest *active* model from the registry.

        Args:
            db: Active SQLAlchemy session.
            model_type: ``'lead'`` or ``'customer'``.

        Returns:
            Tuple of ``(model_object, registry_entry)`` or
            ``(None, None)`` if no active model exists.
        """
        registry: Optional[ModelRegistry] = (
            db.query(ModelRegistry)
            .filter(
                ModelRegistry.model_type == model_type,
                ModelRegistry.status == "active",
            )
            .order_by(desc(ModelRegistry.id))
            .first()
        )

        if registry is None:
            logger.warning(
                f"No active {model_type} model found in the registry."
            )
            return None, None

        if not os.path.exists(registry.model_path):
            logger.error(
                f"Model file missing at {registry.model_path} "
                f"for registry entry id={registry.id}."
            )
            return None, None

        model = joblib.load(registry.model_path)
        logger.info(
            f"Loaded {model_type} model {registry.model_version} "
            f"from {registry.model_path}"
        )
        return model, registry

    @staticmethod
    def get_latest_registry(
        db: Session, model_type: str
    ) -> Optional[ModelRegistry]:
        """Retrieve the latest active ``ModelRegistry`` entry.

        Args:
            db: Active SQLAlchemy session.
            model_type: ``'lead'`` or ``'customer'``.

        Returns:
            The ``ModelRegistry`` row, or ``None``.
        """
        return (
            db.query(ModelRegistry)
            .filter(
                ModelRegistry.model_type == model_type,
                ModelRegistry.status == "active",
            )
            .order_by(desc(ModelRegistry.id))
            .first()
        )

    @staticmethod
    def deactivate_previous_models(
        db: Session, model_type: str, exclude_version: str
    ) -> int:
        """Mark all prior versions as ``'archived'``.

        This is useful after a new model has been successfully trained
        and registered so only one active model exists per type.

        Args:
            db: Active SQLAlchemy session.
            model_type: ``'lead'`` or ``'customer'``.
            exclude_version: The version string to keep active.

        Returns:
            Number of rows updated.
        """
        rows_updated: int = (
            db.query(ModelRegistry)
            .filter(
                ModelRegistry.model_type == model_type,
                ModelRegistry.status == "active",
                ModelRegistry.model_version != exclude_version,
            )
            .update({"status": "archived"})
        )
        if rows_updated:
            db.commit()
            logger.info(
                f"Archived {rows_updated} previous {model_type} model(s)."
            )
        return rows_updated
