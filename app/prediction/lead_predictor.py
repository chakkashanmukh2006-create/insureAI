"""
Lead conversion propensity predictor.

Loads the latest trained lead model and preprocessor, runs inference
on all leads (or individual leads), generates SHAP/feature-importance
explanations, and persists predictions to the database.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.config.settings import settings
from app.explainability.explainer import Explainer
from app.models.lead import Lead
from app.models.prediction import LeadPrediction
from app.models.training import ModelRegistry
from app.training.model_manager import ModelManager
from app.training.preprocessor import DataPreprocessor
from app.utils.logger import logger


class LeadPredictor:
    """Predicts lead conversion propensity using the latest trained model."""

    # Score → category thresholds
    HIGH_THRESHOLD: float = 0.7
    MEDIUM_THRESHOLD: float = 0.4

    # Feature columns (must match training)
    FEATURE_COLS: list[str] = [
        'age', 'gender', 'occupation', 'annual_income', 'city',
        'existing_policy', 'product_interested', 'website_visits',
        'email_opens', 'calls_answered', 'form_submitted',
        'last_interaction_days', 'lead_source',
    ]
    BOOL_COLS: list[str] = ['existing_policy', 'form_submitted']
    CATEGORICAL_COLS: list[str] = [
        'gender', 'occupation', 'city', 'product_interested', 'lead_source',
    ]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def predict_all(self, db: Session) -> list[dict]:
        """Generate predictions for ALL leads using the latest model.

        Steps:
          1. Load the latest active lead model from the registry.
          2. Load the saved preprocessor.
          3. Fetch all leads from the database.
          4. Preprocess features identically to training.
          5. Run batch inference.
          6. Generate per-lead explanations.
          7. Bulk-insert ``LeadPrediction`` rows.

        Args:
            db: Active SQLAlchemy session.

        Returns:
            List of prediction result dictionaries.

        Raises:
            ValueError: If no trained model is available.
        """
        # 1. Load model
        model, registry = ModelManager.load_latest_model(db, 'lead')
        if model is None or registry is None:
            raise ValueError(
                "No trained lead model found. Please train a model first."
            )

        # 2. Load preprocessor
        preprocessor = DataPreprocessor.load(
            settings.MODEL_STORAGE_PATH, 'lead'
        )

        # 3. Process in chunks to prevent Out-Of-Memory errors
        chunk_size = 2000
        offset = 0
        total_results = 0
        
        import shap
        tree_explainer = shap.TreeExplainer(model)
        explainer = Explainer()
        now = datetime.now(timezone.utc)
        
        # Extract primitives to avoid DetachedInstanceError after expunge_all
        model_version = registry.model_version
        model_accuracy = registry.accuracy
        algorithm = registry.algorithm
        training_date = registry.training_date
        
        while True:
            leads = db.query(Lead).order_by(Lead.lead_id).offset(offset).limit(chunk_size).all()
            if not leads:
                break

            # Preprocess chunk
            X_scaled = self._preprocess_batch(leads, preprocessor)
            probabilities: np.ndarray = model.predict_proba(X_scaled)[:, 1]

            shap_values_all = tree_explainer.shap_values(X_scaled)
            if isinstance(shap_values_all, list):
                shap_values_all = shap_values_all[1]

            predictions_to_add: list[LeadPrediction] = []

            for idx, lead in enumerate(leads):
                propensity = float(probabilities[idx])
                score = propensity * 100
                category = self._categorise(propensity)

                abs_shap = np.abs(shap_values_all[idx]).flatten()
                if abs_shap.shape[0] != len(preprocessor.feature_names):
                    abs_shap = abs_shap[: len(preprocessor.feature_names)]
                feature_importance = sorted(zip(preprocessor.feature_names, abs_shap), key=lambda x: x[1], reverse=True)
                reasons = explainer._pick_top_reasons(feature_importance, explainer.LEAD_REASON_MAP)[:3]

                pred = LeadPrediction(
                    prediction_id=str(uuid.uuid4()),
                    lead_id=lead.lead_id,
                    propensity_ratio=propensity,
                    lead_score=score,
                    category=category,
                    top_reasons=reasons[:3],
                    model_version=model_version,
                    model_accuracy=model_accuracy,
                    algorithm=algorithm,
                    prediction_timestamp=now,
                    training_timestamp=training_date,
                    email=lead.email,
                    contact_number=lead.contact_number,
                )
                predictions_to_add.append(pred)

            # Bulk insert chunk and clear session
            db.bulk_save_objects(predictions_to_add)
            db.commit()
            db.expunge_all()
            
            total_results += len(leads)
            offset += chunk_size

        logger.info(
            f"Generated {total_results} lead predictions using model "
            f"{model_version}"
        )
        return []

    def predict_single(self, db: Session, lead_id: str) -> dict:
        """Generate a prediction for a single lead.

        Args:
            db: Active SQLAlchemy session.
            lead_id: The ``lead_id`` to predict.

        Returns:
            Prediction result dictionary.

        Raises:
            ValueError: If no model or lead is found.
        """
        model, registry = ModelManager.load_latest_model(db, 'lead')
        if model is None or registry is None:
            raise ValueError("No trained lead model found.")

        lead = db.query(Lead).filter(Lead.lead_id == lead_id).first()
        if lead is None:
            raise ValueError(f"Lead with id '{lead_id}' not found.")

        preprocessor = DataPreprocessor()
        lead_data = {
            'age': lead.age,
            'gender': lead.gender,
            'occupation': lead.occupation,
            'annual_income': lead.annual_income,
            'city': lead.city,
            'existing_policy': lead.existing_policy,
            'product_interested': lead.product_interested,
            'website_visits': lead.website_visits,
            'email_opens': lead.email_opens,
            'calls_answered': lead.calls_answered,
            'form_submitted': lead.form_submitted,
            'last_interaction_days': lead.last_interaction_days,
            'lead_source': lead.lead_source,
        }
        X_scaled = preprocessor.preprocess_lead_single(lead_data)

        propensity = float(model.predict_proba(X_scaled)[:, 1][0])
        score = propensity * 100
        category = self._categorise(propensity)

        explainer = Explainer()
        reasons = explainer.explain_lead(
            model, X_scaled, preprocessor.feature_names, lead
        )

        now = datetime.now(timezone.utc)
        prediction_id = str(uuid.uuid4())

        pred = LeadPrediction(
            prediction_id=prediction_id,
            lead_id=lead.lead_id,
            propensity_ratio=propensity,
            lead_score=score,
            category=category,
            top_reasons=reasons[:3],
            model_version=registry.model_version,
            model_accuracy=registry.accuracy,
            algorithm=registry.algorithm,
            prediction_timestamp=now,
            training_timestamp=registry.training_date,
            email=lead.email,
            contact_number=lead.contact_number,
        )
        db.add(pred)
        db.commit()

        return {
            'prediction_id': prediction_id,
            'lead_id': lead.lead_id,
            'full_name': lead.full_name,
            'propensity_ratio': propensity,
            'lead_score': score,
            'category': category,
            'top_reasons': reasons[:3],
            'email': lead.email,
            'contact_number': lead.contact_number,
            'model_version': registry.model_version,
            'model_accuracy': registry.accuracy,
            'algorithm': registry.algorithm,
            'prediction_timestamp': now,
            'training_timestamp': registry.training_date,
        }

    def get_top20(self, db: Session) -> list[dict]:
        """Get top 20 leads by propensity ratio (latest prediction per lead).

        Uses a subquery to select only the most recent prediction per
        ``lead_id``, then orders descending by ``propensity_ratio``.

        Args:
            db: Active SQLAlchemy session.

        Returns:
            List of up to 20 result dictionaries.
        """
        # Subquery: latest prediction timestamp per lead
        subq = (
            db.query(
                LeadPrediction.lead_id,
                func.max(LeadPrediction.prediction_timestamp).label('max_ts'),
            )
            .group_by(LeadPrediction.lead_id)
            .subquery()
        )

        predictions = (
            db.query(LeadPrediction)
            .join(
                subq,
                (LeadPrediction.lead_id == subq.c.lead_id)
                & (LeadPrediction.prediction_timestamp == subq.c.max_ts),
            )
            .order_by(desc(LeadPrediction.propensity_ratio))
            .limit(20)
            .all()
        )

        if not predictions:
            return []

        lead_ids = [pred.lead_id for pred in predictions]
        leads = db.query(Lead).filter(Lead.lead_id.in_(lead_ids)).all()
        lead_map = {l.lead_id: l for l in leads}

        results: list[dict] = []
        for pred in predictions:
            lead = lead_map.get(pred.lead_id)
            results.append(
                {
                    'name': lead.full_name if lead else 'Unknown',
                    'lead_id': pred.lead_id,
                    'propensity_ratio': pred.propensity_ratio,
                    'lead_score': pred.lead_score,
                    'category': pred.category,
                    'email': pred.email,
                    'contact_number': pred.contact_number,
                    'top_reasons': pred.top_reasons or [],
                    'model_version': pred.model_version,
                    'training_timestamp': pred.training_timestamp,
                    'prediction_timestamp': pred.prediction_timestamp,
                }
            )

        return results

    def get_all_predicted(self, db: Session, page: int = 1, limit: int = 100) -> dict:
        """Get all leads with their latest propensity predictions.

        Uses a subquery to select the most recent prediction per ``lead_id``.

        Args:
            db: Active SQLAlchemy session.
            page: Page number (1-indexed).
            limit: Number of items per page.

        Returns:
            Dictionary containing paginated data and metadata.
        """
        # Subquery: latest prediction timestamp per lead
        subq = (
            db.query(
                LeadPrediction.lead_id,
                func.max(LeadPrediction.prediction_timestamp).label('max_ts'),
            )
            .group_by(LeadPrediction.lead_id)
            .subquery()
        )

        base_query = db.query(LeadPrediction).join(
            subq,
            (LeadPrediction.lead_id == subq.c.lead_id)
            & (LeadPrediction.prediction_timestamp == subq.c.max_ts),
        )

        total_count = base_query.count()
        import math
        total_pages = math.ceil(total_count / limit) if limit > 0 else 0

        predictions = (
            base_query
            .order_by(desc(LeadPrediction.propensity_ratio))
            .offset((page - 1) * limit)
            .limit(limit)
            .all()
        )

        if not predictions:
            return {
                "data": [],
                "total": total_count,
                "page": page,
                "limit": limit,
                "total_pages": total_pages
            }

        lead_ids = [pred.lead_id for pred in predictions]
        leads = db.query(Lead).filter(Lead.lead_id.in_(lead_ids)).all()
        lead_map = {l.lead_id: l for l in leads}

        results: list[dict] = []
        for pred in predictions:
            lead = lead_map.get(pred.lead_id)
            results.append(
                {
                    'name': lead.full_name if lead else 'Unknown',
                    'lead_id': pred.lead_id,
                    'propensity_ratio': pred.propensity_ratio,
                    'lead_score': pred.lead_score,
                    'category': pred.category,
                    'email': pred.email,
                    'contact_number': pred.contact_number,
                    'top_reasons': pred.top_reasons or [],
                    'model_version': pred.model_version,
                    'training_timestamp': pred.training_timestamp,
                    'prediction_timestamp': pred.prediction_timestamp,
                }
            )

        return {
            "data": results,
            "total": total_count,
            "page": page,
            "limit": limit,
            "total_pages": total_pages
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _preprocess_batch(
        self, leads: list[Lead], preprocessor: DataPreprocessor
    ) -> pd.DataFrame:
        """Build a feature matrix from a list of Lead ORM objects using
        the saved preprocessor's encoders and scaler.

        Args:
            leads: List of Lead ORM instances.
            preprocessor: Previously loaded ``DataPreprocessor``.

        Returns:
            Scaled feature DataFrame.
        """
        lead_data = [
            {
                'age': lead.age,
                'gender': lead.gender,
                'occupation': lead.occupation,
                'annual_income': lead.annual_income,
                'city': lead.city,
                'existing_policy': lead.existing_policy,
                'product_interested': lead.product_interested,
                'website_visits': lead.website_visits,
                'email_opens': lead.email_opens,
                'calls_answered': lead.calls_answered,
                'form_submitted': lead.form_submitted,
                'last_interaction_days': lead.last_interaction_days,
                'lead_source': lead.lead_source,
            }
            for lead in leads
        ]
        df = pd.DataFrame(lead_data)
        X = df[self.FEATURE_COLS].copy()

        # Force numeric types to avoid type inference issues on small chunks
        numeric_cols = [c for c in self.FEATURE_COLS if c not in self.CATEGORICAL_COLS]
        for col in numeric_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)

        # Fill missing values
        for col in X.select_dtypes(include=[np.number]).columns:
            if col not in numeric_cols:
                X[col] = X[col].fillna(X[col].median()).fillna(0.0)
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].fillna('Unknown')

        # Convert booleans
        for col in self.BOOL_COLS:
            if col in X.columns:
                X[col] = X[col].astype(int)

        # Label-encode using saved encoders (handle unseen labels)
        for col in self.CATEGORICAL_COLS:
            key = f'lead_{col}'
            if key in preprocessor.label_encoders:
                le = preprocessor.label_encoders[key]
                X[col] = X[col].astype(str).apply(
                    lambda val, _le=le: (
                        _le.transform([val])[0] if val in _le.classes_ else -1
                    )
                )
            else:
                X[col] = 0

        # Scale
        X_scaled = pd.DataFrame(
            preprocessor.scaler.transform(X),
            columns=preprocessor.feature_names,
        )
        return X_scaled

    def _categorise(self, propensity: float) -> str:
        """Map a propensity probability to a human-readable category."""
        if propensity >= self.HIGH_THRESHOLD:
            return 'High'
        if propensity >= self.MEDIUM_THRESHOLD:
            return 'Medium'
        return 'Low'
