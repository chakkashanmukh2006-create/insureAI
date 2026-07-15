"""
Customer churn predictor.

Loads the latest trained customer churn model, runs inference on all
customers (or individual customers), performs sentiment analysis on
feedback text, generates SHAP/feature-importance explanations, and
persists predictions to the database.
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
from app.models.customer import Customer
from app.models.prediction import CustomerPrediction
from app.models.training import ModelRegistry
from app.sentiment.analyzer import SentimentAnalyzer
from app.training.model_manager import ModelManager
from app.training.preprocessor import DataPreprocessor
from app.utils.logger import logger


class CustomerPredictor:
    """Predicts customer churn risk using the latest trained model."""

    # Score → category thresholds
    HIGH_THRESHOLD: float = 0.7
    MEDIUM_THRESHOLD: float = 0.4

    # Feature columns (must match training)
    FEATURE_COLS: list[str] = [
        'age', 'policy_type', 'premium_amount',
        'renewal_history', 'claim_history', 'complaint_count',
        'support_tickets',
    ]
    CATEGORICAL_COLS: list[str] = ['policy_type']

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def predict_all(self, db: Session) -> list[dict]:
        """Generate churn predictions for ALL customers.

        Steps:
          1. Load the latest active customer model from the registry.
          2. Load the saved preprocessor.
          3. Fetch all customers from the database.
          4. Preprocess features identically to training.
          5. Run batch inference.
          6. Run sentiment analysis on each customer's feedback.
          7. Generate per-customer explanations.
          8. Bulk-insert ``CustomerPrediction`` rows.

        Args:
            db: Active SQLAlchemy session.

        Returns:
            List of prediction result dictionaries.

        Raises:
            ValueError: If no trained model is available.
        """
        # 1. Load model
        model, registry = ModelManager.load_latest_model(db, 'customer')
        if model is None or registry is None:
            raise ValueError(
                "No trained customer model found. "
                "Please train a model first."
            )

        # 2. Load preprocessor
        preprocessor = DataPreprocessor.load(
            settings.MODEL_STORAGE_PATH, 'customer'
        )

        # 3. Process in chunks to prevent Out-Of-Memory errors
        chunk_size = 2000
        offset = 0
        total_results = 0
        
        import shap
        tree_explainer = shap.TreeExplainer(model)
        explainer = Explainer()
        sentiment_analyzer = SentimentAnalyzer()
        now = datetime.now(timezone.utc)
        
        # Extract primitives to avoid DetachedInstanceError after expunge_all
        model_version = registry.model_version
        model_accuracy = registry.accuracy
        algorithm = registry.algorithm
        training_date = registry.training_date
        
        while True:
            customers = db.query(Customer).order_by(Customer.customer_id).offset(offset).limit(chunk_size).all()
            if not customers:
                break
                
            # Preprocess chunk
            X_scaled = self._preprocess_batch(customers, preprocessor)
            probabilities: np.ndarray = model.predict_proba(X_scaled)[:, 1]
            
            shap_values_all = tree_explainer.shap_values(X_scaled)
            if isinstance(shap_values_all, list):
                shap_values_all = shap_values_all[1]
                
            predictions_to_add: list[CustomerPrediction] = []
            
            for idx, customer in enumerate(customers):
                churn_prob = float(probabilities[idx])
                risk_category = self._categorise(churn_prob)

                sentiment_result = sentiment_analyzer.analyze(getattr(customer, 'feedback', None))

                abs_shap = np.abs(shap_values_all[idx]).flatten()
                if abs_shap.shape[0] != len(preprocessor.feature_names):
                    abs_shap = abs_shap[: len(preprocessor.feature_names)]
                feature_importance = sorted(zip(preprocessor.feature_names, abs_shap), key=lambda x: x[1], reverse=True)
                reasons = explainer._pick_top_reasons(feature_importance, explainer.CUSTOMER_REASON_MAP)[:3]

                pred = CustomerPrediction(
                    prediction_id=str(uuid.uuid4()),
                    customer_id=customer.customer_id,
                    churn_ratio=churn_prob,
                    risk_category=risk_category,
                    sentiment=sentiment_result['sentiment'],
                    sentiment_score=sentiment_result['sentiment_score'],
                    confidence_score=sentiment_result['confidence_score'],
                    top_reasons=reasons[:3],
                    model_version=model_version,
                    model_accuracy=model_accuracy,
                    algorithm=algorithm,
                    prediction_timestamp=now,
                    training_timestamp=training_date,
                    email=customer.email,
                    contact_number=customer.contact_number,
                )
                predictions_to_add.append(pred)

            # Bulk insert chunk and clear session
            db.bulk_save_objects(predictions_to_add)
            db.commit()
            db.expunge_all()
            
            total_results += len(customers)
            offset += chunk_size

        logger.info(
            f"Generated {total_results} customer churn predictions using "
            f"model {model_version}"
        )
        return []

    def predict_single(self, db: Session, customer_id: str) -> dict:
        """Generate a churn prediction for a single customer.

        Args:
            db: Active SQLAlchemy session.
            customer_id: The ``customer_id`` to predict.

        Returns:
            Prediction result dictionary.

        Raises:
            ValueError: If no model or customer is found.
        """
        model, registry = ModelManager.load_latest_model(db, 'customer')
        if model is None or registry is None:
            raise ValueError("No trained customer model found.")

        customer = (
            db.query(Customer)
            .filter(Customer.customer_id == customer_id)
            .first()
        )
        if customer is None:
            raise ValueError(
                f"Customer with id '{customer_id}' not found."
            )

        preprocessor = DataPreprocessor()
        customer_data = {
            'age': customer.age,
            'policy_type': customer.policy_type,
            'premium_amount': customer.premium_amount,
            'renewal_history': customer.renewal_history,
            'claim_history': customer.claim_history,
            'complaint_count': customer.complaint_count,
            'support_tickets': customer.support_tickets,
        }
        X_scaled = preprocessor.preprocess_customer_single(customer_data)

        churn_prob = float(model.predict_proba(X_scaled)[:, 1][0])
        risk_category = self._categorise(churn_prob)

        # Sentiment
        sentiment_analyzer = SentimentAnalyzer()
        sentiment_result = sentiment_analyzer.analyze(
            getattr(customer, 'feedback', None)
        )

        # Explanations
        explainer = Explainer()
        reasons = explainer.explain_customer(
            model, X_scaled, preprocessor.feature_names, customer
        )

        now = datetime.now(timezone.utc)
        prediction_id = str(uuid.uuid4())

        pred = CustomerPrediction(
            prediction_id=prediction_id,
            customer_id=customer.customer_id,
            churn_ratio=churn_prob,
            risk_category=risk_category,
            sentiment=sentiment_result['sentiment'],
            sentiment_score=sentiment_result['sentiment_score'],
            confidence_score=sentiment_result['confidence_score'],
            top_reasons=reasons[:3],
            model_version=registry.model_version,
            model_accuracy=registry.accuracy,
            algorithm=registry.algorithm,
            prediction_timestamp=now,
            training_timestamp=registry.training_date,
            email=customer.email,
            contact_number=customer.contact_number,
        )
        db.add(pred)
        db.commit()

        return {
            'prediction_id': prediction_id,
            'customer_id': customer.customer_id,
            'name': customer.name,
            'churn_ratio': churn_prob,
            'risk_category': risk_category,
            'sentiment': sentiment_result['sentiment'],
            'sentiment_score': sentiment_result['sentiment_score'],
            'confidence_score': sentiment_result['confidence_score'],
            'top_reasons': reasons[:3],
            'email': customer.email,
            'contact_number': customer.contact_number,
            'model_version': registry.model_version,
            'model_accuracy': registry.accuracy,
            'algorithm': registry.algorithm,
            'prediction_timestamp': now,
            'training_timestamp': registry.training_date,
        }

    def get_high_risk(self, db: Session) -> list[dict]:
        """Get all customers whose latest prediction is high-risk.

        Uses a subquery to select the most recent prediction per
        ``customer_id``, then filters to ``risk_category = 'High'``.

        Args:
            db: Active SQLAlchemy session.

        Returns:
            List of high-risk customer result dictionaries, ordered by
            churn ratio descending.
        """
        # Subquery: latest prediction timestamp per customer
        subq = (
            db.query(
                CustomerPrediction.customer_id,
                func.max(CustomerPrediction.prediction_timestamp).label(
                    'max_ts'
                ),
            )
            .group_by(CustomerPrediction.customer_id)
            .subquery()
        )

        predictions = (
            db.query(CustomerPrediction)
            .join(
                subq,
                (CustomerPrediction.customer_id == subq.c.customer_id)
                & (
                    CustomerPrediction.prediction_timestamp
                    == subq.c.max_ts
                ),
            )
            .filter(CustomerPrediction.risk_category == 'High')
            .order_by(desc(CustomerPrediction.churn_ratio))
            .limit(100)
            .all()
        )

        if not predictions:
            return []

        customer_ids = [pred.customer_id for pred in predictions]
        customers = db.query(Customer).filter(Customer.customer_id.in_(customer_ids)).all()
        customer_map = {c.customer_id: c for c in customers}

        results: list[dict] = []
        for pred in predictions:
            customer = customer_map.get(pred.customer_id)
            results.append(
                {
                    'name': customer.name if customer else 'Unknown',
                    'customer_id': pred.customer_id,
                    'churn_ratio': pred.churn_ratio,
                    'risk_category': pred.risk_category,
                    'sentiment': pred.sentiment,
                    'sentiment_score': pred.sentiment_score,
                    'confidence_score': pred.confidence_score,
                    'top_reasons': pred.top_reasons or [],
                    'email': pred.email,
                    'contact_number': pred.contact_number,
                    'model_version': pred.model_version,
                    'training_timestamp': pred.training_timestamp,
                    'prediction_timestamp': pred.prediction_timestamp,
                }
            )

        return results

    def get_all_predicted(self, db: Session, page: int = 1, limit: int = 100) -> dict:
        """Get all customers with their latest predictions.

        Uses a subquery to select the most recent prediction per
        ``customer_id``.

        Args:
            db: Active SQLAlchemy session.
            page: Page number (1-indexed).
            limit: Number of items per page.

        Returns:
            Dictionary containing paginated data and metadata.
        """
        # Subquery: latest prediction timestamp per customer
        subq = (
            db.query(
                CustomerPrediction.customer_id,
                func.max(CustomerPrediction.prediction_timestamp).label('max_ts'),
            )
            .group_by(CustomerPrediction.customer_id)
            .subquery()
        )

        base_query = db.query(CustomerPrediction).join(
            subq,
            (CustomerPrediction.customer_id == subq.c.customer_id)
            & (CustomerPrediction.prediction_timestamp == subq.c.max_ts),
        )

        total_count = base_query.count()
        import math
        total_pages = math.ceil(total_count / limit) if limit > 0 else 0

        predictions = (
            base_query
            .order_by(desc(CustomerPrediction.churn_ratio))
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

        customer_ids = [pred.customer_id for pred in predictions]
        customers = db.query(Customer).filter(Customer.customer_id.in_(customer_ids)).all()
        customer_map = {c.customer_id: c for c in customers}

        results: list[dict] = []
        for pred in predictions:
            customer = customer_map.get(pred.customer_id)
            results.append(
                {
                    'customer_id': pred.customer_id,
                    'name': customer.name if customer else 'Unknown',
                    'churn_ratio': pred.churn_ratio,
                    'risk_category': pred.risk_category,
                    'sentiment': pred.sentiment,
                    'top_reasons': pred.top_reasons or [],
                    'email': pred.email,
                    'contact_number': pred.contact_number,
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
        self,
        customers: list[Customer],
        preprocessor: DataPreprocessor,
    ) -> pd.DataFrame:
        """Build a feature matrix from Customer ORM objects using the
        saved preprocessor's encoders and scaler.

        Args:
            customers: List of Customer ORM instances.
            preprocessor: Previously loaded ``DataPreprocessor``.

        Returns:
            Scaled feature DataFrame.
        """
        customer_data = [
            {
                'age': cust.age,
                'policy_type': cust.policy_type,
                'premium_amount': cust.premium_amount,
                'renewal_history': cust.renewal_history,
                'claim_history': cust.claim_history,
                'complaint_count': cust.complaint_count,
                'support_tickets': cust.support_tickets,
            }
            for cust in customers
        ]
        df = pd.DataFrame(customer_data)
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

        # Label-encode using saved encoders (handle unseen labels)
        for col in self.CATEGORICAL_COLS:
            key = f'customer_{col}'
            if key in preprocessor.label_encoders:
                le = preprocessor.label_encoders[key]
                X[col] = X[col].astype(str).apply(
                    lambda val, _le=le: (
                        _le.transform([val])[0]
                        if val in _le.classes_
                        else -1
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

    def _categorise(self, churn_prob: float) -> str:
        """Map a churn probability to a human-readable risk category."""
        if churn_prob >= self.HIGH_THRESHOLD:
            return 'High'
        if churn_prob >= self.MEDIUM_THRESHOLD:
            return 'Medium'
        return 'Low'
