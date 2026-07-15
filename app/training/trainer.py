"""
Core ML training orchestrator.

Manages the complete end-to-end training pipeline for both
lead propensity and customer churn XGBoost models, including
data loading, preprocessing, model fitting, evaluation, versioned
persistence, and database registration.
"""

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session
from xgboost import XGBClassifier

from app.config.settings import settings
from app.models.customer import Customer
from app.models.lead import Lead
from app.models.training import ModelRegistry, TrainingHistory
from app.training.model_manager import ModelManager
from app.training.preprocessor import DataPreprocessor
from app.utils.logger import logger


class TrainingService:
    """Orchestrates the complete model training pipeline.

    Workflow executed by ``train_all``:
      1. Validate minimum data requirements.
      2. Train lead propensity model (XGBoost).
      3. Train customer churn model (XGBoost).
      4. Version and persist models + preprocessors.
      5. Register models in ``model_registry``.
      6. Log results in ``training_history``.
      7. Return a summary dictionary of both runs.
    """

    # Default XGBoost hyper-parameters (can be overridden)
    DEFAULT_XGBOOST_PARAMS: dict[str, Any] = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42,
    }

    MIN_RECORDS: int = 10  # minimum records needed for training

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def train_all(
        self,
        db: Session,
        started_by: str,
        notes: Optional[str] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> dict[str, dict]:
        """Train both lead and customer models end-to-end.

        Args:
            db: Active SQLAlchemy session.
            started_by: Username / identifier of who triggered training.
            notes: Optional free-form notes to attach to the run.
            log_callback: Optional callback to stream log messages.

        Returns:
            Dictionary with ``'lead_model'`` and ``'customer_model'``
            keys, each containing training metrics and metadata.

        Raises:
            ValueError: If either table has fewer than ``MIN_RECORDS``.
        """
        if log_callback: log_callback("Validating dataset sizes...")
        lead_count: int = db.query(Lead).count()
        customer_count: int = db.query(Customer).count()

        if lead_count < self.MIN_RECORDS:
            raise ValueError(
                f"Insufficient lead data for training. "
                f"Found {lead_count} records, need at least {self.MIN_RECORDS}."
            )
        if customer_count < self.MIN_RECORDS:
            raise ValueError(
                f"Insufficient customer data for training. "
                f"Found {customer_count} records, need at least {self.MIN_RECORDS}."
            )

        results: dict[str, dict] = {}

        # Train lead propensity model
        if log_callback: log_callback("Training Lead Propensity XGBoost Model...")
        lead_result = self._train_lead_model(
            db, started_by, lead_count, customer_count, notes
        )
        results['lead_model'] = lead_result

        # Train customer churn model
        if log_callback: log_callback("Training Customer Churn XGBoost Model...")
        customer_result = self._train_customer_model(
            db, started_by, lead_count, customer_count, notes
        )
        results['customer_model'] = customer_result

        if log_callback: log_callback("Models trained successfully. Saving artifacts...")
        logger.info(
            f"Training complete. Lead accuracy: "
            f"{lead_result['accuracy']:.4f}, Customer accuracy: "
            f"{customer_result['accuracy']:.4f}"
        )
        return results

    def train_lead_only(
        self,
        db: Session,
        started_by: str,
        notes: Optional[str] = None,
    ) -> dict:
        """Train only the lead propensity model."""
        lead_count = db.query(Lead).count()
        customer_count = db.query(Customer).count()
        if lead_count < self.MIN_RECORDS:
            raise ValueError(
                f"Insufficient lead data. Found {lead_count}, "
                f"need at least {self.MIN_RECORDS}."
            )
        return self._train_lead_model(
            db, started_by, lead_count, customer_count, notes
        )

    def train_customer_only(
        self,
        db: Session,
        started_by: str,
        notes: Optional[str] = None,
    ) -> dict:
        """Train only the customer churn model."""
        lead_count = db.query(Lead).count()
        customer_count = db.query(Customer).count()
        if customer_count < self.MIN_RECORDS:
            raise ValueError(
                f"Insufficient customer data. Found {customer_count}, "
                f"need at least {self.MIN_RECORDS}."
            )
        return self._train_customer_model(
            db, started_by, lead_count, customer_count, notes
        )

    # ------------------------------------------------------------------ #
    # Lead model training
    # ------------------------------------------------------------------ #

    def _train_lead_model(
        self,
        db: Session,
        started_by: str,
        lead_count: int,
        customer_count: int,
        notes: Optional[str],
    ) -> dict:
        """Train the lead propensity XGBoost model.

        Steps:
          1. Create a ``TrainingHistory`` row (status=running).
          2. Load all leads from the database.
          3. Preprocess features.
          4. Split 80/20 with stratification.
          5. Fit XGBoost.
          6. Evaluate on the hold-out set.
          7. Persist model, preprocessor, and feature names.
          8. Register in ``ModelRegistry``.
          9. Update ``TrainingHistory`` (status=success).
        """
        start_time = time.time()
        training_id = str(uuid.uuid4())
        version = ModelManager.get_next_version(db, 'lead')

        # 1. Create training history record
        history = TrainingHistory(
            training_id=training_id,
            model_type='lead',
            model_version=version,
            algorithm='XGBoost',
            lead_records_used=lead_count,
            customer_records_used=customer_count,
            dataset_source='Database (Kaggle + Uploaded)',
            started_by=started_by,
            status='running',
        )
        db.add(history)
        db.commit()

        try:
            # 2. Load data
            leads = db.query(Lead).all()
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
                    'conversion_target': lead.conversion_target,
                }
                for lead in leads
            ]
            df = pd.DataFrame(lead_data)

            # 3. Preprocess
            preprocessor = DataPreprocessor()
            X, y, feature_names = preprocessor.preprocess_leads(df)

            # 4. Train / test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # 5. Train XGBoost
            model = XGBClassifier(**self.DEFAULT_XGBOOST_PARAMS)
            model.fit(X_train, y_train)

            # 6. Evaluate
            metrics = self._evaluate(model, X_test, y_test)
            duration = time.time() - start_time

            # 7. Persist artifacts
            model_path = ModelManager.save_model(model, 'lead', version)
            preprocessor.save(settings.MODEL_STORAGE_PATH, 'lead')

            feature_names_path = os.path.join(
                settings.MODEL_STORAGE_PATH,
                f'lead_feature_names_{version}.joblib',
            )
            joblib.dump(feature_names, feature_names_path)

            # 8. Register model
            registry = ModelRegistry(
                model_id=str(uuid.uuid4()),
                model_type='lead',
                model_version=version,
                accuracy=metrics['accuracy'],
                precision_score=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1_score'],
                algorithm='XGBoost',
                dataset_size=lead_count,
                dataset_source='Database (Kaggle + Uploaded)',
                status='active',
                model_path=model_path,
            )
            db.add(registry)

            # 9. Finalise training history
            history.accuracy = metrics['accuracy']
            history.precision_score = metrics['precision']
            history.recall = metrics['recall']
            history.f1_score = metrics['f1_score']
            history.training_duration_seconds = duration
            history.status = 'success'
            history.notes = notes
            db.commit()

            # Archive previous active models
            ModelManager.deactivate_previous_models(db, 'lead', version)

            logger.info(
                f"Lead model {version} trained successfully. "
                f"Accuracy: {metrics['accuracy']:.4f}, "
                f"F1: {metrics['f1_score']:.4f}, "
                f"Duration: {duration:.1f}s"
            )

            return {
                'model_type': 'lead',
                'model_version': version,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'training_duration': round(duration, 2),
                'records_used': lead_count,
                'status': 'success',
            }

        except Exception as exc:
            history.status = 'failed'
            history.notes = f"Error: {str(exc)}"
            db.commit()
            logger.error(f"Lead model training failed: {exc}")
            raise

    # ------------------------------------------------------------------ #
    # Customer model training
    # ------------------------------------------------------------------ #

    def _train_customer_model(
        self,
        db: Session,
        started_by: str,
        lead_count: int,
        customer_count: int,
        notes: Optional[str],
    ) -> dict:
        """Train the customer churn XGBoost model.

        Follows the same 9-step pattern as ``_train_lead_model`` but
        uses customer feature columns and the ``churn_target``.
        """
        start_time = time.time()
        training_id = str(uuid.uuid4())
        version = ModelManager.get_next_version(db, 'customer')

        # 1. Create training history record
        history = TrainingHistory(
            training_id=training_id,
            model_type='customer',
            model_version=version,
            algorithm='XGBoost',
            lead_records_used=lead_count,
            customer_records_used=customer_count,
            dataset_source='Database (Kaggle + Uploaded)',
            started_by=started_by,
            status='running',
        )
        db.add(history)
        db.commit()

        try:
            # 2. Load data
            customers = db.query(Customer).all()
            customer_data = [
                {
                    'age': cust.age,
                    'policy_type': cust.policy_type,
                    'premium_amount': cust.premium_amount,
                    'renewal_history': cust.renewal_history,
                    'claim_history': cust.claim_history,
                    'complaint_count': cust.complaint_count,
                    'support_tickets': cust.support_tickets,
                    'churn_target': cust.churn_target,
                }
                for cust in customers
            ]
            df = pd.DataFrame(customer_data)

            # 3. Preprocess
            preprocessor = DataPreprocessor()
            X, y, feature_names = preprocessor.preprocess_customers(df)

            # 4. Train / test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # 5. Train XGBoost
            model = XGBClassifier(**self.DEFAULT_XGBOOST_PARAMS)
            model.fit(X_train, y_train)

            # 6. Evaluate
            metrics = self._evaluate(model, X_test, y_test)
            duration = time.time() - start_time

            # 7. Persist artifacts
            model_path = ModelManager.save_model(model, 'customer', version)
            preprocessor.save(settings.MODEL_STORAGE_PATH, 'customer')

            feature_names_path = os.path.join(
                settings.MODEL_STORAGE_PATH,
                f'customer_feature_names_{version}.joblib',
            )
            joblib.dump(feature_names, feature_names_path)

            # 8. Register model
            registry = ModelRegistry(
                model_id=str(uuid.uuid4()),
                model_type='customer',
                model_version=version,
                accuracy=metrics['accuracy'],
                precision_score=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1_score'],
                algorithm='XGBoost',
                dataset_size=customer_count,
                dataset_source='Database (Kaggle + Uploaded)',
                status='active',
                model_path=model_path,
            )
            db.add(registry)

            # 9. Finalise training history
            history.accuracy = metrics['accuracy']
            history.precision_score = metrics['precision']
            history.recall = metrics['recall']
            history.f1_score = metrics['f1_score']
            history.training_duration_seconds = duration
            history.status = 'success'
            history.notes = notes
            db.commit()

            # Archive previous active models
            ModelManager.deactivate_previous_models(db, 'customer', version)

            logger.info(
                f"Customer model {version} trained successfully. "
                f"Accuracy: {metrics['accuracy']:.4f}, "
                f"F1: {metrics['f1_score']:.4f}, "
                f"Duration: {duration:.1f}s"
            )

            return {
                'model_type': 'customer',
                'model_version': version,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'training_duration': round(duration, 2),
                'records_used': customer_count,
                'status': 'success',
            }

        except Exception as exc:
            history.status = 'failed'
            history.notes = f"Error: {str(exc)}"
            db.commit()
            logger.error(f"Customer model training failed: {exc}")
            raise

    # ------------------------------------------------------------------ #
    # Evaluation helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _evaluate(
        model: XGBClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, float]:
        """Compute classification metrics on the hold-out set.

        Args:
            model: Fitted classifier.
            X_test: Test feature matrix.
            y_test: True labels.

        Returns:
            Dictionary with ``accuracy``, ``precision``, ``recall``,
            and ``f1_score``.
        """
        y_pred = model.predict(X_test)
        return {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        }
