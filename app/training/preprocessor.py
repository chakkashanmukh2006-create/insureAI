"""
Data preprocessing pipeline for Insurance AI ML models.

Handles feature engineering, encoding, scaling, and transformation
for both lead propensity and customer churn prediction models.
"""

import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.config.settings import settings
from app.utils.logger import logger


class DataPreprocessor:
    """Handles feature engineering and preprocessing for ML models.

    Manages label encoding for categorical features, standard scaling
    for numeric features, and provides both batch (training) and
    single-record (inference) preprocessing pipelines.
    """

    # Feature column definitions
    LEAD_FEATURE_COLS: list[str] = [
        'age', 'gender', 'occupation', 'annual_income', 'city',
        'existing_policy', 'product_interested', 'website_visits',
        'email_opens', 'calls_answered', 'form_submitted',
        'last_interaction_days', 'lead_source'
    ]

    LEAD_BOOL_COLS: list[str] = ['existing_policy', 'form_submitted']

    LEAD_CATEGORICAL_COLS: list[str] = [
        'gender', 'occupation', 'city', 'product_interested', 'lead_source'
    ]

    CUSTOMER_FEATURE_COLS: list[str] = [
        'age', 'policy_type', 'premium_amount',
        'renewal_history', 'claim_history', 'complaint_count',
        'support_tickets'
    ]

    CUSTOMER_CATEGORICAL_COLS: list[str] = ['policy_type']

    def __init__(self) -> None:
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler: StandardScaler = StandardScaler()
        self.feature_names: list[str] = []

    # ------------------------------------------------------------------ #
    # Batch preprocessing (training)
    # ------------------------------------------------------------------ #

    def preprocess_leads(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """Preprocess lead data for training.

        Args:
            df: Raw lead DataFrame with all feature columns and
                ``conversion_target``.

        Returns:
            Tuple of (X_scaled, y, feature_names).

        Raises:
            ValueError: If the DataFrame is empty or the target column
                is missing.
        """
        if df.empty:
            raise ValueError("Cannot preprocess an empty DataFrame.")
        if 'conversion_target' not in df.columns:
            raise ValueError("Target column 'conversion_target' is missing.")

        # Validate that required feature columns exist
        missing_cols = [c for c in self.LEAD_FEATURE_COLS if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required feature columns: {missing_cols}"
            )

        y = df['conversion_target'].astype(int)
        X = df[self.LEAD_FEATURE_COLS].copy()

        # Handle missing values
        for col in X.select_dtypes(include=[np.number]).columns:
            X[col] = X[col].fillna(X[col].median())
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].fillna('Unknown')

        # Convert boolean columns to int
        for col in self.LEAD_BOOL_COLS:
            if col in X.columns:
                X[col] = X[col].astype(int)

        # Label-encode categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[f'lead_{col}'] = le

        # Scale all features
        self.feature_names = X.columns.tolist()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=self.feature_names,
            index=X.index,
        )

        logger.info(
            f"Preprocessed {len(X_scaled)} lead records with "
            f"{len(self.feature_names)} features."
        )
        return X_scaled, y, self.feature_names

    def preprocess_customers(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """Preprocess customer data for training.

        Args:
            df: Raw customer DataFrame with all feature columns and
                ``churn_target``.

        Returns:
            Tuple of (X_scaled, y, feature_names).

        Raises:
            ValueError: If the DataFrame is empty or the target column
                is missing.
        """
        if df.empty:
            raise ValueError("Cannot preprocess an empty DataFrame.")
        if 'churn_target' not in df.columns:
            raise ValueError("Target column 'churn_target' is missing.")

        missing_cols = [
            c for c in self.CUSTOMER_FEATURE_COLS if c not in df.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Missing required feature columns: {missing_cols}"
            )

        y = df['churn_target'].astype(int)
        X = df[self.CUSTOMER_FEATURE_COLS].copy()

        # Handle missing values
        for col in X.select_dtypes(include=[np.number]).columns:
            X[col] = X[col].fillna(X[col].median())
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].fillna('Unknown')

        # Label-encode categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[f'customer_{col}'] = le

        self.feature_names = X.columns.tolist()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=self.feature_names,
            index=X.index,
        )

        logger.info(
            f"Preprocessed {len(X_scaled)} customer records with "
            f"{len(self.feature_names)} features."
        )
        return X_scaled, y, self.feature_names

    # ------------------------------------------------------------------ #
    # Single-record preprocessing (inference)
    # ------------------------------------------------------------------ #

    def preprocess_lead_single(
        self, lead_data: dict, preprocessor_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Preprocess a single lead record for prediction.

        Uses saved preprocessor artifacts (label encoders and scaler)
        to apply identical transformations as during training.

        Args:
            lead_data: Dictionary of lead feature values.
            preprocessor_path: Path to the saved preprocessor directory.
                If ``None``, uses ``settings.MODEL_STORAGE_PATH``.

        Returns:
            Preprocessed single-row DataFrame ready for model inference.

        Raises:
            FileNotFoundError: If the preprocessor file does not exist.
            ValueError: If ``lead_data`` is empty.
        """
        if not lead_data:
            raise ValueError("lead_data must be a non-empty dictionary.")

        # Load saved preprocessor if not already loaded
        if not self.feature_names:
            load_path = preprocessor_path or settings.MODEL_STORAGE_PATH
            saved = self._load_preprocessor_file(load_path, 'lead')
            self.label_encoders = saved['label_encoders']
            self.scaler = saved['scaler']
            self.feature_names = saved['feature_names']

        df = pd.DataFrame([lead_data])
        X = pd.DataFrame(columns=self.LEAD_FEATURE_COLS)
        X = pd.concat([X, df], ignore_index=True)
        X = X[self.LEAD_FEATURE_COLS]

        # Fill missing numeric values with 0 (no batch median available)
        for col in X.select_dtypes(include=[np.number]).columns:
            X[col] = X[col].fillna(0)
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].fillna('Unknown')

        # Convert boolean columns
        for col in self.LEAD_BOOL_COLS:
            if col in X.columns:
                X[col] = X[col].astype(int)

        # Label-encode using saved encoders (handle unseen labels)
        for col in self.LEAD_CATEGORICAL_COLS:
            key = f'lead_{col}'
            if key in self.label_encoders:
                le = self.label_encoders[key]
                X[col] = X[col].astype(str).apply(
                    lambda val, _le=le: (
                        _le.transform([val])[0]
                        if val in _le.classes_
                        else -1
                    )
                )
            else:
                X[col] = 0

        # Scale using saved scaler
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_names,
        )
        return X_scaled

    def preprocess_customer_single(
        self, customer_data: dict, preprocessor_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Preprocess a single customer record for prediction.

        Uses saved preprocessor artifacts to apply identical
        transformations as during training.

        Args:
            customer_data: Dictionary of customer feature values.
            preprocessor_path: Path to the saved preprocessor directory.
                If ``None``, uses ``settings.MODEL_STORAGE_PATH``.

        Returns:
            Preprocessed single-row DataFrame ready for model inference.

        Raises:
            FileNotFoundError: If the preprocessor file does not exist.
            ValueError: If ``customer_data`` is empty.
        """
        if not customer_data:
            raise ValueError("customer_data must be a non-empty dictionary.")

        # Load saved preprocessor if not already loaded
        if not self.feature_names:
            load_path = preprocessor_path or settings.MODEL_STORAGE_PATH
            saved = self._load_preprocessor_file(load_path, 'customer')
            self.label_encoders = saved['label_encoders']
            self.scaler = saved['scaler']
            self.feature_names = saved['feature_names']

        df = pd.DataFrame([customer_data])
        X = pd.DataFrame(columns=self.CUSTOMER_FEATURE_COLS)
        X = pd.concat([X, df], ignore_index=True)
        X = X[self.CUSTOMER_FEATURE_COLS]

        # Fill missing numeric values with 0
        for col in X.select_dtypes(include=[np.number]).columns:
            X[col] = X[col].fillna(0)
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].fillna('Unknown')

        # Label-encode using saved encoders (handle unseen labels)
        for col in self.CUSTOMER_CATEGORICAL_COLS:
            key = f'customer_{col}'
            if key in self.label_encoders:
                le = self.label_encoders[key]
                X[col] = X[col].astype(str).apply(
                    lambda val, _le=le: (
                        _le.transform([val])[0]
                        if val in _le.classes_
                        else -1
                    )
                )
            else:
                X[col] = 0

        # Scale using saved scaler
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_names,
        )
        return X_scaled

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str, model_type: str) -> str:
        """Save preprocessor artifacts (encoders, scaler, feature names).

        Args:
            path: Directory to save artifacts in.
            model_type: ``'lead'`` or ``'customer'``.

        Returns:
            Full path of the saved file.
        """
        os.makedirs(path, exist_ok=True)
        save_data = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
        }
        filepath = os.path.join(path, f'{model_type}_preprocessor.joblib')
        joblib.dump(save_data, filepath)
        logger.info(f"Saved {model_type} preprocessor to {filepath}")
        return filepath

    @classmethod
    def load(cls, path: str, model_type: str) -> 'DataPreprocessor':
        """Load a preprocessor from saved artifacts.

        Args:
            path: Directory containing the saved artifact.
            model_type: ``'lead'`` or ``'customer'``.

        Returns:
            Populated ``DataPreprocessor`` instance.

        Raises:
            FileNotFoundError: If the artifact file does not exist.
        """
        filepath = os.path.join(path, f'{model_type}_preprocessor.joblib')
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Preprocessor file not found at {filepath}. "
                "Please train a model first."
            )
        saved = joblib.load(filepath)
        preprocessor = cls()
        preprocessor.label_encoders = saved['label_encoders']
        preprocessor.scaler = saved['scaler']
        preprocessor.feature_names = saved['feature_names']
        logger.info(f"Loaded {model_type} preprocessor from {filepath}")
        return preprocessor

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_preprocessor_file(path: str, model_type: str) -> dict:
        """Load raw preprocessor dict from disk.

        Raises:
            FileNotFoundError: If the artifact file does not exist.
        """
        filepath = os.path.join(path, f'{model_type}_preprocessor.joblib')
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Preprocessor file not found at {filepath}. "
                "Please train a model first."
            )
        return joblib.load(filepath)
