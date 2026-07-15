"""
ML model explainability engine.

Generates human-readable reasons for lead propensity and customer
churn predictions using SHAP values (primary) or XGBoost feature
importance (fallback).
"""

from typing import Any, Optional

import numpy as np
import pandas as pd

from app.utils.logger import logger


class Explainer:
    """Generates human-readable explanations for ML predictions.

    Strategies (tried in order):
      1. **SHAP** ``TreeExplainer`` — per-instance feature contributions.
      2. **XGBoost feature importance** — global importance as fallback.
    """

    # Human-readable reason mappings
    LEAD_REASON_MAP: dict[str, str] = {
        'annual_income': 'High Annual Income',
        'existing_policy': 'Existing Insurance Holder',
        'website_visits': 'Multiple Website Visits',
        'email_opens': 'Opened Marketing Emails',
        'calls_answered': 'Responded to Calls',
        'form_submitted': 'Submitted Inquiry Form',
        'last_interaction_days': 'Recent Interaction',
        'product_interested': 'Strong Product Interest',
        'lead_source': 'High-Quality Lead Source',
        'age': 'Favorable Age Demographics',
        'occupation': 'Professional Occupation',
        'gender': 'Target Demographic Match',
        'city': 'High-Conversion Region',
    }

    CUSTOMER_REASON_MAP: dict[str, str] = {
        'complaint_count': 'Multiple Complaints Filed',
        'support_tickets': 'Frequent Support Requests',
        'renewal_history': 'Low Renewal History',
        'claim_history': 'Frequent Claims Filed',
        'premium_amount': 'Premium Amount Concern',
        'policy_type': 'Policy Type Risk Factor',
        'age': 'Age-Related Risk Factor',
    }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def explain_lead(
        self,
        model: Any,
        X_single: pd.DataFrame,
        feature_names: list[str],
        lead: Any,
    ) -> list[str]:
        """Generate top human-readable reasons for a lead prediction.

        Args:
            model: Trained XGBoost model.
            X_single: Single-row preprocessed DataFrame.
            feature_names: Ordered list of feature column names.
            lead: Original Lead ORM object (unused currently, reserved
                for future rule-based enrichment).

        Returns:
            List of up to 3 human-readable reason strings.
        """
        try:
            return self._explain_with_shap(
                model, X_single, feature_names, self.LEAD_REASON_MAP
            )
        except Exception as exc:
            logger.debug(f"SHAP failed for lead explanation, using feature importance: {exc}")
            return self._explain_with_importance(
                model, feature_names, self.LEAD_REASON_MAP, X_single
            )

    def explain_customer(
        self,
        model: Any,
        X_single: pd.DataFrame,
        feature_names: list[str],
        customer: Any,
    ) -> list[str]:
        """Generate top human-readable reasons for a customer churn prediction.

        Args:
            model: Trained XGBoost model.
            X_single: Single-row preprocessed DataFrame.
            feature_names: Ordered list of feature column names.
            customer: Original Customer ORM object (reserved for future use).

        Returns:
            List of up to 3 human-readable reason strings.
        """
        try:
            return self._explain_with_shap(
                model, X_single, feature_names, self.CUSTOMER_REASON_MAP
            )
        except Exception as exc:
            logger.debug(f"SHAP failed for customer explanation, using feature importance: {exc}")
            return self._explain_with_importance(
                model, feature_names, self.CUSTOMER_REASON_MAP, X_single
            )

    # ------------------------------------------------------------------ #
    # SHAP-based explanation
    # ------------------------------------------------------------------ #

    def _explain_with_shap(
        self,
        model: Any,
        X_single: pd.DataFrame,
        feature_names: list[str],
        reason_map: dict[str, str],
    ) -> list[str]:
        """Use SHAP ``TreeExplainer`` to derive per-instance feature contributions.

        Falls back to ``_explain_with_importance`` if SHAP is unavailable
        or raises an error.
        """
        import shap  # type: ignore[import-untyped]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_single)

        # For binary classification, shap_values may be a list of two
        # arrays (one per class). Use the positive-class array.
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Normalise to a 1-D array of absolute contributions
        abs_shap: np.ndarray = np.abs(shap_values).flatten()

        # If somehow the shape still doesn't match, take the first row
        if abs_shap.shape[0] != len(feature_names):
            abs_shap = abs_shap[: len(feature_names)]

        feature_importance = sorted(
            zip(feature_names, abs_shap),
            key=lambda pair: pair[1],
            reverse=True,
        )

        return self._pick_top_reasons(feature_importance, reason_map)

    # ------------------------------------------------------------------ #
    # Feature-importance-based explanation (fallback)
    # ------------------------------------------------------------------ #

    def _explain_with_importance(
        self,
        model: Any,
        feature_names: list[str],
        reason_map: dict[str, str],
        X_single: Optional[pd.DataFrame] = None,
    ) -> list[str]:
        """Use XGBoost global feature importances as a fallback explanation."""
        try:
            importances: np.ndarray = model.feature_importances_
        except AttributeError:
            logger.warning(
                "Model does not expose feature_importances_. "
                "Returning generic reason."
            )
            return ['Model Pattern Recognition']

        feature_importance = sorted(
            zip(feature_names, importances),
            key=lambda pair: pair[1],
            reverse=True,
        )

        return self._pick_top_reasons(feature_importance, reason_map)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pick_top_reasons(
        ranked_features: list[tuple[str, float]],
        reason_map: dict[str, str],
        top_n: int = 3,
    ) -> list[str]:
        """Select *top_n* unique human-readable reasons from ranked features.

        Args:
            ranked_features: ``[(feature_name, importance), ...]`` sorted
                descending by importance.
            reason_map: Mapping of feature name → human-readable string.
            top_n: Maximum number of reasons to return.

        Returns:
            Deduplicated list of reason strings.
        """
        reasons: list[str] = []
        for feat, _imp in ranked_features:
            reason = reason_map.get(
                feat, f"{feat.replace('_', ' ').title()} Impact"
            )
            if reason not in reasons:
                reasons.append(reason)
            if len(reasons) >= top_n:
                break

        return reasons if reasons else ['Model Pattern Recognition']
