"""
Dashboard schemas for aggregated system statistics and summaries.

Provides Pydantic v2 models for the main dashboard, training dashboard,
and model registry dashboard views.
"""

from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from app.schemas.training import TrainingHistoryResponse, ModelRegistryResponse


class DashboardStats(BaseModel):
    """Schema for aggregate system statistics displayed on the main dashboard."""

    total_leads: int
    total_customers: int
    total_predictions_leads: int
    total_predictions_customers: int
    total_training_sessions: int
    latest_lead_model_version: Optional[str] = None
    latest_customer_model_version: Optional[str] = None
    latest_lead_accuracy: Optional[float] = None
    latest_customer_accuracy: Optional[float] = None


class DashboardResponse(BaseModel):
    """Schema for the main dashboard API response."""

    stats: DashboardStats
    recent_training: List[TrainingHistoryResponse]


class TrainingDashboardResponse(BaseModel):
    """Schema for the training dashboard with full training history."""

    training_history: List[TrainingHistoryResponse]
    total_sessions: int


class ModelDashboardResponse(BaseModel):
    """Schema for the model registry dashboard."""

    models: List[ModelRegistryResponse]
    total_models: int
