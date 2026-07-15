"""
Training schemas for ML model training and registry.

Provides Pydantic v2 models for training requests, training history,
model registry entries, and latest training status.
"""

from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class TrainRequest(BaseModel):
    """Schema for initiating a new training session."""

    notes: Optional[str] = None


class TrainingHistoryResponse(BaseModel):
    """Schema for a single training session history entry."""

    training_id: str
    model_type: str
    model_version: str
    training_datetime: datetime
    algorithm: str
    lead_records_used: int
    customer_records_used: int
    dataset_source: str
    accuracy: float
    precision_score: float
    recall: float
    f1_score: float
    training_duration_seconds: float
    started_by: str
    status: str
    notes: Optional[str] = None

    model_config = {"from_attributes": True}


class ModelRegistryResponse(BaseModel):
    """Schema for a registered ML model entry."""

    model_id: str
    model_type: str
    model_version: str
    training_date: datetime
    accuracy: float
    precision_score: float
    recall: float
    f1_score: float
    algorithm: str
    dataset_size: int
    dataset_source: str
    status: str
    model_path: Optional[str] = None

    model_config = {"from_attributes": True}


class TrainingLatestResponse(BaseModel):
    """Schema for the latest training status of both lead and customer models."""

    lead_model: Optional[TrainingHistoryResponse] = None
    customer_model: Optional[TrainingHistoryResponse] = None
