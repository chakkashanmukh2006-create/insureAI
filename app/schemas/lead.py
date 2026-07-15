"""
Lead schemas for lead management and propensity prediction.

Provides Pydantic v2 models for lead creation, response serialization,
prediction results, and top-20 lead rankings.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List


class LeadBase(BaseModel):
    """Base schema with common lead fields."""

    full_name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    occupation: Optional[str] = None
    annual_income: Optional[float] = None
    city: Optional[str] = None
    existing_policy: Optional[bool] = False
    product_interested: Optional[str] = None
    website_visits: Optional[int] = 0
    email_opens: Optional[int] = 0
    calls_answered: Optional[int] = 0
    form_submitted: Optional[bool] = False
    last_interaction_days: Optional[int] = None
    lead_source: Optional[str] = None
    email: Optional[str] = None
    contact_number: Optional[str] = None
    conversion_target: Optional[int] = 0


class LeadCreate(LeadBase):
    """Schema for creating a new lead record."""

    lead_id: str


class LeadResponse(LeadBase):
    """Schema for lead data returned in API responses."""

    id: int
    lead_id: str
    created_at: datetime

    model_config = {"from_attributes": True}


class LeadPredictionResponse(BaseModel):
    """Schema for individual lead propensity prediction results."""

    prediction_id: str
    lead_id: str
    full_name: str
    propensity_ratio: float
    lead_score: float
    category: str
    top_reasons: List[str]
    email: Optional[str] = None
    contact_number: Optional[str] = None
    model_version: str
    model_accuracy: float
    algorithm: str
    prediction_timestamp: datetime
    training_timestamp: datetime

    model_config = {"from_attributes": True}


class Top20LeadResponse(BaseModel):
    """Schema for top-20 highest propensity leads."""

    name: str
    propensity_ratio: float
    email: Optional[str] = None
    contact_number: Optional[str] = None
    top_reasons: List[str]
    model_version: str
    training_timestamp: datetime
    prediction_timestamp: datetime

    model_config = {"from_attributes": True}
