"""
Customer schemas for customer management and churn prediction.

Provides Pydantic v2 models for customer creation, response serialization,
churn prediction results, and high-risk customer identification.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List


class CustomerBase(BaseModel):
    """Base schema with common customer fields."""

    name: str
    age: Optional[int] = None
    policy_type: Optional[str] = None
    premium_amount: Optional[float] = None
    renewal_history: Optional[int] = 0
    claim_history: Optional[int] = 0
    complaint_count: Optional[int] = 0
    support_tickets: Optional[int] = 0
    feedback: Optional[str] = None
    email: Optional[str] = None
    contact_number: Optional[str] = None
    churn_target: Optional[int] = 0


class CustomerCreate(CustomerBase):
    """Schema for creating a new customer record."""

    customer_id: str


class CustomerResponse(CustomerBase):
    """Schema for customer data returned in API responses."""

    id: int
    customer_id: str
    created_at: datetime

    model_config = {"from_attributes": True}


class CustomerPredictionResponse(BaseModel):
    """Schema for individual customer churn prediction results."""

    prediction_id: str
    customer_id: str
    name: str
    churn_ratio: float
    risk_category: str
    sentiment: str
    sentiment_score: float
    confidence_score: float
    top_reasons: List[str]
    email: Optional[str] = None
    contact_number: Optional[str] = None
    model_version: str
    model_accuracy: float
    algorithm: str
    prediction_timestamp: datetime
    training_timestamp: datetime

    model_config = {"from_attributes": True}


class HighRiskCustomerResponse(BaseModel):
    """Schema for high-risk churn customers summary."""

    customer_id: str
    name: str
    churn_ratio: float
    risk_category: str
    sentiment: str
    top_reasons: List[str]
    email: Optional[str] = None
    contact_number: Optional[str] = None
    model_version: str

    model_config = {"from_attributes": True}
