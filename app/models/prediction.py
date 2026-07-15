"""
Prediction models for lead propensity and customer churn predictions.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.database.base import Base


class LeadPrediction(Base):
    """Model storing lead propensity prediction results."""

    __tablename__ = "lead_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(String(50), unique=True, nullable=False, index=True)
    lead_id = Column(String(50), ForeignKey("leads.lead_id"), nullable=False)
    propensity_ratio = Column(Float)
    lead_score = Column(Float)
    category = Column(String(20))  # High / Medium / Low
    top_reasons = Column(JSON)  # list of strings
    model_version = Column(String(20))
    model_accuracy = Column(Float)
    algorithm = Column(String(50), default="XGBoost")
    prediction_timestamp = Column(DateTime, server_default=func.now())
    training_timestamp = Column(DateTime)
    email = Column(String(255))
    contact_number = Column(String(20))

    # Relationships
    lead = relationship("Lead", back_populates="predictions")

    def __repr__(self) -> str:
        return f"<LeadPrediction(id={self.id}, prediction_id='{self.prediction_id}', lead_id='{self.lead_id}')>"


class CustomerPrediction(Base):
    """Model storing customer churn prediction results."""

    __tablename__ = "customer_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(String(50), unique=True, nullable=False, index=True)
    customer_id = Column(String(50), ForeignKey("customers.customer_id"), nullable=False)
    churn_ratio = Column(Float)
    risk_category = Column(String(20))  # High / Medium / Low
    sentiment = Column(String(20))  # Positive / Neutral / Negative
    sentiment_score = Column(Float)
    confidence_score = Column(Float)
    top_reasons = Column(JSON)
    model_version = Column(String(20))
    model_accuracy = Column(Float)
    algorithm = Column(String(50), default="XGBoost")
    prediction_timestamp = Column(DateTime, server_default=func.now())
    training_timestamp = Column(DateTime)
    email = Column(String(255))
    contact_number = Column(String(20))

    # Relationships
    customer = relationship("Customer", back_populates="predictions")

    def __repr__(self) -> str:
        return f"<CustomerPrediction(id={self.id}, prediction_id='{self.prediction_id}', customer_id='{self.customer_id}')>"
