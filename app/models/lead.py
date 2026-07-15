"""
Lead model for insurance lead propensity tracking.
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.database.base import Base


class Lead(Base):
    """Lead model representing potential insurance customers for propensity scoring."""

    __tablename__ = "leads"

    id = Column(Integer, primary_key=True, autoincrement=True)
    lead_id = Column(String(50), unique=True, nullable=False, index=True)
    full_name = Column(String(200), nullable=False)
    age = Column(Integer)
    gender = Column(String(20))
    occupation = Column(String(100))
    annual_income = Column(Float)
    city = Column(String(100))
    existing_policy = Column(Boolean, default=False)
    product_interested = Column(String(100))
    website_visits = Column(Integer, default=0)
    email_opens = Column(Integer, default=0)
    calls_answered = Column(Integer, default=0)
    form_submitted = Column(Boolean, default=False)
    last_interaction_days = Column(Integer)
    lead_source = Column(String(100))
    email = Column(String(255))
    contact_number = Column(String(20))
    conversion_target = Column(Integer, default=0)  # 0 or 1
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    predictions = relationship("LeadPrediction", back_populates="lead")

    def __repr__(self) -> str:
        return f"<Lead(id={self.id}, lead_id='{self.lead_id}', name='{self.full_name}')>"
