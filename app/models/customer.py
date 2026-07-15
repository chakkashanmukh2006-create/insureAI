"""
Customer model for insurance customer churn tracking.
"""

from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.database.base import Base


class Customer(Base):
    """Customer model representing existing insurance customers for churn prediction."""

    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    age = Column(Integer)
    policy_type = Column(String(100))
    premium_amount = Column(Float)
    renewal_history = Column(Integer, default=0)  # number of renewals
    claim_history = Column(Integer, default=0)  # number of claims
    complaint_count = Column(Integer, default=0)
    support_tickets = Column(Integer, default=0)
    feedback = Column(Text)  # can be long text
    email = Column(String(255))
    contact_number = Column(String(20))
    churn_target = Column(Integer, default=0)  # 0 or 1
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    predictions = relationship("CustomerPrediction", back_populates="customer")

    def __repr__(self) -> str:
        return f"<Customer(id={self.id}, customer_id='{self.customer_id}', name='{self.name}')>"
