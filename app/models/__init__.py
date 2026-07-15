"""
Models package.

Imports all models so that Alembic and SQLAlchemy can discover them
via Base.metadata for automatic migration generation.
"""

from app.models.user import User
from app.models.lead import Lead
from app.models.customer import Customer
from app.models.prediction import LeadPrediction, CustomerPrediction
from app.models.training import TrainingHistory, ModelRegistry
from app.models.audit import AuditLog, UploadedFile

__all__ = [
    "User",
    "Lead",
    "Customer",
    "LeadPrediction",
    "CustomerPrediction",
    "TrainingHistory",
    "ModelRegistry",
    "AuditLog",
    "UploadedFile",
]
