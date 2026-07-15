"""
Training history and model registry models.
"""

from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from sqlalchemy.sql import func

from app.database.base import Base


class TrainingHistory(Base):
    """Model tracking ML model training runs and their metrics."""

    __tablename__ = "training_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    training_id = Column(String(50), unique=True, nullable=False, index=True)
    model_type = Column(String(50))  # "lead" or "customer"
    model_version = Column(String(20))
    training_datetime = Column(DateTime, server_default=func.now())
    algorithm = Column(String(50), default="XGBoost")
    lead_records_used = Column(Integer, default=0)
    customer_records_used = Column(Integer, default=0)
    dataset_source = Column(String(200))
    accuracy = Column(Float)
    precision_score = Column(Float)  # 'precision' is reserved
    recall = Column(Float)
    f1_score = Column(Float)
    training_duration_seconds = Column(Float)
    started_by = Column(String(100))
    status = Column(String(20), default="pending")  # pending/running/success/failed
    notes = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<TrainingHistory(id={self.id}, training_id='{self.training_id}', model_type='{self.model_type}')>"


class ModelRegistry(Base):
    """Model registry tracking deployed and archived ML models."""

    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(50), unique=True, nullable=False, index=True)
    model_type = Column(String(50))  # "lead" or "customer"
    model_version = Column(String(20), nullable=False)
    training_date = Column(DateTime, server_default=func.now())
    accuracy = Column(Float)
    precision_score = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    algorithm = Column(String(50), default="XGBoost")
    dataset_size = Column(Integer)
    dataset_source = Column(String(200))
    status = Column(String(20), default="active")  # active/archived
    model_path = Column(String(500))

    def __repr__(self) -> str:
        return f"<ModelRegistry(id={self.id}, model_id='{self.model_id}', version='{self.model_version}')>"
