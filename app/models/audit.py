"""
Audit log and uploaded file tracking models.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.sql import func

from app.database.base import Base


class AuditLog(Base):
    """Model tracking user actions for audit trail compliance."""

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    log_id = Column(String(50), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)
    resource = Column(String(100))
    details = Column(Text, nullable=True)
    timestamp = Column(DateTime, server_default=func.now())

    def __repr__(self) -> str:
        return f"<AuditLog(id={self.id}, log_id='{self.log_id}', action='{self.action}')>"


class UploadedFile(Base):
    """Model tracking uploaded data files (CSV, XLSX, JSON)."""

    __tablename__ = "uploaded_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(String(50), unique=True, nullable=False, index=True)
    filename = Column(String(500), nullable=False)
    file_type = Column(String(20))  # csv, xlsx, json
    upload_timestamp = Column(DateTime, server_default=func.now())
    uploaded_by = Column(String(100))
    record_count = Column(Integer, default=0)
    target_table = Column(String(50))  # "leads" or "customers"
    status = Column(String(20), default="success")  # success/failed

    def __repr__(self) -> str:
        return f"<UploadedFile(id={self.id}, file_id='{self.file_id}', filename='{self.filename}')>"
