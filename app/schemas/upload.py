"""
Upload schemas for file upload responses.

Provides Pydantic v2 models for file upload result serialization.
"""

from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class UploadResponse(BaseModel):
    """Schema for the response returned after a successful file upload."""

    file_id: str
    filename: str
    file_type: str
    record_count: int
    target_table: str
    status: str
    upload_timestamp: datetime
    message: str
