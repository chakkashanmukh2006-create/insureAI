"""
User schemas for authentication and user management.

Provides Pydantic v2 models for user creation, login, response serialization,
and JWT token handling.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class UserCreate(BaseModel):
    """Schema for creating a new user account."""

    username: str = Field(..., min_length=3, max_length=100)
    email: str = Field(..., max_length=255)  # Use str not EmailStr to avoid extra dep
    password: str = Field(..., min_length=6)


class UserLogin(BaseModel):
    """Schema for user login credentials."""

    username: str
    password: str


class UserResponse(BaseModel):
    """Schema for user data returned in API responses."""

    id: int
    username: str
    email: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    """Schema for JWT token pair returned after authentication."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Schema for decoded JWT token payload data."""

    username: Optional[str] = None
    user_id: Optional[int] = None
