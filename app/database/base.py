"""
SQLAlchemy declarative base.

All ORM models must inherit from this Base class to be discovered
by Alembic and registered with the SQLAlchemy metadata.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""

    pass
