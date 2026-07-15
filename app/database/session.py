"""
Database session management.

Provides SQLAlchemy engine, sessionmaker, and a FastAPI dependency
generator `get_db()` for injecting database sessions into route handlers.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config.settings import settings

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """
    FastAPI dependency that provides a SQLAlchemy database session.

    Yields a session and ensures it is closed after the request completes.

    Yields:
        Session: A SQLAlchemy database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
