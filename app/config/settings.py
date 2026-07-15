"""
Application settings module.

Uses pydantic-settings to load configuration from environment variables
and .env file. Provides a singleton `settings` instance for use throughout
the application.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    DATABASE_URL: str = "sqlite:///./insurance.db"

    # JWT Authentication
    JWT_SECRET_KEY: str = "your-super-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 10080

    # Model Storage
    MODEL_STORAGE_PATH: str = "trained_models"
    DATASET_PATH: str = "datasets"
    UPLOAD_PATH: str = "uploads"

    # Sentiment Analysis
    USE_HUGGINGFACE: bool = True

    # App
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = False


settings = Settings()
