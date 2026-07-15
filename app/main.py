"""
Main FastAPI application entry point.

Insurance AI Intelligence System - Lead Propensity & Customer Churn Analysis.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import settings
from app.database.base import Base
from app.database.session import engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    # Startup
    logger.info("🚀 Starting Insurance AI Intelligence System...")

    # Create all database tables
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables created/verified.")

    # Create required directories
    for directory in [
        settings.MODEL_STORAGE_PATH,
        settings.UPLOAD_PATH,
        settings.DATASET_PATH,
        "training_history",
    ]:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"📁 Directory ensured: {directory}")

    logger.info("✅ Insurance AI Intelligence System is ready!")
    logger.info(f"📍 Running on {settings.APP_HOST}:{settings.APP_PORT}")

    yield

    # Shutdown
    logger.info("🛑 Shutting down Insurance AI Intelligence System...")


# Create FastAPI application
app = FastAPI(
    title="Insurance AI Intelligence System",
    description=(
        "AI-powered Insurance Lead Propensity & Customer Churn Intelligence System. "
        "This platform provides ML-driven insights for insurance lead conversion prediction, "
        "customer churn risk assessment, sentiment analysis, and actionable business intelligence. "
        "Features include XGBoost-based propensity scoring, SHAP explainability, "
        "NLP-powered sentiment analysis, and comprehensive dashboard analytics."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with graceful handling for missing modules
try:
    from app.api.auth_router import router as auth_router

    app.include_router(auth_router, prefix="", tags=["Authentication"])
    logger.info("✅ Auth router loaded.")
except ImportError:
    logger.warning("⚠️ Auth router not available yet.")

try:
    from app.api.lead_router import router as lead_router

    app.include_router(lead_router, prefix="", tags=["Leads"])
    logger.info("✅ Lead router loaded.")
except ImportError:
    logger.warning("⚠️ Lead router not available yet.")

try:
    from app.api.customer_router import router as customer_router

    app.include_router(customer_router, prefix="", tags=["Customers"])
    logger.info("✅ Customer router loaded.")
except ImportError:
    logger.warning("⚠️ Customer router not available yet.")

try:
    from app.api.training_router import router as training_router

    app.include_router(training_router, prefix="", tags=["Training"])
    logger.info("✅ Training router loaded.")
except ImportError:
    logger.warning("⚠️ Training router not available yet.")

try:
    from app.api.prediction_router import router as prediction_router

    app.include_router(prediction_router, prefix="", tags=["Predictions"])
    logger.info("✅ Prediction router loaded.")
except ImportError:
    logger.warning("⚠️ Prediction router not available yet.")

try:
    from app.api.dashboard_router import router as dashboard_router

    app.include_router(dashboard_router, prefix="", tags=["Dashboard"])
    logger.info("✅ Dashboard router loaded.")
except ImportError:
    logger.warning("⚠️ Dashboard router not available yet.")

from fastapi.staticfiles import StaticFiles
import os

# Mount frontend UI
if os.path.exists("frontend"):
    app.mount("/ui", StaticFiles(directory="frontend", html=True), name="frontend")
    logger.info("✅ UI frontend mounted at /ui")

from fastapi.responses import RedirectResponse

@app.get("/", tags=["Root"])
async def root():
    """Redirect root to the UI dashboard."""
    return RedirectResponse(url="/ui/")


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring and load balancer probes."""
    return {
        "status": "healthy",
        "service": "Insurance AI Intelligence System",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }
