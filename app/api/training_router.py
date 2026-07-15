from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.models.user import User
from app.models.training import TrainingHistory, ModelRegistry
from app.schemas.training import TrainingHistoryResponse, TrainingLatestResponse, TrainRequest
from app.auth.dependencies import get_current_user
from app.training.trainer import TrainingService
from app.prediction.lead_predictor import LeadPredictor
from app.prediction.customer_predictor import CustomerPredictor
from app.utils.audit import log_audit
from app.utils.logger import logger

router = APIRouter()


@router.post("/train",
             summary="Train Models",
             description="Retrain both lead propensity and customer churn models using all available data.")
def train_models(
    request: TrainRequest = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Train/retrain both ML models.
    
    This endpoint:
    1. Loads ALL data from leads and customers tables
    2. Trains new XGBoost models for both lead propensity and customer churn
    3. Saves models with incremented version numbers (v1, v2, v3...)
    4. Updates model registry and training history
    5. Future predictions automatically use the latest model
    """
    trainer = TrainingService()
    notes = request.notes if request else None
    
    try:
        results = trainer.train_all(db, started_by=current_user.username, notes=notes)
        log_audit(db, current_user.id, "train", "models", f"Training completed: {results}")
        
        # After training, generate predictions for all records
        try:
            lead_predictor = LeadPredictor()
            lead_predictor.predict_all(db)
            
            customer_predictor = CustomerPredictor()
            customer_predictor.predict_all(db)
        except Exception as e:
            logger.warning(f"Post-training prediction generation failed: {e}")
        
        return {
            "status": "success",
            "message": "Models trained successfully",
            "results": results
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/training/history", response_model=List[TrainingHistoryResponse],
            summary="Training History",
            description="Get complete training history in chronological order.")
def get_training_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all training sessions ordered by most recent first."""
    history = db.query(TrainingHistory).order_by(TrainingHistory.training_datetime.desc()).all()
    return history


@router.get("/training/latest", response_model=TrainingLatestResponse,
            summary="Latest Training",
            description="Get the latest training session for each model type.")
def get_latest_training(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get the most recent successful training session for each model type (lead and customer)."""
    lead_latest = db.query(TrainingHistory).filter(
        TrainingHistory.model_type == 'lead',
        TrainingHistory.status == 'success'
    ).order_by(TrainingHistory.training_datetime.desc()).first()
    
    customer_latest = db.query(TrainingHistory).filter(
        TrainingHistory.model_type == 'customer',
        TrainingHistory.status == 'success'
    ).order_by(TrainingHistory.training_datetime.desc()).first()
    
    return TrainingLatestResponse(
        lead_model=lead_latest,
        customer_model=customer_latest
    )


@router.get("/model/latest",
            summary="Latest Model Info",
            description="Get the latest registered model information for both lead and customer models.")
def get_latest_model(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get the latest model registry entries for both model types."""
    from app.training.model_manager import ModelManager
    lead_reg = ModelManager.get_latest_registry(db, 'lead')
    customer_reg = ModelManager.get_latest_registry(db, 'customer')
    return {
        "lead_model": lead_reg,
        "customer_model": customer_reg
    }
