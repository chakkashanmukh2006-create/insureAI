from typing import List
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.database.session import get_db, SessionLocal
from app.models.user import User
from app.models.training import TrainingHistory, ModelRegistry
from app.schemas.training import TrainingHistoryResponse, TrainingLatestResponse, TrainRequest
from app.auth.dependencies import get_current_user
from app.training.trainer import TrainingService
from app.prediction.lead_predictor import LeadPredictor
from app.prediction.customer_predictor import CustomerPredictor
from app.utils.audit import log_audit
from app.utils.logger import logger
from app.training import job_manager

router = APIRouter()


def train_models_task(job_id: str, db_session: Session, user_id: int, username: str, notes: str):
    try:
        def log_cb(msg: str):
            job_manager.append_log(job_id, msg)
        
        trainer = TrainingService()
        log_cb("Starting AI training pipeline...")
        results = trainer.train_all(db_session, started_by=username, notes=notes, log_callback=log_cb)
        log_audit(db_session, user_id, "train", "models", f"Training completed: {results}")
        
        # After training, generate predictions for all records
        log_cb("Generating predictions for all Lead records (this may take a moment)...")
        lead_predictor = LeadPredictor()
        lead_predictor.predict_all(db_session)
        
        log_cb("Generating predictions for all Customer records (this may take a moment)...")
        customer_predictor = CustomerPredictor()
        customer_predictor.predict_all(db_session)
            
        log_cb("Pipeline finished successfully!")
        job_manager.mark_completed(job_id, results)
    except Exception as e:
        logger.error(f"Background training failed: {e}")
        job_manager.mark_failed(job_id, str(e))
    finally:
        db_session.close()


@router.post("/train",
             summary="Train Models",
             description="Retrain both lead propensity and customer churn models using all available data.")
def train_models(
    background_tasks: BackgroundTasks,
    request: TrainRequest = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Start model training in the background."""
    job_id = job_manager.create_job()
    notes = request.notes if request else None
    
    bg_db = SessionLocal()
    
    background_tasks.add_task(
        train_models_task, 
        job_id, 
        bg_db, 
        current_user.id, 
        current_user.username, 
        notes
    )
    
    return {
        "status": "started",
        "job_id": job_id,
        "message": "Model training started in background."
    }

@router.get("/train/status/{job_id}", summary="Get Training Status")
def get_training_status(job_id: str):
    """Retrieve the status and logs for a background training job."""
    return job_manager.get_job_status(job_id)


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
