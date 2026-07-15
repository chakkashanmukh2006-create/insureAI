from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.models.user import User
from app.schemas.dashboard import DashboardResponse, TrainingDashboardResponse, ModelDashboardResponse
from app.auth.dependencies import get_current_user

router = APIRouter()


@router.get("/dashboard", response_model=DashboardResponse,
            summary="System Dashboard",
            description="Overall system dashboard with stats and recent training.")
def get_dashboard(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get the overall system dashboard.
    
    Returns aggregated statistics including total leads, customers, predictions,
    training sessions, and the latest model versions with their accuracies.
    Also includes the 10 most recent training sessions.
    """
    from app.dashboard.service import DashboardService
    service = DashboardService()
    return service.get_dashboard(db)


@router.get("/dashboard/training", response_model=TrainingDashboardResponse,
            summary="Training Dashboard",
            description="Training history dashboard showing all training sessions.")
def get_training_dashboard(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get the training-focused dashboard.
    
    Returns the complete training history ordered by most recent first,
    along with the total number of training sessions.
    """
    from app.dashboard.service import DashboardService
    service = DashboardService()
    return service.get_training_dashboard(db)


@router.get("/dashboard/model", response_model=ModelDashboardResponse,
            summary="Model Dashboard",
            description="Model registry dashboard showing all model versions.")
def get_model_dashboard(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get the model registry dashboard.
    
    Returns all registered model versions ordered by most recent first,
    along with the total number of models.
    """
    from app.dashboard.service import DashboardService
    service = DashboardService()
    return service.get_model_dashboard(db)
