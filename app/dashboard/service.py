from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.models.lead import Lead
from app.models.customer import Customer
from app.models.prediction import LeadPrediction, CustomerPrediction
from app.models.training import TrainingHistory, ModelRegistry
from app.schemas.dashboard import (
    DashboardResponse,
    DashboardStats,
    TrainingDashboardResponse,
    ModelDashboardResponse
)


class DashboardService:
    """Provides dashboard data aggregation for the Insurance AI system.
    
    Aggregates data from leads, customers, predictions, training history,
    and model registry to provide comprehensive dashboard views.
    """
    
    def get_dashboard(self, db: Session) -> DashboardResponse:
        """Get the overall system dashboard with stats and recent training.
        
        Computes aggregate statistics across all entities and returns
        the 10 most recent training sessions.
        
        Args:
            db: Database session.
        
        Returns:
            DashboardResponse with stats and recent training history.
        """
        total_leads = db.query(Lead).count()
        total_customers = db.query(Customer).count()
        total_lead_preds = db.query(LeadPrediction).count()
        total_cust_preds = db.query(CustomerPrediction).count()
        total_training = db.query(TrainingHistory).count()
        
        # Latest active models
        latest_lead = db.query(ModelRegistry).filter(
            ModelRegistry.model_type == 'lead', ModelRegistry.status == 'active'
        ).order_by(desc(ModelRegistry.id)).first()
        
        latest_customer = db.query(ModelRegistry).filter(
            ModelRegistry.model_type == 'customer', ModelRegistry.status == 'active'
        ).order_by(desc(ModelRegistry.id)).first()
        
        stats = DashboardStats(
            total_leads=total_leads,
            total_customers=total_customers,
            total_predictions_leads=total_lead_preds,
            total_predictions_customers=total_cust_preds,
            total_training_sessions=total_training,
            latest_lead_model_version=latest_lead.model_version if latest_lead else None,
            latest_customer_model_version=latest_customer.model_version if latest_customer else None,
            latest_lead_accuracy=latest_lead.accuracy if latest_lead else None,
            latest_customer_accuracy=latest_customer.accuracy if latest_customer else None
        )
        
        recent = db.query(TrainingHistory).order_by(
            desc(TrainingHistory.training_datetime)
        ).limit(10).all()
        
        return DashboardResponse(stats=stats, recent_training=recent)
    
    def get_training_dashboard(self, db: Session) -> TrainingDashboardResponse:
        """Get the training history dashboard.
        
        Returns all training sessions ordered by most recent first.
        
        Args:
            db: Database session.
        
        Returns:
            TrainingDashboardResponse with full history and count.
        """
        history = db.query(TrainingHistory).order_by(
            desc(TrainingHistory.training_datetime)
        ).all()
        return TrainingDashboardResponse(
            training_history=history,
            total_sessions=len(history)
        )
    
    def get_model_dashboard(self, db: Session) -> ModelDashboardResponse:
        """Get the model registry dashboard.
        
        Returns all registered models ordered by most recent training date.
        
        Args:
            db: Database session.
        
        Returns:
            ModelDashboardResponse with all models and count.
        """
        models = db.query(ModelRegistry).order_by(
            desc(ModelRegistry.training_date)
        ).all()
        return ModelDashboardResponse(
            models=models,
            total_models=len(models)
        )
