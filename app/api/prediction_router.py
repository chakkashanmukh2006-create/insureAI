from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.models.lead import Lead
from app.models.customer import Customer
from app.models.prediction import LeadPrediction, CustomerPrediction
from app.models.user import User
from app.schemas.lead import LeadPredictionResponse
from app.schemas.customer import CustomerPredictionResponse
from app.auth.dependencies import get_current_user

router = APIRouter()


@router.get("/predictions/lead/{lead_id}", response_model=LeadPredictionResponse,
            summary="Get Lead Prediction",
            description="Get the latest prediction for a specific lead with full metadata.")
def get_lead_prediction(
    lead_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get the latest propensity prediction for a specific lead.
    
    Returns the most recent prediction including propensity score, category,
    top contributing reasons, and model metadata.
    
    Args:
        lead_id: The unique identifier for the lead.
    
    Raises:
        HTTPException: 404 if no prediction exists for the given lead.
    """
    # Get latest prediction for this lead
    prediction = db.query(LeadPrediction).filter(
        LeadPrediction.lead_id == lead_id
    ).order_by(LeadPrediction.prediction_timestamp.desc()).first()
    
    if not prediction:
        raise HTTPException(status_code=404, detail=f"No prediction found for lead {lead_id}")
    
    lead = db.query(Lead).filter(Lead.lead_id == lead_id).first()
    
    return LeadPredictionResponse(
        prediction_id=prediction.prediction_id,
        lead_id=prediction.lead_id,
        full_name=lead.full_name if lead else 'Unknown',
        propensity_ratio=prediction.propensity_ratio,
        lead_score=prediction.lead_score,
        category=prediction.category,
        top_reasons=prediction.top_reasons or [],
        email=prediction.email,
        contact_number=prediction.contact_number,
        model_version=prediction.model_version,
        model_accuracy=prediction.model_accuracy,
        algorithm=prediction.algorithm,
        prediction_timestamp=prediction.prediction_timestamp,
        training_timestamp=prediction.training_timestamp
    )


@router.get("/predictions/customer/{customer_id}", response_model=CustomerPredictionResponse,
            summary="Get Customer Prediction",
            description="Get the latest churn prediction for a specific customer.")
def get_customer_prediction(
    customer_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get the latest churn prediction for a specific customer.
    
    Returns the most recent prediction including churn probability, risk category,
    sentiment analysis, top contributing reasons, and model metadata.
    
    Args:
        customer_id: The unique identifier for the customer.
    
    Raises:
        HTTPException: 404 if no prediction exists for the given customer.
    """
    prediction = db.query(CustomerPrediction).filter(
        CustomerPrediction.customer_id == customer_id
    ).order_by(CustomerPrediction.prediction_timestamp.desc()).first()
    
    if not prediction:
        raise HTTPException(status_code=404, detail=f"No prediction found for customer {customer_id}")
    
    customer = db.query(Customer).filter(Customer.customer_id == customer_id).first()
    
    return CustomerPredictionResponse(
        prediction_id=prediction.prediction_id,
        customer_id=prediction.customer_id,
        name=customer.name if customer else 'Unknown',
        churn_ratio=prediction.churn_ratio,
        risk_category=prediction.risk_category,
        sentiment=prediction.sentiment or 'Neutral',
        sentiment_score=prediction.sentiment_score or 0.0,
        confidence_score=prediction.confidence_score or 0.0,
        top_reasons=prediction.top_reasons or [],
        email=prediction.email,
        contact_number=prediction.contact_number,
        model_version=prediction.model_version,
        model_accuracy=prediction.model_accuracy,
        algorithm=prediction.algorithm,
        prediction_timestamp=prediction.prediction_timestamp,
        training_timestamp=prediction.training_timestamp
    )
