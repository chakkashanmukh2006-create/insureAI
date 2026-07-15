from typing import List
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Query
from fastapi.responses import Response
import csv
import io
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.models.customer import Customer
from app.models.user import User
from app.schemas.customer import CustomerResponse, HighRiskCustomerResponse
from app.schemas.pagination import PaginatedResponse
from app.schemas.upload import UploadResponse
from app.auth.dependencies import get_current_user
from app.prediction.customer_predictor import CustomerPredictor
from app.uploads.upload_service import UploadService
from app.utils.audit import log_audit
from app.utils.logger import logger

router = APIRouter()

@router.get("/customers/export", summary="Export Customers to CSV")
def export_customers(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Export all customers to a CSV file."""
    customers = db.query(Customer).all()
    
    if not customers:
        raise HTTPException(status_code=404, detail="No customers found")
        
    output = io.StringIO()
    # Get columns from the first customer
    columns = [column.name for column in Customer.__table__.columns if column.name != 'id']
    
    writer = csv.writer(output)
    writer.writerow(columns)
    
    for customer in customers:
        writer.writerow([getattr(customer, col) for col in columns])
        
    csv_content = output.getvalue()
    
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=customers_export.csv"
        }
    )


@router.get("/customers", response_model=List[CustomerResponse],
            summary="List All Customers",
            description="Retrieve all customers with pagination support using skip and limit query parameters.")
def get_customers(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all customers with pagination.
    
    Args:
        skip: Number of records to skip (default: 0)
        limit: Maximum number of records to return (default: 100)
    """
    customers = db.query(Customer).offset(skip).limit(limit).all()
    return customers


@router.get("/customers/high-risk", response_model=List[HighRiskCustomerResponse],
            summary="Get High Risk Customers",
            description="Returns customers with the highest churn risk scores.")
def get_high_risk_customers(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get high-risk customers most likely to churn.
    
    If no predictions exist yet, the system will automatically generate
    predictions for all customers before returning the high-risk ones.
    """
    from app.training.model_manager import ModelManager
    model, registry = ModelManager.load_latest_model(db, 'customer')
    if model is None or registry is None:
        raise HTTPException(status_code=400, detail="No trained customer model found. Please train a model first.")
        
    predictor = CustomerPredictor()
    return predictor.get_high_risk(db)


@router.get("/customers/predicted/all", response_model=PaginatedResponse[HighRiskCustomerResponse],
            summary="Get All Predicted Customers",
            description="Returns all customers with their latest prediction scores.")
def get_all_predicted_customers(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(100, ge=1, le=1000, description="Items per page"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all customers and their latest predictions.
    
    If no predictions exist yet, the system will automatically generate
    predictions for all customers before returning the results.
    """
    from app.training.model_manager import ModelManager
    model, registry = ModelManager.load_latest_model(db, 'customer')
    if model is None or registry is None:
        raise HTTPException(status_code=400, detail="No trained customer model found. Please train a model first.")
        
    predictor = CustomerPredictor()
    return predictor.get_all_predicted(db, page=page, limit=limit)


@router.get("/customers/{customer_id}", response_model=CustomerResponse,
            summary="Get Customer by ID",
            description="Retrieve a single customer by their unique customer ID.")
def get_customer(
    customer_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a single customer by ID.
    
    Args:
        customer_id: The unique identifier for the customer.
    
    Raises:
        HTTPException: 404 if the customer is not found.
    """
    customer = db.query(Customer).filter(Customer.customer_id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    return customer


@router.post("/upload/customers", response_model=UploadResponse,
             summary="Upload Customer Dataset",
             description="Upload CSV, Excel, or JSON file containing customer data.")
async def upload_customers(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Upload a customer dataset file.
    
    Accepts CSV, Excel (.xlsx, .xls), or JSON files containing customer data.
    The data is validated and inserted into the database.
    """
    service = UploadService()
    result = await service.process_upload(file, "customers", current_user.username, db)
    log_audit(db, current_user.id, "upload", "customers", f"Uploaded {file.filename}")
    return result
