from typing import List
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Query
from fastapi.responses import Response
import csv
import io
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.models.lead import Lead
from app.models.user import User
from app.schemas.lead import LeadResponse, Top20LeadResponse
from app.schemas.pagination import PaginatedResponse
from app.schemas.upload import UploadResponse
from app.auth.dependencies import get_current_user
from app.prediction.lead_predictor import LeadPredictor
from app.uploads.upload_service import UploadService
from app.utils.audit import log_audit
from app.utils.logger import logger

router = APIRouter()


@router.get("/leads/export", summary="Export Leads to CSV")
def export_leads(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Export all leads to a CSV file."""
    leads = db.query(Lead).all()
    
    if not leads:
        raise HTTPException(status_code=404, detail="No leads found")
        
    output = io.StringIO()
    # Get columns from the first lead
    columns = [column.name for column in Lead.__table__.columns if column.name != 'id']
    
    writer = csv.writer(output)
    writer.writerow(columns)
    
    for lead in leads:
        writer.writerow([getattr(lead, col) for col in columns])
        
    csv_content = output.getvalue()
    
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=leads_export.csv"
        }
    )


@router.get("/leads", response_model=List[LeadResponse],
            summary="List All Leads",
            description="Retrieve all leads with pagination support using skip and limit query parameters.")
def get_leads(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all leads with pagination.
    
    Args:
        skip: Number of records to skip (default: 0)
        limit: Maximum number of records to return (default: 100)
    """
    leads = db.query(Lead).offset(skip).limit(limit).all()
    return leads


@router.get("/leads/top20", response_model=List[Top20LeadResponse],
            summary="Get Top 20 Leads",
            description="Returns the 20 leads with highest propensity to convert.")
def get_top20_leads(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get the top 20 leads by propensity score.
    
    If no predictions exist yet, the system will automatically generate
    predictions for all leads before returning the top 20.
    """
    from app.training.model_manager import ModelManager
    model, registry = ModelManager.load_latest_model(db, 'lead')
    if model is None or registry is None:
        raise HTTPException(status_code=400, detail="No trained lead model found. Please train a model first.")
        
    predictor = LeadPredictor()
    return predictor.get_top20(db)


@router.get("/leads/predicted/all", response_model=PaginatedResponse[Top20LeadResponse],
            summary="Get All Predicted Leads",
            description="Returns all leads with their latest prediction scores.")
def get_all_predicted_leads(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(100, ge=1, le=1000, description="Items per page"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all leads and their latest predictions.
    
    If no predictions exist yet, the system will automatically generate
    predictions for all leads before returning the results.
    """
    from app.training.model_manager import ModelManager
    model, registry = ModelManager.load_latest_model(db, 'lead')
    if model is None or registry is None:
        raise HTTPException(status_code=400, detail="No trained lead model found. Please train a model first.")
        
    predictor = LeadPredictor()
    return predictor.get_all_predicted(db, page=page, limit=limit)


@router.get("/leads/{lead_id}", response_model=LeadResponse,
            summary="Get Lead by ID",
            description="Retrieve a single lead by its unique lead ID.")
def get_lead(
    lead_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a single lead by ID.
    
    Args:
        lead_id: The unique identifier for the lead.
    
    Raises:
        HTTPException: 404 if the lead is not found.
    """
    lead = db.query(Lead).filter(Lead.lead_id == lead_id).first()
    if not lead:
        raise HTTPException(status_code=404, detail=f"Lead {lead_id} not found")
    return lead


@router.post("/upload/leads", response_model=UploadResponse,
             summary="Upload Lead Dataset",
             description="Upload CSV, Excel, or JSON file containing lead data.")
async def upload_leads(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Upload a lead dataset file.
    
    Accepts CSV, Excel (.xlsx, .xls), or JSON files containing lead data.
    The data is validated and inserted into the database.
    """
    service = UploadService()
    result = await service.process_upload(file, "leads", current_user.username, db)
    log_audit(db, current_user.id, "upload", "leads", f"Uploaded {file.filename}")
    return result
