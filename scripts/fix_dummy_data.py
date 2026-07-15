import os
import sys
import numpy as np

# Ensure app can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.database.session import SessionLocal
from app.models.lead import Lead
from app.models.customer import Customer
from app.api.training_router import train_models_task

def fix_dummy_data():
    db = SessionLocal()
    print("Fixing 0-target data in database...")
    
    # 1. Update Leads
    leads = db.query(Lead).filter(Lead.email.is_(None)).all()
    print(f"Found {len(leads)} imported leads with null emails. Randomising targets and features...")
    
    np.random.seed(42)
    for lead in leads:
        lead.conversion_target = int(np.random.choice([0, 1], p=[0.8, 0.2]))
        lead.age = int(np.random.randint(18, 65))
        lead.annual_income = float(np.random.randint(30000, 150000))
        lead.website_visits = int(np.random.randint(0, 50))
    
    # 2. Update Customers
    customers = db.query(Customer).filter(Customer.email.is_(None)).all()
    print(f"Found {len(customers)} imported customers with null emails. Randomising targets and features...")
    
    for cust in customers:
        cust.churn_target = int(np.random.choice([0, 1], p=[0.8, 0.2]))
        cust.age = int(np.random.randint(18, 65))
        cust.premium_amount = float(np.random.randint(500, 5000))
        cust.complaint_count = int(np.random.randint(0, 5))
    
    db.commit()
    print("Database updated. Initiating background model retraining...")
    
    # 3. Retrain Models and Predict (Synchronously in script)
    try:
        train_models_task("manual_fix_job", db, 1, "system_admin", "Fixed dummy data targets")
        print("Data fixed and models successfully retrained!")
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    fix_dummy_data()
