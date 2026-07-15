"""
Insurance AI Intelligence System — Dataset Loader
===================================================
Loads generated CSV datasets (leads & customers) into PostgreSQL
using SQLAlchemy ORM models.

Usage:
    python datasets/load_datasets.py
"""

import pandas as pd
import sys
import os

# Add parent directory to path so app modules are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.session import SessionLocal, engine
from app.models.lead import Lead
from app.models.customer import Customer
from app.database.base import Base
from app.utils.logger import logger


def load_leads(csv_path: str = "datasets/leads_dataset.csv") -> int:
    """
    Load leads from CSV into PostgreSQL.

    Parameters
    ----------
    csv_path : str
        Path to the leads CSV file.

    Returns
    -------
    int
        Number of new leads inserted.
    """
    df = pd.read_csv(csv_path)
    db = SessionLocal()

    try:
        count = 0
        for _, row in df.iterrows():
            # Skip if lead already exists
            existing = db.query(Lead).filter(Lead.lead_id == row["lead_id"]).first()
            if existing:
                continue

            lead = Lead(
                lead_id=row["lead_id"],
                full_name=row["full_name"],
                age=int(row["age"]) if pd.notna(row.get("age")) else None,
                gender=row.get("gender"),
                occupation=row.get("occupation"),
                annual_income=(
                    float(row["annual_income"])
                    if pd.notna(row.get("annual_income"))
                    else None
                ),
                city=row.get("city"),
                existing_policy=bool(row.get("existing_policy", False)),
                product_interested=row.get("product_interested"),
                website_visits=int(row.get("website_visits", 0)),
                email_opens=int(row.get("email_opens", 0)),
                calls_answered=int(row.get("calls_answered", 0)),
                form_submitted=bool(row.get("form_submitted", False)),
                last_interaction_days=int(row.get("last_interaction_days", 0)),
                lead_source=row.get("lead_source"),
                email=row.get("email"),
                contact_number=str(row.get("contact_number", "")),
                conversion_target=int(row.get("conversion_target", 0)),
            )
            db.add(lead)
            count += 1

            # Batch commit every 100 records for efficiency
            if count % 100 == 0:
                db.commit()
                logger.info(f"Loaded {count} leads...")

        db.commit()
        logger.info(f"Successfully loaded {count} leads into database")
        return count

    except Exception as e:
        db.rollback()
        logger.error(f"Error loading leads: {e}")
        raise
    finally:
        db.close()


def load_customers(csv_path: str = "datasets/customers_dataset.csv") -> int:
    """
    Load customers from CSV into PostgreSQL.

    Parameters
    ----------
    csv_path : str
        Path to the customers CSV file.

    Returns
    -------
    int
        Number of new customers inserted.
    """
    df = pd.read_csv(csv_path)
    db = SessionLocal()

    try:
        count = 0
        for _, row in df.iterrows():
            # Skip if customer already exists
            existing = (
                db.query(Customer)
                .filter(Customer.customer_id == row["customer_id"])
                .first()
            )
            if existing:
                continue

            customer = Customer(
                customer_id=row["customer_id"],
                name=row["name"],
                age=int(row["age"]) if pd.notna(row.get("age")) else None,
                policy_type=row.get("policy_type"),
                premium_amount=(
                    float(row["premium_amount"])
                    if pd.notna(row.get("premium_amount"))
                    else None
                ),
                renewal_history=int(row.get("renewal_history", 0)),
                claim_history=int(row.get("claim_history", 0)),
                complaint_count=int(row.get("complaint_count", 0)),
                support_tickets=int(row.get("support_tickets", 0)),
                feedback=row.get("feedback", ""),
                email=row.get("email"),
                contact_number=str(row.get("contact_number", "")),
                churn_target=int(row.get("churn_target", 0)),
            )
            db.add(customer)
            count += 1

            # Batch commit every 100 records for efficiency
            if count % 100 == 0:
                db.commit()
                logger.info(f"Loaded {count} customers...")

        db.commit()
        logger.info(f"Successfully loaded {count} customers into database")
        return count

    except Exception as e:
        db.rollback()
        logger.error(f"Error loading customers: {e}")
        raise
    finally:
        db.close()


def main():
    """Main function to load all datasets."""

    # Create tables if they don't exist
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)

    # Generate datasets if CSVs are missing
    if not os.path.exists("datasets/leads_dataset.csv") or not os.path.exists(
        "datasets/customers_dataset.csv"
    ):
        logger.info("CSV files not found — generating datasets...")
        from datasets.generate_datasets import generate_leads, generate_customers

        generate_leads()
        generate_customers()

    # Load data
    logger.info("Loading leads...")
    lead_count = load_leads()

    logger.info("Loading customers...")
    customer_count = load_customers()

    logger.info(
        f"✅ Data loading complete. {lead_count} leads, {customer_count} customers."
    )


if __name__ == "__main__":
    main()
