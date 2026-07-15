"""
Audit logging utility for tracking user actions.

Provides a helper function to log audit events to the database
for compliance and traceability purposes.
"""

import uuid
from sqlalchemy.orm import Session
from app.models.audit import AuditLog


def log_audit(
    db: Session,
    user_id: int | None,
    action: str,
    resource: str,
    details: str = None,
) -> None:
    """
    Log an audit event to the database.

    Creates an AuditLog record capturing who performed what action
    on which resource, with optional details.

    Args:
        db: The active database session.
        user_id: The ID of the user performing the action, or None for system actions.
        action: The action performed (e.g., 'CREATE', 'DELETE', 'TRAIN').
        resource: The resource affected (e.g., 'lead', 'customer', 'model').
        details: Optional additional details about the action.
    """
    audit = AuditLog(
        log_id=str(uuid.uuid4()),
        user_id=user_id,
        action=action,
        resource=resource,
        details=details,
    )
    db.add(audit)
    db.commit()
