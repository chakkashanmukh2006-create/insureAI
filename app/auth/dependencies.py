"""
FastAPI authentication dependencies.

Provides OAuth2 scheme and dependency injection for extracting
the current authenticated user from JWT bearer tokens.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from jose import JWTError

from app.auth.security import verify_token
from app.database.session import get_db
from app.models.user import User

# OAuth2 password bearer scheme pointing to the login endpoint
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")


def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    """
    Dependency that extracts and validates the current user from a JWT token.

    Decodes the bearer token, looks up the user in the database,
    and returns the User ORM instance.

    Args:
        token: The JWT bearer token extracted from the Authorization header.
        db: The database session dependency.

    Returns:
        The authenticated User ORM model instance.

    Raises:
        HTTPException: 401 Unauthorized if the token is invalid, expired,
            or the user does not exist / is inactive.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        token_data = verify_token(token)
    except JWTError:
        raise credentials_exception

    if token_data.username is None:
        raise credentials_exception

    user = db.query(User).filter(User.username == token_data.username).first()

    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )

    return user
