from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.models.user import User
from app.schemas.user import UserCreate, UserLogin, UserResponse, TokenResponse
from app.auth.security import get_password_hash, verify_password, create_access_token, create_refresh_token, verify_token
from app.auth.dependencies import get_current_user, oauth2_scheme
from app.utils.audit import log_audit
from app.utils.logger import logger

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=201,
             summary="Register a new user",
             description="Create a new user account with username, email, and password.")
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user account.
    
    Validates that both username and email are unique before creating the account.
    Passwords are hashed before storage.
    """
    # Check if username exists
    existing = db.query(User).filter(User.username == user_data.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Check if email exists  
    existing_email = db.query(User).filter(User.email == user_data.email).first()
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    log_audit(db, user.id, "register", "users", f"User {user.username} registered")
    logger.info(f"New user registered: {user.username}")
    return user


from fastapi.security import OAuth2PasswordRequestForm

@router.post("/login", response_model=TokenResponse,
             summary="Login and get JWT tokens",
             description="Authenticate with username and password to receive access and refresh tokens.")
def login(user_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Authenticate a user and return JWT tokens.
    
    Returns both an access token (short-lived) and a refresh token (long-lived).
    """
    user = db.query(User).filter(User.username == user_data.username).first()
    if not user or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    access_token = create_access_token(data={"sub": user.username, "user_id": user.id})
    refresh_token = create_refresh_token(data={"sub": user.username, "user_id": user.id})
    
    log_audit(db, user.id, "login", "users", f"User {user.username} logged in")
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh-token", response_model=TokenResponse,
             summary="Refresh JWT tokens",
             description="Use a valid refresh token to get new access and refresh tokens.")
def refresh_token(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Refresh expired JWT tokens.
    
    Accepts a valid refresh token and returns a new pair of access and refresh tokens.
    """
    token_data = verify_token(token)
    if token_data is None or token_data.username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    user = db.query(User).filter(User.username == token_data.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    new_access = create_access_token(data={"sub": user.username, "user_id": user.id})
    new_refresh = create_refresh_token(data={"sub": user.username, "user_id": user.id})
    
    return TokenResponse(access_token=new_access, refresh_token=new_refresh)
