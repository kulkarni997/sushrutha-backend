import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from jose import jwt, JWTError
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

load_dotenv()

JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    raise EnvironmentError("JWT_SECRET is not set in environment variables.")

ALGORITHM = "HS256"
TOKEN_EXPIRY_DAYS = 7


def create_token(user_id: str, role: str, name: str) -> str:
    payload = {
        "sub": user_id,
        "role": role,
        "name": name,
        "exp": datetime.utcnow() + timedelta(days=TOKEN_EXPIRY_DAYS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)


def verify_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(HTTPBearer()),
) -> dict:
    return verify_token(credentials.credentials)


def require_patient(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "patient":
        raise HTTPException(status_code=403, detail="Access restricted to patients.")
    return user


def require_doctor(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "doctor":
        raise HTTPException(status_code=403, detail="Access restricted to doctors.")
    return user
