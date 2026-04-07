from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from db.supabase_client import supabase
from auth.jwt_handler import require_patient
from datetime import datetime

router = APIRouter()

class ClaimPayload(BaseModel):
    token: str
    email: str
    password: str
    full_name: str

@router.post("/claim")
async def claim_scan(payload: ClaimPayload):
    # Find guest scan by token
    session = supabase.table("guest_scans")\
        .select("*")\
        .eq("claim_token", payload.token)\
        .single()\
        .execute()
    
    if not session.data:
        raise HTTPException(404, "Invalid claim token")
    
    # Check expiry
    expires = session.data["token_expires_at"]
    if expires and datetime.utcnow().isoformat() > expires:
        raise HTTPException(400, "Claim token has expired")
    
    # Check already claimed
    if session.data["claimed_by"]:
        raise HTTPException(400, "This scan has already been claimed")
    
    # Create user account
    import hashlib
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    hashed = pwd_context.hash(payload.password)
    
    user = supabase.table("users").insert({
        "email": payload.email,
        "hashed_password": hashed,
        "role": "patient",
        "full_name": payload.full_name
    }).execute()
    
    new_user_id = user.data[0]["id"]
    
    # Link scan to new user
    supabase.table("guest_scans")\
        .update({"claimed_by": new_user_id})\
        .eq("id", session.data["id"])\
        .execute()
    
    return {"message": "Scan claimed successfully", "user_id": new_user_id}