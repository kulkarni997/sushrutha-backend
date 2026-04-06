import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from passlib.context import CryptContext
from db.supabase_client import supabase
from auth.jwt_handler import create_token

router = APIRouter()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

VALID_ROLES = {"patient", "doctor"}


class RegisterRequest(BaseModel):
    full_name: str
    email: str
    password: str
    role: str
    bams_number: str = None


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/register")
def register(req: RegisterRequest):
    if req.role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail="Role must be 'patient' or 'doctor'.")

    if req.role == "doctor" and not req.bams_number:
        raise HTTPException(status_code=400, detail="BAMS number required for doctors.")

    existing = (
        supabase.table("users").select("id").eq("email", req.email).execute()
    )
    if existing.data:
        raise HTTPException(status_code=400, detail="Email already registered.")

    hashed_password = pwd_context.hash(req.password)
    user_id = str(uuid.uuid4())

    supabase.table("users").insert({
        "id": user_id,
        "email": req.email,
        "hashed_password": hashed_password,
        "role": req.role,
        "full_name": req.full_name,
    }).execute()

    if req.role == "doctor":
        supabase.table("doctors").insert({
            "id": user_id,
            "bams_number": req.bams_number,
            "verified": False,
            "subscription_active": False,
        }).execute()

    token = create_token(user_id, req.role, req.full_name)
    return {"token": token, "role": req.role, "user_id": user_id, "full_name": req.full_name}


@router.post("/login")
def login(req: LoginRequest):
    result = (
        supabase.table("users")
        .select("id, email, hashed_password, role, full_name")
        .eq("email", req.email)
        .execute()
    )

    if not result.data:
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    user = result.data[0]

    if not pwd_context.verify(req.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    if user["role"] == "doctor":
        doctor = (
            supabase.table("doctors").select("verified").eq("id", user["id"]).execute()
        )
        if not doctor.data or not doctor.data[0]["verified"]:
            raise HTTPException(
                status_code=403,
                detail="Account pending verification. Contact admin.",
            )

    token = create_token(user["id"], user["role"], user["full_name"])
    return {
        "token": token,
        "role": user["role"],
        "user_id": user["id"],
        "full_name": user["full_name"],
    }
