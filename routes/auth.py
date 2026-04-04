from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from passlib.context import CryptContext
from db.supabase_client import supabase
from auth.jwt_handler import create_token
import uuid

router = APIRouter()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


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
async def register(body: RegisterRequest):
    if body.role not in ("patient", "doctor"):
        raise HTTPException(status_code=400, detail="Role must be 'patient' or 'doctor'")

    if body.role == "doctor" and not body.bams_number:
        raise HTTPException(status_code=400, detail="BAMS number required for doctors")

    existing = supabase.table("users").select("id").eq("email", body.email).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = pwd_context.hash(body.password)
    user_id = str(uuid.uuid4())

    supabase.table("users").insert({
        "id": user_id,
        "email": body.email,
        "hashed_password": hashed_password,
        "role": body.role,
        "full_name": body.full_name,
    }).execute()

    if body.role == "doctor":
        supabase.table("doctors").insert({
            "id": user_id,
            "bams_number": body.bams_number,
            "verified": False,
            "subscription_active": False,
        }).execute()

    token = create_token(user_id, body.role, body.full_name)
    return {"token": token, "role": body.role, "user_id": user_id, "full_name": body.full_name}


@router.post("/login")
async def login(body: LoginRequest):
    result = supabase.table("users").select("*").eq("email", body.email).execute()

    if not result.data:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user = result.data[0]

    if not pwd_context.verify(body.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if user["role"] == "doctor":
        doctor = supabase.table("doctors").select("verified").eq("id", user["id"]).execute()
        if not doctor.data or not doctor.data[0]["verified"]:
            raise HTTPException(status_code=403, detail="Account pending verification. Contact admin.")

    token = create_token(user["id"], user["role"], user["full_name"])
    return {"token": token, "role": user["role"], "user_id": user["id"], "full_name": user["full_name"]}
