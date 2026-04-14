from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from passlib.context import CryptContext
from db.supabase_client import supabase
from auth.jwt_handler import create_token, get_current_user
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

    plan = 'free'
    token = create_token(user_id, body.role, body.full_name, plan)
    # return {"token": token, "role": body.role, "user_id": user_id, "full_name": body.full_name}
    token = create_token(user_id, body.role, body.full_name, plan)
    return {
        "token": token, 
        "role": body.role, 
        "user_id": user_id, 
        "full_name": body.full_name,
        "plan": plan
    }


@router.post("/login")
async def login(body: LoginRequest):
    result = supabase.table("users").select("*").eq("email", body.email).execute()

    if not result.data:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user = result.data[0]

    if not pwd_context.verify(body.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if user["role"] == "patient":
        plan = user.get("patient_plan", "free")
    else:
        doctor_full = supabase.table("doctors").select("subscription_tier, verified").eq("id", user["id"]).single().execute()
        plan = doctor_full.data.get("subscription_tier", "free") if doctor_full.data else "free"
        if not doctor_full.data.get("verified", False):
            raise HTTPException(status_code=403, detail="Your account is pending verification. Please contact admin.")

    token = create_token(user["id"], user["role"], user["full_name"], plan)
    return {
        "token": token,
        "role": user["role"],
        "user_id": user["id"],
        "full_name": user["full_name"],
        "plan": plan
    }

class SubscribeRequest(BaseModel):
    plan: str


@router.post("/subscribe")
async def subscribe(body: SubscribeRequest,
                    user: dict = Depends(get_current_user)):

    valid_plans = ['free', 'basic', 'pro', 'pro_family']
    if body.plan not in valid_plans:
        raise HTTPException(status_code=400,
                            detail="Invalid plan selected")

    if user['role'] == 'patient':
        supabase.table('users')\
            .update({'patient_plan': body.plan})\
            .eq('id', user['sub'])\
            .execute()
    else:
        raise HTTPException(status_code=400,
                            detail="Use doctor upgrade endpoint")

    return {
        "message": f"Plan upgraded to {body.plan} successfully",
        "plan": body.plan
    }