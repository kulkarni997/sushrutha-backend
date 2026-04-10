from fastapi import APIRouter, Depends
from db.supabase_client import supabase
from auth.jwt_handler import require_patient

router = APIRouter()

@router.get("/history")
async def get_history(user: dict = Depends(require_patient)):
    scans = supabase.table("scans")\
        .select("*, results(*)")\
        .eq("user_id", user["sub"])\
        .order("created_at", desc=True)\
        .execute()
    return scans.data