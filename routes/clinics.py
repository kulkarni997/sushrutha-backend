from fastapi import APIRouter, Depends
from db.supabase_client import supabase
from auth.jwt_handler import require_patient

router = APIRouter()

@router.get("/clinics")
async def get_clinics(user: dict = Depends(require_patient)):
    clinics = supabase.table("doctors")\
        .select("id, clinic_name, lat, lng, subscription_tier, map_priority")\
        .eq("verified", True)\
        .order("map_priority", desc=True)\
        .execute()
    return clinics.data