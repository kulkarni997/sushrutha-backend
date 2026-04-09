from fastapi import APIRouter, Depends
from db.supabase_client import supabase
from auth.jwt_handler import require_patient

router = APIRouter()

@router.get("/history/{user_id}")
async def get_history(user_id: str, user: dict = Depends(require_patient)):
    # Get scans
    scans_res = supabase.table("scans")\
        .select("*")\
        .eq("user_id", user["sub"])\
        .order("created_at", desc=True)\
        .execute()
    
    scans = scans_res.data or []
    
    # For each scan, get its result
    for scan in scans:
        result = supabase.table("results")\
            .select("*")\
            .eq("scan_id", scan["id"])\
            .execute()
        scan["results"] = result.data[0] if result.data else None
    
    return scans