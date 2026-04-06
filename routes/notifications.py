from fastapi import APIRouter, Depends
from db.supabase_client import supabase
from auth.jwt_handler import require_patient, require_doctor

router = APIRouter()

@router.get("/notifications")
async def get_notifications(user: dict = Depends(require_patient)):
    notifs = supabase.table("notifications")\
        .select("*")\
        .eq("user_id", user["sub"])\
        .order("created_at", desc=True)\
        .limit(20)\
        .execute()
    return notifs.data

@router.patch("/notifications/{notif_id}/seen")
async def mark_seen(notif_id: str, user: dict = Depends(require_patient)):
    supabase.table("notifications")\
        .update({"seen": True})\
        .eq("id", notif_id)\
        .eq("user_id", user["sub"])\
        .execute()
    return {"ok": True}