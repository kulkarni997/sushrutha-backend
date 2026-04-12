from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from db.supabase_client import supabase
from auth.jwt_handler import require_patient
from pydantic import BaseModel

class SharePayload(BaseModel):
    doctor_id: str

router = APIRouter()

class ScanCreate(BaseModel):
    symptoms_text: str
    shared: Optional[bool] = False

@router.patch("/scans/{scan_id}/share")
async def share_scan(scan_id: str, payload: SharePayload, user: dict = Depends(require_patient)):
    result = supabase.table("scans")\
        .update({"shared": True, "shared_with": payload.doctor_id})\
        .eq("id", scan_id)\
        .eq("user_id", user["sub"])\
        .execute()
    return result.data[0]

@router.post("/scans")
async def create_scan(payload: ScanCreate, user: dict = Depends(require_patient)):
    result = supabase.table("scans").insert({
        "user_id": user["sub"],
        "symptoms_text": payload.symptoms_text,
        "shared": payload.shared,
    }).execute()

    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create scan")

    return result.data[0]

@router.get("/scans/{scan_id}")
async def get_scan(scan_id: str, user: dict = Depends(require_patient)):
    scan = supabase.table("scans")\
        .select("*")\
        .eq("id", scan_id)\
        .eq("user_id", user["sub"])\
        .single()\
        .execute()

    if not scan.data:
        raise HTTPException(status_code=404, detail="Scan not found")

    # Fetch results separately
    results = supabase.table("results")\
        .select("*")\
        .eq("scan_id", scan_id)\
        .execute()

    return {**scan.data, "results": results.data or []}