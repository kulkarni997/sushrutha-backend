from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from db.supabase_client import supabase
from auth.jwt_handler import require_doctor
import uuid
from datetime import datetime, timedelta

router = APIRouter()

class WalkinCreate(BaseModel):
    patient_name: str

class ResultsOverride(BaseModel):
    doctor_notes: Optional[str] = None
    override_dosha: Optional[str] = None

@router.get("/patients")
async def get_patients(user: dict = Depends(require_doctor)):
    scans = supabase.table("scans")\
        .select("*")\
        .eq("shared_with", user["sub"])\
        .eq("shared", True)\
        .order("created_at", desc=True)\
        .execute()
    
    scans_data = scans.data or []
    
    for scan in scans_data:
        # Get result
        result = supabase.table("results").select("*").eq("scan_id", scan["id"]).execute()
        scan["results"] = result.data[0] if result.data else None
        # Get user name
        user_data = supabase.table("users").select("full_name").eq("id", scan["user_id"]).execute()
        scan["full_name"] = user_data.data[0]["full_name"] if user_data.data else None
    
    return scans_data

@router.get("/patient/{scan_id}")
async def get_patient(scan_id: str, user: dict = Depends(require_doctor)):
    scan = supabase.table("scans")\
        .select("*")\
        .eq("id", scan_id)\
        .eq("shared_with", user["sub"])\
        .execute()
    
    if not scan.data:
        raise HTTPException(404, "Scan not found or not shared with you")
    
    result = supabase.table("results").select("*").eq("scan_id", scan_id).execute()
    scan.data[0]["results"] = result.data[0] if result.data else None
    return scan.data[0]

@router.get("/walkin/{session_id}")
async def get_walkin(session_id: str, user: dict = Depends(require_doctor)):
    session = supabase.table("guest_scans")\
        .select("*")\
        .eq("id", session_id)\
        .eq("doctor_id", user["sub"])\
        .execute()
    
    if not session.data:
        raise HTTPException(404, "Session not found")
    
    result = supabase.table("results").select("*").eq("scan_id", session_id).execute()
    session.data[0]["results"] = result.data[0] if result.data else None
    return session.data[0]

@router.get("/walkin/{session_id}")
async def get_walkin(session_id: str, user: dict = Depends(require_doctor)):
    session = supabase.table("guest_scans")\
        .select("*")\
        .eq("id", session_id)\
        .eq("doctor_id", user["sub"])\
        .single()\
        .execute()
    if not session.data:
        raise HTTPException(404, "Session not found")
    
    result = supabase.table("results").select("*").eq("scan_id", session_id).execute()
    session.data["results"] = result.data[0] if result.data else None
    return session.data

@router.patch("/results/{result_id}")
async def override_results(result_id: str, payload: ResultsOverride, user: dict = Depends(require_doctor)):
    update = supabase.table("results")\
        .update({
            "doctor_notes": payload.doctor_notes,
            "override_dosha": payload.override_dosha
        })\
        .eq("id", result_id)\
        .execute()
    return update.data[0]

@router.post("/finalise/{result_id}")
async def finalise_report(result_id: str, user: dict = Depends(require_doctor)):
    result = supabase.table("results")\
        .update({"finalised": True})\
        .eq("id", result_id)\
        .execute()
    scan_id = result.data[0]["scan_id"]
    scan = supabase.table("scans")\
        .select("user_id")\
        .eq("id", scan_id)\
        .single()\
        .execute()
    if scan.data:
        supabase.table("notifications").insert({
            "user_id": scan.data["user_id"],
            "type": "report_finalised",
            "reference_id": scan_id,
            "seen": False
        }).execute()
    return result.data[0]