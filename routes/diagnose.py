from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from db.supabase_client import supabase
from auth.jwt_handler import require_patient

router = APIRouter()

class DiagnosePayload(BaseModel):
    scan_id: str
    symptoms_text: str
    image_data: Optional[str] = None
    audio_data: Optional[str] = None
    pulse_used: Optional[bool] = False

def assess_severity(vata: int, pitta: int, kapha: int) -> str:
    dominant = max(vata, pitta, kapha)
    scores = sorted([vata, pitta, kapha], reverse=True)
    if dominant >= 70:
        return "severe"
    elif dominant >= 55 and scores[1] >= 30:
        return "moderate"
    else:
        return "mild"

@router.post("/diagnose")
async def diagnose(payload: DiagnosePayload, user: dict = Depends(require_patient)):
    # Mock ML output for now — real models wired on Day 10
    vata, pitta, kapha = 58, 28, 14
    severity = assess_severity(vata, pitta, kapha)

    # Save result to Supabase
    result = supabase.table("results").insert({
        "scan_id": payload.scan_id,
        "vata_pct": vata,
        "pitta_pct": pitta,
        "kapha_pct": kapha,
        "severity": severity,
        "pulse_used": payload.pulse_used,
        "recipe_text": "Ashwagandha 500mg twice daily. Triphala at night.",
        "finalised": False
    }).execute()

    return {
        "vata": vata,
        "pitta": pitta,
        "kapha": kapha,
        "severity": severity,
        "pulse_used": payload.pulse_used,
        "result_id": result.data[0]["id"]
    }