from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from db.supabase_client import supabase
from auth.jwt_handler import require_patient, require_doctor
from datetime import datetime
import asyncio

# Reuse the real ML pipeline helpers from the patient /diagnose route
from routes.diagnose import (
    DEMO_MODE,
    DEMO_RESPONSES,
    assess_severity,
    get_dominant_dosha,
    get_pulse_averages,
    run_vision,
    run_voice,
    run_recipe,
)

router = APIRouter()


# ────────────────────────────────────────────────────────────────────────────
# PATIENT-SIDE: Claim a walk-in scan (existing logic, unchanged)
# ────────────────────────────────────────────────────────────────────────────

class ClaimPayload(BaseModel):
    token: str
    email: str
    password: str
    full_name: str

@router.post("/claim")
async def claim_scan(payload: ClaimPayload):
    session = supabase.table("guest_scans")\
        .select("*")\
        .eq("claim_token", payload.token)\
        .single()\
        .execute()

    if not session.data:
        raise HTTPException(404, "Invalid claim token")

    expires = session.data["token_expires_at"]
    if expires and datetime.utcnow().isoformat() > expires:
        raise HTTPException(400, "Claim token has expired")

    if session.data["claimed_by"]:
        raise HTTPException(400, "This scan has already been claimed")

    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    hashed = pwd_context.hash(payload.password)

    user = supabase.table("users").insert({
        "email": payload.email,
        "hashed_password": hashed,
        "role": "patient",
        "full_name": payload.full_name
    }).execute()

    new_user_id = user.data[0]["id"]

    supabase.table("guest_scans")\
        .update({"claimed_by": new_user_id})\
        .eq("id", session.data["id"])\
        .execute()

    return {"message": "Scan claimed successfully", "user_id": new_user_id}


# ────────────────────────────────────────────────────────────────────────────
# DOCTOR-SIDE: Run a walk-in scan diagnosis
# ────────────────────────────────────────────────────────────────────────────

class GuestDiagnosePayload(BaseModel):
    session_id: str                        # guest_scans.id
    symptoms_text: str
    image_data: Optional[str] = None       # base64 tongue image
    audio_data: Optional[str] = None       # base64 audio
    pulse_used: Optional[bool] = False


@router.post("/diagnose")
async def guest_diagnose(
    payload: GuestDiagnosePayload,
    user: dict = Depends(require_doctor)
):
    # Verify session belongs to this doctor
    session = supabase.table("guest_scans")\
        .select("*")\
        .eq("id", payload.session_id)\
        .eq("doctor_id", user["sub"])\
        .single()\
        .execute()

    if not session.data:
        raise HTTPException(404, "Walk-in session not found")

    # Save symptoms text to the guest_scans row
    try:
        supabase.table("guest_scans")\
            .update({"symptoms_text": payload.symptoms_text})\
            .eq("id", payload.session_id)\
            .execute()
    except Exception as e:
        print(f"Could not save symptoms to guest_scans: {e}")

    # ── DEMO MODE ────────────────────────────────────────────────────────
    if DEMO_MODE:
        symptoms_lower = payload.symptoms_text.lower()
        if any(w in symptoms_lower for w in ["acidity", "anger", "heat", "rash", "headache"]):
            demo = DEMO_RESPONSES["Pitta"]
        elif any(w in symptoms_lower for w in ["weight", "congestion", "heavy", "slow", "mucus"]):
            demo = DEMO_RESPONSES["Kapha"]
        else:
            demo = DEMO_RESPONSES["Vata"]

        try:
            result = supabase.table("results").insert({
                "scan_id":        None,
                "guest_scan_id":  payload.session_id,
                "vata_pct":       demo["vata"],
                "pitta_pct":      demo["pitta"],
                "kapha_pct":      demo["kapha"],
                "severity":       demo["severity"],
                "pulse_used":     payload.pulse_used,
                "recipe_text":    demo["recipe"],
                "finalised":      False
            }).execute()
            result_id = result.data[0]["id"] if result.data else None
        except Exception as e:
            print(f"Walk-in demo insert error: {e}")
            result_id = None

        return {
            "vata":           demo["vata"],
            "pitta":          demo["pitta"],
            "kapha":          demo["kapha"],
            "severity":       demo["severity"],
            "dominant_dosha": demo["dominant_dosha"],
            "pulse_used":     payload.pulse_used,
            "avg_bpm":        demo["avg_bpm"] if payload.pulse_used else None,
            "avg_spo2":       demo["avg_spo2"] if payload.pulse_used else None,
            "recipe":         demo["recipe"],
            "vision":         demo["vision"],
            "voice":          demo["voice"],
            "result_id":      result_id,
            "session_id":     payload.session_id
        }

    # ── REAL MODE: Full ML pipeline ──────────────────────────────────────
    vision_result, voice_result = await asyncio.gather(
        run_vision(payload.image_data),
        run_voice(payload.audio_data)
    )

    avg_bpm, avg_spo2 = None, None
    if payload.pulse_used:
        # Pulse readings are keyed by session_id for walk-ins (same as patient scan_id)
        avg_bpm, avg_spo2 = get_pulse_averages(payload.session_id)

    from ml.svm_ensemble import run_svm_ensemble
    scores = run_svm_ensemble(
        vision_result=vision_result,
        voice_result=voice_result,
        pulse_used=payload.pulse_used and avg_bpm is not None,
        bpm=avg_bpm,
        spo2=avg_spo2,
        symptoms_text=payload.symptoms_text,
    )

    vata  = scores["vata_pct"]
    pitta = scores["pitta_pct"]
    kapha = scores["kapha_pct"]

    severity       = assess_severity(vata, pitta, kapha)
    dominant_dosha = get_dominant_dosha(vata, pitta, kapha)

    recipe_text = await run_recipe(dominant_dosha, payload.symptoms_text, severity)

    try:
        result = supabase.table("results").insert({
            "scan_id":       None,
            "guest_scan_id": payload.session_id,
            "vata_pct":      vata,
            "pitta_pct":     pitta,
            "kapha_pct":     kapha,
            "severity":      severity,
            "pulse_used":    payload.pulse_used,
            "recipe_text":   recipe_text,
            "finalised":     False
        }).execute()
        result_id = result.data[0]["id"] if result.data else None
    except Exception as e:
        print(f"Walk-in results insert error: {e}")
        result_id = None

    return {
        "vata":           vata,
        "pitta":          pitta,
        "kapha":          kapha,
        "severity":       severity,
        "dominant_dosha": dominant_dosha,
        "pulse_used":     payload.pulse_used,
        "avg_bpm":        avg_bpm,
        "avg_spo2":       avg_spo2,
        "recipe":         recipe_text,
        "vision":         vision_result,
        "voice":          voice_result,
        "result_id":      result_id,
        "session_id":     payload.session_id
    }