from fastapi import APIRouter, Depends, UploadFile, File
from auth.jwt_handler import require_patient

router = APIRouter()

@router.post("/voice")
async def analyze_voice(
    scan_id: str,
    audio: UploadFile = File(...),
    user: dict = Depends(require_patient)
):
    # Mock output — Whisper wired on Day 8
    return {
        "transcript": "I have been feeling tired and bloated lately.",
        "voice_dosha": "Vata",
        "confidence": 0.72
    }