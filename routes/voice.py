from fastapi import APIRouter, Depends, UploadFile, File
from auth.jwt_handler import require_patient

router = APIRouter()

@router.post("/voice")
async def analyze_voice(
    scan_id: str,
    audio: UploadFile = File(...),
    user: dict = Depends(require_patient)
):
    try:
        from ml.whisper_model import analyze_voice as run_whisper
        audio_bytes = await audio.read()
        result = run_whisper(audio_bytes, audio.filename or "audio.webm")
        return result
    except Exception as e:
        print(f"Voice model error: {e}")
        return {
            "transcript": "",
            "voice_dosha": "Vata",
            "confidence": 0.5,
            "language": "en"
        }