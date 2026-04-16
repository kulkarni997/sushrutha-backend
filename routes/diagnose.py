import asyncio
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from db.supabase_client import supabase
from auth.jwt_handler import require_patient

router = APIRouter()

class DiagnosePayload(BaseModel):
    scan_id: str
    symptoms_text: str
    image_data: Optional[str] = None      # base64 tongue image
    audio_data: Optional[str] = None      # base64 audio (used if no upload)
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

def get_dominant_dosha(vata: int, pitta: int, kapha: int) -> str:
    scores = {"Vata": vata, "Pitta": pitta, "Kapha": kapha}
    return max(scores, key=scores.get)

def get_pulse_averages(scan_id: str) -> tuple:
    """Fetch last 10 pulse readings and return avg bpm, avg spo2"""
    try:
        res = supabase.table("pulse_readings") \
            .select("bpm, spo2") \
            .eq("scan_id", scan_id) \
            .order("timestamp", desc=True) \
            .limit(10) \
            .execute()

        if res.data:
            bpms  = [r["bpm"]  for r in res.data if r["bpm"]  > 0]
            spo2s = [r["spo2"] for r in res.data if r["spo2"] > 0]
            avg_bpm  = round(sum(bpms)  / len(bpms),  2) if bpms  else None
            avg_spo2 = round(sum(spo2s) / len(spo2s), 2) if spo2s else None
            return avg_bpm, avg_spo2
    except Exception as e:
        print(f"Pulse fetch error: {e}")
    return None, None

async def run_vision(image_data: Optional[str]) -> dict:
    fallback = {"coating": "thin_pale", "vein_score": 0.5, "dosha_signal": "Vata"}
    if not image_data:
        return fallback
    try:
        from ml.yolo_model import analyze_tongue
        return analyze_tongue(image_data)
    except Exception as e:
        print(f"Vision failed: {e}")
        return fallback

async def run_voice(audio_data: Optional[str]) -> dict:
    fallback = {"voice_dosha": "Vata", "confidence": 0.5, "transcript": "", "language": "en"}
    if not audio_data:
        return fallback
    try:
        import base64
        from ml.whisper_model import analyze_voice
        audio_bytes = base64.b64decode(audio_data)
        return analyze_voice(audio_bytes, "audio.webm")
    except Exception as e:
        print(f"Voice failed: {e}")
        return fallback

async def run_recipe(dominant_dosha: str, symptoms_text: str, severity: str) -> str:
    if severity == "severe":
        return "Severity is high. Please consult a BAMS doctor before taking any herbal remedies."
    try:
        from rag.generator import generate_recipe
        return generate_recipe(dominant_dosha, symptoms_text)
    except Exception as e:
        print(f"RAG failed: {e}")
        defaults = {
            "Vata": "Ashwagandha 500mg twice daily. Sesame oil massage. Warm meals only.",
            "Pitta": "Shatavari 500mg after meals. Coconut water daily. Avoid spicy food.",
            "Kapha": "Trikatu 250mg before meals. Ginger tea morning. Light exercise daily."
        }
        return defaults.get(dominant_dosha, "Triphala 500mg at bedtime with warm water.")

@router.post("/diagnose")
async def diagnose(
    payload: DiagnosePayload,
    user: dict = Depends(require_patient)
):
    scan_id = payload.scan_id

    # DEMO MODE — skip all ML, return realistic hardcoded response instantly
    if DEMO_MODE:
        # Pick demo response based on symptoms keyword
        symptoms_lower = payload.symptoms_text.lower()
        if any(w in symptoms_lower for w in ["acidity", "anger", "heat", "rash", "headache"]):
            demo = DEMO_RESPONSES["Pitta"]
        elif any(w in symptoms_lower for w in ["weight", "congestion", "heavy", "slow", "mucus"]):
            demo = DEMO_RESPONSES["Kapha"]
        else:
            demo = DEMO_RESPONSES["Vata"]

        try:
            result = supabase.table("results").insert({
                "scan_id":    scan_id,
                "vata_pct":   demo["vata"],
                "pitta_pct":  demo["pitta"],
                "kapha_pct":  demo["kapha"],
                "severity":   demo["severity"],
                "pulse_used": payload.pulse_used,
                "recipe_text": demo["recipe"],
                "finalised":  False
            }).execute()
            result_id = result.data[0]["id"] if result.data else None
        except Exception as e:
            print(f"Demo results insert error: {e}")
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
            "result_id":      result_id
        }

    # REAL MODE — full ML pipeline
    vision_result, voice_result = await asyncio.gather(
        run_vision(payload.image_data),
        run_voice(payload.audio_data)
    )

    avg_bpm, avg_spo2 = None, None
    if payload.pulse_used:
        avg_bpm, avg_spo2 = get_pulse_averages(scan_id)

    from ml.svm_ensemble import run_svm_ensemble
    scores = run_svm_ensemble(
        vision_result=vision_result,
        voice_result=voice_result,
        pulse_used=payload.pulse_used and avg_bpm is not None,
        bpm=avg_bpm,
        spo2=avg_spo2
    )

    vata  = scores["vata_pct"]
    pitta = scores["pitta_pct"]
    kapha = scores["kapha_pct"]

    severity       = assess_severity(vata, pitta, kapha)
    dominant_dosha = get_dominant_dosha(vata, pitta, kapha)

    recipe_text = await run_recipe(dominant_dosha, payload.symptoms_text, severity)

    try:
        result = supabase.table("results").insert({
            "scan_id":    scan_id,
            "vata_pct":   vata,
            "pitta_pct":  pitta,
            "kapha_pct":  kapha,
            "severity":   severity,
            "pulse_used": payload.pulse_used,
            "recipe_text": recipe_text,
            "finalised":  False
        }).execute()
        result_id = result.data[0]["id"] if result.data else None
    except Exception as e:
        print(f"Results insert error: {e}")
        result_id = None

    return {
        "vata":          vata,
        "pitta":         pitta,
        "kapha":         kapha,
        "severity":      severity,
        "dominant_dosha": dominant_dosha,
        "pulse_used":    payload.pulse_used,
        "avg_bpm":       avg_bpm,
        "avg_spo2":      avg_spo2,
        "recipe":        recipe_text,
        "vision":        vision_result,
        "voice":         voice_result,
        "result_id":     result_id
    }