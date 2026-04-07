from fastapi import APIRouter, Depends
from pydantic import BaseModel
from auth.jwt_handler import require_patient

router = APIRouter()

class VisionPayload(BaseModel):
    image_data: str
    scan_id: str

@router.post("/vision")
async def analyze_vision(payload: VisionPayload, user: dict = Depends(require_patient)):
    # Mock output — YOLOv10 wired on Day 8
    return {
        "coating": "thick_white",
        "vein_score": 0.7,
        "dosha_signal": "Kapha"
    }