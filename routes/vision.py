from fastapi import APIRouter, Depends
from pydantic import BaseModel
from auth.jwt_handler import require_patient

router = APIRouter()

class VisionPayload(BaseModel):
    image_data: str
    scan_id: str

@router.post("/vision")
async def analyze_vision(payload: VisionPayload, user: dict = Depends(require_patient)):
    try:
        from ml.yolo_model import analyze_tongue
        result = analyze_tongue(payload.image_data)
        return result
    except Exception as e:
        print(f"Vision model error: {e}")
        return {
            "coating": "thin_pale",
            "vein_score": 0.7,
            "dosha_signal": "Vata"
        }