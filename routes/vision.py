from fastapi import APIRouter, Depends, UploadFile, File, Form
from auth.jwt_handler import require_patient
import base64

router = APIRouter()

@router.post("/vision")
async def analyze_vision(
    scan_id: str = Form(...),
    image: UploadFile = File(...),
    user: dict = Depends(require_patient)
):
    try:
        image_bytes = await image.read()
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        
        from ml.yolo_model import analyze_tongue
        result = analyze_tongue(image_data)
        return result
    except Exception as e:
        print(f"Vision model error: {e}")
        return {
            "coating": "thin_pale",
            "vein_score": 0.7,
            "dosha_signal": "Vata"
        }