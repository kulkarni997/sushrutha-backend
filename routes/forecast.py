from fastapi import APIRouter, Depends
from auth.jwt_handler import require_patient

router = APIRouter()

@router.get("/forecast/{user_id}")
async def get_forecast(user_id: str, user: dict = Depends(require_patient)):
    # Mock output — Prophet model wired on Day 11
    forecast = []
    base = 58
    for day in range(1, 15):
        base = max(30, base - 2 + (day % 3))
        forecast.append({
            "day": day,
            "vata": base,
            "pitta": max(10, 28 - day),
            "kapha": max(5, 14 - day // 2),
            "healing_score": round(min(100, 40 + day * 4.2), 1)
        })
    return {"forecast": forecast, "days": 14}