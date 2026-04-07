from fastapi import APIRouter, Depends
from auth.jwt_handler import require_patient

router = APIRouter()

@router.get("/recipe/{dosha}")
async def get_recipe(dosha: str, user: dict = Depends(require_patient)):
    # Mock output — LangChain RAG + Llama 3.2 wired on Day 7
    recipes = {
        "vata": {
            "herbs": [
                {"name": "Ashwagandha", "dosage": "500mg", "timing": "twice daily with warm milk"},
                {"name": "Shatavari", "dosage": "500mg", "timing": "once daily after meals"},
                {"name": "Triphala", "dosage": "1 tsp", "timing": "with lukewarm water at night"}
            ],
            "yoga": ["Balasana", "Viparita Karani", "Nadi Shodhana pranayama"],
            "diet": "Prefer warm, oily, grounding foods. Avoid cold and dry foods."
        },
        "pitta": {
            "herbs": [
                {"name": "Shatavari", "dosage": "500mg", "timing": "twice daily with cool water"},
                {"name": "Amalaki", "dosage": "500mg", "timing": "once daily before meals"},
                {"name": "Brahmi", "dosage": "300mg", "timing": "once daily in the morning"}
            ],
            "yoga": ["Sheetali pranayama", "Chandra Namaskar", "Shavasana"],
            "diet": "Prefer cool, sweet, bitter foods. Avoid spicy and oily foods."
        },
        "kapha": {
            "herbs": [
                {"name": "Trikatu", "dosage": "500mg", "timing": "twice daily before meals"},
                {"name": "Guggulu", "dosage": "500mg", "timing": "twice daily with warm water"},
                {"name": "Ginger", "dosage": "1 tsp powder", "timing": "with honey in the morning"}
            ],
            "yoga": ["Surya Namaskar", "Bhastrika pranayama", "Trikonasana"],
            "diet": "Prefer light, dry, warm foods. Avoid heavy, oily, sweet foods."
        }
    }
    
    dosha_lower = dosha.lower()
    if dosha_lower not in recipes:
        dosha_lower = "vata"
    
    return recipes[dosha_lower]