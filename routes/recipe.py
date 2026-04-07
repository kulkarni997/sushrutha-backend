from fastapi import APIRouter, Depends
from auth.jwt_handler import require_patient
from rag.generator import generate_recipe

router = APIRouter()

@router.get("/recipe/{dosha}")
async def get_recipe(
    dosha: str,
    symptoms: str = "",
    vata: int = 0,
    pitta: int = 0,
    kapha: int = 0,
    user: dict = Depends(require_patient)
):
    recipe_text = generate_recipe(
        dosha=dosha,
        symptoms=symptoms,
        vata=vata,
        pitta=pitta,
        kapha=kapha
    )
    return {
        "dosha": dosha,
        "recipe": recipe_text
    }