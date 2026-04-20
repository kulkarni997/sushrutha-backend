import os
from dotenv import load_dotenv

load_dotenv()

try:
    from rag.retriever import retrieve_context
except Exception as e:
    print(f"[rag.generator] FAILED to import retriever: {e}")
    def retrieve_context(dosha, symptoms=""):
        return ""

# Groq client — initialized lazily
_client = None

# Current Groq model. If deprecated, swap to "llama-3.1-8b-instant".
GROQ_MODEL = "llama-3.1-8b-instant"


def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("[rag.generator] GROQ_API_KEY not set — using fallback recipes")
            return None
        try:
            from groq import Groq
            _client = Groq(api_key=api_key)
            print("[rag.generator] Groq client initialized")
        except Exception as e:
            print(f"[rag.generator] Failed to init Groq: {e}")
            return None
    return _client


def generate_recipe(
    dosha: str,
    symptoms: str = "",
    vata: int = 0,
    pitta: int = 0,
    kapha: int = 0,
) -> str:
    # Retrieve grounding passages from Sushruta + Charaka
    context = retrieve_context(dosha, symptoms)

    if not context:
        print("[rag.generator] No context retrieved — using fallback")
        return get_fallback_recipe(dosha)

    client = get_client()
    if client is None:
        return get_fallback_recipe(dosha)

    # Trim context to stay well under Groq's context window.
    # 3B model handles ~8k tokens. 4000 chars ≈ 1000 tokens of context is plenty.
    if len(context) > 4000:
        context = context[:4000]

    system_prompt = (
        "You are an Ayurvedic doctor trained in the classical texts of "
        "Sushruta Samhita and Charaka Samhita. You provide personalized "
        "herbal recipes grounded strictly in the passages provided. "
        "You do not invent remedies. If the passages do not support a "
        "recommendation, say so briefly and give a safe general guideline."
    )

    user_prompt = f"""Classical text passages (use these as your source):

{context}

---

Patient profile:
- Dominant dosha: {dosha}
- Dosha balance: Vata {vata}%, Pitta {pitta}%, Kapha {kapha}%
- Reported symptoms: {symptoms if symptoms else "none specified"}

Provide a concise personalized recommendation with exactly these sections:

**Herbs** (3 items, each with dosage and timing)
**Yoga** (2 poses or pranayama with duration)
**Diet** (1-2 lines on what to eat and avoid)

Keep the total response under 250 words. Use plain English. No disclaimers."""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.6,
            max_tokens=500,
        )
        answer = response.choices[0].message.content.strip()
        if not answer:
            return get_fallback_recipe(dosha)
        return answer

    except Exception as e:
        print(f"[rag.generator] Groq call failed: {e}")
        return get_fallback_recipe(dosha)


def get_fallback_recipe(dosha: str) -> str:
    recipes = {
        "vata": "Ashwagandha 500mg twice daily with warm milk.\nTriphala 1 tsp at night.\nYoga: Balasana, Nadi Shodhana.\nDiet: Warm, oily, grounding foods.",
        "pitta": "Shatavari 500mg twice daily with cool water.\nAmalaki 500mg before meals.\nYoga: Sheetali pranayama, Shavasana.\nDiet: Cool, sweet, bitter foods.",
        "kapha": "Trikatu 500mg before meals.\nGuggulu 500mg twice daily.\nYoga: Surya Namaskar, Bhastrika.\nDiet: Light, dry, warm foods.",
    }
    return recipes.get(dosha.lower(), recipes["vata"])