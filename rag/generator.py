from rag.retriever import retrieve_context
import os

def generate_recipe(dosha: str, symptoms: str = "", vata: int = 0, pitta: int = 0, kapha: int = 0) -> str:
    context = retrieve_context(dosha, symptoms)
    
    if not context:
        # Fallback if no FAISS index yet
        return get_fallback_recipe(dosha)
    
    try:
        from transformers import pipeline
        generator = pipeline(
            "text-generation",
            model="openlm-research/open_llama_3b",
            max_new_tokens=300,
            temperature=0.7
        )
        
        prompt = f"""You are an Ayurvedic doctor. Based on the following classical texts:

{context}

Patient dosha: {dosha} dominant (Vata: {vata}%, Pitta: {pitta}%, Kapha: {kapha}%)
Symptoms: {symptoms}

Provide a personalized herbal recipe with:
1. 3 herbs with dosage and timing
2. 2 yoga poses
3. Diet recommendation

Answer:"""
        
        result = generator(prompt)[0]["generated_text"]
        answer = result.split("Answer:")[-1].strip()
        return answer
        
    except Exception as e:
        print(f"LLM generation failed: {e}")
        return get_fallback_recipe(dosha)

def get_fallback_recipe(dosha: str) -> str:
    recipes = {
        "vata": "Ashwagandha 500mg twice daily with warm milk.\nTriphala 1 tsp at night.\nYoga: Balasana, Nadi Shodhana.\nDiet: Warm, oily, grounding foods.",
        "pitta": "Shatavari 500mg twice daily with cool water.\nAmalaki 500mg before meals.\nYoga: Sheetali pranayama, Shavasana.\nDiet: Cool, sweet, bitter foods.",
        "kapha": "Trikatu 500mg before meals.\nGuggulu 500mg twice daily.\nYoga: Surya Namaskar, Bhastrika.\nDiet: Light, dry, warm foods."
    }
    return recipes.get(dosha.lower(), recipes["vata"])