import numpy as np
from typing import Optional
import os
import joblib

# Dosha encoding
DOSHA_MAP = {"Vata": 0, "Pitta": 1, "Kapha": 2}

# Paths and model cache
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "svm_model.pkl")
SCALER_PATH = os.path.join(_HERE, "svm_scaler.pkl")

_svm_model = None
_svm_scaler = None
_svm_load_attempted = False


def _load_svm():
    """Lazy-load trained SVM + scaler. Returns (model, scaler) or (None, None)."""
    global _svm_model, _svm_scaler, _svm_load_attempted
    if _svm_load_attempted:
        return _svm_model, _svm_scaler
    _svm_load_attempted = True

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print(f"[svm_ensemble] Model files not found at {MODEL_PATH} -- using rule-based fallback")
        return None, None
    try:
        _svm_model = joblib.load(MODEL_PATH)
        _svm_scaler = joblib.load(SCALER_PATH)
        print(f"[svm_ensemble] Loaded trained SVM from {MODEL_PATH}")
    except Exception as e:
        print(f"[svm_ensemble] Failed to load SVM: {e} -- using rule-based fallback")
    return _svm_model, _svm_scaler


def svm_scores(features: np.ndarray) -> dict:
    """Trained-SVM path. Returns dict with vata_pct/pitta_pct/kapha_pct summing to 100."""
    model, scaler = _load_svm()
    if model is None:
        raise RuntimeError("SVM model unavailable")

    X = features.reshape(1, -1)
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[0]

    class_order = list(model.classes_)
    vata_p = float(probs[class_order.index(0)])
    pitta_p = float(probs[class_order.index(1)])
    kapha_p = float(probs[class_order.index(2)])

    vata_pct = round(vata_p * 100)
    pitta_pct = round(pitta_p * 100)
    kapha_pct = 100 - vata_pct - pitta_pct

    return {"vata_pct": int(vata_pct), "pitta_pct": int(pitta_pct), "kapha_pct": int(kapha_pct)}


def encode_dosha(dosha: str) -> list:
    """One-hot encode dosha string -> [vata, pitta, kapha]"""
    v = [0.0, 0.0, 0.0]
    idx = DOSHA_MAP.get(dosha, 0)
    v[idx] = 1.0
    return v


def normalize_bpm(bpm: float) -> float:
    """Vata = high BPM (>80), Pitta = mid (70-80), Kapha = low (<70)"""
    if bpm <= 0:
        return 0.5
    return min(max((bpm - 50) / 60.0, 0.0), 1.0)


def normalize_spo2(spo2: float) -> float:
    """Higher SpO2 = more Kapha grounding"""
    if spo2 <= 0:
        return 0.5
    return min(max((spo2 - 94) / 6.0, 0.0), 1.0)


# Symptom keyword lexicon — standard Ayurvedic correspondences
VATA_KEYWORDS = [
    "anxiety", "worry", "insomnia", "restless", "dry skin", "constipation",
    "gas", "bloating", "cold hand", "cold feet", "nervousness", "tremor",
    "cracking joint", "irregular", "fear", "panic", "stiff", "weight loss",
    "trouble sleeping", "racing thought"
]

PITTA_KEYWORDS = [
    "acidity", "heartburn", "reflux", "rash", "inflammation", "anger",
    "irritable", "frustration", "headache", "burning", "hot flush", "fever",
    "skin redness", "ulcer", "sharp pain", "impatience", "sweating",
    "loose stool", "diarrhea", "red eye"
]

KAPHA_KEYWORDS = [
    "weight gain", "congestion", "mucus", "heavy", "lethargy", "sleepy",
    "tired", "slow digestion", "water retention", "swelling", "cough",
    "cold", "allergies", "attachment", "stubborn", "depressed", "sluggish",
    "daytime sleepiness", "feeling heavy", "phlegm"
]


def extract_symptom_dosha_scores(symptoms_text: str) -> tuple:
    """
    Keyword-match the symptoms text to dosha-specific lexicons.
    Returns (vata_score, pitta_score, kapha_score) normalized to [0, 1] each,
    independently. Multiple doshas can be high simultaneously.
    """
    if not symptoms_text or not isinstance(symptoms_text, str):
        return 0.0, 0.0, 0.0

    text = symptoms_text.lower()
    v_hits = sum(1 for kw in VATA_KEYWORDS  if kw in text)
    p_hits = sum(1 for kw in PITTA_KEYWORDS if kw in text)
    k_hits = sum(1 for kw in KAPHA_KEYWORDS if kw in text)

    # Normalize by lexicon size, cap at 1.0 so a massive dump of Vata words
    # doesn't blow past the bounds of the feature distribution
    # Softer saturation: caps at 0.7 so symptoms can't dominate other signals
    v = min(v_hits / 3.0, 0.7)
    p = min(p_hits / 3.0, 0.7)
    k = min(k_hits / 3.0, 0.7)
    return float(v), float(p), float(k)


def build_feature_vector(
    vision_dosha: str,
    vein_score: float,
    voice_dosha: str,
    voice_confidence: float,
    bpm: Optional[float] = None,
    spo2: Optional[float] = None,
    pulse_used: bool = False,
    symptoms_text: str = ""
) -> np.ndarray:
    """
    Feature vector (13 dims):
    [vision_vata, vision_pitta, vision_kapha,   # 3  one-hot tongue
     vein_score,                                 # 1  0-1 float
     voice_vata, voice_pitta, voice_kapha,       # 3  one-hot voice
     voice_confidence,                           # 1  0-1 float
     bpm_norm, spo2_norm,                        # 2  0 if no pulse
     sym_vata, sym_pitta, sym_kapha]             # 3  keyword-derived
    """
    vision_enc = encode_dosha(vision_dosha)
    voice_enc = encode_dosha(voice_dosha)

    bpm_norm = normalize_bpm(bpm) if pulse_used and bpm else 0.0
    spo2_norm = normalize_spo2(spo2) if pulse_used and spo2 else 0.0

    sym_v, sym_p, sym_k = extract_symptom_dosha_scores(symptoms_text)

    features = (
        vision_enc +
        [vein_score] +
        voice_enc +
        [voice_confidence, bpm_norm, spo2_norm,
         sym_v, sym_p, sym_k]
    )
    return np.array(features, dtype=np.float32)


def rule_based_scores(features: np.ndarray, pulse_used: bool) -> dict:
    """
    Weighted rule-based SVM substitute.
    Weights: vision=0.35, voice=0.25, pulse=0.15, symptoms=0.25
    (redistributed when pulse absent)
    """
    vision_vata, vision_pitta, vision_kapha = features[0], features[1], features[2]
    vein_score = features[3]
    voice_vata, voice_pitta, voice_kapha = features[4], features[5], features[6]
    voice_conf = features[7]
    bpm_norm = features[8]
    spo2_norm = features[9]
    sym_vata, sym_pitta, sym_kapha = features[10], features[11], features[12]

    if pulse_used:
        w_vision, w_voice, w_pulse, w_symp = 0.35, 0.25, 0.15, 0.25
    else:
        w_vision, w_voice, w_pulse, w_symp = 0.40, 0.30, 0.0, 0.30

    vata_v = vision_vata * (1.0 + 0.2 * vein_score)
    pitta_v = vision_pitta * 1.0
    kapha_v = vision_kapha * (1.0 + 0.2 * (1 - vein_score))

    vata_vo = voice_vata * voice_conf
    pitta_vo = voice_pitta * voice_conf
    kapha_vo = voice_kapha * voice_conf

    if pulse_used:
        vata_p = bpm_norm * 0.6
        pitta_p = bpm_norm * 0.4
        kapha_p = (1 - bpm_norm) * 0.5 + spo2_norm * 0.5
    else:
        vata_p = pitta_p = kapha_p = 0.0

    raw_vata  = w_vision * vata_v  + w_voice * vata_vo  + w_pulse * vata_p  + w_symp * sym_vata
    raw_pitta = w_vision * pitta_v + w_voice * pitta_vo + w_pulse * pitta_p + w_symp * sym_pitta
    raw_kapha = w_vision * kapha_v + w_voice * kapha_vo + w_pulse * kapha_p + w_symp * sym_kapha

    total = raw_vata + raw_pitta + raw_kapha
    if total == 0:
        total = 1.0

    vata_pct = round((raw_vata / total) * 100)
    pitta_pct = round((raw_pitta / total) * 100)
    kapha_pct = 100 - vata_pct - pitta_pct

    return {
        "vata_pct": int(vata_pct),
        "pitta_pct": int(pitta_pct),
        "kapha_pct": int(kapha_pct)
    }


def run_svm_ensemble(
    vision_result: dict,
    voice_result: dict,
    pulse_used: bool = False,
    bpm: Optional[float] = None,
    spo2: Optional[float] = None,
    symptoms_text: str = ""
) -> dict:
    """
    Main entry point. Called from /diagnose route.
    Returns: { vata_pct, pitta_pct, kapha_pct, _method } -- sum always = 100
    """
    features = build_feature_vector(
        vision_dosha=vision_result.get("dosha_signal", "Vata"),
        vein_score=float(vision_result.get("vein_score", 0.5)),
        voice_dosha=voice_result.get("voice_dosha", "Vata"),
        voice_confidence=float(voice_result.get("confidence", 0.5)),
        bpm=bpm,
        spo2=spo2,
        pulse_used=pulse_used,
        symptoms_text=symptoms_text
    )

    try:
        scores = svm_scores(features)
        scores["_method"] = "trained_svm"
    except Exception as e:
        print(f"[svm_ensemble] SVM inference failed: {e} -- using rule-based")
        scores = rule_based_scores(features, pulse_used)
        scores["_method"] = "rule_based"

    return scores