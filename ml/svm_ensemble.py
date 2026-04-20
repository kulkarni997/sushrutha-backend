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


def build_feature_vector(
    vision_dosha: str,
    vein_score: float,
    voice_dosha: str,
    voice_confidence: float,
    bpm: Optional[float] = None,
    spo2: Optional[float] = None,
    pulse_used: bool = False
) -> np.ndarray:
    vision_enc = encode_dosha(vision_dosha)
    voice_enc = encode_dosha(voice_dosha)

    bpm_norm = normalize_bpm(bpm) if pulse_used and bpm else 0.0
    spo2_norm = normalize_spo2(spo2) if pulse_used and spo2 else 0.0

    features = (
        vision_enc +
        [vein_score] +
        voice_enc +
        [voice_confidence, bpm_norm, spo2_norm]
    )
    return np.array(features, dtype=np.float32)


def rule_based_scores(features: np.ndarray, pulse_used: bool) -> dict:
    """
    Weighted rule-based SVM substitute.
    Weights: vision=0.45, voice=0.35, pulse=0.20 (0 if no pulse, redistributed)
    """
    vision_vata, vision_pitta, vision_kapha = features[0], features[1], features[2]
    vein_score = features[3]
    voice_vata, voice_pitta, voice_kapha = features[4], features[5], features[6]
    voice_conf = features[7]
    bpm_norm = features[8]
    spo2_norm = features[9]

    if pulse_used:
        w_vision, w_voice, w_pulse = 0.45, 0.35, 0.20
    else:
        w_vision, w_voice, w_pulse = 0.55, 0.45, 0.0

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

    raw_vata = w_vision * vata_v + w_voice * vata_vo + w_pulse * vata_p
    raw_pitta = w_vision * pitta_v + w_voice * pitta_vo + w_pulse * pitta_p
    raw_kapha = w_vision * kapha_v + w_voice * kapha_vo + w_pulse * kapha_p

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
    spo2: Optional[float] = None
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
        pulse_used=pulse_used
    )

    try:
        scores = svm_scores(features)
        scores["_method"] = "trained_svm"
    except Exception as e:
        print(f"[svm_ensemble] SVM inference failed: {e} -- using rule-based")
        scores = rule_based_scores(features, pulse_used)
        scores["_method"] = "rule_based"

    return scores