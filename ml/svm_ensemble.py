import numpy as np
from typing import Optional

# Dosha encoding
DOSHA_MAP = {"Vata": 0, "Pitta": 1, "Kapha": 2}

def encode_dosha(dosha: str) -> list:
    """One-hot encode dosha string → [vata, pitta, kapha]"""
    v = [0.0, 0.0, 0.0]
    idx = DOSHA_MAP.get(dosha, 0)
    v[idx] = 1.0
    return v

def normalize_bpm(bpm: float) -> float:
    """Vata = high BPM (>80), Pitta = mid (70-80), Kapha = low (<70)"""
    if bpm <= 0:
        return 0.5  # neutral if no pulse
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
    """
    Feature vector (10 dims):
    [vision_vata, vision_pitta, vision_kapha,   # 3 — one-hot from tongue
     vein_score,                                 # 1 — 0-1 float
     voice_vata, voice_pitta, voice_kapha,       # 3 — one-hot from voice
     voice_confidence,                           # 1 — 0-1 float
     bpm_norm,                                   # 1 — 0 if no pulse
     spo2_norm]                                  # 1 — 0 if no pulse
    """
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

    # Vision contribution
    # vein_score: high = Vata, low = Kapha
    vata_v  = vision_vata * (1.0 + 0.2 * vein_score)
    pitta_v = vision_pitta * 1.0
    kapha_v = vision_kapha * (1.0 + 0.2 * (1 - vein_score))

    # Voice contribution (weighted by confidence)
    vata_vo  = voice_vata  * voice_conf
    pitta_vo = voice_pitta * voice_conf
    kapha_vo = voice_kapha * voice_conf

    # Pulse contribution
    # High BPM → Vata/Pitta, Low BPM + high SpO2 → Kapha
    if pulse_used:
        vata_p  = bpm_norm * 0.6
        pitta_p = bpm_norm * 0.4
        kapha_p = (1 - bpm_norm) * 0.5 + spo2_norm * 0.5
    else:
        vata_p = pitta_p = kapha_p = 0.0

    # Combine
    raw_vata  = w_vision * vata_v  + w_voice * vata_vo  + w_pulse * vata_p
    raw_pitta = w_vision * pitta_v + w_voice * pitta_vo + w_pulse * pitta_p
    raw_kapha = w_vision * kapha_v + w_voice * kapha_vo + w_pulse * kapha_p

    total = raw_vata + raw_pitta + raw_kapha
    if total == 0:
        total = 1.0  # avoid div by zero

    vata_pct  = round((raw_vata  / total) * 100)
    pitta_pct = round((raw_pitta / total) * 100)
    kapha_pct = 100 - vata_pct - pitta_pct  # ensure sum = 100

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

    Args:
        vision_result: output from analyze_tongue()
            { dosha_signal, vein_score, coating, ... }
        voice_result: output from analyze_voice()
            { voice_dosha, confidence, transcript, language }
        pulse_used: whether ESP32 data was captured
        bpm: average BPM from pulse_readings (None if no pulse)
        spo2: average SpO2 from pulse_readings (None if no pulse)

    Returns:
        { vata_pct, pitta_pct, kapha_pct }  — sum always = 100
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

    scores = rule_based_scores(features, pulse_used)
    return scores