import whisper
import librosa
import numpy as np
import tempfile
import os

_model = None

def get_whisper_model():
    global _model
    if _model is None:
        _model = whisper.load_model("base")
    return _model

def extract_mfcc_features(audio_path: str) -> dict:
    try:
        y, sr = librosa.load(audio_path, duration=10)
        
        energy = float(np.mean(librosa.feature.rms(y=y)))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

        # Score each dosha — higher score wins
        # Vata: fast/irregular speech, thin/breathy voice
        # ZCR threshold raised — 0.1 fires on almost all speech
        vata_score  = (1.0 if zcr > 0.18 else 0.4) + (1.0 if tempo > 130 else 0.3)
        # Pitta: strong energy, fast but controlled
        pitta_score = (1.0 if energy > 0.04 else 0.3) + (1.0 if 90 < tempo <= 130 else 0.2)
        # Kapha: slow tempo, low energy, low ZCR
        kapha_score = (1.0 if zcr < 0.08 else 0.3) + (1.0 if tempo < 90 else 0.2) + (1.0 if energy < 0.02 else 0.2)

        scores = {"Vata": vata_score, "Pitta": pitta_score, "Kapha": kapha_score}
        voice_dosha = max(scores, key=scores.get)

        # Real confidence: how much does the winner beat second place
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1]
        # margin range is roughly 0-2, normalize to 0.5-0.9
        confidence = round(min(0.5 + (margin / 2.0) * 0.4, 0.92), 2)

        return {
            "voice_dosha": voice_dosha,
            "confidence": confidence,
            "energy": energy,
            "tempo": tempo,
            "zcr": zcr
        }
    except Exception as e:
        print(f"MFCC extraction failed: {e}")
        return {"voice_dosha": "Vata", "confidence": 0.5}

def analyze_voice(audio_bytes: bytes, filename: str = "audio.webm") -> dict:
    try:
        # Save to temp file
        suffix = os.path.splitext(filename)[1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        # Transcribe with Whisper
        model = get_whisper_model()
        result = model.transcribe(tmp_path)
        transcript = result["text"].strip()
        
        # Extract MFCC features
        mfcc_data = extract_mfcc_features(tmp_path)
        
        os.unlink(tmp_path)
        
        return {
            "transcript": transcript,
            "voice_dosha": mfcc_data["voice_dosha"],
            "confidence": mfcc_data["confidence"],
            "language": result.get("language", "en")
        }
    except Exception as e:
        print(f"Voice analysis failed: {e}")
        return {
            "transcript": "",
            "voice_dosha": "Vata",
            "confidence": 0.5,
            "language": "en"
        }