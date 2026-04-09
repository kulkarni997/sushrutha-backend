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
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # Map MFCC features to dosha signals
        energy = np.mean(librosa.feature.rms(y=y))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Vata: high ZCR, irregular tempo, thin voice
        # Pitta: medium energy, sharp articulation
        # Kapha: low ZCR, slow tempo, heavy voice
        if zcr > 0.1 or tempo > 120:
            voice_dosha = "Vata"
            confidence = 0.72
        elif energy > 0.05:
            voice_dosha = "Pitta"
            confidence = 0.68
        else:
            voice_dosha = "Kapha"
            confidence = 0.65
            
        return {
            "voice_dosha": voice_dosha,
            "confidence": confidence,
            "energy": float(energy),
            "tempo": float(tempo),
            "zcr": float(zcr)
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