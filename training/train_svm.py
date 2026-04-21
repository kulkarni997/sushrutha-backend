"""
Train the SVM ensemble on synthetic data derived from classical Ayurvedic
tongue / voice / pulse correspondences. Run once. Outputs go to ml/.

Usage:  python training/train_svm.py
"""
import os
import sys
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.svm_ensemble import build_feature_vector

RNG = np.random.default_rng(42)

# ---- classical rules encoded as sampling priors ----
# Each dosha archetype: plausible (dosha_label, vein_score range, confidence range,
# pulse profile). Noise added so SVM learns a smooth boundary.

VATA_SYMPTOM_POOL = [
    "anxiety dry skin insomnia",
    "constipation gas bloating",
    "cold hands nervousness restless",
    "trouble sleeping worry racing thoughts",
    "irregular digestion dry skin panic",
    "weight loss stiffness tremors",
]
PITTA_SYMPTOM_POOL = [
    "acidity heartburn anger",
    "skin rash inflammation irritable",
    "burning sensation headache impatience",
    "loose stool sweating hot flush",
    "sharp pain red eyes frustration",
    "ulcer reflux fever",
]
KAPHA_SYMPTOM_POOL = [
    "weight gain congestion mucus",
    "daytime sleepiness lethargy heavy",
    "slow digestion tired swelling",
    "cough cold allergies phlegm",
    "feeling heavy sluggish depressed",
    "water retention stubborn attachment",
]

def sample_vata():
    """Vata: thin/cracked tongue, high vein score, fast pulse, lower spo2 — with realistic confusion."""
    vision_dosha = RNG.choice(["Vata", "Pitta", "Kapha"], p=[0.55, 0.25, 0.20])
    vein_score = float(np.clip(RNG.normal(0.70, 0.15), 0.0, 1.0))
    voice_dosha = RNG.choice(["Vata", "Pitta", "Kapha"], p=[0.55, 0.25, 0.20])
    voice_conf = float(np.clip(RNG.normal(0.70, 0.15), 0.3, 0.99))
    bpm = float(np.clip(RNG.normal(90, 10), 55, 120))
    spo2 = float(np.clip(RNG.normal(95, 1.5), 90, 100))
    pulse_used = bool(RNG.random() > 0.3)
    roll = RNG.random()
    if roll < 0.30:
        symptoms = ""
    elif roll < 0.55:
        symptoms = str(RNG.choice(PITTA_SYMPTOM_POOL + KAPHA_SYMPTOM_POOL))
    else:
        symptoms = str(RNG.choice(VATA_SYMPTOM_POOL))
    return vision_dosha, vein_score, voice_dosha, voice_conf, bpm, spo2, pulse_used, symptoms

def sample_pitta():
    """Pitta: red/yellow tongue, mid features — overlaps with both Vata and Kapha."""
    vision_dosha = RNG.choice(["Pitta", "Vata", "Kapha"], p=[0.55, 0.25, 0.20])
    vein_score = float(np.clip(RNG.normal(0.50, 0.15), 0.0, 1.0))
    voice_dosha = RNG.choice(["Pitta", "Vata", "Kapha"], p=[0.55, 0.25, 0.20])
    voice_conf = float(np.clip(RNG.normal(0.70, 0.15), 0.3, 0.99))
    bpm = float(np.clip(RNG.normal(78, 8), 55, 110))
    spo2 = float(np.clip(RNG.normal(96, 1.3), 90, 100))
    pulse_used = bool(RNG.random() > 0.3)
    roll = RNG.random()
    if roll < 0.30:
        symptoms = ""
    elif roll < 0.45:
        symptoms = str(RNG.choice(VATA_SYMPTOM_POOL + KAPHA_SYMPTOM_POOL))
    else:
        symptoms = str(RNG.choice(PITTA_SYMPTOM_POOL))
    return vision_dosha, vein_score, voice_dosha, voice_conf, bpm, spo2, pulse_used, symptoms


def sample_kapha():
    """Kapha: thick/chubby tongue, low vein score, slow BPM, high spo2 — with realistic confusion."""
    vision_dosha = RNG.choice(["Kapha", "Pitta", "Vata"], p=[0.55, 0.25, 0.20])
    vein_score = float(np.clip(RNG.normal(0.30, 0.15), 0.0, 1.0))
    voice_dosha = RNG.choice(["Kapha", "Pitta", "Vata"], p=[0.55, 0.25, 0.20])
    voice_conf = float(np.clip(RNG.normal(0.70, 0.15), 0.3, 0.99))
    bpm = float(np.clip(RNG.normal(65, 8), 50, 95))
    spo2 = float(np.clip(RNG.normal(97, 1.2), 92, 100))
    pulse_used = bool(RNG.random() > 0.3)
    roll = RNG.random()
    if roll < 0.30:
        symptoms = ""
    elif roll < 0.45:
        symptoms = str(RNG.choice(VATA_SYMPTOM_POOL + PITTA_SYMPTOM_POOL))
    else:
        symptoms = str(RNG.choice(KAPHA_SYMPTOM_POOL))
    return vision_dosha, vein_score, voice_dosha, voice_conf, bpm, spo2, pulse_used, symptoms
SAMPLERS = [sample_vata, sample_pitta, sample_kapha]
LABELS = ["Vata", "Pitta", "Kapha"]
N_PER_CLASS = 500


def generate_dataset():
    X, y = [], []
    for class_idx, sampler in enumerate(SAMPLERS):
        for _ in range(N_PER_CLASS):
            vd, vs, vod, vc, bpm, spo2, pu, symp = sampler()
            feat = build_feature_vector(
                vision_dosha=vd,
                vein_score=vs,
                voice_dosha=vod,
                voice_confidence=vc,
                bpm=bpm,
                spo2=spo2,
                pulse_used=pu,
                symptoms_text=symp,
            )
            X.append(feat)
            y.append(class_idx)
    return np.array(X), np.array(y)


def main():
    print("[train_svm] Generating synthetic dataset...")
    X, y = generate_dataset()
    print(f"[train_svm] Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("[train_svm] Training SVC(rbf, probability=True)...")
    model = SVC(kernel="rbf", C=0.5, gamma="scale", probability=True, random_state=42)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"[train_svm] Test accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=LABELS))

    # Save
    here = os.path.dirname(os.path.abspath(__file__))
    ml_dir = os.path.join(os.path.dirname(here), "ml")
    os.makedirs(ml_dir, exist_ok=True)
    model_path = os.path.join(ml_dir, "svm_model.pkl")
    scaler_path = os.path.join(ml_dir, "svm_scaler.pkl")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"[train_svm] Saved: {model_path}")
    print(f"[train_svm] Saved: {scaler_path}")


if __name__ == "__main__":
    main()