import cv2
import numpy as np
import base64
import math
import os

_model = None

# TMC class name → Ayurvedic signal mapping
# Matches the 7 classes you trained on (jiankangshe dropped)
CLASS_TO_DOSHA = {
    "baitaishe":   "Kapha",   # white coating      — mAP 0.88
    "huangtaishe": "Pitta",   # yellow coating     — mAP 0.78
    "hongshe":     "Pitta",   # red tongue         — mAP 0.53
    "zishe":       "Vata",    # purple tongue      — mAP 0.18
    "liewenshe":   "Vata",    # cracked tongue     — mAP 0.38
    "pangdashe":   "Kapha",   # chubby tongue      — mAP 0.23
    "shoushe":     "Vata",    # thin tongue        — mAP 0.67
}

# TMC class name → legacy `coating` value (for backward compat)
CLASS_TO_COATING = {
    "baitaishe":   "thick_white",
    "huangtaishe": "yellow_orange",
    "hongshe":     "red",
    "zishe":       "thin_pale",      # purple falls under Vata visually
    "liewenshe":   "cracked",
    "pangdashe":   "thick_white",    # chubby often with white coat
    "shoushe":     "thin_pale",
}

# Confidence below this → treat as no detection
CONF_THRESHOLD = 0.25

# Path resolution — works regardless of where uvicorn is launched from
_HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(_HERE, "best.pt")


def safe_float(val):
    f = float(val)
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return round(f, 4)


def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        if not os.path.exists(WEIGHTS_PATH):
            print(f"[yolo_model] WARNING: {WEIGHTS_PATH} not found. HSV fallback will be used.")
            return None
        print(f"[yolo_model] Loading trained weights from {WEIGHTS_PATH}")
        _model = YOLO(WEIGHTS_PATH)
    return _model


def _hsv_fallback(img):
    """Legacy HSV rule-based analysis. Used when model unavailable or no detections."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    center = hsv[h//4:3*h//4, w//4:3*w//4]

    if center.size == 0:
        return get_fallback_result()

    avg_hue = float(np.mean(center[:, :, 0]))
    avg_sat = float(np.mean(center[:, :, 1]))

    if avg_sat < 50:
        coating, dosha_signal, vein_score = "thick_white", "Kapha", 0.3
    elif avg_hue < 30 or avg_hue > 150:
        coating, dosha_signal, vein_score = "yellow_orange", "Pitta", 0.6
    else:
        coating, dosha_signal, vein_score = "thin_pale", "Vata", 0.7

    return {
        "coating": coating,
        "vein_score": safe_float(vein_score),
        "dosha_signal": dosha_signal,
        "avg_hue": safe_float(avg_hue),
        "avg_saturation": safe_float(avg_sat),
        "detected_features": [],
        "top_class": None,
        "top_confidence": 0.0,
        "model_used": "hsv_fallback",
    }


def analyze_tongue(image_base64: str) -> dict:
    try:
        img_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return get_fallback_result()

        # Compute HSV stats regardless (SVM may consume these)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]
        center = hsv[h//4:3*h//4, w//4:3*w//4]
        avg_hue = float(np.mean(center[:, :, 0])) if center.size else 0.0
        avg_sat = float(np.mean(center[:, :, 1])) if center.size else 0.0

        model = get_model()
        if model is None:
            # Weights missing — fall back to HSV rules
            return _hsv_fallback(img)

        # Run detection
        results = model.predict(img, conf=CONF_THRESHOLD, verbose=False)

        detections = []
        if len(results) > 0:
            r = results[0]
            names = r.names  # dict: id -> class name
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = names.get(cls_id, f"class_{cls_id}")
                    detections.append({"class": cls_name, "confidence": safe_float(conf)})

        # Nothing detected → fall back to HSV
        if not detections:
            print("[yolo_model] No detections above threshold — using HSV fallback")
            return _hsv_fallback(img)

        # Pick highest-confidence detection
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        top = detections[0]
        top_class = top["class"]
        top_conf = top["confidence"]

        dosha_signal = CLASS_TO_DOSHA.get(top_class, "Vata")
        coating = CLASS_TO_COATING.get(top_class, "thin_pale")

        return {
            "coating": coating,
            "vein_score": safe_float(top_conf),
            "dosha_signal": dosha_signal,
            "avg_hue": safe_float(avg_hue),
            "avg_saturation": safe_float(avg_sat),
            "detected_features": detections,
            "top_class": top_class,
            "top_confidence": safe_float(top_conf),
            "model_used": "yolov8_tongue",
        }

    except Exception as e:
        print(f"[yolo_model] Tongue analysis failed: {e}")
        return get_fallback_result()


def get_fallback_result() -> dict:
    return {
        "coating": "thin_pale",
        "vein_score": 0.7,
        "dosha_signal": "Vata",
        "avg_hue": 0.0,
        "avg_saturation": 0.0,
        "detected_features": [],
        "top_class": None,
        "top_confidence": 0.0,
        "model_used": "error_fallback",
    }