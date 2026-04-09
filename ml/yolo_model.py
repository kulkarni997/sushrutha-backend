import cv2
import numpy as np
import base64
import math

_model = None

def safe_float(val):
    f = float(val)
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return round(f, 4)

def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolov10n.pt")
    return _model

def analyze_tongue(image_base64: str) -> dict:
    try:
        img_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return get_fallback_result()
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        h, w = img.shape[:2]
        center = hsv[h//4:3*h//4, w//4:3*w//4]
        
        if center.size == 0:
            return get_fallback_result()
        
        avg_hue = np.mean(center[:,:,0])
        avg_sat = np.mean(center[:,:,1])
        
        if avg_sat < 50:
            coating = "thick_white"
            dosha_signal = "Kapha"
            vein_score = 0.3
        elif avg_hue < 30 or avg_hue > 150:
            coating = "yellow_orange"
            dosha_signal = "Pitta"
            vein_score = 0.6
        else:
            coating = "thin_pale"
            dosha_signal = "Vata"
            vein_score = 0.7
            
        return {
            "coating": coating,
            "vein_score": safe_float(vein_score),
            "dosha_signal": dosha_signal,
            "avg_hue": safe_float(avg_hue),
            "avg_saturation": safe_float(avg_sat)
        }
        
    except Exception as e:
        print(f"Tongue analysis failed: {e}")
        return get_fallback_result()

def get_fallback_result() -> dict:
    return {
        "coating": "thin_pale",
        "vein_score": 0.7,
        "dosha_signal": "Vata"
    }