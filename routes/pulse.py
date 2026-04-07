from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from db.supabase_client import supabase
import json

router = APIRouter()

@router.websocket("/ws/pulse")
async def pulse_websocket(websocket: WebSocket):
    await websocket.accept()
    scan_id = None
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            scan_id = payload.get("scan_id")
            bpm = payload.get("bpm")
            spo2 = payload.get("spo2")
            timestamp = payload.get("timestamp")

            if scan_id and bpm:
                supabase.table("pulse_readings").insert({
                    "scan_id": scan_id,
                    "bpm": bpm,
                    "spo2": spo2,
                    "timestamp": timestamp
                }).execute()

            await websocket.send_text(json.dumps({
                "bpm": bpm,
                "spo2": spo2,
                "status": "ok"
            }))

    except WebSocketDisconnect:
        pass