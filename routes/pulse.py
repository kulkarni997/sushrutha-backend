from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from db.supabase_client import supabase
import json

router = APIRouter()

@router.websocket("/ws/pulse")
async def pulse_websocket(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)

            bpm = payload.get("bpm", 0)
            spo2 = payload.get("spo2", 0)
            scan_id = payload.get("scan_id", "")

            # Save to Supabase (only if valid scan_id)
            try:
                if scan_id and bpm > 0:
                    supabase.table("pulse_readings").insert({
                        "scan_id": scan_id,
                        "bpm": round(float(bpm), 2),
                        "spo2": round(float(spo2), 2),
                        "timestamp": "now()"
                    }).execute()
            except Exception as db_error:
                print(f"DB insert skipped: {db_error}")

            # Always send ack back
            await websocket.send_json({
                "status": "ok",
                "bpm": bpm,
                "spo2": spo2
            })

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")