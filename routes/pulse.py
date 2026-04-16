from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from db.supabase_client import supabase
from typing import Set
from collections import deque
import json

router = APIRouter()

# Track all connected clients (ESP32 + frontend listeners)
connected_clients: Set[WebSocket] = set()

# Buffer last 30 seconds of readings per scan (for ML CNN later)
pulse_buffers: dict = {}  # {scan_id: deque of readings}

@router.websocket("/ws/pulse")
async def pulse_websocket(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"Client connected. Total: {len(connected_clients)}")

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)

            bpm = float(payload.get("bpm", 0))
            spo2 = float(payload.get("spo2", 0))
            scan_id = payload.get("scan_id", "")

            # Save to Supabase
            try:
                if scan_id and bpm > 0:
                    supabase.table("pulse_readings").insert({
                        "scan_id": scan_id,
                        "bpm": round(bpm, 2),
                        "spo2": round(spo2, 2),
                        "timestamp": "now()"
                    }).execute()
            except Exception as db_error:
                print(f"DB insert skipped: {db_error}")

            # Buffer last 30 seconds (for ML CNN)
            if scan_id:
                if scan_id not in pulse_buffers:
                    pulse_buffers[scan_id] = deque(maxlen=30)
                pulse_buffers[scan_id].append({"bpm": bpm, "spo2": spo2})

            # Broadcast to ALL connected clients (frontend listeners)
            broadcast_msg = json.dumps({
                "type": "pulse",
                "bpm": round(bpm, 2),
                "spo2": round(spo2, 2),
                "timestamp": payload.get("timestamp", 0),
            })
            disconnected = []
            for client in connected_clients:
                try:
                    await client.send_text(broadcast_msg)
                except Exception:
                    disconnected.append(client)
            for c in disconnected:
                connected_clients.discard(c)

    except WebSocketDisconnect:
        connected_clients.discard(websocket)
        print(f"Client disconnected. Total: {len(connected_clients)}")
    except Exception as e:
        connected_clients.discard(websocket)
        print(f"WebSocket error: {e}")


# Helper function for ML pipeline to fetch buffered readings
def get_pulse_buffer(scan_id: str) -> list:
    return list(pulse_buffers.get(scan_id, []))