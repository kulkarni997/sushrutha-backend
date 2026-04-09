
import asyncio
import websockets
import json
import random

async def mock_esp32():
    uri = "ws://127.0.0.1:8000/ws/pulse"
    
    print("Mock ESP32 connecting...")
    
    async with websockets.connect(uri) as websocket:
        print("Connected! Sending pulse data...")
        
        scan_id = "test-scan-123"
        
        while True:
            bpm = round(random.uniform(65, 85), 2)
            spo2 = round(random.uniform(96, 99), 2)
            
            payload = {
                "bpm": bpm,
                "spo2": spo2,
                "scan_id": scan_id
            }
            
            await websocket.send(json.dumps(payload))
            print(f"Sent → BPM: {bpm}, SpO2: {spo2}")
            
            response = await websocket.recv()
            print(f"Server ack: {response}")
            
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(mock_esp32())