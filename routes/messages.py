from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import uuid
from db.supabase_client import supabase
from auth.jwt_handler import require_patient, require_doctor

router = APIRouter()

def get_thread_id(sender_id: str, receiver_id: str, scan_id: str) -> str:
    ids = sorted([sender_id, receiver_id])
    raw = f"{ids[0]}-{ids[1]}-{scan_id}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw))

class MessageCreate(BaseModel):
    receiver_id: str
    scan_id: str
    body: str

@router.post("/messages")
async def send_message(payload: MessageCreate, user: dict = Depends(require_patient)):
    thread_id = get_thread_id(user["sub"], payload.receiver_id, payload.scan_id)
    msg = supabase.table("messages").insert({
        "thread_id": thread_id,
        "sender_id": user["sub"],
        "receiver_id": payload.receiver_id,
        "scan_id": payload.scan_id,
        "body": payload.body,
        "read": False
    }).execute()
    supabase.table("notifications").insert({
        "user_id": payload.receiver_id,
        "type": "new_message",
        "reference_id": msg.data[0]["id"],
        "seen": False
    }).execute()
    return msg.data[0]

@router.get("/messages/{thread_id}")
async def get_messages(thread_id: str, user: dict = Depends(require_patient)):
    msgs = supabase.table("messages")\
        .select("*")\
        .eq("thread_id", thread_id)\
        .order("sent_at", desc=False)\
        .execute()
    if not msgs.data:
        return []
    first = msgs.data[0]
    if user["sub"] not in [first["sender_id"], first["receiver_id"]]:
        raise HTTPException(403, "Access denied")
    return msgs.data

@router.post("/doctor/messages")
async def send_message_doctor(payload: MessageCreate, user: dict = Depends(require_doctor)):
    thread_id = get_thread_id(user["sub"], payload.receiver_id, payload.scan_id)
    msg = supabase.table("messages").insert({
        "thread_id": thread_id,
        "sender_id": user["sub"],
        "receiver_id": payload.receiver_id,
        "scan_id": payload.scan_id,
        "body": payload.body,
        "read": False
    }).execute()
    supabase.table("notifications").insert({
        "user_id": payload.receiver_id,
        "type": "new_message",
        "reference_id": msg.data[0]["id"],
        "seen": False
    }).execute()
    return msg.data[0]