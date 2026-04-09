import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routes.auth import router as auth_router
from routes.messages import router as messages_router
from routes.notifications import router as notifications_router
from routes.doctor import router as doctor_router
from routes.guest import router as guest_router
from routes.clinics import router as clinics_router
from routes.history import router as history_router
from routes.diagnose import router as diagnose_router
from routes.vision import router as vision_router
from routes.voice import router as voice_router
from routes.pulse import router as pulse_router
from routes.recipe import router as recipe_router
from routes.forecast import router as forecast_router

logger = logging.getLogger(__name__)

app = FastAPI(title="Sushrutha AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("FRONTEND_URL", "http://localhost:5173")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(messages_router, tags=["messages"])
app.include_router(notifications_router, tags=["notifications"])
app.include_router(doctor_router, prefix="/doctor", tags=["doctor"])
app.include_router(guest_router, prefix="/guest", tags=["guest"])
app.include_router(clinics_router, tags=["clinics"])
app.include_router(history_router, tags=["history"])
app.include_router(diagnose_router, tags=["diagnose"])
app.include_router(vision_router, tags=["vision"])
app.include_router(voice_router, tags=["voice"])
app.include_router(pulse_router, tags=["pulse"])
app.include_router(recipe_router, tags=["recipe"])
app.include_router(forecast_router, tags=["forecast"])

@app.get("/")
async def root():
    return {"message": "Sushrutha AI API", "status": "running", "version": "1.0.0"}

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status_code": 500},
    )