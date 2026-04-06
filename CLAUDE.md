# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Sushrutha** is a medical AI backend built with FastAPI and Supabase. It integrates multiple ML pipelines (vision, audio, biosignal, forecasting) with a RAG system for clinical knowledge retrieval.

## Environment Setup

Copy `.env.example` to `.env` and fill in values:
- `SUPABASE_URL` / `SUPABASE_KEY` — Supabase project credentials
- `JWT_SECRET` — secret for signing/verifying JWTs
- `CLOUDINARY_URL` — media storage (optional)

Install dependencies and run:
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Architecture

### Entry Point
`main.py` — FastAPI app instance; registers all routers from `routes/`.

### Layers

| Layer | Path | Responsibility |
|-------|------|----------------|
| Routes | `routes/` | HTTP endpoints, request/response handling |
| Auth | `auth/jwt_handler.py` | JWT creation and verification |
| DB | `db/supabase_client.py` | Supabase client singleton |
| ML | `ml/` | Model inference wrappers |
| RAG | `rag/` | Vector ingestion, retrieval, generation |

### Routes
Each file in `routes/` maps to a domain:
- `auth.py` — registration, login, token refresh
- `diagnose.py` — AI-based symptom/image diagnosis
- `vision.py` — YOLO-based image analysis (`ml/yolo_model.py`)
- `voice.py` — Whisper-based audio transcription (`ml/whisper_model.py`)
- `pulse.py` — biosignal classification via CNN (`ml/pulse_cnn.py`)
- `forecast.py` — time-series health forecasting via Prophet (`ml/prophet_model.py`)
- `history.py` — patient consultation history
- `messages.py` — doctor-patient messaging
- `notifications.py` — push/in-app notifications
- `clinics.py` / `doctor.py` — clinic and doctor profiles
- `recipe.py` — medication/treatment recommendations
- `guest.py` — unauthenticated access endpoints

### ML Models (`ml/`)
- `yolo_model.py` — YOLO for medical image/object detection
- `whisper_model.py` — OpenAI Whisper for voice-to-text
- `pulse_cnn.py` — CNN for biosignal (pulse/ECG) classification
- `svm_ensemble.py` — SVM ensemble for structured clinical data
- `prophet_model.py` — Facebook Prophet for health metric forecasting

Model weights (`.pt`, `.pth`, `.pkl`) and FAISS indexes (`faiss_index/`) are gitignored and must be provided separately.

### RAG Pipeline (`rag/`)
- `ingest.py` — chunk and embed medical documents into FAISS
- `retriever.py` — semantic search over the FAISS index
- `generator.py` — LLM generation conditioned on retrieved context

### Training Notebooks (`training/`)
- `train_cnn.ipynb` — CNN pulse model training
- `train_yolo.ipynb` — YOLO fine-tuning

## Key Conventions

- All protected routes authenticate via JWT Bearer token (verified in `auth/jwt_handler.py`).
- Database access goes through the Supabase client in `db/supabase_client.py`; do not instantiate a second client.
- ML model files are loaded once at startup (module-level), not per-request.
- FAISS index lives at `faiss_index/` (gitignored); rebuild it by running `rag/ingest.py` after adding documents.
