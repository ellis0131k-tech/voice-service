"""
Whisper transcription API — Cyclone (RTX 5070)
Model: large-v3 on CUDA
"""

import io
import time
import tempfile
import os
from contextlib import asynccontextmanager

import torch
import whisper
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3")
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[whisper] Loading {MODEL_NAME} on {device}...")
    model = whisper.load_model(MODEL_NAME, device=device)
    print(f"[whisper] Ready")
    yield
    model = None


app = FastAPI(title="Whisper API", lifespan=lifespan)


@app.get("/health")
def health():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await audio.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # Write to temp file (whisper needs a path)
    suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        start = time.time()
        result = whisper.transcribe(model, tmp_path, fp16=torch.cuda.is_available())
        elapsed = round(time.time() - start, 2)
        return JSONResponse({
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "duration": elapsed,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)
