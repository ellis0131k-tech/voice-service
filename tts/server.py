"""
Qwen3-TTS synthesis API — Cyclone (RTX 5070 Ti)
Model: Qwen3-TTS-12Hz-1.7B-CustomVoice on CUDA
"""

import io
import time
from contextlib import asynccontextmanager

import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
model = None


class SynthesizeRequest(BaseModel):
    text: str
    speaker: str = "Ryan"
    language: str = "English"
    instruct: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    from qwen_tts import Qwen3TTSModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[tts] Loading {MODEL_ID} on {device}...")

    # Try flash_attention_2, fall back to default (e.g. on Windows)
    try:
        model = Qwen3TTSModel.from_pretrained(
            MODEL_ID,
            device_map=f"{device}:0" if device == "cuda" else device,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        print("[tts] Using flash_attention_2")
    except Exception:
        model = Qwen3TTSModel.from_pretrained(
            MODEL_ID,
            device_map=f"{device}:0" if device == "cuda" else device,
            dtype=torch.bfloat16,
        )
        print("[tts] Using default attention")

    print("[tts] Ready -- http://0.0.0.0:8200")
    yield
    model = None


app = FastAPI(title="Qwen3-TTS API", lifespan=lifespan)


@app.get("/health")
def health():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        start = time.time()
        wavs, sr = model.generate_custom_voice(
            text=req.text,
            speaker=req.speaker,
            language=req.language,
            instruct=req.instruct if req.instruct else "",
        )
        elapsed = round(time.time() - start, 3)

        buf = io.BytesIO()
        sf.write(buf, wavs[0], sr, format="WAV")
        buf.seek(0)

        return Response(
            content=buf.read(),
            media_type="audio/wav",
            headers={"X-Duration": str(elapsed)},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
