"""Async httpx proxy to the FastAPI services."""

import base64

import httpx

TIMEOUT = httpx.Timeout(120.0, connect=10.0)

SERVICE_URLS = {
    "whisper": "http://localhost:8100",
    "tts": "http://localhost:8200",
}


def _url(service: str) -> str:
    if service not in SERVICE_URLS:
        raise ValueError(f"Unknown service: {service!r}")
    return SERVICE_URLS[service]


async def health(service: str) -> dict:
    """GET /health on a service."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(f"{_url(service)}/health")
        resp.raise_for_status()
        return resp.json()


async def transcribe(audio_base64: str, filename: str = "audio.wav") -> dict:
    """POST /transcribe with multipart audio file. Returns transcription dict."""
    audio_bytes = base64.b64decode(audio_base64)
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(
            f"{_url('whisper')}/transcribe",
            files={"audio": (filename, audio_bytes)},
        )
        resp.raise_for_status()
        return resp.json()


async def synthesize(
    text: str,
    speaker: str = "Ryan",
    language: str = "English",
    instruct: str = "",
) -> dict:
    """POST /synthesize, returns base64-encoded WAV + metadata."""
    payload = {
        "text": text,
        "speaker": speaker,
        "language": language,
        "instruct": instruct,
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(f"{_url('tts')}/synthesize", json=payload)
        resp.raise_for_status()
        wav_b64 = base64.b64encode(resp.content).decode("ascii")
        return {
            "audio_base64": wav_b64,
            "format": "wav",
            "size_bytes": len(resp.content),
            "synthesis_duration": resp.headers.get("X-Duration"),
        }
