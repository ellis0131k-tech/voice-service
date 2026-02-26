"""
MCP Voice Service Control Panel
================================
Streamable-HTTP MCP server on port 8000 providing 12 tools for managing
the Whisper STT and Qwen3 TTS FastAPI services.
"""

from mcp.server.fastmcp import FastMCP

from process_manager import SERVICE_CONFIG
import process_manager
import service_proxy
import gpu_monitor

mcp = FastMCP(
    "Voice Service Control Panel",
    host="0.0.0.0",
    port=8000,
)

# ---------------------------------------------------------------------------
# Service Management
# ---------------------------------------------------------------------------

@mcp.tool()
async def service_start(service: str) -> dict:
    """Start a voice service subprocess (whisper or tts)."""
    return process_manager.start(service)


@mcp.tool()
async def service_stop(service: str) -> dict:
    """Gracefully stop a voice service (CTRL_BREAK, force-kill after 10s)."""
    return await process_manager.stop(service)


@mcp.tool()
async def service_restart(service: str) -> dict:
    """Restart a voice service (stop then start)."""
    return await process_manager.restart(service)


@mcp.tool()
async def service_status(service: str) -> dict:
    """Get status of a voice service: PID, running state, port, uptime."""
    return process_manager.status(service)


@mcp.tool()
async def service_health(service: str) -> dict:
    """HTTP health check against a running service's /health endpoint."""
    try:
        return await service_proxy.health(service)
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}


@mcp.tool()
async def services_overview() -> dict:
    """Combined status, health, and GPU info for all services."""
    results = {}
    for name in SERVICE_CONFIG:
        status = process_manager.status(name)
        try:
            health = await service_proxy.health(name)
        except Exception as e:
            health = {"status": "unreachable", "error": str(e)}
        results[name] = {**status, "health": health}
    try:
        gpu = await gpu_monitor.gpu_info()
    except Exception as e:
        gpu = {"error": str(e)}
    return {"services": results, "gpu": gpu}


# ---------------------------------------------------------------------------
# API Actions
# ---------------------------------------------------------------------------

@mcp.tool()
async def transcribe_audio(audio_base64: str, filename: str = "audio.wav") -> dict:
    """Transcribe audio via Whisper. Audio must be base64-encoded."""
    return await service_proxy.transcribe(audio_base64, filename)


@mcp.tool()
async def synthesize_speech(
    text: str,
    speaker: str = "Ryan",
    language: str = "English",
    instruct: str = "",
) -> dict:
    """Synthesize speech via Qwen3 TTS. Returns base64-encoded WAV."""
    return await service_proxy.synthesize(text, speaker, language, instruct)


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

@mcp.tool()
async def view_logs(service: str, lines: int = 50) -> dict:
    """View last N lines of a service's stdout log buffer (max 500)."""
    return process_manager.view_logs(service, lines)


@mcp.tool()
async def gpu_info() -> dict:
    """GPU VRAM, utilization, temperature, and power via nvidia-smi."""
    return await gpu_monitor.gpu_info()


@mcp.tool()
async def gpu_process_list() -> dict:
    """List processes currently using GPU resources."""
    return await gpu_monitor.gpu_process_list()


@mcp.tool()
async def model_info() -> dict:
    """Static metadata about the deployed voice models."""
    return {
        "whisper": {
            "model": "openai/whisper large-v3",
            "task": "speech-to-text",
            "port": 8100,
            "endpoint": "/transcribe",
            "input": "audio file (multipart upload)",
            "output": "JSON with text, language, duration",
        },
        "tts": {
            "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "task": "text-to-speech",
            "port": 8200,
            "endpoint": "/synthesize",
            "input": "JSON with text, speaker, language, instruct",
            "output": "WAV audio",
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
