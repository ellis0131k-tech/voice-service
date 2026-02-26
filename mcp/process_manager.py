"""Subprocess lifecycle manager for whisper & tts services on Windows."""

import asyncio
import signal
import subprocess
import time
from collections import deque
from pathlib import Path

# voice-service root (one level up from mcp/)
REPO_ROOT = Path(__file__).resolve().parent.parent

SERVICE_CONFIG = {
    "whisper": {
        "cwd": REPO_ROOT / "whisper",
        "uvicorn": REPO_ROOT / "whisper" / "venv" / "Scripts" / "uvicorn.exe",
        "app": "server:app",
        "port": 8100,
    },
    "tts": {
        "cwd": REPO_ROOT / "tts",
        "uvicorn": REPO_ROOT / "tts" / "venv" / "Scripts" / "uvicorn.exe",
        "app": "server:app",
        "port": 8200,
    },
}

LOG_BUFFER_SIZE = 500


class ManagedService:
    """Tracks a running subprocess and its log ring buffer."""

    def __init__(self, name: str):
        self.name = name
        self.process: subprocess.Popen | None = None
        self.logs: deque[str] = deque(maxlen=LOG_BUFFER_SIZE)
        self.started_at: float | None = None
        self._reader_task: asyncio.Task | None = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process else None

    @property
    def uptime_seconds(self) -> float | None:
        if self.started_at and self.running:
            return round(time.time() - self.started_at, 1)
        return None

    def status_dict(self) -> dict:
        cfg = SERVICE_CONFIG[self.name]
        return {
            "service": self.name,
            "running": self.running,
            "pid": self.pid,
            "port": cfg["port"],
            "uptime_seconds": self.uptime_seconds,
        }


# Global registry
_services: dict[str, ManagedService] = {}


def _get(name: str) -> ManagedService:
    if name not in SERVICE_CONFIG:
        raise ValueError(f"Unknown service: {name!r}. Must be 'whisper' or 'tts'.")
    if name not in _services:
        _services[name] = ManagedService(name)
    return _services[name]


async def _log_reader(svc: ManagedService):
    """Read stdout+stderr lines in a background task into the ring buffer."""
    proc = svc.process
    if proc is None or proc.stdout is None:
        return
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, proc.stdout.readline)
        if not line:
            break
        svc.logs.append(line.decode("utf-8", errors="replace").rstrip("\n"))


def start(name: str) -> dict:
    """Start a service subprocess. Returns status dict."""
    svc = _get(name)
    if svc.running:
        return {**svc.status_dict(), "message": "Already running"}

    cfg = SERVICE_CONFIG[name]
    uvicorn = str(cfg["uvicorn"])
    cmd = [uvicorn, cfg["app"], "--host", "0.0.0.0", "--port", str(cfg["port"])]

    svc.logs.clear()
    svc.process = subprocess.Popen(
        cmd,
        cwd=str(cfg["cwd"]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    svc.started_at = time.time()

    # Kick off async log reader
    loop = asyncio.get_running_loop()
    svc._reader_task = loop.create_task(_log_reader(svc))

    return {**svc.status_dict(), "message": "Started"}


async def stop(name: str) -> dict:
    """Graceful CTRL_BREAK, then force-kill after 10s."""
    svc = _get(name)
    if not svc.running:
        return {**svc.status_dict(), "message": "Not running"}

    proc = svc.process
    # Send CTRL_BREAK to the process group
    try:
        proc.send_signal(signal.CTRL_BREAK_EVENT)
    except OSError:
        pass

    # Wait up to 10s for graceful shutdown
    loop = asyncio.get_running_loop()
    try:
        await asyncio.wait_for(
            loop.run_in_executor(None, proc.wait), timeout=10.0
        )
    except asyncio.TimeoutError:
        proc.kill()
        await loop.run_in_executor(None, proc.wait)

    # Cancel log reader
    if svc._reader_task and not svc._reader_task.done():
        svc._reader_task.cancel()

    return {**svc.status_dict(), "message": "Stopped"}


async def restart(name: str) -> dict:
    """Stop then start."""
    await stop(name)
    return start(name)


def status(name: str) -> dict:
    """Return current status."""
    svc = _get(name)
    return svc.status_dict()


def view_logs(name: str, lines: int = 50) -> dict:
    """Return last N lines from the ring buffer."""
    svc = _get(name)
    lines = max(1, min(lines, LOG_BUFFER_SIZE))
    recent = list(svc.logs)[-lines:]
    return {
        "service": name,
        "lines_requested": lines,
        "lines_returned": len(recent),
        "logs": recent,
    }
