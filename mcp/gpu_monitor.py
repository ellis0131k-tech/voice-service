"""nvidia-smi wrapper for GPU monitoring."""

import asyncio
import json
import subprocess


async def _run_smi(*args: str) -> str:
    """Run nvidia-smi with the given args and return stdout."""
    proc = await asyncio.create_subprocess_exec(
        "nvidia-smi", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {stderr.decode().strip()}")
    return stdout.decode().strip()


async def gpu_info() -> dict:
    """VRAM, utilization, temperature, power for each GPU."""
    csv = await _run_smi(
        "--query-gpu=index,name,memory.total,memory.used,memory.free,"
        "utilization.gpu,temperature.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    )
    gpus = []
    for line in csv.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9:
            continue
        gpus.append({
            "index": int(parts[0]),
            "name": parts[1],
            "vram_total_mb": int(parts[2]),
            "vram_used_mb": int(parts[3]),
            "vram_free_mb": int(parts[4]),
            "utilization_pct": int(parts[5]),
            "temperature_c": int(parts[6]),
            "power_draw_w": float(parts[7]),
            "power_limit_w": float(parts[8]),
        })
    return {"gpus": gpus}


async def gpu_process_list() -> dict:
    """Processes currently using GPU resources."""
    csv = await _run_smi(
        "--query-compute-apps=pid,name,gpu_uuid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    )
    processes = []
    for line in csv.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        processes.append({
            "pid": int(parts[0]),
            "process_name": parts[1],
            "gpu_uuid": parts[2],
            "vram_used_mb": int(parts[3]),
        })
    return {"processes": processes}
