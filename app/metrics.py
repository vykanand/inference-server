import os
import re
import subprocess
import time

import psutil


def _run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=5, creationflags=0x08000000)
        return p.stdout
    except Exception:
        return ""


def gpu_info():
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 6:
            try:
                gpus.append({
                    "name": parts[0],
                    "vram_total_mb": int(float(parts[1])),
                    "vram_used_mb": int(float(parts[2])),
                    "vram_free_mb": int(float(parts[3])),
                    "util_pct": int(float(parts[4])),
                    "temp_c": int(float(parts[5])),
                })
            except ValueError:
                continue
    return gpus


def ram_info():
    vm = psutil.virtual_memory()
    return {
        "total_gb": round(vm.total / 2**30, 1),
        "used_gb": round(vm.used / 2**30, 1),
        "free_gb": round(vm.available / 2**30, 1),
        "used_pct": vm.percent,
    }


def total_vram():
    gpus = gpu_info()
    if not gpus:
        return 0, 0
    t = sum(g["vram_total_mb"] for g in gpus)
    f = sum(g["vram_free_mb"] for g in gpus)
    return t, f


def gpu_pid_memory():
    """Map {pid: vram_mb} for every process currently holding GPU memory."""
    out = {}
    out_lines = _run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"])
    for line in out_lines.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[1].isdigit():
            try:
                out[int(parts[0])] = int(parts[1])
            except ValueError:
                continue
    return out


def process_ram_mb(pid):
    try:
        p = psutil.Process(pid)
        return p.memory_info().rss / 2**20
    except Exception:
        return None


def hardware():
    return {"gpus": gpu_info(), "ram": ram_info(), "cpu_count": os.cpu_count() or 1}
