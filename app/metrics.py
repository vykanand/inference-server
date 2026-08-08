import os
import re
import subprocess
import time

import psutil

RESERVE_MB = 320


def _run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=6, creationflags=0x08000000)
        return p.stdout
    except Exception:
        return ""


def _wmi_video():
    """Fallback so Intel/AMD-sans-driver-tools still report a GPU name."""
    try:
        out = _run(["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_VideoController | ForEach-Object { '{}|{}'.format($_.Name,$_.AdapterRAM) })"])
        gpus = []
        for line in out.strip().splitlines():
            parts = line.strip().split("|")
            if not parts or not parts[0]:
                continue
            total = int(parts[1]) if len(parts) > 1 and parts[1] and parts[1].isdigit() else 0
            if total and total < 128 * 1024:
                continue
            gpus.append({"name": parts[0], "vram_total_mb": total,
                         "vram_used_mb": 0, "vram_free_mb": total,
                         "util_pct": 0, "temp_c": None, "provider": "fallback"})
        return gpus
    except Exception:
        return []


def _wmic_video():
    try:
        out = _run(["wmic", "path", "win32_VideoController", "get",
                    "name,AdapterRAM", "/format:list"])
        name = None
        total = 0
        gpus = []
        raw = re.search(r"name=([^\r\n]*)", out, re.I)
        ram = re.search(r"AdapterRAM=(\d+)", out)
        if raw:
            name = raw.group(1).strip()
            if ram:
                total = int(int(ram.group(1)) / 1024 / 1024)
            gpus.append({"name": name, "vram_total_mb": total, "vram_used_mb": 0,
                         "vram_free_mb": total, "util_pct": 0, "temp_c": None,
                         "provider": "fallback"})
        return gpus
    except Exception:
        return []


def _nvidia_gpus():
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
                    "provider": "nvidia",
                })
            except ValueError:
                continue
    return gpus


def _amd_rocm_gpus():
    """AMD via ROCm rocm-smi (Linux / ROCm Windows)."""
    try:
        ver = _run(["rocm-smi", "--showdriverversion"])
        if "roc" not in ver.lower() and "amd" not in ver.lower():
            return []
    except Exception:
        return []
    info = _run(["rocm-smi", "--showmeminfo", "vram", "--json"])
    names = _run(["rocm-smi", "--showproductname", "--csv", "--unique"])
    gpus = []
    try:
        import json as _json
        data = _json.loads(info or "{}")
    except Exception:
        data = {}
    for key in sorted(data.keys()):
        if not isinstance(key, str) or not key.startswith("card"):
            continue
        try:
            total = int(data[key].get("vram_total", 0))
            used = int(data[key].get("vram_used", 0))
            free = max(0, total - used)
            name = "AMD GPU"
            m = re.search(rf"{re.escape(key)},([^,\r\n]+)", names)
            if m:
                name = m.group(1).strip()
            gpus.append({"name": name, "vram_total_mb": total, "vram_used_mb": used,
                         "vram_free_mb": free, "util_pct": 0, "temp_c": None,
                         "provider": "amd"})
        except Exception:
            continue
    return gpus


def gpu_info():
    """Real-time GPU list. NVIDIA first, then AMD ROCm, then WMI fallback."""
    gpus = _nvidia_gpus()
    if gpus:
        return gpus
    gpus = _amd_rocm_gpus()
    if gpus:
        return gpus
    gpus = _wmi_video()
    if gpus:
        return gpus
    return _wmi_video() or []


def active_gpu():
    """The GPU we will actually target: most free VRAM, fallback most VRAM."""
    gpus = gpu_info()
    if not gpus:
        return None
    return max(gpus, key=lambda g: (g.get("vram_free_mb") or 0, g.get("vram_total_mb") or 0))


def total_vram():
    """Returns (total, free) VRAM across the active-adopted GPUs."""
    gpu = active_gpu()
    if not gpu:
        return 0, 0
    return gpu.get("vram_total_mb", 0), gpu.get("vram_free_mb", 0)


def total_vram_all():
    gpus = gpu_info()
    if not gpus:
        return 0, 0
    return sum(g.get("vram_total_mb", 0) for g in gpus), sum(g.get("vram_free_mb", 0) for g in gpus)


def ram_info():
    vm = psutil.virtual_memory()
    return {
        "total_gb": round(vm.total / 2 ** 30, 1),
        "used_gb": round(vm.used / 2 ** 30, 1),
        "free_gb": round(vm.available / 2 ** 30, 1),
        "used_pct": vm.percent,
    }


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
        return p.memory_info().rss / 2 ** 20
    except Exception:
        return None


def hardware():
    gpu = active_gpu()
    gpus = gpu_info()
    return {"gpus": gpus, "active_gpu": gpu, "ram": ram_info(),
            "cpu_count": os.cpu_count() or 1,
            "backend": _backend()}


def _backend():
    """Which llama build is preferred: cuda / vulkan / cpu-only."""
    from .engine import BIN_DIR
    import os as _os
    d = BIN_DIR
    if _os.path.isfile(_os.path.join(d, "ggml-cuda.dll")):
        return "cuda"
    if _os.path.isfile(_os.path.join(d, "ggml-vulkan.dll")):
        return "vulkan"
    return "cpu"


def kill_llama_processes():
    """Terminate every llama-server process to free GPU + RAM. Returns freed info."""
    freed = []
    for p in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            name = p.info.get("name")
            if name and "llama-server" in name.lower():
                pid = p.info["pid"]
                mem = p.memory_info()
                ram = (mem.rss / 2**20) if mem else 0
                p.terminate()
                try:
                    p.wait(timeout=3)
                except (psutil.TimeoutExpired, Exception):
                    p.kill()
                freed.append({"pid": pid, "ram_mb": round(ram, 1)})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return freed