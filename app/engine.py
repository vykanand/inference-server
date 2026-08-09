import os
import re
import socket
import subprocess
import threading
import time
from collections import deque

import psutil
import requests

LLAMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llama-server")
CUDA_CANDIDATES = [r"C:\dev\clearml\tools\llama-cuda", LLAMA_DIR]
PIDS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine_pids.json")
import json

# Port the inference-server (FastAPI) proxy listens on. External tools such as
# opencode / Claude Code / Cline must point here (NOT the raw llama-server port)
# so context-window overflow is auto-truncated. start.py sets PORT env before launch.
SERVER_PORT = int(os.environ.get("PORT", "8080"))


def read_pids():
    try:
        with open(PIDS_FILE, "r", encoding="utf-8") as f:
            return {int(k): int(v) for k, v in json.load(f).items()}
    except Exception:
        return {}


def write_pids(pids):
    try:
        with open(PIDS_FILE, "w", encoding="utf-8") as f:
            json.dump(pids, f)
    except Exception:
        pass


def find_bin_dir():
    env = os.environ.get("LLAMA_BIN")
    if env and os.path.isfile(os.path.join(env, "llama-server.exe")):
        return env
    for d in CUDA_CANDIDATES:
        if os.path.isfile(os.path.join(d, "llama-server.exe")) and os.path.isfile(os.path.join(d, "ggml-cuda.dll")):
            return d
    for d in CUDA_CANDIDATES:
        if os.path.isfile(os.path.join(d, "llama-server.exe")):
            return d
    return LLAMA_DIR


BIN_DIR = find_bin_dir()
LLAMA_SERVER = os.path.join(BIN_DIR, "llama-server.exe")


def free_port(start=8081):
    p = start
    while p < 9000:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                p += 1
    return 8081


class Engine:
    def __init__(self, base_dir, port=None):
        self.base_dir = base_dir
        self.port = port or free_port()
        self.proc = None
        self.log_thread = None
        self.logs = deque(maxlen=400)
        self.params = {}
        self.model_path = None
        self.model_name = None
        self.ready = False
        self.info = {}
        # When True the watchdog keeps this engine alive (auto-restart on crash).
        self.keep_alive = False
        self.baseline_vram = None
        self.lock = threading.Lock()
        self.start_time = None

    @property
    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def base_url(self):
        if self.port:
            return f"http://127.0.0.1:{self.port}"
        url = os.environ.get("LLAMA_BASE_URL")
        return url or f"http://127.0.0.1:{self.port or free_port()}"

    def connect(self):
        """Details external tools (opencode, claude-code, Cline/Roo, etc.) need.

        IMPORTANT: external tools MUST talk to the inference-server proxy
        (port SERVER_PORT), NOT the raw llama-server port. The proxy adds
        context-window truncation + auto engine selection so large opencode
        conversations can never hit llama-server's hard "exceeds context size"
        400. Pointing tools at the raw engine port is the classic cause of that
        error.
        """
        base = f"http://127.0.0.1:{SERVER_PORT}"
        engine_base = self.base_url()
        return {
            "base_url": base,
            "api_url": f"{base}/v1/chat/completions",
            "models_url": f"{base}/v1/models",
            "health_url": f"{engine_base}/health",
            "api_style": "OpenAI-compatible",
            "headers": {"Content-Type": "application/json",
                        "Authorization": "Bearer local"},
            "curl": f"curl -X POST {base}/v1/chat/completions -H 'Content-Type: application/json' "
                    f"-d '{{\"model\":\"local\",\"messages\":[{{\"role\":\"user\",\"content\":\"ping\"}}}}'",
            "opencode": {"provider": "openai", "base_url": f"{base}/v1"},
            "claude_code": {"base_url": f"{base}/v1", "model": "local"},
            "note": "Route through the inference server (port %d), not the raw llama-server "
                    "port, so context overflow is auto-truncated instead of erroring." % SERVER_PORT,
        }

    def _tail(self, line):
        self.logs.append(line)
        self._parse(line)

    def _parse(self, line):
        m = re.search(r"offloaded (\d+)/(\d+) layers to GPU", line)
        if m:
            self.info["layers_offloaded"] = int(m.group(1))
            self.info["layers_total"] = int(m.group(2))
        m = re.search(r"total VRAM used:\s*([\d.]+)\s*MiB", line)
        if m:
            self.info["vram_used_mib"] = float(m.group(1))
        m = re.search(r"llama_model_load: n_layer\s*=\s*(\d+)", line)
        if m and "layers_total" not in self.info:
            self.info["layers_total"] = int(m.group(1))
        m = re.search(r"llama_kv_cache_init:\s+.*VRAM", line)
        if m:
            self.info.setdefault("kv_lines", []).append(line.strip())
        m = re.search(r"(model size|model size =)\s*([\d.]+)\s*(B|GiB)", line)
        if m and "model_size_gb" not in self.info:
            self.info["model_size_gb"] = float(m.group(2))
        m = re.search(r"HTTP server is listening", line)
        if m:
            self.ready = True

    def _reader(self):
        for line in self.proc.stdout:
            self._tail(line.rstrip("\r\n"))
        if self.proc.poll() is None:
            try:
                self.proc.wait(timeout=3)
            except Exception:
                pass
        self._tail("--- llama-server process exited (code %s) ---" % self.proc.poll())

    def build_cmd(self, model_path, params):
        p = params
        cmd = [LLAMA_SERVER, "-m", model_path, "--host", "127.0.0.1", "--port", str(self.port),
               "--metrics"]
        if p.get("n_gpu_layers", 0) != 0:
            cmd += ["-ngl", str(p.get("n_gpu_layers", 0))]
        if p.get("ctx"):
            cmd += ["-c", str(p["ctx"])]
        if p.get("batch"):
            cmd += ["-b", str(p["batch"])]
        if p.get("ubatch"):
            cmd += ["-ub", str(p["ubatch"])]
        if p.get("parallel", 1) > 1:
            cmd += ["-np", str(p["parallel"])]
        if p.get("threads"):
            cmd += ["-t", str(p["threads"])]
        if p.get("threads_batch"):
            cmd += ["-tb", str(p["threads_batch"])]
        if p.get("flash_attn"):
            cmd += ["--flash-attn", "on"]
        if p.get("kv_type"):
            cmd += ["-ctk", p["kv_type"], "-ctv", p["kv_type"]]
        if not p.get("mmap", True):
            cmd += ["--no-mmap"]
        if p.get("mlock"):
            cmd += ["--mlock"]
        if p.get("cache_reuse"):
            cmd += ["--cache-reuse", "1"]
        if p.get("temp") is not None:
            cmd += ["--temp", str(p["temp"])]
        if p.get("top_k") is not None:
            cmd += ["--top-k", str(p["top_k"])]
        if p.get("top_p") is not None:
            cmd += ["--top-p", str(p["top_p"])]
        if p.get("min_p") is not None:
            cmd += ["--min-p", str(p["min_p"])]
        if p.get("repeat_penalty") is not None:
            cmd += ["--repeat-penalty", str(p["repeat_penalty"])]
        return cmd

    def _track(self):
        if self.proc:
            pids = read_pids()
            pids[self.port] = self.proc.pid
            write_pids(pids)

    def _untrack(self):
        pids = read_pids()
        if self.port in pids:
            del pids[self.port]
            write_pids(pids)

    def start(self, model_path, params, model_name=None):
        with self.lock:
            self.stop()
            self.info = {}
            self.ready = False
            self.logs.clear()
            self.model_path = model_path
            self.model_name = model_name or os.path.basename(model_path)
            self.params = dict(params)
            self.keep_alive = True
            self.start_time = time.time()
            from .metrics import total_vram
            _, self.baseline_vram = total_vram()
            cmd = self.build_cmd(model_path, params)
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, encoding="utf-8", errors="replace",
                                         bufsize=1, creationflags=0x08000000, cwd=BIN_DIR)
            self.log_thread = threading.Thread(target=self._reader, daemon=True)
            self.log_thread.start()
            ok = self.wait_ready(90)
            if ok:
                time.sleep(1.0)
                self._record_allocation()
            self._track()
        return self.status()

    def _record_allocation(self):
        from .metrics import gpu_pid_memory, total_vram
        used = 0
        try:
            if self.proc:
                gmap = gpu_pid_memory()
                pid = self.proc.pid
                kids = {c.pid for c in self.proc.children(recursive=True)}
                used = sum(int(v) for k, v in gmap.items()
                           if (k == pid or k in kids) and str(v).isdigit())
        except Exception:
            used = 0
        if not used:
            try:
                _, free_now = total_vram()
                used = max(0, (self.baseline_vram or 0) - free_now)
            except Exception:
                used = 0
        self.info["vram_used_mib"] = used
        self.info["vram_allocated_mb"] = used

    def stop(self):
        self.keep_alive = False
        if self.proc is not None:
            if self.proc.poll() is None:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=8)
                except Exception:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
            self.proc = None
        self.ready = False
        self._untrack()

    def wait_ready(self, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.running:
                return False
            try:
                r = requests.get(self.base_url() + "/health", timeout=1)
                if r.status_code == 200:
                    self.ready = True
                    return True
            except Exception:
                pass
            time.sleep(0.3)
        return False

    def _req(self, path, timeout=3):
        try:
            return requests.get(self.base_url() + path, timeout=timeout).json()
        except Exception:
            return None

    def live(self):
        slots = self._req("/slots")
        metrics = self._req("/metrics")
        out = {"running": self.running, "ready": self.ready,
               "uptime_s": (time.time() - self.start_time) if self.start_time else 0}
        if slots:
            out["slots"] = slots
            try:
                out["kv_usage"] = max((s.get("n_past", 0) / max(s.get("n_ctx", 1), 1) for s in slots), default=0)
            except Exception:
                out["kv_usage"] = 0
        if metrics and isinstance(metrics, str):
            parsed = {}
            for line in metrics.splitlines():
                if not line.startswith("llamacpp:"):
                    continue
                m = re.match(r"llamacpp:(\w+)[^{]*\{[^}]*\}\s+([\d.eE+-]+)", line)
                if not m:
                    m = re.match(r"llamacpp:(\w+)\s+([\d.eE+-]+)", line)
                if m:
                    try:
                        parsed.setdefault(m.group(1), []).append(float(m.group(2)))
                    except ValueError:
                        pass
            groups = {
                "prompt_tokens_seconds": "prompt_speed",
                "predicted_tokens_seconds": "predicted_speed",
                "prompt_tokens_total": "prompt_tokens",
                "tokens_predicted_total": "predicted_tokens",
                "kv_cache_usage_ratio": "kv_usage",
                "requests_processing": "requests_processing",
                "requests_deferred": "requests_deferred",
                "n_busy_slots_per_decode": "busy_slots",
            }
            mmm = {}
            for raw, nice in groups.items():
                if parsed.get(raw):
                    fn = max if raw.endswith("_seconds") else sum
                    v = fn(parsed[raw])
                    mmm[nice] = round(v, 4) if isinstance(v, float) else v
            out["metrics"] = mmm
        from .metrics import gpu_info, process_ram_mb
        gpus = gpu_info()
        out["gpus"] = gpus
        pid = self.proc.pid if self.proc else None
        out["pid"] = pid
        out["vram_used_engine_mb"] = self.info.get("vram_allocated_mb", 0)
        out["ram_used_engine_mb"] = process_ram_mb(pid) if pid else None
        total = self.info.get("layers_total")
        req = self.params.get("n_gpu_layers", 0)
        if self.info.get("layers_offloaded") is None and total:
            self.info["layers_offloaded"] = min(req, total) if req else 0
        return out

    def status(self):
        info = dict(self.info)
        info.update({
            "port": self.port,
            "running": self.running,
            "ready": self.ready,
            "model_path": self.model_path,
            "model_name": self.model_name,
            "params": dict(self.params),
            "baseline_vram": self.baseline_vram,
        })
        info.update(self.live())
        info.update(self._vram_split())
        info["connect"] = self.connect()
        return info

    def _vram_split(self):
        out = {"est_vram_mb": 0, "gpu_weights_mb": 0, "ram_weights_mb": 0, "model_size_mb": 0}
        try:
            if not self.model_path or not os.path.isfile(self.model_path):
                return out
            size_mb = os.path.getsize(self.model_path) / 2**20
            out["model_size_mb"] = round(size_mb, 1)
            total = self.info.get("layers_total")
            if not total:
                total = self._meta_layers()
            ngl = self.params.get("n_gpu_layers", 0)
            ratio = min(1.0, ngl / total) if total else (1.0 if ngl >= 999 else 0.0)
            weights_gpu = size_mb * ratio
            ctx = self.params.get("ctx") or 4096
            kv_mb = (ctx / 1024) * total * 0.35
            pre = 48
            out["gpu_weights_mb"] = round(weights_gpu, 1)
            out["ram_weights_mb"] = round(size_mb - weights_gpu, 1)
            out["est_vram_mb"] = round(weights_gpu + kv_mb + pre, 1)
            out["gpu_fit_pct"] = round(ratio * 100, 1)
        except Exception:
            pass
        return out

    def _meta_layers(self):
        try:
            from .gguf_meta import read_gguf_meta
            m = read_gguf_meta(self.model_path)
            return m.get("block_count") or 0
        except Exception:
            return 0


class EngineManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.engines = {}
        self._lock = threading.Lock()
        if not os.environ.get("INFERENCE_NO_CLEANUP"):
            self._cleanup_orphans()
        self._watchdog_thread = None

    def _cleanup_orphans(self):
        # Kill registered orphan pids AND any llama-server still alive,
        # so restarting the manager starts with fully clean RAM/GPU.
        try:
            from .metrics import kill_llama_processes
            kill_llama_processes()
        except Exception:
            pass
        for port, pid in read_pids().items():
            try:
                p = psutil.Process(pid)
                if p.is_running() and p.name().lower().startswith("llama-server"):
                    p.terminate()
                    try:
                        p.wait(timeout=6)
                    except Exception:
                        p.kill()
            except Exception:
                try:
                    os.kill(int(pid), 9)
                except Exception:
                    pass
        write_pids({})

    def start_watchdog(self, interval=15):
        """Background thread that keeps every keep_alive engine running 24x7.

        If an engine's llama-server process dies (crash / OOM / VRAM eviction),
        it is transparently restarted with the same model + params so clients
        (opencode, Cline, Copilot, ...) never see a hard failure.
        """
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return

        def _watch():
            while True:
                time.sleep(interval)
                try:
                    with self._lock:
                        snapshot = list(self.engines.values())
                    for e in snapshot:
                        if getattr(e, "keep_alive", False) and not e.running:
                            try:
                                e.logs.append("--- watchdog: engine died, auto-restarting ---")
                                e.start(e.model_path, e.params, e.model_name)
                            except Exception as ex:
                                try:
                                    e.logs.append("--- watchdog restart failed: %s ---" % ex)
                                except Exception:
                                    pass
                except Exception:
                    pass

        t = threading.Thread(target=_watch, name="engine-watchdog", daemon=True)
        t.start()
        self._watchdog_thread = t

    def list(self):
        with self._lock:
            return [e.status() for e in self.engines.values()]

    def get(self, port=None):
        if port is None:
            with self._lock:
                if not self.engines:
                    return None
                port = sorted(self.engines.keys())[0]
        return self.engines.get(port)

    def create(self, port=None):
        with self._lock:
            eng = Engine(self.base_dir, port=port)
            self.engines[eng.port] = eng
            return eng

    def load(self, model_path, params, model_name=None, port=None):
        # If a port is specified, stop any engine on that port first
        if port is not None:
            self.stop(port)
        eng = self.create(port)
        st = eng.start(model_path, params, model_name)
        if not st["ready"]:
            self.stop(eng.port)
            raise RuntimeError("engine failed to start:\n" + "\n".join(list(eng.logs)[-15:]))
        return eng

    def stop(self, port):
        with self._lock:
            eng = self.engines.pop(port, None)
            if eng:
                eng.stop()
