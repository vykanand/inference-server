import json
import os
import re
import threading
import time
import asyncio

import requests
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import defaults, hub, metrics, settings
from .configchat import build_system_prompt, chat_stream, free_models
from .engine import EngineManager
from .gguf_meta import file_gb, read_gguf_meta

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(BASE, "web")
app = FastAPI(title="llama.cpp Inference Server Manager")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

manager = EngineManager(BASE)
manager.start_watchdog()
app.mount("/static", StaticFiles(directory=WEB), name="static")

VERSION = "2.2.0"
_startup = time.time()

# ── lightweight inference log (opencode ↔ server) ──────────────────────
_inf_log = []
_log_lock = threading.Lock()

def _log(tag, **kw):
    entry = {"ts": time.time(), "tag": tag}
    entry.update(kw)
    with _log_lock:
        _inf_log.append(entry)
        if len(_inf_log) > 300:
            _inf_log[:] = _inf_log[-200:]

# ── cancellation token for client-disconnect propagation ────────────────
class CancellationToken:
    def __init__(self):
        self._evt = threading.Event()

    def cancel(self):
        self._evt.set()

    @property
    def is_cancelled(self):
        return self._evt.is_set()

# ── OpenAI-compliant error helpers ──────────────────────────────────────
def _err(message, errtype="server_error", code=None, param=None, retryable=False):
    return {"error": {"message": str(message), "type": errtype,
                       "code": code or errtype, "param": param, "retryable": retryable}}

def _sse_err(message, errtype="server_error"):
    return "data: " + json.dumps(_err(message, errtype)) + "\n\n"


def gguf_files():
    return hub.local_models()


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"))


@app.get("/api/hardware")
def hw():
    return metrics.hardware()


@app.get("/api/spec")
def spec():
    spec_out = []
    for s in defaults.param_spec():
        d = dict(s)
        cat = s.get("cat", "performance")
        d["cat_label"] = defaults.CAT_LABEL.get(cat, "Performance")
        d["cat_color"] = defaults.CAT_COLOR.get(cat, "green")
        spec_out.append(d)
    return spec_out


@app.get("/api/autotune")
def auto_tune():
    """Returns recommended params for the ACTIVE GPU right now."""
    hw = metrics.hardware()
    gpu = hw.get("active_gpu")
    return {"params": defaults.auto_tune(hw, 0, None), "gpu": gpu, "ram": hw.get("ram")}


@app.get("/api/settings")
def get_settings():
    s = settings.load()
    return {"openrouter_model": s.get("openrouter_model"), "key_set": bool(s.get("openrouter_key"))}


@app.post("/api/settings")
def save_settings(body: dict):
    patch = {k: v for k, v in body.items() if k in ("openrouter_key", "openrouter_model", "hf_token")}
    s = settings.save(patch)
    return {"ok": True, "key_set": bool(s.get("openrouter_key"))}


@app.get("/api/free-models")
def or_free():
    s = settings.load()
    return free_models(s.get("openrouter_key"))


@app.get("/api/search")
def search(q: str = Query("instruct", min_length=1), limit: int = 40):
    s = settings.load()
    return hub.search_fit_vram(q, _free_vram(), limit, s.get("hf_token"))


@app.get("/api/hub")
def hub_home():
    """Auto-populated hub: only models that FULLY fit the active GPU."""
    s = settings.load()
    try:
        rows = hub.compatible(_free_vram(), limit=30, token=s.get("hf_token"))
        return {"models": rows, "gpu": metrics.active_gpu(), "vram_free_mb": _free_vram(),
                "backend": metrics._backend()}
    except Exception as e:
        return {"models": [], "gpu": metrics.active_gpu(), "vram_free_mb": _free_vram(),
                "backend": metrics._backend(), "error": str(e)}


@app.get("/api/model/{repo:path}")
def model_detail(repo: str):
    s = settings.load()
    return hub.detail(repo, s.get("hf_token"))


@app.get("/api/local")
def local():
    return hub.local_models()


@app.post("/api/download")
def download(body: dict):
    s = settings.load()
    repo = body["repo"]
    _file = body["file"]
    return hub.start_download(repo, _file, s.get("hf_token"))


@app.get("/api/downloads")
def progress():
    st = []
    for v in hub.downloads().values():
        st.append({"repo": v["repo"], "file": v["file"], "done": v["done"],
                   "total": v["total"], "speed": v["speed"], "status": v["status"],
                   "error": v.get("error")})
    return st


@app.get("/api/gguf-meta")
def gguf_info(path: str):
    if not os.path.isfile(path):
        raise HTTPException(404, "file not found")
    meta = read_gguf_meta(path)
    meta["size_gb"] = round(file_gb(path), 2)
    return meta


@app.get("/api/estimate")
def estimate(path: str):
    return auto_estimate(path)


def auto_estimate(path):
    meta = read_gguf_meta(path)
    size_mb = os.path.getsize(path) / 2**20
    layers = meta.get("block_count") or 0
    _, free_vram = metrics.total_vram()
    reserve = 300
    usable = max(0, free_vram - reserve)
    tunekv = defaults.auto_tune(metrics.hardware(), size_mb, layers or None,
                                ctx_limit=meta.get("context_length") or None)
    fits = size_mb <= usable
    ngl = tunekv.get("n_gpu_layers", layers if fits else 999)
    ngl = max(0, min(ngl, (layers or 999)))
    per_layer = max(20, (size_mb - min(900, size_mb * 0.35)) / max(layers, 1))
    est_vram = size_mb if fits else per_layer * ngl + min(900, size_mb * 0.35) * min(1, ngl / max(layers, 1))
    return {"layers": layers, "size_mb": round(size_mb, 1), "size_gb": round(size_mb / 1024, 2),
            "fits_fully": fits, "suggested_ngl": ngl,
            "tuned": tunekv,
            "est_vram_mb": round(min(est_vram + 128, free_vram), 0),
            "free_vram_mb": free_vram, "ctx_vram_est_mb": 128}

def _free_vram():
    _, free = metrics.total_vram()
    return free


@app.get("/api/engines")
def engines():
    return manager.list()


@app.post("/api/engines/load")
def load(body: dict):
    path = body.get("path")
    if not path or not os.path.isfile(path):
        raise HTTPException(400, "model file not found")
    set_params = dict(body.get("params", {}))
    if body.get("auto_tune", True):
        meta = read_gguf_meta(path)
        layers = meta.get("block_count")
        size_mb = os.path.getsize(path) / 2**20
        tuned = defaults.auto_tune(metrics.hardware(), size_mb, layers or None,
                                   ctx_limit=meta.get("context_length") or None)
        params = {**defaults.default_params(), **tuned, **{k: v for k, v in set_params.items() if k not in tuned}}
    else:
        params = {**defaults.default_params(), **set_params}
    name = body.get("name") or os.path.basename(path)
    meta = read_gguf_meta(path)
    layers = meta.get("block_count") or params.get("n_gpu_layers")
    if params.get("n_gpu_layers") and params["n_gpu_layers"] > layers:
        params["n_gpu_layers"] = layers if layers else params["n_gpu_layers"]
    eng = manager.load(path, params, name)
    eng.info.setdefault("layers_total", layers)
    out = eng.status()
    out["_info"] = {"layers": layers, "meta": meta}
    return out


@app.post("/api/engines/{port}/stop")
def stop(port: int):
    manager.stop(port)
    return {"ok": True}


@app.post("/api/engines/{port}/restart")
def restart(port: int, body: dict):
    eng = manager.get(port)
    if not eng:
        raise HTTPException(404, "engine not found")
    params = dict(eng.params)
    params.update(body.get("params", {}) or {})
    eng.start(eng.model_path, params, eng.model_name)
    out = eng.status()
    if not eng.ready:
        raise HTTPException(500, "restart failed:\n" + "\n".join(list(eng.logs)[-20:]))
    return out


@app.get("/api/engines/{port}/logs")
def logs(port: int, n: int = Query(120, ge=1)):
    eng = manager.get(port)
    if not eng:
        raise HTTPException(404, "engine not found")
    return {"logs": list(eng.logs)[-n:]}


def _stream_llama(port, payload):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    eng = manager.get(port)
    if not eng:
        yield json.dumps({"error": "engine not running"}) + "\n\n"
        return
    start = time.time()
    first = None
    started_chars = 0
    tkn = 0
    nctx = None
    try:
        with requests.post(url, json=payload, stream=True, timeout=300) as r:
            if r.status_code != 200:
                err = json.dumps({"error": {"message": "llama-server %s: %s" % (r.status_code, r.text[:500])}})
                yield "data: " + err + "\n\n"
                yield "data: [DONE]\n\n"
                return
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                d = raw[5:].strip()
                if d == "[DONE]":
                    continue
                try:
                    obj = json.loads(d)
                except Exception:
                    continue
                if first is None:
                    first = time.time()
                choices = obj.get("choices") or []
                if choices:
                    delta = (choices[0].get("delta") or {}).get("content")
                    if delta:
                        started_chars += len(delta)
                usage = obj.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    tkn = usage.get("completion_tokens")
                    try:
                        nctx = usage.get("prompt_tokens") or None
                    except Exception:
                        pass
                yield "data: " + json.dumps(obj) + "\n\n"
    except Exception as e:
        yield "data: " + json.dumps({"error": {"message": str(e)}}) + "\n\n"
        yield "data: [DONE]\n\n"
        return
    total = time.time() - start
    if tkn <= 0:
        tkn = max(1, int(started_chars / 4))
    meta = {"elapsed_s": round(total, 2),
            "tps": round(tkn / total, 1) if total > 0 else 0,
            "tokens": tkn,
            "first_token_s": round((first - start), 3) if first else None,
            "prompt_tokens": nctx,
            "interactive": False}
    yield "event: meta\ndata: " + json.dumps(meta) + "\n\n"
    yield "data: [DONE]\n\n"


@app.post("/api/cleanup")
def cleanup():
    """Stop all engines + kill leftover llama-server processes: frees RAM + GPU."""
    stopped = []
    for e in list(manager.engines.values()):
        nm = e.model_name
        e.stop()
        stopped.append(nm)
    freed = metrics.kill_llama_processes()
    _, free = metrics.total_vram()
    return {"ok": True, "stopped": stopped, "killed_pids": [f["pid"] for f in freed],
            "ram_freed_mb": round(sum(f["ram_mb"] for f in freed), 1),
            "gpu_free_mb": free, "hardware": metrics.hardware()}


@app.post("/api/engines/{port}/chat")
def chat(port: int, body: dict):
    eng = manager.get(port)
    if not eng or not eng.running:
        raise HTTPException(400, "no running engine")
    messages = body.get("messages")
    if not messages:
        raise HTTPException(400, "messages required")
    # Truncate defensively so even direct engine calls never 400 on ctx overflow.
    ctx_limit = int(eng.params.get("ctx", 4096))
    max_tokens = int(body.get("max_tokens") or 512)
    budget = max(512, ctx_limit - min(512, max_tokens))
    payload = {"messages": messages, "stream": True, "max_tokens": max_tokens}
    for k in ("temperature", "top_p", "top_k", "min_p", "repeat_penalty", "presence_penalty",
              "frequency_penalty", "stop", "tools", "tool_choice", "parallel_tool_calls",
              "stream_options", "response_format", "seed", "logprobs", "top_logprobs", "n"):
        if body.get(k) is not None:
            payload[k] = body[k]
    if isinstance(messages, list) and messages:
        if _has_tool_messages(messages):
            messages = _normalize_messages(messages)
        payload["messages"] = messages
    if body.get("cache_prompt") is True:
        payload["cache_prompt"] = True
    return StreamingResponse(_stream_llama(port, payload), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# OpenAI-compatible API surface for opencode / GitHub Copilot / Cline / Roo.
# No auth headers required: just point base_url at this server.
# ---------------------------------------------------------------------------

def _default_model_path():
    """Pick a .gguf to auto-load when nothing is running (zero-config)."""
    import glob
    preferred = os.environ.get("DEFAULT_MODEL", "")
    roots = [os.path.join(BASE, "models")]
    found = []
    for root in roots:
        for f in glob.glob(os.path.join(root, "**", "*.gguf"), recursive=True):
            found.append(f)
    if not found:
        return None
    if preferred:
        for f in found:
            if preferred.lower() in f.lower():
                return f
    for f in found:
        low = f.lower()
        if "coder" in low and "instruct" in low:
            return f  # Qwen2.5-Coder: best tool-calling at any temp
    for f in found:
        low = f.lower()
        if "phi" in low and "instruct" in low and "mini" in low:
            return f  # Phi-4-mini: good multi-turn, weaker tool names
    for f in found:
        low = f.lower()
        if "instruct" in low:
            return f
    return sorted(found, key=os.path.getsize)[0]


def _escape_json(s):
    """Escape a string so it is safe to embed inside JSON."""
    return json.dumps(str(s))


# Conservative chars-per-token estimate. Real code tokenizers (llama.cpp BPE)
# pack ~1.8 chars/token for dense code, so we deliberately OVER-estimate the
# token count (using 1.6) to guarantee the truncated prompt always fits and
# never 400s on the backend. Slightly over-truncating is safe; overflowing is not.
CHAR_PER_TOKEN = 1.6


def _sanitize_content(text):
    """Strip control/non-printable characters the model sometimes emits at
    high temperature or with corrupted context. Preserves newlines, tabs,
    and all printable Unicode. Also strips the Unicode replacement char."""
    if not isinstance(text, str):
        return text
    out = []
    for ch in text:
        cp = ord(ch)
        if cp == 0xFFFD:   # replacement char — drop
            continue
        if cp < 0x20 and cp not in (9, 10, 13):  # control (keep \t \n \r)
            continue
        if 0x7F <= cp < 0xA0:  # DEL + C1 controls
            continue
        out.append(ch)
    return "".join(out)


def truncate_messages(messages, ctx_limit):
    """Drop oldest messages so the prompt fits within ctx_limit tokens.

    Tool-interaction pairs (assistant tool_call + tool result) are dropped
    as a unit to prevent orphaned result messages that confuse the model.
    Code-heavy prompts tokenize densely, so we use a conservative token
    estimate and reserve room for the completion. As a last resort we hard-cap
    each remaining message so total tokens stay under budget.
    """
    if not messages or not isinstance(messages, list):
        return messages, False, 0

    def _toks(m):
        c = m.get("content") or ""
        tc = len(str(c)) // CHAR_PER_TOKEN
        for tc_item in (m.get("tool_calls") or []):
            fn = tc_item.get("function") or {}
            tc += len(str(fn.get("arguments") or "")) // CHAR_PER_TOKEN
            tc += len(str(fn.get("name") or "")) // CHAR_PER_TOKEN
        return max(tc, 1)

    # Compute tool-interaction run boundaries: a run is an assistant message
    # with tool_calls followed by zero or more `role:"tool"` results.
    # When truncating we drop whole runs to keep pair integrity.
    runs = []  # list of (start_idx, end_idx_inclusive)
    i = 0
    while i < len(messages):
        m = messages[i]
        role = m.get("role")
        has_tc = bool(m.get("tool_calls"))
        if role == "assistant" and has_tc:
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1

    # total tokens: sum of all
    def run_toks(start, end):
        return sum(_toks(messages[k]) for k in range(start, end + 1))

    total_toks = sum(_toks(m) for m in messages)
    if total_toks <= ctx_limit:
        return messages, False, 0

    # Build result list — keep system prompt, drop oldest runs/messages
    head = [messages[0]] if messages and messages[0].get("role") == "system" else []
    tail = list(range(1 if head else 0, len(messages)))
    freed = 0

    while tail and (sum(_toks(messages[k]) for k in tail) + (run_toks(0, 0) if head else 0)) > ctx_limit:
        # Never drop the last message — the model must have something to respond to
        if len(tail) <= 1:
            break
        idx = tail.pop(0)
        # If this message is part of a tool run, drop the whole run
        run_found = None
        for si, ei in runs:
            if si <= idx <= ei:
                run_found = (si, ei)
                break
        if run_found:
            si, ei = run_found
            # Remove all members of the run from tail
            drop_set = set(range(si, ei + 1))
            tail = [t for t in tail if t not in drop_set]
            freed += run_toks(si, ei)
        else:
            freed += _toks(messages[idx])

    result = head + [messages[k] for k in tail]

    # Last resort: even the minimal set may overflow (e.g. one giant file dump).
    # Hard-cap every message's characters so total tokens stay under budget.
    if sum(_toks(m) for m in result) > ctx_limit:
        per = max(200, int(ctx_limit * CHAR_PER_TOKEN) // max(len(result), 1))
        new_result = []
        for m in result:
            m = dict(m)
            c = m.get("content")
            if isinstance(c, str) and len(c) > per:
                m["content"] = c[:per] + "\n...[context truncated to fit window]"
            new_result.append(m)
        result = new_result
        freed += 1
    return result, True, freed


def _pick_engine(model):
    with manager._lock:
        running = [e for e in manager.engines.values() if e.running]
    if not running:
        return None
    m = str(model or "")
    for e in running:
        if m and (e.model_name == m or str(e.port) == m or m in e.model_name):
            return e
    return running[0]


def _ensure_engine(model_req):
    """Return a running engine, auto-loading a default model if needed."""
    eng = _pick_engine(model_req)
    if eng and eng.running:
        return eng
    path = _default_model_path()
    if not path:
        return None
    meta = read_gguf_meta(path)
    layers = meta.get("block_count")
    size_mb = os.path.getsize(path) / 2 ** 20
    tuned = defaults.auto_tune(metrics.hardware(), size_mb, layers or None,
                               ctx_limit=meta.get("context_length") or None)
    params = {**defaults.default_params(), **tuned}
    return manager.load(path, params, os.path.basename(path))


def _proxy_completion(target, payload, model_req):
    # Normalize OpenAI tool/tool-result messages for llama-server (no native
    # tool template) so non-streaming tool use also works across turns.
    msgs = payload.get("messages")
    if isinstance(msgs, list) and _has_tool_messages(msgs):
        payload["messages"] = _normalize_messages(msgs)
    try:
        r = requests.post(target, json=payload, timeout=600)
    except Exception as e:
        return 502, {"error": {"message": str(e)}}
    try:
        obj = r.json()
    except Exception:
        return r.status_code, {"error": {"message": r.text[:500]}}
    if isinstance(obj, dict) and obj.get("model"):
        obj["model"] = model_req
    # Many local GGUFs (e.g. Qwen2.5-Coder) emit a tool call as a fenced
    # ```json block instead of native tool_calls. Convert it so editors that
    # expect OpenAI tool_calls get a properly structured call.
    if payload.get("tools") and isinstance(obj, dict):
        obj = _convert_tool_text(obj, payload.get("tools"))
    # Sanitize model output — strip control/illegal characters from content
    if isinstance(obj, dict):
        for choice in (obj.get("choices") or []):
            msg = choice.get("message") or {}
            if not msg.get("tool_calls"):
                c = msg.get("content")
                if isinstance(c, str):
                    msg["content"] = _sanitize_content(c)
    return r.status_code, obj


# Whole-string fenced tool call, OR a fenced block embedded in text.
_TOOL_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(\{.*?\})\s*```\s*$", re.S | re.I)
_FENCE_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S | re.I)


def _tool_names(tools):
    out = []
    for t in (tools or []):
        fn = t.get("function") or t
        n = fn.get("name")
        if n:
            out.append(n)
    return out


def _canonicalize_tool_name(name, tools):
    """Models often emit a near-miss tool name (e.g. `question` for
    `ask_question`). Map it to the exact name from the request's tool list so
    editors (opencode/Cline/Copilot/Roo) accept the call instead of rejecting
    an unknown tool. Falls back to the original name if no match."""
    names = _tool_names(tools)
    if not names or name in names:
        return name
    a = name.lower().replace("_", "").replace("-", "")
    for n in names:
        b = n.lower().replace("_", "").replace("-", "")
        if a == b or a in b or b in a:
            return n
    return name


def _parse_tool_call_obj(obj, tools):
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    if not name or not isinstance(name, str):
        return None
    args = obj.get("arguments")
    if not isinstance(args, dict):
        args = {}
    return _canonicalize_tool_name(name, tools), args


def _repair_json(text):
    """Repair truncated JSON from models that miss closing braces/brackets.
    E.g. `{"name":"x","arguments":{"k":"v"}` → add missing `}}`."""
    t = text.strip()
    if t.startswith("```"):
        t = t[3:].strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    for _ in range(8):
        try:
            return json.loads(t)
        except Exception:
            pass
        # Try adding closing characters
        last = t[-1] if t else ""
        if last in ("}", "]"):
            # Already closed — try adding comma-like closings
            # This handles `{"a":1}{"b":2` → `{"a":1},{"b":2}`
            # But the simpler case: just add more closing braces
            pass
        # Count open/close
        opens = t.count("{") + t.count("[")
        closes = t.count("}") + t.count("]")
        if opens > closes:
            t += "}" * (opens - closes)
        else:
            t += "}"
    return None


def _extract_tool_calls(content, tools=None):
    """Extract ALL tool calls from fenced or bare JSON text.
    Handles single-object, array, and nested {"tool_calls":[...]} shapes.
    Includes JSON repair for models that miss closing braces (Phi-mini)."""
    if not isinstance(content, str):
        return []
    text = content.strip()
    if not text:
        return []
    m = _TOOL_FENCE_RE.match(text)
    src = m.group(1) if m else text
    if not m:
        if not (text.startswith("{") and '"name"' in text) and not (text.startswith("[") and '"name"' in text):
            return []
    # Try strict parse, then repair if broken JSON (common on Phi/1-4B models)
    try:
        obj = json.loads(src)
    except Exception:
        obj = _repair_json(src)
        if obj is None:
            return []
    # Unwrap common wrappers
    if isinstance(obj, dict) and "tool_calls" in obj and isinstance(obj.get("tool_calls"), list):
        obj = obj["tool_calls"]
    if isinstance(obj, dict):
        tc = _parse_tool_call_obj(obj, tools)
        return [tc] if tc else []
    if isinstance(obj, list):
        out = []
        for item in obj:
            if isinstance(item, dict) and "function" in item and isinstance(item.get("function"), dict):
                item = item["function"]
            tc = _parse_tool_call_obj(item, tools)
            if tc:
                out.append(tc)
        return out
    return []


def _split_tool_and_text(full, tools):
    """Split response into (text_before, [(name,args),...], text_after).
    Handles multiple fenced blocks. Returns None when no tool calls are present."""
    segments = []
    pos = 0
    for m in _FENCE_BLOCK_RE.finditer(full):
        text_before = full[pos:m.start()].strip()
        tcs = _extract_tool_calls(m.group(1), tools)
        if tcs:
            segments.append(("text", text_before) if text_before else None)
            segments.append(("tools", tcs))
            pos = m.end()
    if not segments:
        # Also try bare JSON at the end
        tcs = _extract_tool_calls(full.strip(), tools)
        if tcs:
            return "", tcs, ""
        return None
    # Flatten: first text segment + all tool calls + trailing text
    after = full[pos:].strip()
    first_text = ""
    all_tcs = []
    for seg in segments:
        if seg is None:
            continue
        kind, val = seg
        if kind == "text" and not first_text:
            first_text = val
        elif kind == "tools":
            all_tcs.extend(val)
    return first_text, all_tcs, after


def _looks_like_tool_call(text):
    """Heuristic used during streaming to decide whether to buffer+convert
    (tool call) or stream live (normal chat)."""
    t = text.strip()
    if not t:
        return False
    if t.startswith("```"):
        return True
    return t.startswith("{") and '"name"' in t and '"arguments"' in t


def _augment_tool_system(messages, tools):
    """Override system prompt with a short, tool-focused directive for GGUFs
    that lack a native chat template. Small models (1-7B) get overwhelmed by
    opencode's verbose system prompt and role-play instead of calling tools.
    
    This REPLACES the system message with our own concise agent directive.
    Action tools (bash/read/grep/glob/edit) are listed first with clear
    descriptions. question is demoted to last-resort to prevent the model
    from clarifying instead of acting."""
    # Sort: action tools first, question-type tools last
    priority_tools = ("bash", "read", "grep", "glob", "edit", "write", "task", "skill",
                      "todowrite", "question", "webfetch")
    ordered = []
    rest = list(tools or [])
    for pn in priority_tools:
        for t in rest[:]:
            if t.get("function", {}).get("name") == pn:
                ordered.append(t)
                rest.remove(t)
                break
    ordered.extend(rest)
    
    tool_desc = []
    for t in ordered[:20]:
        fn = t.get("function") or {}
        n = fn.get("name", "")
        desc = (fn.get("description") or "")[:80]
        params = fn.get("parameters", {}).get("properties", {})
        pnames = list(params.keys())[:4]
        pmap = {}  # noqa - only used below
        # Add note for special tools
        note = ""
        if n == "question":
            note = " [LAST RESORT: act first, ask only if stuck]"
        elif n in ("bash", "read", "grep", "glob", "edit"):
            note = " [USE THIS FIRST]"
        tool_desc.append(
            '{name}: {desc}{params}{note}'.format(
                name=n, desc=desc,
                params=(" — params: " + ", ".join("'{}'".format(p) for p in pnames)) if pnames else "",
                note=note
            )
        )
    
    tool_format = (
        "You are a coding agent. ALWAYS use bash/read/grep/glob/edit first.\n"
        "question is ONLY for when you are genuinely stuck and must clarify.\n\n"
        "TOOLS:\n" + "\n".join(tool_desc) + "\n\n"
        "TO CALL A TOOL, output ONLY:\n"
        "```json\n"
        '{"name":"EXACT_tool_name_from_above","arguments":{exact_params}}\n'
        "```\n"
        "Do NOT invent tool names. Use EXACTLY the names listed in TOOLS above.\n"
        "After tool results, give your final answer."
    )
    
    has_results = _has_tool_messages(messages)
    if has_results:
        tool_format += "\nTool results are present. Answer the user — do NOT call more tools."
    
    msgs = list(messages)
    idx = None
    for i in range(len(msgs)):
        if msgs[i].get("role") == "system":
            idx = i
            break
    if idx is not None:
        msgs[idx] = {"role": "system", "content": tool_format}
    else:
        msgs.insert(0, {"role": "system", "content": tool_format})
    return msgs


def _normalize_messages(messages):
    """Convert OpenAI tool-call / tool-result messages back into the plain-text
    shape the non-native-tool model understands. The model sees a linear
    conversation: user → assistant(tool_call as text) → system(tool_result)
    so it clearly knows the tool result is the response to its own call.

    - assistant with tool_calls -> rendered as fenced-JSON text
    - role:"tool" result      -> system message: [tool_result] content
    """
    id_name = {}
    out = []
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                if tc.get("id"):
                    id_name[tc["id"]] = fn.get("name", "")
            parts = []
            if m.get("content"):
                parts.append(str(m["content"]))
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = fn.get("arguments", {})
                parts.append("```json\n" + json.dumps({"name": fn.get("name",""), "arguments": args},
                                                        ensure_ascii=False) + "\n```")
            out.append({"role": "assistant", "content": "\n".join(parts)})
        elif role == "tool":
            tcid = m.get("tool_call_id")
            name = id_name.get(tcid, "")
            content = m.get("content") or ""
            # Distinguish success from error — model needs to know when a tool call failed
            is_error = any(w in str(content).lower()[:200] for w in ("error", "invalid", "failed", "cannot", "denied"))
            if is_error:
                prefix = "ERROR: your %s call failed: " % name if name else "ERROR: your tool call failed: "
            else:
                prefix = "Result of %s: " % name if name else "Result: "
            out.append({"role": "user", "content": prefix + content})
        else:
            out.append(m)
    return out


def _has_tool_messages(messages):
    return any((m.get("role") == "tool") or
               (m.get("role") == "assistant" and m.get("tool_calls")) for m in messages)


def _wrap_tool_calls(tcs, call_id=None):
    """Wrap a list of (name, args) tuples into OpenAI tool_call deltas.
    Strips null values from args dict — models often include null for
    optional params, which fails schema validation on the client side."""
    out = []
    base_id = call_id or "call_%d" % int(time.time() * 1000)
    for i, (name, args) in enumerate(tcs):
        cid = "%s_%d" % (base_id, i) if len(tcs) > 1 else base_id
        if isinstance(args, dict):
            args = {k: v for k, v in args.items() if v is not None}
        out.append({"id": cid, "type": "function",
                     "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return out


def _sse_chunk(content=None, tool_calls=None, finish=None):
    """OpenAI-compliant streaming chunk: finish_reason at CHOICE level, not inside delta."""
    d = {}
    if content is not None:
        d["content"] = _sanitize_content(content)
    if tool_calls is not None:
        d["tool_calls"] = tool_calls
    choice = {"index": 0, "delta": d}
    if finish:
        choice["finish_reason"] = finish
    return "data: " + json.dumps({"choices": [choice]}) + "\n\n"


def _convert_tool_text(obj, tools=None):
    """When the assistant message content is a single JSON tool call, move it
    into the OpenAI `tool_calls` field so editors parse it natively."""
    try:
        choices = obj.get("choices") or []
        if not choices:
            return obj
        msg = (choices[0].get("message") or {})
        if msg.get("tool_calls"):  # already native — leave untouched
            return obj
        content = msg.get("content") or ""
        tcs = _extract_tool_calls(content, tools)
        if not tcs:
            return obj
        valid_names = _tool_names(tools)
        valid_tcs = [(n, a) for n, a in tcs if n in valid_names]
        if not valid_tcs:
            return obj  # all tool names invalid — leave as text
        msg["tool_calls"] = _wrap_tool_calls(valid_tcs)
        msg["content"] = ""
        choices[0]["finish_reason"] = "tool_calls"
        obj["choices"] = choices
    except Exception:
        return obj
    return obj


@app.get("/v1/models")
def v1_models():
    data = []
    for e in manager.list():
        if e.get("running"):
            ctx = e.get("params", {}).get("ctx", 4096)
            data.append({
                "id": e.get("model_name") or "local",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
                "port": e.get("port"),
                "context_length": ctx,
                "max_output_tokens": ctx,
                "capabilities": {
                    "tool_calling": True,
                    "streaming": True,
                    "reasoning": False,
                    "structured_output": False,
                    "vision": False,
                    "prompt_caching": False,
                },
            })
    if not data:
        data.append({"id": "local", "object": "model", "created": 0, "owned_by": "local",
                       "context_length": 0, "max_output_tokens": 0,
                       "capabilities": {"tool_calling": False, "streaming": True, "reasoning": False}})
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def v1_chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    model_req = body.get("model") or "local"
    eng = await run_in_threadpool(_ensure_engine, model_req)
    _log("request", model=model_req, msgs=len(body.get("messages") or []),
         tools=len(body.get("tools") or []), stream=bool(body.get("stream")))
    _req_start = time.time()
    if not eng or not eng.running:
        return JSONResponse(status_code=503, content=_err(
            "No local model is loaded and no .gguf model found to auto-load. "
            "Load a model via the UI (/api/engines/load) or set DEFAULT_MODEL.",
            "no_model_loaded"))
    target = f"http://127.0.0.1:{eng.port}/v1/chat/completions"
    payload = dict(body)
    want_stream = bool(payload.get("stream", False))

    # Client-disconnect cancellation hook
    ct = CancellationToken()
    async def _watch_disconnect():
        try:
            while not ct.is_cancelled:
                if await request.is_disconnected():
                    ct.cancel()
                    return
                await asyncio.sleep(0.5)
        except Exception:
            pass
    _ = asyncio.ensure_future(_watch_disconnect())

    # --- Context-aware message truncation (fixes "exceeds available context size") ---
    # Reserve room for the completion so we never overflow ctx on the reply either.
    ctx_limit = int(eng.params.get("ctx", 4096))
    messages = payload.get("messages")
    if isinstance(messages, list) and messages:
        # Normalize tool_calls/tool-results to text (llama-server compatibility).
        # This is the ONLY message transformation — no truncation, no dropping.
        if _has_tool_messages(messages):
            messages = _normalize_messages(messages)
        # Minimal hint: prepend tool format instruction for models without
        # chat templates. Essential — without it, tool calling is impossible.
        if payload.get("tools") and _tool_names(payload["tools"]):
            idx = None
            for i in range(len(messages)):
                if messages[i].get("role") == "system":
                    idx = i
                    break
            names = _tool_names(payload["tools"])
            # Build project-context prefix (OS, dir, git status, languages)
            try:
                pc_parts = ["CWD: %s" % os.getcwd(), "OS: Windows"]
                import glob
                py_files = len(glob.glob("**/*.py", recursive=True))
                js_files = len(glob.glob("**/*.{js,ts,tsx,jsx}", recursive=True))
                if py_files: pc_parts.append("Python files: %d" % py_files)
                if js_files: pc_parts.append("JS/TS files: %d" % js_files)
                gf = ".git" if os.path.isdir(".git") else None
                if gf: pc_parts.append("Git repo: yes")
                project_ctx = "ENV: " + " | ".join(pc_parts) + "\n"
            except Exception:
                project_ctx = ""
            tool_hint = (
                project_ctx +
                "You have tools. ALWAYS call them to act — never ask the user to do anything.\n"
                "Your FIRST response to ANY request must be a tool call, not text.\n"
                "Only use text after receiving tool results to summarize findings.\n"
                "Format: ```json\n{\"name\":\"<tool>\",\"arguments\":{...}}\n```\n"
                "Available: %s.\n\n" % ", ".join(names[:15])
            )
            if idx is not None:
                if "tool-calling coding agent" not in str(messages[idx].get("content", "")):
                    messages[idx] = dict(messages[idx])
                    messages[idx]["content"] = tool_hint + str(messages[idx].get("content", ""))
            else:
                messages.insert(0, {"role": "system", "content": tool_hint.strip()})
        payload["messages"] = messages
        # NO preemptive truncation — the full conversation reaches the model.
        # Truncation only happens as a fallback if llama-server returns 400.

    def _stream_gen():
        has_tools = bool(payload.get("tools"))
        tools = payload.get("tools")
        buf = []           # accumulated content for text-to-tool conversion
        saw_native = False
        saw_finish = None
        chunk_count = 0
        content_chars = 0
        live_emit = True   # progressive streaming on; off when tool-call fence detected
        try:
            for attempt in (0, 1):
                buf.clear(); saw_native = False; saw_finish = None
                try:
                    with requests.post(target, json=payload, stream=True, timeout=600) as r:
                        if r.status_code != 200:
                            err_text = r.text[:600]
                            if attempt == 0 and r.status_code == 400 and (
                                "context" in err_text.lower() or "exceed" in err_text.lower()):
                                msgs = payload.get("messages") or []
                                if isinstance(msgs, list) and len(msgs) > 2:
                                    sysmsg = [msgs[0]] if msgs and msgs[0].get("role") == "system" else []
                                    rest = msgs[1:] if sysmsg else msgs
                                    keep = max(1, len(rest) // 2)
                                    payload["messages"] = sysmsg + rest[-keep:]
                                    _log("truncate", kept=len(payload["messages"]), reason="ctx_400")
                                    continue
                            yield _sse_err("llama-server %s: %s" % (r.status_code, err_text),
                                          "upstream_error" if r.status_code >= 500 else "invalid_request_error")
                            yield "data: [DONE]\n\n"
                            _log("response", model=model_req, error=True, status=r.status_code,
                                 dur_s=round(time.time() - _req_start, 2))
                            return

                        for raw in r.iter_lines(decode_unicode=True):
                            if ct.is_cancelled:
                                r.close()
                                yield _sse_err("client disconnected", "cancelled")
                                yield "data: [DONE]\n\n"
                                _log("response", model=model_req, cancelled=True,
                                     dur_s=round(time.time() - _req_start, 2))
                                return
                            if not raw or not raw.startswith("data:"):
                                continue
                            chunk_count += 1
                            d = raw[5:].strip()
                            if d == "[DONE]":
                                continue

                            # --- non-tool path: relay raw SSE directly ---
                            if not has_tools:
                                yield raw + "\n\n"
                                try:
                                    obj = json.loads(d)
                                    dc = (obj.get("choices") or [{}])[0].get("delta") or {}
                                    c = dc.get("content")
                                    if c:
                                        c = _sanitize_content(c)
                                        content_chars += len(c)
                                        buf.append(c)
                                    fr = dc.get("finish_reason")
                                    if fr:
                                        saw_finish = fr
                                except Exception:
                                    pass
                                continue

                            # --- tool path ---
                            try:
                                obj = json.loads(d)
                            except Exception:
                                yield _sse_err("invalid JSON: %s" % str(raw)[:200], "invalid_response_error")
                                continue
                            if isinstance(obj, dict) and obj.get("model"):
                                obj["model"] = model_req
                            delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                            if delta.get("tool_calls"):
                                if buf and not saw_native:
                                    yield _sse_chunk(content="".join(buf))
                                    buf.clear()
                                saw_native = True
                                # Sanitize content in native tool_call chunks + strip null args
                                for tc in (delta.get("tool_calls") or []):
                                    fn = tc.get("function") or {}
                                    a = fn.get("arguments")
                                    if isinstance(a, str):
                                        fn["arguments"] = _sanitize_content(a)
                                    # Strip null values from parsed args to avoid schema errors
                                    try:
                                        parsed = json.loads(fn.get("arguments", "{}"))
                                        if isinstance(parsed, dict):
                                            parsed = {k: v for k, v in parsed.items() if v is not None}
                                            fn["arguments"] = json.dumps(parsed, ensure_ascii=False)
                                    except Exception:
                                        pass
                                content_chars += len(json.dumps(obj))
                                yield "data: " + json.dumps(obj) + "\n\n"
                                continue
                            c = delta.get("content")
                            if c:
                                c = _sanitize_content(c)
                                content_chars += len(c)
                                buf.append(c)
                                if saw_native:
                                    (obj.get("choices") or [{}])[0].get("delta", {})["content"] = c
                                    yield "data: " + json.dumps(obj) + "\n\n"
                                elif live_emit:
                                    trailing = "".join(buf)[-200:]
                                    if "```json" in trailing:
                                        live_emit = False
                                    else:
                                        yield _sse_chunk(content=c)
                                # else: already in buffering mode — accumulate only
                            fr = delta.get("finish_reason")
                            if fr:
                                saw_finish = fr

                        # --- stream ended: emit final chunk ---
                        if has_tools and not saw_native:
                            full = "".join(buf)
                            if full:
                                seg = _split_tool_and_text(full, tools)
                                if seg:
                                    before, tcs, after = seg
                                    # Content was already live-emitted via progressive
                                    # streaming. Only emit tool_calls + trailing text.
                                    if tcs:
                                        valid_names = _tool_names(tools)
                                        valid_tcs = [(n, a) for n, a in tcs if n in valid_names]
                                        bad_names = [n for n, _ in tcs if n not in valid_names]
                                        if valid_tcs:
                                            yield _sse_chunk(tool_calls=_wrap_tool_calls(valid_tcs))
                                            yield _sse_chunk(finish="tool_calls")
                                        elif bad_names:
                                            msg = ("Model tried to call unavailable tool(s): " +
                                                   ", ".join(bad_names) +
                                                   ". Available: " + ", ".join(valid_names[:15]) + ".")
                                            yield _sse_chunk(content=msg, finish="stop")
                                        else:
                                            yield _sse_chunk(finish="stop")
                                    if after:
                                        yield _sse_chunk(content=after, finish="stop")
                                    elif not tcs:
                                        if not saw_finish:
                                            yield _sse_chunk(finish="stop")
                                else:
                                    # No tool call detected. If we stopped live
                                    # emission (false fence), emit buffered text.
                                    if not live_emit:
                                        yield _sse_chunk(content=full, finish="stop")
                                    elif not saw_finish:
                                        yield _sse_chunk(finish="stop")
                            else:
                                yield _sse_chunk(finish="stop")
                        elif has_tools and saw_native:
                            if not saw_finish:
                                yield _sse_chunk(finish="stop")
                        elif not has_tools:
                            if not saw_finish:
                                yield _sse_chunk(finish="stop")
                        break  # success — exit retry loop
                except Exception as e:
                    if attempt == 1:
                        yield _sse_err(str(e), "server_error")
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield _sse_err(str(e), "server_error")
            yield "data: [DONE]\n\n"
        _log("response", model=model_req, tool_calls=saw_native, has_tools=has_tools,
             chunks=chunk_count, content_chars=content_chars, finish=saw_finish,
             buf_preview="".join(buf)[:200] if buf else "",
             dur_s=round(time.time() - _req_start, 2))

    if want_stream:
        return StreamingResponse(_stream_gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                                           "X-Context-Limit": str(ctx_limit),
                                           "X-Context-Type": "openai-chat",
                                           "X-Model": eng.model_name or "local"})

    status, obj = await run_in_threadpool(_proxy_completion, target, payload, model_req)
    return JSONResponse(status_code=status, content=obj,
                        headers={"X-Context-Limit": str(ctx_limit),
                                  "X-Model": eng.model_name or "local"})


@app.post("/api/config-chat")
def config_chat(body: dict):
    s = settings.load()
    key = body.get("key") or s.get("openrouter_key")
    if not key:
        return {"error": "no OpenRouter key"}
    model = body.get("model") or s.get("openrouter_model")
    messages = body.get("messages") or []
    eng = manager.get(body.get("port"))
    status = eng.status() if eng else None
    sysp = build_system_prompt(status, defaults.param_spec(), metrics.hardware())
    full = [{"role": "system", "content": sysp}] + messages

    def gen():
        text_chunks = []
        for ev in chat_stream(full, model, key):
            if "error" in ev:
                yield f'event: error\ndata: {json.dumps({"error": ev["error"]})}\n\n'
                return
            if ev.get("delta"):
                text_chunks.append(ev["delta"])
                yield f'data: {json.dumps({"delta": ev["delta"]})}\n\n'
        final_text = "".join(text_chunks)
        block = _extract_json(final_text)
        applied = None
        if block and block.get("changes") and eng and eng.running and body.get("auto_apply", True):
            try:
                spec = {x["key"]: x for x in defaults.param_spec()}
                changes = {k: v for k, v in block["changes"].items() if k in spec}
                new = {**eng.params, **changes}
                eng.start(eng.model_path, new, eng.model_name)
                applied = {"port": eng.port, "params_after": new, "reason": block.get("reason")}
            except Exception as e:
                applied = {"error": str(e)}
        yield f'event: done\ndata: {json.dumps({"changes": (block or {}).get("changes") or {}, "applied": applied})}\n\n'

    return StreamingResponse(gen(), media_type="text/event-stream")


def _extract_json(text):
    """Pulls the trailing openrouter 'changes' JSON block out of a reply."""
    m = re.search(r'\{"changes"\s*:\s*\{.*?\},\s*"reason"\s*:\s*".*?"\}', text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    for m in re.finditer(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.S):
        try:
            return json.loads(m.group(1))
        except Exception:
            continue
    m = re.search(r'(\{.*\})', text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None


@app.post("/api/apply-config")
def apply_config(body: dict):
    port = body.get("port")
    eng = manager.get(port)
    if not eng or not eng.running:
        raise HTTPException(404, "engine not found")
    changes = body.get("changes") or {}
    spec = {x["key"]: x for x in defaults.param_spec()}
    invalid = [k for k in changes if k not in spec]
    if invalid:
        raise HTTPException(400, f"unknown params: {invalid}")
    new = {**eng.params, **changes}
    eng.start(eng.model_path, new, eng.model_name)
    st = eng.status()
    return {"ok": True, "status": st}


@app.post("/api/engines/{port}/benchmark")
def benchmark(port: int, body: dict):
    """Real-time benchmark against the running engine (tokens/sec, TTFT)."""
    eng = manager.get(port)
    if not eng or not eng.running:
        raise HTTPException(400, "no running engine")
    runs = max(1, min(int(body.get("runs", 3)), 10))
    prompt = body.get("prompt") or "Write a short Python function that computes the first 20 Fibonacci numbers. Output only code."
    max_tokens = int(body.get("max_tokens", 200))
    results = []
    for i in range(runs):
        payload = {"messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "temperature": 0, "stream": True}
        t0 = time.time()
        first = None
        chars = 0
        cpn = 0
        try:
            with requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                               json=payload, stream=True, timeout=300) as r:
                if r.status_code != 200:
                    results.append({"error": f"{r.status_code}"})
                    continue
                usage = None
                for raw in r.iter_lines(decode_unicode=True):
                    if not raw or not raw.startswith("data:"):
                        continue
                    d = raw[5:].strip()
                    if d == "[DONE]":
                        continue
                    try:
                        obj = json.loads(d)
                    except Exception:
                        continue
                    if first is None:
                        first = time.time()
                    choices = obj.get("choices") or []
                    if choices:
                        delta = (choices[0].get("delta") or {}).get("content")
                        if delta:
                            chars += len(delta)
                    if obj.get("usage") and obj["usage"].get("completion_tokens"):
                        cpn = obj["usage"].get("completion_tokens") or 0
                elapsed = time.time() - t0
                if cpn <= 0:
                    cpn = max(1, int(chars / 4))
                ttft = (first - t0) if first else elapsed
                results.append({"run": i + 1, "tokens": cpn, "elapsed_s": round(elapsed, 3),
                                "tps": round(cpn / elapsed, 2) if elapsed > 0 else 0,
                                "ttft_s": round(ttft, 3),
                                "prompt_tokens": (obj or {}).get("usage", {}).get("prompt_tokens")})
        except Exception as e:
            results.append({"run": i + 1, "error": str(e)})
    ok = [r for r in results if r.get("tps")]
    best = max(ok, key=lambda r: r["tps"]) if ok else None
    avg = round(sum(r["tps"] for r in ok) / len(ok), 2) if ok else 0
    return {"engine": eng.model_name, "port": port,
            "params": {k: eng.params.get(k) for k in ("ctx", "n_gpu_layers", "flash_attn", "kv_type", "batch")},
            "results": results, "best_tps": best["tps"] if best else 0,
            "avg_tps": avg, "best_ttft_s": best["ttft_s"] if best else 0}


def _tool_test_payload():
    tools = [
        {"type": "function", "function": {
            "name": "get_weather", "description": "Get current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
        {"type": "function", "function": {
            "name": "search_web", "description": "Search the web for a query",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
        {"type": "function", "function": {
            "name": "run_python", "description": "Execute python code and return output",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
        {"type": "function", "function": {
            "name": "read_file", "description": "Read contents of a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    ]
    return tools


@app.post("/api/engines/{port}/tool-test")
def tool_test(port: int, body: dict):
    """Tests whether the loaded model emits valid tool_calls (multi-tool capable)."""
    eng = manager.get(port)
    if not eng or not eng.running:
        raise HTTPException(404, "engine not running")
    prompt = body.get("prompt") or "Use tools to answer: What is the weather in Berlin and Mumbai? Search the web for 'llama.cpp' and run Python to print hello."
    max_tokens = int(body.get("max_tokens", 400))
    payload = {"messages": [{"role": "user", "content": prompt}],
               "tools": _tool_test_payload(), "parallel_tool_calls": True,
               "max_tokens": max_tokens, "temperature": 0, "stream": True}
    text = ""
    first = None
    t0 = time.time()
    tool_calls = []
    buffer = {}
    errors = []
    try:
        with requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                           json=payload, stream=True, timeout=300) as r:
            if r.status_code != 200:
                return {"ok": False, "errors": [f"llama-server {r.status_code}: {r.text[:200]}"]}
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                ddat = raw[5:].strip()
                if ddat == "[DONE]":
                    continue
                try:
                    obj = json.loads(ddat)
                except Exception:
                    continue
                if first is None:
                    first = time.time()
                ch = obj.get("choices") or []
                if not ch:
                    continue
                delta = ch[0].get("delta") or {}
                if delta.get("content"):
                    text += delta["content"]
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    fn = tc.get("function") or {}
                    buf = buffer.setdefault(idx, {"name": "", "args": ""})
                    if fn.get("name"):
                        buf["name"] += fn["name"]
                    if fn.get("arguments"):
                        buf["args"] += fn["arguments"]
    except Exception as e:
        errors.append(str(e))
    elapsed = time.time() - t0
    calls = []
    for idx in sorted(buffer):
        b = buffer[idx]
        args_json = b["args"]
        valid_json = False
        try:
            import json as _j
            parsed = _j.loads(args_json)
            valid_json = isinstance(parsed, dict)
        except Exception:
            parsed = args_json
        calls.append({"name": b["name"], "arguments_raw": args_json, "valid_json": valid_json,
                      "arguments": parsed if valid_json else None})
    score = 0
    if calls:
        score = sum(1 for c in calls if c["valid_json"] and c["name"])
    has_tool = bool(calls)
    finished = bool(calls) or (len(text.strip()) > 0)
    return {"engine": eng.model_name, "port": port,
            "tool_calls_made": len(calls), "calls": calls,
            "supports_tools": has_tool,
            "tool_score_pct": round(score / len(calls) * 100, 0) if calls else 0,
            "text_fallback": text.strip()[:500] if not calls else None,
            "elapsed_s": round(elapsed, 2), "errors": errors[:5]}


@app.post("/api/engines/{port}/code-edit-test")
def code_edit_test(port: int, body: dict):
    """Runs a code-editing task through the local engine and scores the edit."""
    eng = manager.get(port)
    if not eng or not eng.running:
        raise HTTPException(404, "engine not running")
    code = body.get("code")
    if not code:
        code = (
            "def sum_evens(numbers):\n"
            "    total = 0\n"
            "    for i, n in enumerate(numbers):\n"
            "        if i % 2 == 0:\n"
            "            total += n\n"
            "    return total\n"
        )
    task = body.get("task") or "Fix the bug: this function sums elements at even indexes instead of even values. Return only corrected code."
    sys_edit = ("You are a code editor. Make the requested edit, return ONLY the edited code in a code block, "
                "no explanations.")
    payload = {"messages": [{"role": "system", "content": sys_edit},
                            {"role": "user", "content": f"{task}\n\n```python\n{code}\n```"}],
               "max_tokens": int(body.get("max_tokens", 256)), "temperature": 0, "stream": True}
    text = ""
    first = None
    t0 = time.time()
    try:
        with requests.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                           json=payload, stream=True, timeout=300) as r:
            if r.status_code != 200:
                return {"error": f"llama-server {r.status_code}: {r.text[:200]}"}
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                ddat = raw[5:].strip()
                if ddat == "[DONE]":
                    continue
                try:
                    obj = json.loads(ddat)
                except Exception:
                    continue
                if first is None:
                    first = time.time()
                ch = obj.get("choices") or []
                if ch:
                    delta = ch[0].get("delta") or {}
                    if delta.get("content"):
                        text += delta["content"]
    except Exception as e:
        text += f"\n[error] {e}"
    elapsed = time.time() - t0
    tokens = max(1, int(len(text) / 4))
    # collect all code blocks from edit
    blocks = _extract_code_blocks(text)
    edited = blocks[-1] if blocks else None
    import ast
    ast_ok = False
    if edited:
        try:
            ast.parse(edited)
            ast_ok = True
        except SyntaxError:
            ast_ok = False
    return {"engine": eng.model_name, "port": port,
            "output": text, "code_blocks": blocks, "last_block": edited,
            "valid_python": ast_ok, "tokens": tokens,
            "elapsed_s": round(elapsed, 2),
            "tps": round(tokens / elapsed, 1) if elapsed > 0 else 0}


def _extract_code_blocks(text):
    blocks = []
    for m in re.finditer(r"```(?:(\w+))?\s*\n(.*?)```", text, re.S):
        if m.group(2).strip():
            blocks.append(m.group(2).strip())
    if not blocks:
        cleaned = re.sub(r"^```\w*\s*|\s*```$", "", text.strip())
        blocks = [cleaned] if cleaned else []
    return blocks


@app.get("/health")
@app.get("/ready")
def health():
    return {"status": "ok", "version": VERSION, "uptime_s": round(time.time() - _startup, 1)}

@app.get("/version")
def version():
    return {"version": VERSION, "backend": "llama.cpp", "api": "OpenAI-compatible /v1"}


@app.get("/api/log")
def inference_log(n: int = Query(50, ge=1, le=200)):
    """Last N inference request/response log entries (opencode ↔ server)."""
    with _log_lock:
        tail = _inf_log[-n:]
    return {"log": tail, "total": len(_inf_log)}


@app.get("/api/status")
def status():
    return {"managers": manager.list(), "hardware": metrics.hardware(),
            "settings": {"openrouter_model": settings.load().get("openrouter_model"), "key_set": bool(settings.load().get("openrouter_key"))}}


@app.post("/{path:path}")
async def catch_all_chat(request: Request, path: str):
    # Safety net: tolerate clients that append /chat/completions to the full URL.
    if "chat/completions" in path:
        return await v1_chat_completions(request)
    raise HTTPException(404, "not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8899")))