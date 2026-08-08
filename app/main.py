import json
import os
import re
import time

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
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
app.mount("/static", StaticFiles(directory=WEB), name="static")


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
    return defaults.param_spec()


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
    return hub.search(q, limit, s.get("hf_token"))


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
    reserve = 400
    usable = max(0, free_vram - reserve)
    emb_out = min(900, size_mb * 0.35)
    per_layer = max(20, (size_mb - emb_out) / max(layers, 1))
    fits = size_mb <= usable
    ngl = layers if fits else int((usable - 0) / per_layer)
    ngl = max(0, min(ngl, layers + (layers or 2)))
    est_vram = size_mb if fits else per_layer * ngl + emb_out * min(1, ngl / max(layers, 1))
    return {"layers": layers, "size_mb": round(size_mb, 1), "size_gb": round(size_mb / 1024, 2),
            "fits_fully": fits, "suggested_ngl": ngl, "est_vram_mb": round(est_vram, 0),
            "free_vram_mb": free_vram, "ctx_vram_est_mb": 128}


@app.get("/api/engines")
def engines():
    return manager.list()


@app.post("/api/engines/load")
def load(body: dict):
    path = body.get("path")
    if not path or not os.path.isfile(path):
        raise HTTPException(400, "model file not found")
    params = {**defaults.default_params(), **body.get("params", {})}
    name = body.get("name") or os.path.basename(path)
    meta = read_gguf_meta(path)
    layers = meta.get("block_count") or params.get("n_gpu_layers")
    if params.get("n_gpu_layers") and params["n_gpu_layers"] > layers:
        params["n_gpu_layers"] = layers
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
        yield 'data: {"error":"engine not running"}\n\n'
        return
    start = time.time()
    first = None
    started_chars = 0
    tkn = 0
    nctx = None
    try:
        with requests.post(url, json=payload, stream=True, timeout=300) as r:
            if r.status_code != 200:
                yield f'data: {{"error":"llama-server {r.status_code}: {r.text[:200]}"}}\n\n'
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
                yield f"data: {d}\n\n"
    except Exception as e:
        yield f'data: {{"error":"{e}"}}\n\n'
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
    yield f"event: meta\ndata: {json.dumps(meta)}\n\n"


@app.post("/api/engines/{port}/chat")
def chat(port: int, body: dict):
    eng = manager.get(port)
    if not eng or not eng.running:
        raise HTTPException(400, "no running engine")
    messages = body.get("messages")
    if not messages:
        raise HTTPException(400, "messages required")
    payload = {"messages": messages, "stream": True, "max_tokens": body.get("max_tokens", 512)}
    for k in ("temperature", "top_p", "top_k", "min_p", "repeat_penalty", "presence_penalty", "frequency_penalty", "stop"):
        if body.get(k) is not None:
            payload[k] = body[k]
    if body.get("cache_prompt") is True:
        payload["cache_prompt"] = True
    return StreamingResponse(_stream_llama(port, payload), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


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


@app.get("/api/status")
def status():
    return {"managers": manager.list(), "hardware": metrics.hardware(),
            "settings": {"openrouter_model": settings.load().get("openrouter_model"), "key_set": bool(settings.load().get("openrouter_key"))}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8899")))