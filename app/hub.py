import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

HF_API = "https://huggingface.co/api/models"
HF_TREE = "https://huggingface.co/api/models/{repo}/tree/main"
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{file}"
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

_downloads = {}
_dl_lock = threading.Lock()
_tree_cache = {}
_tree_lock = threading.Lock()

# Auto-populated hub: curated instruct-strong family roots, resolved to concrete
# repos at run-time. These families give reliable tool-calling chat templates.
COMPATIBLE_SEEDS = [
    "bartowski/Qwen3-Coder-4B-Instruct-GGUF",
    "bartowski/Qwen3-8B-Instruct-GGUF",
    "bartowski/Qwen3-4B-Instruct-GGUF",
    "unsloth/Qwen3-4B-Instruct-GGUF",
    "bartowski/Qwen2.5-Coder-3B-Instruct-GGUF",
    "bartowski/Qwen2.5-7B-Instruct-GGUF",
    "unsloth/Llama-3.2-3B-Instruct-GGUF",
    "bartowski/Llama-3.2-3B-Instruct-GGUF",
    "bartowski/Llama-3.2-1B-Instruct-GGUF",
    "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
    "unsloth/mistral-7b-instruct-v0.3-GGUF",
    "bartowski/Phi-3.5-mini-instruct-GGUF",
    "bartowski/Phi-3-mini-4k-instruct-GGUF",
    "bartowski/gemma-2-2b-it-GGUF",
    "ggml-org/gemma-2-2b-it-GGUF",
    "bartowski/DEV-Pro-Llama-3.1-8B-Instruct-GGUF",
]
TOOL_SAFE_SUBSTRINGS = ["instruct", "qtools", "abcq", "qwen", "llama-3", "v3", "mistral", "phi", "gemma"]


def _sanitize(repo_id):
    return repo_id.replace("/", "__")


def _headers(token=None):
    h = {"User-Agent": "llama-server-manager/1.0"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _get_json(url, token=None, **kw):
    try:
        r = requests.get(url, headers=_headers(token), timeout=15, **kw)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def search(q, limit=60, token=None):
    r = _get_json(HF_API, token=token,
                  params={"search": q, "filter": "gguf", "sort": "downloads",
                          "direction": -1, "limit": int(limit)})
    if not r:
        return []
    out = []
    for m in r:
        out.append({"id": m.get("id"),
                    "downloads": m.get("downloads", 0),
                    "likes": m.get("likes", 0),
                    "pipeline_tag": m.get("pipeline_tag"),
                    "tags": [t for t in m.get("tags", []) if not t.startswith("gguf")][:12],
                    "gguf_quant": (m.get("gguf") or {}).get("quantization") or None})
    return out


def repo_tree(repo_id, token=None, timeout=15):
    """GGUF files with real byte sizes from the HF tree API (cached)."""
    key = f"{repo_id}|{bool(token)}"
    with _tree_lock:
        if key in _tree_cache:
            return _tree_cache[key]
    try:
        r = requests.get(HF_TREE.format(repo=repo_id), headers=_headers(token), timeout=timeout)
        if r.status_code != 200:
            files = []
        else:
            files = []
            for e in r.json():
                p = e.get("path", "")
                if p.lower().endswith(".gguf") and not p.endswith(".part"):
                    files.append({"name": p, "size": e.get("size", 0)})
        files.sort(key=lambda f: f["name"])
        with _tree_lock:
            _tree_cache[key] = files
        return files
    except Exception:
        return []


def _has_tool_template(repo_id):
    low = repo_id.lower()
    return not any(x in low for x in ["base", "raw", "pretrain", "-embed", "non-chat"])


def compatible(vram_mb, limit=24, token=None, ctx_estimate_mb=900):
    """Auto-populated list of instruct/agent-ready GGUF repos, enriched with real
    file sizes and a GPU-fit rating so only fully-compatible models are shown."""
    usable = max(0.0, vram_mb - ctx_estimate_mb) / 1024.0  # GiB headroom for KV + buffers
    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(repo_tree, r, token): r for r in COMPATIBLE_SEEDS}
        sizes = {r: f.result() for r, f in ((fut, f) for fut in futs)}
    for repo in COMPATIBLE_SEEDS:
        files = sizes[repo]
        if not files or not _has_tool_template(repo):
            continue
        best = None
        for f in files:
            gb = f["size"] / 2**30
            if gb <= usable:
                if best is None or gb > best["size"] / 2**30:
                    best = {"name": f["name"], "size": f["size"]}
        rows.append({
            "id": repo,
            "fits_gpu": best is not None,
            "best_file": best["name"] if best else None,
            "best_size_gb": round(best["size"] / 2**30, 2) if best else None,
            "file_count": len(files),
        })
    rows = [r for r in rows if r["fits_gpu"]]
    rows.sort(key=lambda x: -x["best_size_gb"])
    return rows[:limit]


def search_fit_vram(q, vram_mb, limit=40, token=None, ctx_estimate_mb=4):
    """Search repos whose smallest GGUF resolves in GPU VRAM (strict full-offload)."""
    usable = max(0.0, vram_mb - ctx_estimate_mb) / 2**30
    rows = search(q, limit=limit, token=token)
    out = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(repo_tree, row["id"], token): row for row in rows}
        for fut in futs:
            row = futs[fut]
            files = fut.result()
            gguf = [f for f in files if f["size"] > 0]
            if not gguf:
                continue
            gguf.sort(key=lambda x: x["size"])
            fits = gguf[0]["size"] / 2**30 <= usable
            out.append({**row, "fits": fits,
                        "files": gguf[:6],
                        "min_gb": round(gguf[0]["size"] / 2**30, 2)})
    return out


def detail(repo_id, token=None):
    r = _get_json(f"{HF_API}/{repo_id}?blobs=true", token=token)
    files = []
    if r:
        for s in r.get("siblings", []):
            fn = s.get("rfilename", "")
            if fn.lower().endswith(".gguf"):
                files.append({"name": fn, "size": s.get("size", 0)})
    return {"id": repo_id,
            "files": sorted(files, key=lambda f: f.get("name")),
            "downloads": (r or {}).get("downloads", 0),
            "likes": (r or {}).get("likes", 0),
            "pipeline_tag": (r or {}).get("pipeline_tag"),
            "created_at": (r or {}).get("created_at"),
            "random": (r or {}).get("random"),
            "gated": (r or {}).get("gated", False)}


def local_models():
    out = []
    if not os.path.isdir(MODELS_DIR):
        return out
    for repo in sorted(os.listdir(MODELS_DIR)):
        rr = os.path.join(MODELS_DIR, repo)
        if not os.path.isdir(rr):
            continue
        for fn in sorted(os.listdir(rr)):
            if fn.lower().endswith(".gguf") and not fn.endswith(".part"):
                p = os.path.join(rr, fn)
                out.append({"repo": repo.replace("__", "/"), "file": fn, "path": p,
                            "size_gb": round(os.path.getsize(p) / 2**30, 2)})
    return out


def _resolve(repo, filename, token=None):
    url = HF_RESOLVE.format(repo=repo, file=filename)
    r = requests.head(url, headers=_headers(token), allow_redirects=True, timeout=15)
    return r.url, int(r.headers.get("content-length", 0))


def _part_path(repo, filename):
    return os.path.join(MODELS_DIR, _sanitize(repo), os.path.basename(filename) + ".part")


def _dest_path(repo, filename):
    return os.path.join(MODELS_DIR, _sanitize(repo), os.path.basename(filename))


def start_download(repo_id, filename, token=None):
    key = f"{repo_id}/{filename}"
    with _dl_lock:
        if key in _downloads:
            return _downloads[key]
    os.makedirs(os.path.join(MODELS_DIR, _sanitize(repo_id)), exist_ok=True)
    part = _part_path(repo_id, filename)
    done = os.path.getsize(part) if os.path.exists(part) else 0
    state = {"repo": repo_id, "file": filename, "done": done, "total": 0, "speed": 0,
             "status": "starting", "error": None}
    with _dl_lock:
        _downloads[key] = state
    threading.Thread(target=_worker, args=(key, repo_id, filename, token, state), daemon=True).start()
    return state


def _worker(key, repo, filename, token, state):
    try:
        url, total = _resolve(repo, filename, token)
        state["total"] = total
        part = _part_path(repo, filename)
        done = state["done"]
        headers = _headers(token)
        if done and done < total:
            headers["Range"] = f"bytes={done}-"
        state["status"] = "downloading"
        t0 = time.time()
        with requests.get(url, headers=headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(part, "ab") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    f.write(chunk)
                    state["done"] += len(chunk)
                    el = time.time() - t0
                    if el > 1.0:
                        state["speed"] = (state["done"] - done) / el
        final = _dest_path(repo, filename)
        if os.path.exists(final):
            os.remove(final)
        os.rename(part, final)
        state["status"] = "done"
        state["speed"] = 0
    except Exception as e:
        state["status"] = "error"
        state["error"] = str(e)


def cancel_download(repo_id, filename):
    key = f"{repo_id}/{filename}"
    with _dl_lock:
        s = _downloads.get(key)
        if s:
            s["status"] = "cancelled"


def downloads():
    with _dl_lock:
        return {k: dict(v) for k, v in _downloads.items()}