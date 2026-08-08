import json

import requests

OR_URL = "https://openrouter.ai/api/v1/chat/completions"
OR_MODELS = "https://openrouter.ai/api/v1/models"


def free_models(key=None):
    try:
        h = {"Authorization": f"Bearer {key}"} if key else {}
        h["User-Agent"] = "llama-manager"
        r = requests.get(OR_MODELS, headers=h, timeout=15)
        if r.status_code != 200:
            return []
        out = []
        for m in r.json().get("data", []):
            pr = m.get("pricing", {}) or {}
            if pr.get("prompt", "0") == "0" and pr.get("completion", "0") == "0":
                out.append({"id": m["id"], "name": m.get("name"), "context": m.get("context_length")})
        return sorted(out, key=lambda x: x["id"])
    except Exception:
        return []


def build_system_prompt(status, spec, hardware):
    hw = {"gpus": hardware.get("gpus"), "ram": hardware.get("ram"),
          "cpu_count": hardware.get("cpu_count")}
    if status is None or not status.get("running"):
        obj = "No local model is currently running. Answer briefly with the JSON with empty changes, or advise on model choice."
    else:
        obj = json.dumps({
            "model": status.get("model_name"),
            "model_path": status.get("model_path"),
            "port": status.get("port"),
            "layers_offloaded": status.get("layers_offloaded"),
            "layers_total": status.get("layers_total"),
            "model_vram_used_mib": status.get("vram_used_mib"),
            "engine_vram_mb": status.get("vram_used_engine_mb"),
            "kv_usage": status.get("kv_usage"),
            "current_params": status.get("params"),
        }, indent=2)
    return f"""You are a real-time ML engineer who tunes a local llama.cpp inference server through JSON knobs.

HARDWARE: {json.dumps(hw)}
CURRENT STATE: {obj}
TUNABLE PARAMETERS SCHEMA: {spec_json}

Rules:
- One NVIDIA GPU with 4GB VRAM, ~15GB RAM.
- Flash attention + kv_type q8_0 shrink KV-cache VRAM a lot. Higher ctx raises VRAM.
- Speed lever: n_gpu_layers, batch, flash_attn, kv_type, threads.
- Accuracy lever: kv f16, temp ~0.7, top_p ~0.9, repr 1.05-1.15. Lower min_p for quality.
- Only propose values within schema min/max/step. Do not invent keys.

Answer the user conversationally in at most 6 short sentences. Then on the FINAL line, output ONLY a JSON object matching EXACTLY:
{{"changes": {{"key": value, ...}}, "reason": "one-line justification"}}
Only include keys that exist in the schema. Empty changes object if nothing changes. Do not print the JSON anywhere else."""


def chat_stream(messages, model, key, temperature=0.4):
    """Streams openrouter response. Yields dicts: {'delta': str, 'text': str...}."""
    h = {"Authorization": f"Bearer {key}",
         "Content-Type": "application/json",
         "X-Title": "llama-server-manager"}
    payload = {"model": model, "messages": messages, "temperature": temperature,
               "stream": True, "include_reasoning": True}
    try:
        with requests.post(OR_URL, headers=h, json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            acc = []
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                c = choices[0]
                d = c.get("delta") or {}
                piece = d.get("content")
                if piece:
                    acc.append(piece)
                    yield {"delta": piece, "text": "".join(acc)}
                if c.get("finish_reason"):
                    break
    except Exception as e:
        yield {"error": str(e)}