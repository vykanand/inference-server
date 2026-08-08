PARAM_SPEC = [
    {"key": "n_gpu_layers", "label": "GPU layers (offload)", "group": "GPU offload",
     "cat": "performance", "type": "int", "min": 0, "max": 999, "step": 1, "default": 999,
     "help": "How many model layers to offload to the GPU. Set to total layer count (or 999) to fit the whole model on GPU if VRAM allows. Lower = more RAM, slower."},
    {"key": "ctx", "label": "Context size", "group": "Context", "cat": "context", "type": "int",
     "min": 512, "max": 131072, "step": 256, "default": 4096,
     "help": "Total prompt+completion token window."},
    {"key": "batch", "label": "Batch size", "group": "Performance", "cat": "performance", "type": "int",
     "min": 32, "max": 8192, "step": 32, "default": 512,
     "help": "Prompt-processing batch. Higher = faster prefill, more memory."},
    {"key": "ubatch", "label": "Ubatch", "group": "Performance", "cat": "performance", "type": "int",
     "min": 32, "max": 4096, "step": 32, "default": 512,
     "help": "Compute batch. Usually same as batch."},
    {"key": "parallel", "label": "Parallel slots", "group": "Performance", "cat": "performance", "type": "int",
     "min": 1, "max": 8, "step": 1, "default": 1,
     "help": "Concurrent request slots (multiplies KV cache memory)."},
    {"key": "threads", "label": "CPU threads", "group": "Performance", "cat": "performance", "type": "int",
     "min": 0, "max": 64, "step": 1, "default": 0,
     "help": "0 = auto (all physical cores)."},
    {"key": "threads_batch", "label": "Batch threads", "group": "Performance", "cat": "performance", "type": "int",
     "min": 0, "max": 64, "step": 1, "default": 0,
     "help": "0 = auto."},
    {"key": "flash_attn", "label": "Flash attention", "group": "KV cache", "cat": "performance", "type": "bool",
     "default": True, "help": "Massively reduces KV cache VRAM. Keep on."},
    {"key": "kv_type", "label": "KV cache type", "group": "KV cache", "cat": "memory", "type": "enum",
     "options": ["f16", "q8_0", "q4_0", "i8"], "default": "f16",
     "help": "q8_0/q4_0 shrink KV cache VRAM with minor quality loss."},
    {"key": "cache_reuse", "label": "Prompt cache reuse", "group": "KV cache", "cat": "performance", "type": "bool",
     "default": True, "help": "Reuse cached KV across requests with same prefix. Big speed win for editors/tools."},
    {"key": "temp", "label": "Temperature", "group": "Sampling", "cat": "accuracy", "type": "float",
     "min": 0, "max": 2, "step": 0.01, "default": 0.7, "help": "Creativity. 0 = greedy. Low = deterministic tool calls."},
    {"key": "top_k", "label": "Top-K", "group": "Sampling", "cat": "accuracy", "type": "int",
     "min": 1, "max": 100, "step": 1, "default": 40, "help": "Keep top K tokens."},
    {"key": "top_p", "label": "Top-P", "group": "Sampling", "cat": "accuracy", "type": "float",
     "min": 0, "max": 1, "step": 0.01, "default": 0.95, "help": "Nucleus sampling."},
    {"key": "min_p", "label": "Min-P", "group": "Sampling", "cat": "accuracy", "type": "float",
     "min": 0, "max": 1, "step": 0.01, "default": 0.05, "help": "Cut low-probability tokens."},
    {"key": "repeat_penalty", "label": "Repeat penalty", "group": "Sampling", "cat": "accuracy", "type": "float",
     "min": 1, "max": 2, "step": 0.01, "default": 1.0, "help": "Damp repeats."},
    {"key": "mmap", "label": "Memory-mapped weights", "group": "Memory", "cat": "memory", "type": "bool",
     "default": True, "help": "mmap lets OS page weights in from disk as needed."},
    {"key": "mlock", "label": "Lock weights in RAM", "group": "Memory", "cat": "memory", "type": "bool",
     "default": False, "help": "Pin CPU-layer weights in RAM so they never swap. Best for speed when partially offloaded."},
]

CAT_LABEL = {
    "performance": "Performance / speed",
    "accuracy": "Accuracy / quality",
    "context": "Context window",
    "memory": "Memory (RAM / KV)",
}

CAT_COLOR = {
    "performance": "green",
    "accuracy": "blue",
    "context": "purple",
    "memory": "amber",
}

CAT_ORDER = ["performance", "context", "accuracy", "memory"]


def default_params():
    return {s["key"]: s["default"] for s in PARAM_SPEC}


def param_spec():
    return PARAM_SPEC


def spec_map():
    return {s["key"]: s for s in PARAM_SPEC}


def auto_tune(hw, file_size_mb, layers=None, ctx_limit=None):
    """Pick params that maximize speed+accuracy while fit fully inside VRAM.

    Reality-checked every load: uses the ACTIVE GPU free VRAM, RAM headroom
    and CPU count so the whole model + KV cache lands on the GPU.
    """
    gpu = (hw or {}).get("active_gpu") or {}
    gpus = (hw or {}).get("gpus") or []
    free_vram = int(gpu.get("vram_free_mb") or 0)
    total_vram = int(gpu.get("vram_total_mb") or 0) or free_vram
    ram = (hw or {}).get("ram") or {}
    ram_free_gb = float(ram.get("free_gb") or 0)
    cpu = int((hw or {}).get("cpu_count") or os_cpu_count())

    p = default_params()
    p["flash_attn"] = True
    p["cache_reuse"] = True

    # --- offload: fit the WHOLE model on GPU when possible, else max layers ---
    reserve = 300  # MB for CUDA context + buffers
    usable = max(0, free_vram - reserve)
    if file_size_mb and layers:
        if file_size_mb <= usable:
            p["n_gpu_layers"] = layers  # full offload
        else:
            per_layer = file_size_mb / max(layers, 1)
            ngl = int((usable * 0.98) / per_layer)
            p["n_gpu_layers"] = max(0, min(ngl, layers))
    elif file_size_mb and not layers:
        p["n_gpu_layers"] = 999
    else:
        p["n_gpu_layers"] = 999

    # --- context: biggest window that still fits ---
    # Use q8_0 KV cache by default — halves VRAM vs f16, doubling ctx budget.
    # f16 is faster but agents need context more than they need KV cache speed.
    p["kv_type"] = "q8_0"
    if layers:
        # q8_0 KV ≈ n_layer * 0.002 MB per token
        kv_mb_per_token = max(0.04, layers * 0.002)
        head = max(0, free_vram - file_size_mb - reserve)
        max_ctx = int(head / kv_mb_per_token) if head > 0 else 2048
    else:
        max_ctx = 8192
    if ctx_limit:
        max_ctx = min(max_ctx, ctx_limit)
    ctx = max(512, (max_ctx // 256) * 256)
    p["ctx"] = ctx

    # --- batch / threads tuned for GPUs (bigger is faster on CUDA) ---
    if free_vram >= 8000:
        p["batch"], p["ubatch"] = 2048, 512
    elif free_vram >= 3500:
        p["batch"], p["ubatch"] = 512, 512
    else:
        p["batch"], p["ubatch"] = 256, 256
    p["threads"] = 0
    p["threads_batch"] = 0
    p["parallel"] = 1 if free_vram < 6000 else 2

    # --- RAM: precommit only what we won't page out; mlock only when layers live in RAM ---
    partial = layers and p["n_gpu_layers"] < layers
    p["mlock"] = bool(partial and ram_free_gb > 2)
    p["mmap"] = True
    p["temp"], p["top_k"], p["top_p"], p["min_p"], p["repeat_penalty"] = 0.7, 40, 0.95, 0.05, 1.0
    return p


def os_cpu_count():
    import os
    return os.cpu_count() or 1