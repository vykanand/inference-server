PARAM_SPEC = [
    {"key": "n_gpu_layers", "label": "GPU layers (offload)", "group": "GPU offload",
     "type": "int", "min": 0, "max": 999, "step": 1, "default": 999,
     "help": "How many model layers to offload to the GPU. Set to total layer count (or 999) to fit the whole model on GPU if VRAM allows. Lower = more RAM, slower."},
    {"key": "ctx", "label": "Context size", "group": "Context", "type": "int",
     "min": 512, "max": 131072, "step": 256, "default": 4096,
     "help": "Total prompt+completion token window."},
    {"key": "batch", "label": "Batch size", "group": "Performance", "type": "int",
     "min": 32, "max": 8192, "step": 32, "default": 512,
     "help": "Prompt-processing batch. Higher = faster prefill, more memory."},
    {"key": "ubatch", "label": "Ubatch", "group": "Performance", "type": "int",
     "min": 32, "max": 4096, "step": 32, "default": 512,
     "help": "Compute batch. Usually same as batch."},
    {"key": "parallel", "label": "Parallel slots", "group": "Performance", "type": "int",
     "min": 1, "max": 8, "step": 1, "default": 1,
     "help": "Concurrent request slots (multiplies KV cache memory)."},
    {"key": "threads", "label": "CPU threads", "group": "Performance", "type": "int",
     "min": 1, "max": 64, "step": 1, "default": 0,
     "help": "0 = auto (all cores)."},
    {"key": "threads_batch", "label": "Batch threads", "group": "Performance", "type": "int",
     "min": 0, "max": 64, "step": 1, "default": 0,
     "help": "0 = auto."},
    {"key": "flash_attn", "label": "Flash attention", "group": "KV cache", "type": "bool",
     "default": True, "help": "Massively reduces KV cache VRAM. Keep on."},
    {"key": "kv_type", "label": "KV cache type", "group": "KV cache", "type": "enum",
     "options": ["f16", "q8_0", "q4_0", "i8"], "default": "f16",
     "help": "q8_0/q4_0 shrink KV cache VRAM with minor quality loss."},
    {"key": "cache_reuse", "label": "Prompt cache reuse", "group": "KV cache", "type": "bool",
     "default": False, "help": "Reuse cached KV across requests with same prefix."},
    {"key": "temp", "label": "Temperature", "group": "Sampling", "type": "float",
     "min": 0, "max": 2, "step": 0.01, "default": 0.8, "help": "Creativity. 0 = greedy."},
    {"key": "top_k", "label": "Top-K", "group": "Sampling", "type": "int",
     "min": 1, "max": 100, "step": 1, "default": 40, "help": "Keep top K tokens."},
    {"key": "top_p", "label": "Top-P", "group": "Sampling", "type": "float",
     "min": 0, "max": 1, "step": 0.01, "default": 0.95, "help": "Nucleus sampling."},
    {"key": "min_p", "label": "Min-P", "group": "Sampling", "type": "float",
     "min": 0, "max": 1, "step": 0.01, "default": 0.05, "help": "Cut low-probability tokens."},
    {"key": "repeat_penalty", "label": "Repeat penalty", "group": "Sampling", "type": "float",
     "min": 1, "max": 2, "step": 0.01, "default": 1.0, "help": "Damp repeats."},
    {"key": "mmap", "label": "Memory-mapped weights", "group": "Memory", "type": "bool",
     "default": True, "help": "mmap lets OS page weights in from disk as needed."},
    {"key": "mlock", "label": "Lock weights in RAM", "group": "Memory", "type": "bool",
     "default": False, "help": "Prevent OS from swapping weights out. Uses real RAM."},
]


def default_params():
    return {s["key"]: s["default"] for s in PARAM_SPEC}


def param_spec():
    return PARAM_SPEC
