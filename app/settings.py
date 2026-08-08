import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.json")

DEFAULTS = {
    "openrouter_key": "",
    "openrouter_model": "meta-llama/llama-3.1-8b-instruct:free",
    "hf_token": "",
    "download_dir": "models",
}


def load():
    data = dict(DEFAULTS)
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data.update(json.load(f))
    except Exception:
        pass
    return data


def save(patch):
    data = load()
    data.update(patch)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data