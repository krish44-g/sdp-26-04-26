import json
import os

CACHE_FILE = "/tmp/spineai_cache.json"

def store_analysis(analysis_id: str, data: dict):
    cache = _load()
    cache[analysis_id] = data
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

def get_analysis(analysis_id: str) -> dict:
    return _load().get(analysis_id)

def _load() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}