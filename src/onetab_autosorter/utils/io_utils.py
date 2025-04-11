import os
import json
import hashlib
from typing import List, Dict


class PythonSetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


def compute_hash(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()

def cache_path(base_dir: str, name: str, ext="json"):
    return os.path.join(base_dir, f"{name}.{ext}")


def get_cache_file(input_path: str, stage_name: str, cache_dir: str) -> str:
    hash_str = compute_hash(input_path)
    return os.path.join(cache_dir, f"{stage_name}_{hash_str}.json")


def load_json(file_path: str) -> Dict:
    #? NOTE: might just import and use this within the JSON parser I made for no reason
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as fptr:
        return dict(json.load(fptr))


def save_json(file_path: str, data: Dict):
    with open(file_path, "w", encoding="utf-8") as fptr:
        json.dump(data, fptr, indent=2, cls=PythonSetEncoder)  # Use the custom encoder for sets


