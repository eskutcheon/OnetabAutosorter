import os
import json
import hashlib
import functools
import pickle
from typing import List, Dict, Any, Callable


class PythonSetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


def compute_hash(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()

# def cache_path(base_dir: str, name: str, ext="json"):
#     return os.path.join(base_dir, f"{name}.{ext}")


def get_hashed_path_from_string(input_path: str, stage_name: str, cache_dir: str) -> str:
    hash_str = compute_hash(input_path)
    return get_hashed_path_from_hash(hash_str, stage_name, cache_dir)

def get_hashed_path_from_hash(hash_str: str, stage_name: str, cache_dir: str) -> str:
    file_prefix = "" if not stage_name else f"{stage_name}_"
    return os.path.join(cache_dir, f"{file_prefix}{hash_str}.json")


def load_json(file_path: str) -> Any:
    #? NOTE: might just import and use this within the JSON parser I made for no reason
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as fptr:
        return json.load(fptr)


def save_json(file_path: str, data: Dict):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as fptr:
        json.dump(data, fptr, indent=2, cls=PythonSetEncoder)  # Use the custom encoder for sets



# # TODO: create variant of this that accepts a boolean in the decorator to perform caching or not
# def cache_output(func):
#     """A decorator that caches the results of a function using a hash of its arguments."""
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         # Create a unique key for the function call
#         # TODO: use md5 instead of sha256
#         key = hashlib.sha256()
#         key.update(pickle.dumps((args, kwargs)))
#         cache_key = key.hexdigest()
#         # TODO: remove this hard-coding of the cache directory
#         cache_dir = ".cache"
#         os.makedirs(cache_dir, exist_ok=True)
#         cache_path = os.path.join(cache_dir, cache_key + ".pkl")
#         # Check if the result is already cached
#         if os.path.exists(cache_path):
#             with open(cache_path, "rb") as f:
#                 return pickle.load(f)
#         # If not cached, compute and cache the result
#         result = func(*args, **kwargs)
#         with open(cache_path, "wb") as f:
#             pickle.dump(result, f)
#         return result
#     return wrapper


#~ FIXCHANGE: rather than a decorator, this may be good to have as a wrapper function to execute
    #~ the utility functions so that the utils can be more general without decoration
#~ it would be great for avoiding the super high dependence on the config object
    #~ also really think the config objects should only be managed within the pipeline factory
    #~ while it acts as dispatcher for wrapping the stages' utility functions
def cache_stage(stage_name: str):
    """Decorator to automatically cache the result of a stage based on input arguments."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(config, *args, **kwargs):
            cache_file = get_hashed_path_from_string(config.input_file, stage_name, config.checkpoints.cache_dir)
            reuse_flag = getattr(config.checkpoints, f"reuse_{stage_name}", True)
            save_flag = getattr(config.checkpoints, f"save_{stage_name}", True)
            if reuse_flag and os.path.exists(cache_file):
                print(f"Loading cached data for '{stage_name}' stage from {cache_file}")
                return load_json(cache_file)
            print(f"No cached data found or reuse disabled for '{stage_name}'. Running stage...")
            result = func(config, *args, **kwargs)
            if save_flag:
                save_json(cache_file, result)
                print(f"Saved cached data for '{stage_name}' stage at {cache_file}")
            return result
        return wrapper
    return decorator