import json
from .config import cfg_from_dict, Cfg

def load_config(config_path: str, overrides=None)->dict:
    with open(config_path) as f:
        cfg = json.load(f)
    overrides = overrides or []
    for s in overrides:
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        try:
            v = json.loads(v)
        except (json.JSONDecodeError, ValueError):
            pass
        cfg[k] = v
    return cfg

def build_cfg_from_json(j: dict) -> Cfg:
    # for train
    return cfg_from_dict(j)

def build_cfg_from_checkpoint(checkpoint: dict) -> Cfg:
    # for test 
    return cfg_from_dict(checkpoint.get("config", {}))