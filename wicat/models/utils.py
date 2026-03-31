from typing import Dict, Tuple


def get_nested(d: Dict, key: str, default=None):
    cur = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def split_state_dict_by_prefix(state_dict: Dict, prefix: str) -> Dict:
    out = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            out[key[len(prefix) :]] = value
    return out


def load_prefixed_weights(module, state_dict: Dict, prefix: str, strict: bool = False) -> Tuple[list, list]:
    sub_state = split_state_dict_by_prefix(state_dict, prefix)
    missing, unexpected = module.load_state_dict(sub_state, strict=strict)
    return missing, unexpected
