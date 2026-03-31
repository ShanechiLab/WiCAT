from typing import Dict, Optional, Tuple

import torch

from wicat.data.metadata import Metadata
from wicat.models.model import WiCAT


def build_model(model_config: Dict, metadata: Metadata) -> WiCAT:
    return WiCAT(model_config=model_config["model"], metadata=metadata)


def _normalize_state_dict(ckpt_obj: Dict) -> Dict:
    state_dict = ckpt_obj.get("state_dict", ckpt_obj)

    if any(k.startswith("model.") for k in state_dict.keys()):
        return {k[len("model.") :]: v for k, v in state_dict.items() if k.startswith("model.")}
    return state_dict


def load_model_from_checkpoint(
    model: WiCAT,
    checkpoint_path: str,
    strict: bool = False,
    map_location: str = "cpu",
) -> Tuple[list, list]:
    ckpt_obj = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    state_dict = _normalize_state_dict(ckpt_obj)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return missing, unexpected


def resolve_metadata(metadata_csv: Optional[str], checkpoint_path: Optional[str]) -> Metadata:
    if metadata_csv:
        return Metadata(load_path=metadata_csv)

    raise ValueError(
        "Metadata must be provided via metadata_csv in config; checkpoint metadata fallback is disabled."
    )
