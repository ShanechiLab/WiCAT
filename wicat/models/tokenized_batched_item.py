from dataclasses import dataclass
from typing import List

import torch


@dataclass
class TokenizedBatchedItem:
    tokens: torch.Tensor
    seq_lens: List[int]
    position_ids: torch.Tensor
    patch_ids: torch.Tensor
    token_add_mask: torch.Tensor
