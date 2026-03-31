from typing import List

import torch
import torch.nn as nn

from wicat.models.tokenized_batched_item import TokenizedBatchedItem
from wicat.models.tokenizer import ImagePatchTokenizer
from wicat.models.transformer import Transformer
from wicat.models.MLP import MLP


def _build_tokenizer(config: dict, metadata):
    class_name = config.get("class_name", "ImagePatchTokenizer")
    if class_name != "ImagePatchTokenizer":
        raise ValueError(
            f"Only ImagePatchTokenizer is supported in WiCAT, got: {class_name}"
        )
    kwargs = dict(config)
    kwargs.pop("class_name", None)
    kwargs["session_d_input_dict"] = metadata.get_subject_session_d_input()
    kwargs["subjects"] = metadata.get_subjects()
    return ImagePatchTokenizer(**kwargs)


class WiCAT(nn.Module):
    def __init__(self, model_config: dict, metadata):
        super().__init__()
        self.metadata = metadata

        self.tokenizer = _build_tokenizer(
            config=model_config["tokenizer"],
            metadata=metadata,
        )

        self.backbone = Transformer(**model_config["backbone"])
        self.d_hidden = model_config["backbone"]["d_hidden"]
        self.decoder_config = dict(model_config.get("decoder", {}))

        out_dim = self._infer_output_dim_from_metadata()
        self.time_bins = int(self.decoder_config.pop("time_bins", 205))
        self.patches_per_time = int(self.decoder_config.pop("patches_per_time", 16))
        self.patch_merge = str(self.decoder_config.pop("patch_merge", "mean")).lower()
        self.decoder_head = None
        self.create_downstream_head(output_dim=out_dim)

    def _infer_output_dim_from_metadata(self) -> int:
        d_out_dict = self.metadata.get_subject_session_d_out()
        if not d_out_dict:
            raise ValueError("No d_kinem information found in metadata to build decoder head.")

        unique_dims = sorted({int(v) for v in d_out_dict.values()})
        if len(unique_dims) != 1:
            raise ValueError(
                f"Expected a single d_kinem across sessions, got: {unique_dims}."
            )
        return unique_dims[0]

    def create_downstream_head(self, output_dim: int):
        class_name = str(self.decoder_config.get("class_name", "MLP"))
        if class_name != "MLP":
            raise ValueError(f"Only MLP decoder is supported, got: {class_name}")

        decoder_kwargs = dict(self.decoder_config)
        decoder_kwargs.pop("class_name", None)
        self.decoder_head = MLP(
            d_input=self.d_hidden,
            d_out=output_dim,
            **decoder_kwargs,
        )

    def tokenize(self, x: torch.Tensor, subject_sessions: List[str], subjects: List[str]) -> TokenizedBatchedItem:
        _, tokens, position_ids, patch_ids, seq_lens, token_add_mask = self.tokenizer(
            x=x,
            subject_sessions=subject_sessions,
            subjects=subjects,
        )
        return TokenizedBatchedItem(
            tokens=tokens,
            seq_lens=seq_lens,
            position_ids=position_ids,
            patch_ids=patch_ids,
            token_add_mask=token_add_mask,
        )

    def get_latent_embeddings(self, x: torch.Tensor, subject_sessions: List[str], subjects: List[str]):
        tokenized = self.tokenize(x, subject_sessions, subjects)
        _, latents = self.backbone(
            x=tokenized.tokens,
            seq_lens=tokenized.seq_lens,
            position_ids=tokenized.position_ids,
        )
        return latents

    def forward(self, x: torch.Tensor, subject_sessions: List[str], subjects: List[str]):
        latents = self.get_latent_embeddings(x, subject_sessions, subjects)

        if x.dim() < 2:
            raise ValueError("Expected x to have batch dimension.")
        batch_size = x.shape[0]

        expected_tokens = self.time_bins * self.patches_per_time
        if latents.shape[1] != expected_tokens:
            raise ValueError(
                f"Expected token count {expected_tokens} (= {self.time_bins}*{self.patches_per_time}), got {latents.shape[1]}"
            )

        latents_reshaped = latents.reshape(
            batch_size,
            self.time_bins,
            self.patches_per_time,
            latents.shape[-1],
        )

        if self.patch_merge == "mean":
            merged = latents_reshaped.mean(dim=2)
        elif self.patch_merge == "sum":
            merged = latents_reshaped.sum(dim=2)
        else:
            raise ValueError(f"Unsupported patch_merge mode: {self.patch_merge}")

        predictions = self.decoder_head(merged)
        return predictions

    def get_task_params(self):
        return [*self.decoder_head.named_parameters()]

    def get_upstream_params(self):
        return [*self.tokenizer.named_parameters(), *self.backbone.named_parameters()]
