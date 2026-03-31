from functools import partial
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat

from wicat.enums import EmbeddingAddType, TemporalPoolingType
from wicat.models.MLP import MLP
from wicat.models.CNN import CNN

from wicat.utility.utils import init_logger

std_logger = init_logger("Tokenizer")


class ImagePatchTokenizer(nn.Module):
    def __init__(
        self,
        temporal_patch_size: int,
        spatial_patch_size: int,
        session_d_input_dict: Dict,
        subjects: List,
        d_hidden: int,
        layer_list: List = [512],
        activation: str = "tanh",
        dropout: float = 0.1,
        learn_patch_embedding: bool = False,
        learn_global_patch: bool = False,
        use_conv_for_image: bool = False,
        kernel_size: int = 5,
        dilation: int = 1,
        stride: int = 2,
        max_count: int = 5,
        initialization_std: float = 1,
        do_temporal_pool: bool = False,
        temporal_pool_type: str = "average",
        **kwargs,
    ):
        super().__init__()

        self.temporal_patch_size = temporal_patch_size
        self.spatial_patch_size = spatial_patch_size
        self.d_hidden = d_hidden
        self.layer_list = layer_list
        self.activation = activation
        self.dropout = dropout
        self.learn_patch_embedding = learn_patch_embedding
        self.learn_global_patch = learn_global_patch
        self.use_conv_for_image = use_conv_for_image
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.max_count = max_count
        self.session_d_input_dict = session_d_input_dict
        self.subjects = subjects
        self.initialization_std = initialization_std
        self.do_temporal_pool = do_temporal_pool
        self.temporal_pool_type = temporal_pool_type
        self.temporal_pool_type_enum = TemporalPoolingType.get_modes(temporal_pool_type)
        self.stride = stride

        if self.do_temporal_pool:
            self.temporal_multiplier = 1
            if self.temporal_pool_type_enum == TemporalPoolingType.LEARNABLE:
                self.temporal_pooler = nn.Linear(self.temporal_patch_size, 1)
            elif self.temporal_pool_type_enum == TemporalPoolingType.AVERAGE:
                self.temporal_pooler = partial(torch.mean, dim=-1, keepdim=True)
        else:
            self.temporal_multiplier = self.temporal_patch_size


        if self.use_conv_for_image:
            self.embedder = CNN(
                temporal_patch_size=self.temporal_patch_size,
                spatial_patch_size=self.spatial_patch_size,
                d_hidden=self.d_hidden,
                layer_list=self.layer_list,
                dropout=self.dropout,
                activation=self.activation,
                kernel_size=self.kernel_size,
                stride=self.stride,
            )
            self.pad_value = -0.00314
        else:
            self.embedder = MLP(
                d_input=(self.temporal_multiplier * self.spatial_patch_size * self.spatial_patch_size),
                d_out=self.d_hidden,
                layer_list=self.layer_list,
                activation=self.activation,
                dropout=self.dropout,
            )
            self.pad_value = -0.00314
        self.embedder.apply(self._init_weights)


        self.set_patch_embeddings()
        if self.patch_embeddings:
            self.patch_embeddings.apply(self._init_weights)

    def _init_weights(
        self,
        module,
    ):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.initialization_std)

    def _create_embedding_dict(
        self,
        is_learnable: bool,
        num_tokens: int,
        embedding_dim: int,
        keys: Union[list, dict],
    ):
        if not is_learnable:
            return None

        embedding = nn.ModuleDict()
        for k in keys:
            embedding[k] = nn.Embedding(
                num_tokens,
                embedding_dim,
            )
        return embedding

    def get_num_spatial_patches(self, height, width):
        num_patches_h = height // self.spatial_patch_size + (1 if height % self.spatial_patch_size > 0 else 0)
        num_patches_w = width // self.spatial_patch_size + (1 if width % self.spatial_patch_size > 0 else 0)
        return num_patches_h * num_patches_w

    def set_patch_embeddings(self):
        if self.learn_patch_embedding:
            self.patch_embeddings = nn.ModuleDict()
            if self.learn_global_patch:
                for ss, shape in self.session_d_input_dict.items():
                    height, width = int(shape[1]), int(shape[1])
                    break
                num_patches = self.get_num_spatial_patches(height, width)
                ss = 'global'
                self.patch_embeddings[ss] = nn.Embedding(num_patches, self.d_hidden)

            else:
                for ss, shape in self.session_d_input_dict.items():
                    if len(shape) == 3:
                        height, width = shape[1], shape[2]
                        if height == 1:
                            height = shape[0]
                    else:
                        height, width = shape[0], shape[1]
                    num_patches = self.get_num_spatial_patches(height, width)
                    self.patch_embeddings[ss] = nn.Embedding(num_patches, self.d_hidden)
        else:
            self.patch_embeddings = None


    def _create_patches_for_batched_tensor(self, x: torch.Tensor):
        assert x.dim() == 5
        B, N, C, H, W = x.shape

        pad_h = (self.spatial_patch_size - H % self.spatial_patch_size) % self.spatial_patch_size
        pad_w = (self.spatial_patch_size - W % self.spatial_patch_size) % self.spatial_patch_size
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), value=self.pad_value)
        _, _, _, padded_h, padded_w = x.shape

        temporal_pad_size = (self.temporal_patch_size - N % self.temporal_patch_size) % self.temporal_patch_size
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, temporal_pad_size), value=self.pad_value)
        _, padded_n, _, _, _ = x.shape

        if self.use_conv_for_image:
            x_patched = rearrange(
                x,
                "b (n tp) c (h sp1) (w sp2) -> b n (h w) c tp sp1 sp2",
                tp=self.temporal_patch_size,
                sp1=self.spatial_patch_size,
                sp2=self.spatial_patch_size,
            )
        else:
            x_patched = rearrange(
                x,
                "b (n tp) c (h sp1) (w sp2) -> b n (h w) tp (sp1 sp2 c)",
                tp=self.temporal_patch_size,
                sp1=self.spatial_patch_size,
                sp2=self.spatial_patch_size,
            )

        position_ids_patched = torch.arange(x_patched.shape[1]).to(x_patched.device)
        position_ids_patched = repeat(
            position_ids_patched,
            "n -> b (n nsp)",
            b=B,
            nsp=x_patched.shape[2],
        )

        patch_ids = torch.arange(x_patched.shape[2]).to(x_patched.device)
        patch_ids = repeat(patch_ids, "nsp -> b nsp", b=B)
        patch_ids = repeat(patch_ids, "b nsp -> b (nt nsp)", nt=x_patched.shape[1])

        if not self.use_conv_for_image:
            x_patched = rearrange(x_patched, "b nt nsp tps (sps c) -> b (nt nsp) tps (sps c)", sps=self.spatial_patch_size**2, c=C)
            token_add_mask = (x_patched != self.pad_value).any(dim=-1)
        else:
            x_patched_flat = rearrange(x_patched, "b nt nsp c tp sp1 sp2 -> b (nt nsp) tp (sp1 sp2 c)")
            token_add_mask = (x_patched_flat != self.pad_value).any(dim=-1) 
        seq_lens_patched = [patch_ids.shape[1]] * patch_ids.shape[0]
        
        return (
            x_patched,
            position_ids_patched,
            patch_ids,
            seq_lens_patched,
            token_add_mask,
        )

    def create_patches(
        self,
        x: Union[torch.Tensor, List],
    ):
        if isinstance(x, torch.Tensor):
            (
                x_patched,
                position_ids_patched,
                patch_ids,
                seq_lens_patched,
                token_add_mask,
            ) = self._create_patches_for_batched_tensor(x)
        else:
            (
                x_patched,
                position_ids_patched,
                patch_ids,
                seq_lens_patched,
                token_add_mask,
            ) = ([], [], [], [], [])

            for x_this in x:
                (
                    x_patched_batched,
                    position_ids_patched_batched,
                    patch_ids_batched,
                    seq_lens_patched_batched,
                    token_add_mask_batched,
                ) = self._create_patches_for_batched_tensor(
                    x_this.reshape(1, *x_this.shape)
                )

                x_patched.append(x_patched_batched[0])
                position_ids_patched.append(position_ids_patched_batched[0])
                patch_ids.append(patch_ids_batched[0])
                seq_lens_patched.append(seq_lens_patched_batched[0])
                token_add_mask.append(token_add_mask_batched[0])

            x_patched = torch.cat(x_patched).unsqueeze(dim=0)
            position_ids_patched = torch.cat(position_ids_patched).unsqueeze(dim=0)
            patch_ids = torch.cat(patch_ids).unsqueeze(dim=0)
            token_add_mask = torch.cat(token_add_mask).unsqueeze(dim=0)
        return (
            x_patched,
            position_ids_patched,
            patch_ids,
            seq_lens_patched,
            token_add_mask,
        )

    def add_patch_embeddings(
        self, x, x_patched, patch_ids, seq_lens_patched, subject_sessions
    ):
        if self.patch_embeddings:
            patch_embeddings = []
            if isinstance(x, list):
                patch_ids_split = torch.split(patch_ids, seq_lens_patched, dim=1)
            else:
                patch_ids_split = patch_ids


            for i, ss in enumerate(subject_sessions):
                if self.learn_global_patch:
                    patch_embedding_i = self.patch_embeddings['global'](patch_ids_split[i])
                else:
                    patch_embedding_i = self.patch_embeddings[ss](patch_ids_split[i])
                patch_embeddings.append(patch_embedding_i)

            if isinstance(x, list):
                patch_embeddings = torch.cat(patch_embeddings, dim=1)
            else:
                patch_embeddings = torch.stack(patch_embeddings, dim=0)
            x_patched += patch_embeddings
        return x_patched

    def forward(
        self,
        x: Union[torch.Tensor, List],
        subject_sessions: List,
        subjects: List,
        **kwargs,
    ):
        x_patched, position_ids_patched, patch_ids, seq_lens_patched, token_add_mask = (
            self.create_patches(x=x)
        )
        token_add_mask = token_add_mask.all(-1)

        if self.use_conv_for_image:
            data_patched = rearrange(x_patched, "b nt nsp c tps sp1 sp2 -> b (nt nsp) tps (sp1 sp2 c)")
            x_patched = self.embedder(x_patched)

        else:
            data_patched = x_patched
            if not self.do_temporal_pool or self.temporal_patch_size == 1:
                x_patched = rearrange(x_patched, "b n tps sps -> b n (tps sps)")

            x_patched = self.embedder(x_patched)

            if self.do_temporal_pool and self.temporal_patch_size > 1:
                x_patched = rearrange(x_patched, "b n tps d -> b n d tps")
                x_patched = self.temporal_pooler(x_patched).squeeze(dim=-1)

        x_patched = self.add_patch_embeddings(
            x=x,
            x_patched=x_patched,
            patch_ids=patch_ids,
            seq_lens_patched=seq_lens_patched,
            subject_sessions=subject_sessions,
        )

        return (
            data_patched,
            x_patched,
            position_ids_patched,
            patch_ids,
            seq_lens_patched,
            token_add_mask,
        )

