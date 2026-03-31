import torch
import torch.nn as nn
import xformers.ops as xops
from einops import rearrange, repeat

from wicat.utility.torch_utils import get_activation_function


class RotaryEmbedding(nn.Module):
    def __init__(self, d_head, base=10000, max_position=1024, device=None):
        super().__init__()

        self.d_head = d_head
        self.max_position = max_position
        self.device = device

        inv_freq = 1 / (
            base
            ** (torch.arange(0, self.d_head, 2).float().to(self.device) / self.d_head)
        )
        self.register_buffer("inv_freq", inv_freq)
        self.build_cache()

    def build_cache(self):
        t = torch.arange(
            self.max_position,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, position_ids):
        cos = self.cos_cached[position_ids].unsqueeze(2)
        sin = self.sin_cached[position_ids].unsqueeze(2)
        return cos, sin


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos.to(q)) + (rotate_half(q) * sin.to(q))
    k_embed = (k * cos.to(k)) + (rotate_half(k) * sin.to(k))
    return q_embed, k_embed


class RMSNorm(nn.Module):
    def __init__(self, d_hidden, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_hidden))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * x).to(input_dtype)


class RotarySelfAttention(nn.Module):
    def __init__(
        self,
        d_hidden,
        num_heads=8,
        max_position=1024,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.d_head = self.d_hidden // self.num_heads
        self.dropout = nn.Dropout(dropout)
        self.max_position = max_position

        assert (
            self.d_hidden % self.num_heads == 0
        ), f"Number of attention heads: {self.num_heads} must divide embedding dimension: {self.d_hidden}."

        self.qkv_proj = nn.Linear(self.d_hidden, 3 * self.d_hidden, bias=True)
        self.o_proj = nn.Linear(self.d_hidden, self.d_hidden, bias=True)
        self.rotary_emb = RotaryEmbedding(self.d_head, max_position=self.max_position)

    def get_qkv(self, x):
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = rearrange(q, "b n (h d_h) -> b n h d_h", h=self.num_heads)
        k = rearrange(k, "b n (h d_h) -> b n h d_h", h=self.num_heads)
        v = rearrange(v, "b n (h d_h) -> b n h d_h", h=self.num_heads)
        return q, k, v

    def get_attention_out(self, q, k, v, seq_lens=None, causal=False):
        if seq_lens is not None and q.shape[0] == 1:
            if causal:
                attn_bias = xops.fmha.attn_bias.BlockDiagonalCausalMask.from_seqlens(seq_lens).to(q.device)
            else:
                attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(seq_lens).to(q.device)
        else:
            attn_bias = None

        attention_out = xops.memory_efficient_attention(
            q,
            k,
            v,
            p=0,
            attn_bias=attn_bias,
        )
        attention_out = self.dropout(attention_out)
        attention_out = rearrange(attention_out, "b n h d_h -> b n (h d_h)")
        return self.o_proj(attention_out)

    def forward(self, x, position_ids=None, seq_lens=None, causal=False):
        if seq_lens is None and x.shape[0] == 1:
            raise ValueError(
                "'seq_lens' for memory efficient attention with variable length sequences (x.shape[0] == 1) must be non-None."
            )

        if position_ids is None:
            if x.shape[0] == 1:
                position_ids = [torch.arange(seq_len_) for seq_len_ in seq_lens]
                position_ids = torch.cat(position_ids).unsqueeze(dim=0).int().to(x.device)
            else:
                position_ids = repeat(torch.arange(x.shape[1]), "n -> b n", b=x.shape[0]).int().to(x.device)

        q, k, v = self.get_qkv(x)
        cos, sin = self.rotary_emb(position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        v = v.to(q)

        return self.get_attention_out(q, k, v, seq_lens=seq_lens, causal=causal)


class GatedTransformerMLP(nn.Module):
    def __init__(self, d_hidden, mlp_ratio=4, activation="silu", dropout=0.1):
        super().__init__()
        d_feedforward = mlp_ratio * d_hidden
        self.gate_proj = nn.Linear(d_hidden, d_feedforward, bias=True)
        self.down_proj = nn.Linear(d_feedforward, d_hidden, bias=True)
        self.up_proj = nn.Linear(d_hidden, d_feedforward, bias=True)
        self.activation_fn = get_activation_function(activation)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(self.activation_fn(self.gate_proj(x)) * self.up_proj(x))
        return self.dropout2(self.down_proj(x))


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_hidden,
        mlp_ratio=4,
        norm="rmsnorm",
        norm_eps=1e-6,
        activation="silu",
        num_heads=8,
        dropout=0.1,
        norm_first=True,
        **attention_module_kwargs,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.norm_first = norm_first

        self.attention = RotarySelfAttention(
            d_hidden=d_hidden,
            num_heads=num_heads,
            dropout=dropout,
            **attention_module_kwargs,
        )
        self.mlp = GatedTransformerMLP(
            d_hidden=d_hidden,
            mlp_ratio=mlp_ratio,
            activation=activation,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

        if norm.lower() == "rmsnorm":
            self.norm1 = RMSNorm(d_hidden, eps=norm_eps)
            self.norm2 = RMSNorm(d_hidden, eps=norm_eps)
        elif norm.lower() == "layernorm":
            self.norm1 = nn.LayerNorm(d_hidden, eps=norm_eps)
            self.norm2 = nn.LayerNorm(d_hidden, eps=norm_eps)
        else:
            raise NotImplementedError(
                f"Requested normalization layer: {norm} is not implemented for RotaryTransformerEncoderLayer."
            )

    def forward(self, x, position_ids=None, seq_lens=None, causal=False):
        residual = x
        if self.norm_first:
            x = self.norm1(x)
            x = self.attention(x=x, position_ids=position_ids, seq_lens=seq_lens, causal=causal)
            x = self.dropout(x)
            x = residual + x

            residual = x
            x = self.norm2(x)
            x = self.mlp(x)
            x = residual + x
        else:
            x = self.attention(
                x=x,
                position_ids=position_ids,
                seq_lens=seq_lens,
                causal=causal,
            )
            x = self.dropout(x)
            x = residual + x
            x = self.norm1(x)

            residual = x
            x = self.mlp(x)
            x = residual + x
            x = self.norm2(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers,
        d_hidden,
        mlp_ratio=4,
        norm="rmsnorm",
        norm_eps=1e-6,
        activation="gelu",
        num_heads=8,
        dropout=0.1,
        norm_first=True,
        use_final_norm=True,
        return_states=False,
        **attention_module_kwargs,
    ):
        self.d_hidden = d_hidden
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CustomTransformerEncoderLayer(
                    d_hidden=d_hidden,
                    mlp_ratio=mlp_ratio,
                    norm=norm,
                    norm_eps=norm_eps,
                    activation=activation,
                    num_heads=num_heads,
                    dropout=dropout,
                    norm_first=norm_first,
                    **attention_module_kwargs,
                )
                for _ in range(num_layers)
            ]
        )
        if use_final_norm:
            if norm.lower() == "rmsnorm":
                self.norm = RMSNorm(d_hidden, eps=norm_eps)
            elif norm.lower() == "layernorm":
                self.norm = nn.LayerNorm(d_hidden, eps=norm_eps)
            else:
                raise NotImplementedError(
                    f"Requested normalization layer: {norm} is not implemented for Transformer final norm."
                )
        else:
            self.norm = None

        self.return_states = return_states

    def forward(self, x, position_ids=None, seq_lens=None, **kwargs):
        states = []
        for layer in self.layers:
            x = layer(
                x=x,
                position_ids=position_ids,
                seq_lens=seq_lens,
                **kwargs,
            )
            if self.return_states:
                states.append(x)
        if self.norm:
            x = self.norm(x)

        return states, x
