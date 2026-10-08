"""NomicBert architecture for MLX (nomic-embed-text models).

``mlx-embeddings`` (Blaizzy/mlx-embeddings) ships no ``nomic_bert`` module,
but nomic-embed-text-v1.5 is a standard NomicBert:

- word + token-type embeddings, **no learned position embeddings**
- **RoPE** rotary position embeddings (base 1000, full head dim)
- **fused bias-free QKV** projection (``attn.Wqkv``)
- **SwiGLU** MLP (``mlp.fc11`` / ``fc12`` / ``fc2``), bias-free
- post-norm transformer blocks (``norm1`` after attention, ``norm2`` after MLP)

This module ports the reference implementation from
ml-explore/mlx-swift-lm (``Libraries/MLXEmbedders/Models/NomicBert.swift``)
to the mlx-embeddings loader contract (``ModelArgs`` + ``Model`` +
``sanitize``). ``register_nomic_bert()`` injects it as
``mlx_embeddings.models.nomic_bert`` so the package's loader finds it via
``importlib.import_module``.
"""

from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_embeddings.models.base import BaseModelArgs, BaseModelOutput, normalize_embeddings


def _pool_by_config(token_embeddings, attention_mask, pooling_config):
    """Pool token embeddings per the model's pooling config.

    Local copy of mlx-embeddings' ``pool_by_config`` — the released 0.1.0
    wheel predates the ``models/pooling`` module, so we don't import it.
    Supports both the simple ``{"pooling_mode": ...}`` format and the legacy
    sentence-transformers flag format (``pooling_mode_mean_tokens: true``).
    """
    cfg = dict(pooling_config or {})
    legacy = {
        "pooling_mode_cls_token": "cls",
        "pooling_mode_max_tokens": "max",
        "pooling_mode_mean_tokens": "mean",
        "pooling_mode_lasttoken": "lasttoken",
    }
    if "pooling_mode" not in cfg:
        active = [name for key, name in legacy.items() if cfg.get(key)]
        cfg["pooling_mode"] = active[0] if active else "mean"
    mode = cfg["pooling_mode"]
    b, l, d = token_embeddings.shape

    if mode == "mean":
        mask = mx.broadcast_to(mx.expand_dims(attention_mask, -1), token_embeddings.shape).astype(mx.float32)
        return mx.sum(token_embeddings * mask, axis=1) / mx.maximum(mx.sum(mask, axis=1), 1e-9)
    if mode == "cls":
        idx = mx.broadcast_to(mx.argmax(attention_mask, axis=1)[:, None, None], (b, 1, d))
        return mx.squeeze(mx.take_along_axis(token_embeddings, idx, axis=1), axis=1)
    if mode == "max":
        mask = mx.broadcast_to(mx.expand_dims(attention_mask, -1), token_embeddings.shape).astype(token_embeddings.dtype)
        return mx.max(mx.where(mask == 0, -float("inf"), token_embeddings), axis=1)
    if mode == "lasttoken":
        flipped = attention_mask[:, ::-1]
        flip_idx = mx.argmax(flipped, axis=1)
        has_any = mx.max(flipped, axis=1)
        flip_idx = mx.where(has_any == 0, l - 1, flip_idx)
        last_idx = l - flip_idx - 1
        idx = mx.broadcast_to(last_idx[:, None, None], (b, 1, d))
        mask = mx.broadcast_to(attention_mask[:, :, None], token_embeddings.shape).astype(token_embeddings.dtype)
        return mx.squeeze(mx.take_along_axis(token_embeddings * mask, idx, axis=1), axis=1)
    raise ValueError(f"Unsupported pooling mode {mode!r}")


@dataclass
class ModelArgs(BaseModelArgs):
    # Field names must match the keys in the HF config.json (from_dict
    # filters the config dict to these names).
    model_type: str = "nomic_bert"
    vocab_size: int = 30528
    n_embd: int = 768
    n_head: int = 12
    n_inner: int = 3072
    n_layer: int = 12
    layer_norm_epsilon: float = 1e-12
    qkv_proj_bias: bool = False
    mlp_fc1_bias: bool = False
    mlp_fc2_bias: bool = False
    rotary_emb_base: float = 1000.0
    rotary_emb_fraction: float = 1.0
    rotary_emb_interleaved: bool = False
    rotary_scaling_factor: Optional[float] = None
    type_vocab_size: int = 2
    max_position_embeddings: int = 0
    pooling_config: dict = field(default_factory=lambda: {"pooling_mode": "mean"})


class NomicEmbedding(nn.Module):
    """word + token-type embeddings + LayerNorm (no position embeddings)."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.n_embd)
        self.norm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def __call__(self, input_ids, token_type_ids=None):
        x = self.word_embeddings(input_ids)
        if token_type_ids is None:
            token_type_ids = mx.zeros_like(input_ids)
        x = x + self.token_type_embeddings(token_type_ids)
        return self.norm(x)


class Attention(nn.Module):
    """Fused bias-free QKV + RoPE on q/k + scaled dot-product attention."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.n_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        # NOTE: attribute name must be "Wqkv" (capital W) — it is the weight
        # key in the safetensors file (the Swift reference hides this behind
        # @ModuleInfo(key: "Wqkv")).
        self.Wqkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.qkv_proj_bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.qkv_proj_bias)
        self.rotary_dim = int(self.head_dim * config.rotary_emb_fraction)
        # mx.fast.rope strictly requires float base/scale — JSON config gives ints.
        self.base = float(config.rotary_emb_base)
        self.scale = float(config.rotary_scaling_factor or 1.0)
        self.traditional = bool(config.rotary_emb_interleaved)

    def _rope(self, x):
        # x: (B, H, L, D)
        return mx.fast.rope(
            x, self.rotary_dim, traditional=self.traditional, base=self.base, scale=self.scale
        )

    def __call__(self, x, mask=None):
        b, l = x.shape[0], x.shape[1]
        q, k, v = mx.split(self.Wqkv(x), 3, axis=-1)
        shape = (b, l, self.n_heads, -1)
        q = q.reshape(shape).transpose(0, 2, 1, 3)
        k = k.reshape(shape).transpose(0, 2, 1, 3)
        v = v.reshape(shape).transpose(0, 2, 1, 3)
        if self.rotary_dim > 0:
            q = self._rope(q)
            k = self._rope(k)
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        probs = mx.softmax(scores, axis=-1)
        out = mx.matmul(probs, v).transpose(0, 2, 1, 3).reshape(b, l, -1)
        return self.out_proj(out)


class MLP(nn.Module):
    """SwiGLU: down(up(x) * silu(gate(x)))."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.fc11 = nn.Linear(config.n_embd, config.n_inner, bias=config.mlp_fc1_bias)
        self.fc12 = nn.Linear(config.n_embd, config.n_inner, bias=config.mlp_fc1_bias)
        self.fc2 = nn.Linear(config.n_inner, config.n_embd, bias=config.mlp_fc2_bias)

    def __call__(self, x):
        return self.fc2(self.fc11(x) * mx.nn.silu(self.fc12(x)))


class TransformerBlock(nn.Module):
    """Post-norm block: LN(x + Attn(x)) then LN(h + MLP(h))."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.attn = Attention(config)
        self.norm1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.norm2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def __call__(self, x, mask=None):
        h = self.norm1(self.attn(x, mask) + x)
        return self.norm2(self.mlp(h) + h)


class Encoder(nn.Module):
    """Stack of transformer blocks.

    The list must live on an ``Encoder`` module (not directly on ``Model``)
    so MLX's parameter keys come out as ``encoder.layers.<i>.*`` — matching
    the safetensors keys — instead of ``encoder.<i>.*``.
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.layers = [TransformerBlock(config) for _ in range(config.n_layer)]

    def __call__(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.embeddings = NomicEmbedding(config)
        self.encoder = Encoder(config)

    def __call__(self, input_ids, token_type_ids=None, attention_mask=None):
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        b, l = input_ids.shape
        max_pos = self.config.max_position_embeddings
        if max_pos > 0 and l > max_pos:
            input_ids = input_ids[:, :max_pos]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :max_pos]
            l = max_pos
        if attention_mask is None:
            attention_mask = mx.ones((b, l))
        mask = (1.0 - attention_mask[:, None, None, :]) * -1e9

        x = self.embeddings(input_ids, token_type_ids)
        x = self.encoder(x, mask)

        text_embeds = _pool_by_config(x, attention_mask, self.config.pooling_config)
        text_embeds = normalize_embeddings(text_embeds)
        return BaseModelOutput(last_hidden_state=x, text_embeds=text_embeds)

    def sanitize(self, weights):
        """Map HF weight names onto this module's structure."""
        out = {}
        for key, value in weights.items():
            if key.startswith("emb_ln."):
                key = "embeddings.norm." + key[len("emb_ln.") :]
            out[key] = value
        return out


def register_nomic_bert() -> None:
    """Expose this module as ``mlx_embeddings.models.nomic_bert``.

    The mlx-embeddings loader resolves architectures with
    ``importlib.import_module(f"mlx_embeddings.models.{model_type}")``, which
    checks ``sys.modules`` first — so injecting the module here is enough.
    """
    name = "mlx_embeddings.models.nomic_bert"
    if name in sys.modules:
        return
    import mlx_embeddings.models  # noqa: F401  (ensure parent package)

    mod = types.ModuleType(name)
    mod.Model = Model
    mod.ModelArgs = ModelArgs
    sys.modules[name] = mod
