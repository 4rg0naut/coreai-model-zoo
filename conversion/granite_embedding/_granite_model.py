"""Static, plain-PyTorch authoring graph for Granite Embedding 97M R2.

The graph is re-authored from the raw checkpoint schema and gated separately
against Hugging Face ModernBertModel. It neither wraps nor imports Transformers.
knowledge/granite-embedding-97m-port.md records the architectural decisions.

Inputs are batch-one int32 token IDs and a binary attention mask on a fixed grid.
The output is a 384-dimensional L2-normalized CLS embedding. Diagnostic methods
also expose the embedding LayerNorm output, every residual layer, final LayerNorm
output, and unnormalized CLS vector.

The mutations deliberately change only one behavior at a time. In particular,
all_global/all_local alter attention masks while retaining the original per-layer
RoPE theta; wrong_pooling uses masked mean pooling. They are negative controls,
never export variants to be presented as faithful ports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


MODEL_ID = "ibm-granite/granite-embedding-97m-multilingual-r2"
MODEL_SHA = "835ad14087e140460703cf0fae09f97d469d65c2"
PARAMETER_COUNT = 97_441_152
MUTATIONS = ("none", "all_global", "all_local", "ignore_padding", "wrong_pooling", "window63")


def _validate_config(config: dict[str, Any]) -> None:
    """Reject checkpoints outside this deliberately narrow authoring recipe."""
    required = {
        "model_type": "modernbert",
        "hidden_size": 384,
        "intermediate_size": 1536,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "vocab_size": 180000,
        "pad_token_id": 179935,
        "hidden_activation": "silu",
        "attention_bias": False,
        "mlp_bias": False,
        "norm_bias": False,
        "attention_dropout": 0.0,
        "embedding_dropout": 0.0,
        "mlp_dropout": 0.0,
        "global_attn_every_n_layers": 3,
        "local_attention": 128,
        "global_rope_theta": 150000.0,
        "local_rope_theta": 160000.0,
        "norm_eps": 1e-5,
    }
    mismatches = {key: (config.get(key), expected) for key, expected in required.items()
                  if config.get(key) != expected}
    if mismatches:
        raise ValueError(f"Unsupported Granite configuration (actual, expected): {mismatches}")
    if config.get("rope_scaling") is not None:
        raise ValueError("This checkpoint recipe requires unscaled default RoPE")
    if config.get("partial_rotary_factor", 1.0) != 1.0:
        raise ValueError("This checkpoint recipe requires full-head RoPE")


def _layer_norm(config: dict[str, Any]) -> nn.LayerNorm:
    return nn.LayerNorm(config["hidden_size"], eps=config["norm_eps"], bias=False)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class _Embeddings(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.tok_embeddings = nn.Embedding(
            config["vocab_size"], config["hidden_size"], padding_idx=config["pad_token_id"]
        )
        self.norm = _layer_norm(config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.norm(self.tok_embeddings(input_ids))


class _Attention(nn.Module):
    def __init__(self, config: dict[str, Any], seq_len: int, layer_id: int, mutation: str):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.seq_len = seq_len
        self.Wqkv = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        self.Wo = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        original_local = layer_id % config["global_attn_every_n_layers"] != 0
        self.is_local = (True if mutation == "all_local" else
                         False if mutation == "all_global" else original_local)
        theta = config["local_rope_theta"] if original_local else config["global_rope_theta"]
        inv_freq = 1.0 / (theta ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.int64).float() / self.head_dim
        ))
        # Follow HF's fp32 multiply and half duplication before trig exactly.
        position_ids = torch.arange(seq_len).unsqueeze(0)
        freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
        angles = torch.cat((freqs, freqs), dim=-1).unsqueeze(1)
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)

    def forward(
        self, hidden_states: torch.Tensor, global_mask: torch.Tensor, local_mask: torch.Tensor
    ) -> torch.Tensor:
        qkv = self.Wqkv(hidden_states).reshape(
            1, self.seq_len, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.transpose(3, 1).unbind(dim=2)
        cosine = self.rope_cos.to(query.dtype)
        sine = self.rope_sin.to(query.dtype)
        query = query * cosine + _rotate_half(query) * sine
        key = key * cosine + _rotate_half(key) * sine
        scores = torch.matmul(query, key.transpose(2, 3)) * (self.head_dim ** -0.5)
        scores = scores + (local_mask if self.is_local else global_mask)
        # HF's eager implementation computes the softmax in fp32, then restores
        # the query dtype. No SDPA composite or explicit unstabilized exp is used.
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attended = torch.matmul(probabilities, value)
        attended = attended.transpose(1, 2).contiguous().reshape(1, self.seq_len, self.hidden_size)
        return self.Wo(attended)


class _MLP(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.Wi = nn.Linear(config["hidden_size"], 2 * config["intermediate_size"], bias=False)
        self.Wo = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        inputs, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(F.silu(inputs) * gate)


class _EncoderLayer(nn.Module):
    def __init__(self, config: dict[str, Any], seq_len: int, layer_id: int, mutation: str):
        super().__init__()
        # HF omits the first attention norm; adding one creates a missing weight.
        self.attn_norm = nn.Identity() if layer_id == 0 else _layer_norm(config)
        self.attn = _Attention(config, seq_len, layer_id, mutation)
        self.mlp_norm = _layer_norm(config)
        self.mlp = _MLP(config)

    def forward(
        self, hidden_states: torch.Tensor, global_mask: torch.Tensor, local_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.attn_norm(hidden_states), global_mask, local_mask)
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


class GraniteEmbedding(nn.Module):
    """Batch-one, fixed-sequence graph whose parameter names match the HF weights."""

    def __init__(self, config: dict[str, Any], seq_len: int, mutation: str = "none"):
        super().__init__()
        _validate_config(config)
        if mutation not in MUTATIONS:
            raise ValueError(f"Unknown mutation {mutation!r}; expected one of {MUTATIONS}")
        if not 2 <= seq_len <= config["max_position_embeddings"]:
            raise ValueError("Sequence length must accommodate both special tokens and fit the context")
        self.seq_len = int(seq_len)
        self.mutation = mutation
        self.embeddings = _Embeddings(config)
        self.layers = nn.ModuleList([
            _EncoderLayer(config, self.seq_len, index, mutation)
            for index in range(config["num_hidden_layers"])
        ])
        self.final_norm = _layer_norm(config)
        positions = torch.arange(self.seq_len)
        distance = (positions[:, None] - positions[None, :]).abs()
        radius = 63 if mutation == "window63" else config["local_attention"] // 2
        self.register_buffer("local_allowed", (distance <= radius)[None, None], persistent=False)

    def _masks(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # These are static shape checks; they introduce no tensor-data branching.
        if tuple(input_ids.shape) != (1, self.seq_len):
            raise ValueError(f"Expected input_ids [1,{self.seq_len}], got {tuple(input_ids.shape)}")
        if tuple(attention_mask.shape) != (1, self.seq_len):
            raise ValueError(f"Expected attention_mask [1,{self.seq_len}], got {tuple(attention_mask.shape)}")
        dtype = self.embeddings.tok_embeddings.weight.dtype
        minimum = torch.finfo(dtype).min
        if self.mutation == "ignore_padding":
            attention_mask = torch.ones_like(attention_mask)
        mask = attention_mask[:, None, None, :].to(dtype).expand(1, 1, self.seq_len, self.seq_len)
        inverted = 1.0 - mask
        # Keep HF's finite-min convention, including fully masked padding queries.
        global_mask = inverted.masked_fill(inverted.to(torch.bool), minimum)
        local_mask = global_mask.masked_fill(~self.local_allowed, minimum)
        return global_mask, local_mask

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.mutation == "wrong_pooling":
            weights = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            pooled = (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        else:
            pooled = hidden_states[:, 0]
        # Explicit HF-equivalent clamp avoids the converter's documented
        # F.normalize decomposition losing the epsilon safeguard.
        pooled = pooled.float()
        return pooled * pooled.square().sum(dim=-1, keepdim=True).clamp_min(1e-24).rsqrt()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        global_mask, local_mask = self._masks(input_ids, attention_mask)
        hidden_states = self.embeddings(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, global_mask, local_mask)
        hidden_states = self.final_norm(hidden_states)
        return self._pool(hidden_states, attention_mask)

    def forward_intermediates(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, Any]:
        """Diagnostic eager path; outputs retain tensors and do not detach implicitly.

        HF hidden_states is (embedding_output, *layers): its last entry is BEFORE
        final_norm. HF last_hidden_state corresponds to final_norm below.
        """
        global_mask, local_mask = self._masks(input_ids, attention_mask)
        hidden_states = self.embeddings(input_ids)
        embedding_output = hidden_states
        residuals = []
        for layer in self.layers:
            hidden_states = layer(hidden_states, global_mask, local_mask)
            residuals.append(hidden_states)
        final_norm = self.final_norm(hidden_states)
        return {
            "embedding_output": embedding_output,
            "layers": tuple(residuals),
            "hidden_states": (embedding_output, *residuals),
            "last_hidden_state": final_norm,
            "raw_cls": final_norm[:, 0],
            "embedding": self._pool(final_norm, attention_mask),
        }

    def return_intermediates(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, Any]:
        return self.forward_intermediates(input_ids, attention_mask)


def load_granite(
    model_dir: str | Path,
    seq_len: int,
    dtype: torch.dtype = torch.float32,
    mutation: str = "none",
) -> GraniteEmbedding:
    """Load every raw safetensors tensor strictly, without an HF model object.

    There are no intentionally unused checkpoint tensors. RoPE/mask constants are
    derived and nonpersistent, so they are excluded from the state dictionary.
    A new model should be loaded for each precision, avoiding fp16→fp32 roundtrip
    loss in static trig constants. This function performs no network access.
    """
    from safetensors.torch import load_file

    if dtype not in (torch.float32, torch.float16):
        raise ValueError("Only explicit fp32 and fp16 float baselines are supported")
    source = Path(model_dir)
    config = json.loads((source / "config.json").read_text())
    model = GraniteEmbedding(config, seq_len, mutation=mutation)
    weights = load_file(str(source / "model.safetensors"), device="cpu")
    actual_parameters = sum(value.numel() for value in weights.values())
    if actual_parameters != PARAMETER_COUNT:
        raise ValueError(f"Checkpoint parameter count {actual_parameters} != {PARAMETER_COUNT}")
    model.load_state_dict(weights, strict=True)
    return model.to(dtype=dtype).eval()
