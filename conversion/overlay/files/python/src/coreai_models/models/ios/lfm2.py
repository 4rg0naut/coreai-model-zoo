# LFM2 / LFM2.5 text decoder (short-conv + full-attention hybrid) on Apple's static iOS
# contract: load_embeddings / gather_embeddings / extend / prompt_opt graphs, static
# (cache_len, query_len) shapes, and key_cache / value_cache as the ONLY two runner-bound
# states (Apple's StaticShapeEngine binds nothing else).
#
# Community port — NOT an Apple model. Attention / MLP / norms / RoPE / embedding path are
# the same primitives and layout as models/ios/qwen3.py (LFM2's attention layers are
# Qwen3-shaped: GQA, per-head q/k RMSNorm, full-dim RoPE). The new part is the conv mixer.
#
# How the conv mixer fits the two-state contract
# ----------------------------------------------
# A conv layer's only cross-step state is the previous kernel-1 (= 2) gated-input columns
# Bx[t-1], Bx[t-2] (hidden 2048 each). Rather than a third state, every conv layer writes
# the Bx column of each batch position into ITS OWN row of key_cache / value_cache at
# position in_step.. (exactly where an attention layer writes k / v), and reads the two
# previous columns back through one-hot [cache_len] selectors (multiply + reduce over the
# cache axis; the equivalent [W, L] @ [L, 2] matmul is rejected by the ANE compiler). The selector is
# derived from column 0 of the runner's causal_mask (0 where p <= in_step, -40000
# elsewhere) with fp16 add / clamp / slice / sub only, so no int32 arithmetic has to lower
# on the Neural Engine. The cache is therefore CACHE_WIDTH = hidden/2 = 1024 channels wide
# for every layer: a 2048-wide Bx column is split across the key row and the value row;
# attention layers (8 kv heads x 64 = 512) use channels [0:512) of each row.
# Memory: 16 x 1024 x 2 rows x ctx x 2 B = 268 MB at ctx 4096.
#
# The engine re-runs the aligned batch (batchStart = position // q * q) on every decode step
# and rewrites positions in_step..in_step+q-1, and it switches query lengths
# (prompt_opt q=64 -> extend q=64 -> decode q=8), so the state MUST be addressed by
# position, never "last written" — the position-indexed layout above is what keeps the
# q=64 -> q=8 transition exact.
#
# The depthwise 3-tap conv is written as three shifted multiply-adds, not F.conv1d:
# the grouped conv1d failed the MPSGraph -> ANEC conversion on the iOS-27 ANE in the
# Qwen3.5 port (models/ios/qwen3_5_ios.py), the windowed form lowers.

from __future__ import annotations

import os

import torch
import torch.nn as nn
from transformers.models.lfm2.modeling_lfm2 import Lfm2Config
from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM as HFLfm2ForCausalLM
from typing_extensions import override

from coreai_models._constants import (
    CAUSAL_MASK_INPUT_NAME,
    EXTEND_FUNCTION_NAME,
    GATHER_EMBEDDINGS_FUNCTION_NAME,
    KEY_CACHE_INPUT_NAME,
    LOAD_EMBEDDINGS_FUNCTION_NAME,
    POSITION_IDS_INPUT_NAME,
    TOKEN_IDS_INPUT_NAME,
    TRANSFORMER_INPUT_NAME,
    VALUE_CACHE_INPUT_NAME,
)
from coreai_models._hf import resolve_rope_theta
from coreai_models.models.base import BaseForCausalLMForiOS, TraceSpec
from coreai_models.primitives._ops import mutable_cache_update_and_fetch
from coreai_models.primitives.ios.mlp import MLP
from coreai_models.primitives.ios.quantization import (
    dequantize_per_tensor,
    quantize_per_tensor,
)
from coreai_models.primitives.ios.rope import RoPECache, apply_rope
from coreai_models.primitives.ios.sdpa import SDPA


# Diagnostic knobs for the ANE lowering bisect (comma list in LFM2_IOS_DIAG; production = unset):
#   nohist   conv layers write their Bx columns but read no history (prev = 0)
#   nocache  conv layers never touch the cache (prev = 0)
#   convzero conv mixer returns 0 (no in_proj / gating / window)
#   narrow   cache rows are kv_width wide (attention-standard); implies nocache
_DIAG = {d for d in os.environ.get("LFM2_IOS_DIAG", "").split(",") if d}
# Mac-side precision bisect (s6_lfm2_check.py --fp32-parts): parts computed in fp32 inside an fp16 eager run.
# Empty in production; never set during export.
FP32_PARTS: set[str] = set()


def _up(x: torch.Tensor, part: str) -> torch.Tensor:
    return x.float() if part in FP32_PARTS else x


def _down(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return x.to(ref.dtype) if x.dtype != ref.dtype else x


# --------------------------------------------------------------------------- #
# Config helpers (HF Lfm2Config)
# --------------------------------------------------------------------------- #
def _head_dim(config) -> int:
    head_dim = getattr(config, "head_dim", None)
    if not isinstance(head_dim, int):
        head_dim = config.hidden_size // config.num_attention_heads
    return head_dim


def _kv_width(config) -> int:
    return config.num_key_value_heads * _head_dim(config)


def _cache_width(config) -> int:
    """Channels per cache row: half a conv column (the other half rides in the value row),
    never narrower than an attention layer's k/v."""
    if "narrow" in _DIAG:
        return _kv_width(config)
    return max((config.hidden_size + 1) // 2, _kv_width(config))


def _ff_dim(config) -> int:
    """Mirrors HF Lfm2MLP.__init__ (1.2B: 12288 -> 8192)."""
    inter = config.intermediate_size
    if getattr(config, "block_auto_adjust_ff_dim", True):
        inter = int(2 * inter / 3)
        mult = getattr(config, "block_ffn_dim_multiplier", 1.0)
        if mult is not None:
            inter = int(mult * inter)
            m = int(getattr(config, "block_multiple_of", 256))
            inter = m * ((inter + m - 1) // m)
    return inter


def _layer_types(config) -> list[str]:
    layer_types = getattr(config, "layer_types", None)
    if layer_types:
        return list(layer_types)
    full = set(getattr(config, "full_attn_idxs", None) or [])
    return ["full_attention" if i in full else "conv" for i in range(config.num_hidden_layers)]


# --------------------------------------------------------------------------- #
# RMSNorm computed at a larger scale (exact): the ANE flushes fp16 subnormals
# --------------------------------------------------------------------------- #
class RMSNormK(nn.Module):
    """x / sqrt(mean(x^2) + eps) * w, computed on y = K * x with eps' = K^2 * eps
    (mathematically identical). LFM2.5's residual stream is tiny (RMS ~0.01; mean(x^2) down to
    1e-5 for some tokens, q/k projections down to 2e-5) and its eps is 1e-5: all below the fp16
    normal range (6.1e-5). On the h18p ANE the subnormal eps constant is flushed to zero (the
    2026-09-18 device gate flipped exactly where a Mac twin with eps = 0 flips), so the norm's
    mean-square and eps are kept in the normal range here: K = 8 for the stream norms
    (inputs up to ~12 -> squares ~1e4, 7x below fp16 overflow), K = 32 for the q/k norms
    (inputs < 1). Same `weight` parameter name as HF / Apple's RMSNorm."""

    def __init__(self, dim: int, eps: float, k: float) -> None:
        super().__init__()
        with torch.device("cpu"):
            self.weight = nn.Parameter(torch.zeros(dim))
            self._eps = nn.Buffer(torch.tensor(eps * k * k), persistent=False)
            self._k = nn.Buffer(torch.tensor(float(k)), persistent=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        y = input * self._k
        mean_square = (y * y).mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_square + self._eps)
        return y * inv_rms * self.weight


NORM_K_STREAM = 8.0
NORM_K_QK = 32.0


# --------------------------------------------------------------------------- #
# Hybrid cache handler: one row per layer, per-call channel width
# --------------------------------------------------------------------------- #
class _HybridCacheHandler:
    """KV cache rows [n_layers, 1, cache_width, 1, max_seq] shared by both layer kinds.

    Attention layers write k / v into channels [0:kv_width) of the key / value row and
    read the whole row back (the SDPA slices the width it needs). Conv layers write the
    two halves of their Bx column into channels [0:conv_k) of the key row and
    [0:conv_v) of the value row.
    """

    def __init__(self, n_layers: int, cache_width: int, widths: tuple[int, ...]) -> None:
        self._k_cache: torch.Tensor | None = None
        self._v_cache: torch.Tensor | None = None
        self.cache_width = cache_width
        with torch.device("cpu"):
            self._zero = torch.zeros(1, dtype=torch.int32)
            self._one = torch.ones(1, dtype=torch.int32)
            self._layer_indices = torch.arange(n_layers, dtype=torch.int32).unsqueeze(1)
            self._layer_indices_end = torch.arange(1, n_layers + 1, dtype=torch.int32).unsqueeze(1)
            # the channel counts written per call: kv_width (attention), split_k / split_v (conv)
            self._widths = {w: torch.tensor([w], dtype=torch.int32) for w in sorted(set(widths)) if w > 0}

    def register_kv_cache(self, key_cache: torch.Tensor, value_cache: torch.Tensor) -> None:
        assert key_cache.shape == value_cache.shape, (key_cache.shape, value_cache.shape)
        self._k_cache = key_cache
        self._v_cache = value_cache

    def _slice_args(self, layer_idx: int, offset: torch.Tensor, width: int, n_tokens: int):
        dev = offset.device
        begin = torch.cat(
            [
                self._layer_indices[layer_idx].to(dev),
                self._zero.to(dev),
                self._zero.to(dev),
                self._zero.to(dev),
                offset,
            ]
        )
        end = torch.cat(
            [
                self._layer_indices_end[layer_idx].to(dev),
                self._one.to(dev),
                self._widths[width].to(dev),
                self._one.to(dev),
                offset + n_tokens,
            ]
        )
        return begin, end

    def update_and_fetch(
        self,
        layer_idx: int,
        offset: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        n_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write k into the key row and v into the value row at [offset, offset+n_tokens),
        each over its own channel count, and return both full rows [1, cache_width, 1, L]."""
        assert self._k_cache is not None and self._v_cache is not None
        torch._check_is_size(layer_idx)
        torch._check(layer_idx < self._k_cache.size(0))
        bk, ek = self._slice_args(layer_idx, offset, k.shape[1], n_tokens)
        k_row = mutable_cache_update_and_fetch(
            x=self._k_cache, update=k, begin=bk, end=ek, layer_idx=layer_idx, seq_dim=-1, seq_len=None
        )
        bv, ev = self._slice_args(layer_idx, offset, v.shape[1], n_tokens)
        v_row = mutable_cache_update_and_fetch(
            x=self._v_cache, update=v, begin=bv, end=ev, layer_idx=layer_idx, seq_dim=-1, seq_len=None
        )
        return k_row, v_row

    @property
    def k_cache(self) -> torch.Tensor:
        return self._k_cache

    @property
    def v_cache(self) -> torch.Tensor:
        return self._v_cache


# --------------------------------------------------------------------------- #
# Full attention (Qwen3-shaped: GQA, per-head q/k RMSNorm, full-dim RoPE)
# --------------------------------------------------------------------------- #
class Attention(nn.Module):
    def __init__(self, config: Lfm2Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = _head_dim(config)
        self.kv_width = self.n_kv_heads * self.head_dim

        self.q_proj = nn.Conv2d(dim, self.n_heads * self.head_dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(dim, self.kv_width, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(dim, self.kv_width, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(self.n_heads * self.head_dim, dim, kernel_size=1, bias=False)

        # HF names: q_layernorm / k_layernorm (plain weight gain, eps = norm_eps)
        self.q_layernorm = RMSNormK(self.head_dim, config.norm_eps, NORM_K_QK)
        self.k_layernorm = RMSNormK(self.head_dim, config.norm_eps, NORM_K_QK)
        self.sdpa = SDPA(head_dim=self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        cache: _HybridCacheHandler | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _, _ = x.shape
        n_heads, n_kv_heads = self.n_heads, self.n_kv_heads

        x = x.transpose(-3, -1)  # [b, d, 1, q]
        if "attn_proj" in FP32_PARTS:
            xf = x.float()
            query = torch.nn.functional.conv2d(xf, self.q_proj.weight.float()).to(x.dtype)
            key = torch.nn.functional.conv2d(xf, self.k_proj.weight.float()).to(x.dtype)
            value = torch.nn.functional.conv2d(xf, self.v_proj.weight.float()).to(x.dtype)
        else:
            query = self.q_proj(x)
            key = self.k_proj(x)
            value = self.v_proj(x)

        query = (
            query.transpose(-3, -1)
            .reshape(batch_size, query_len, n_heads, self.head_dim)
            .transpose(-2, -3)
        )
        key = (
            key.transpose(-3, -1)
            .reshape(batch_size, query_len, n_kv_heads, self.head_dim)
            .transpose(-2, -3)
        )
        if "norms" in FP32_PARTS:
            query = _down(self.q_layernorm(query.float()), query)
            key = _down(self.k_layernorm(key.float()), key)
        else:
            query = self.q_layernorm(query)
            key = self.k_layernorm(key)

        seq_len = rope_cos.shape[1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)

        if "rope" in FP32_PARTS:
            query = _down(apply_rope(query.float(), rope_cos.float(), rope_sin.float()), query)
            key = _down(apply_rope(key.float(), rope_cos.float(), rope_sin.float()), key)
        else:
            query = apply_rope(query, rope_cos, rope_sin)
            key = apply_rope(key, rope_cos, rope_sin)

        query = (
            query.transpose(-2, -3)
            .reshape(batch_size, query_len, 1, n_heads * self.head_dim)
            .transpose(-3, -1)
        )
        key = (
            key.transpose(-3, -2)
            .reshape(batch_size, query_len, 1, self.kv_width)
            .transpose(-3, -1)
        )

        if cache is not None:
            key, value = cache.update_and_fetch(self.layer_idx, in_step, key, value, query_len)
            if key.shape[1] != self.kv_width:
                # rows are cache_width wide; k / v live in the first kv_width channels
                key = key[:, : self.kv_width]
                value = value[:, : self.kv_width]

        if "scores" in FP32_PARTS:
            output = _down(self.sdpa(query.float(), key.float(), value.float(), causal_mask.float()), query)
        else:
            output = self.sdpa(query, key, value, causal_mask)
        if "attn_proj" in FP32_PARTS:
            output = torch.nn.functional.conv2d(output.float(), self.out_proj.weight.float()).to(output.dtype)
        else:
            output = self.out_proj(output)
        return output.transpose(-3, -1)


# --------------------------------------------------------------------------- #
# Short-conv mixer with the conv history in the KV rows
# --------------------------------------------------------------------------- #
class _ConvTaps(nn.Module):
    """The depthwise taps as a bare parameter under the HF name (`conv.weight`, [d, 1, k]).
    Not an nn.Conv1d: the palettizer leaves bare parameters alone and the graph never
    emits a grouped conv."""

    def __init__(self, dim: int, kernel: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, 1, kernel))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None


def history_selector(causal_mask: torch.Tensor, n_prev: int, pad: torch.Tensor) -> torch.Tensor:
    """One-hot selectors for cache positions in_step-n_prev .. in_step-1 (oldest first).

    causal_mask: [1, L, 1, q]; its column 0 is 0 where p <= in_step and the sentinel
    (-40000) elsewhere — the runner's contract (InputHandler+StaticBucket.swift). With
    a[p] = clamp(mask[p, 0] + 1, 0, 1) (exact in fp16: 1 for p <= in_step, else 0),
    onehot(in_step - j)[p] = a[p + j] - a[p + j + 1]. Returns [1, 1, L, n_prev].
    """
    a = torch.clamp(causal_mask[:, :, :, 0:1] + 1.0, 0.0, 1.0)  # [1, L, 1, 1]
    cache_len = a.shape[1]
    torch._check_is_size(cache_len)
    a = torch.cat([a, pad], dim=1)  # [1, L + n_prev + 1, 1, 1]
    cols = []
    for j in range(n_prev, 0, -1):
        cols.append(a[:, j : j + cache_len] - a[:, j + 1 : j + 1 + cache_len])
    sel = torch.cat(cols, dim=-1)  # [1, L, 1, n_prev]
    return sel.reshape(1, 1, cache_len, n_prev)


class ShortConv(nn.Module):
    def __init__(self, config: Lfm2Config, layer_idx: int, cache_width: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        d = config.hidden_size
        self.dim = d
        self.kernel = int(config.conv_L_cache)
        self.n_prev = self.kernel - 1
        bias = bool(getattr(config, "conv_bias", False))
        self.cache_width = cache_width
        # Bx column split: channels [0:split_k) -> key row, [split_k:d) -> value row
        self.split_k = min(cache_width, d)
        self.split_v = d - self.split_k
        self.use_cache = not ({"nocache", "narrow", "convzero"} & _DIAG)
        self.use_hist = self.use_cache and "nohist" not in _DIAG
        assert not self.use_cache or 0 <= self.split_v <= cache_width, (d, cache_width)

        self.in_proj = nn.Conv2d(d, 3 * d, kernel_size=1, bias=bias)
        self.out_proj = nn.Conv2d(d, d, kernel_size=1, bias=bias)
        self.conv = _ConvTaps(d, self.kernel, bias)
        with torch.device("cpu"):
            self._pad = nn.Buffer(torch.zeros(1, self.n_prev + 1, 1, 1), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        cache: _HybridCacheHandler | None = None,
        history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _, _ = x.shape
        torch._check_is_size(query_len)
        d, n_prev = self.dim, self.n_prev
        if "convzero" in _DIAG:
            return torch.zeros_like(x)

        x = x.transpose(-3, -1)  # [b, d, 1, q]
        bcx = self.in_proj(x)  # [b, 3d, 1, q]
        gate_b, gate_c, xv = torch.split(bcx, d, dim=1)
        bx = gate_b * xv  # [b, d, 1, q]

        if cache is not None and self.use_cache:
            k_row, v_row = cache.update_and_fetch(
                self.layer_idx, in_step, bx[:, : self.split_k], bx[:, self.split_k :], query_len
            )
            if self.use_hist:
                # rows: [1, cache_width, 1, L]; selector columns: [1, 1, L, n_prev] one-hots.
                # Read = broadcast multiply + reduce over L, NOT a [W, L] @ [L, n_prev] matmul:
                # the h18p ANE compiler rejects that matmul (ANECCompileOffline, empty
                # ErrorList, whole model falls back to the GPU) even against a constant
                # selector, while the multiply-reduce form lowers (S6 bisect, 2026-09-17).
                cache_len = k_row.shape[-1]
                sel = history_selector(causal_mask, n_prev, self._pad.to(bx.dtype))
                cols = []
                for j in range(n_prev):
                    col = sel[:, :, :, j].reshape(1, 1, 1, cache_len)
                    cols.append(torch.cat([(k_row * col).sum(-1, keepdim=True)[:, : self.split_k],
                                           (v_row * col).sum(-1, keepdim=True)[:, : self.split_v]], dim=1))
                prev = torch.cat(cols, dim=-1)  # [1, d, 1, n_prev]
            else:
                prev = torch.zeros(batch_size, d, 1, n_prev, dtype=bx.dtype, device=bx.device)
        elif history is not None:
            prev = history  # [1, d, 1, n_prev] (eager tests)
        else:
            prev = torch.zeros(batch_size, d, 1, n_prev, dtype=bx.dtype, device=bx.device)

        window = torch.cat([prev, bx], dim=-1)  # [b, d, 1, n_prev + q]
        taps = self.conv.weight  # [d, 1, k]: out[t] = sum_j w[j] * Bx[t - (k-1) + j]
        conv_out = taps[:, 0, 0].reshape(1, d, 1, 1) * window[..., 0:query_len]
        for j in range(1, self.kernel):
            conv_out = conv_out + taps[:, 0, j].reshape(1, d, 1, 1) * window[..., j : j + query_len]
        if self.conv.bias is not None:
            conv_out = conv_out + self.conv.bias.reshape(1, d, 1, 1)

        y = gate_c * conv_out
        return self.out_proj(y).transpose(-3, -1)


# --------------------------------------------------------------------------- #
# Decoder layer / model / extend graph
# --------------------------------------------------------------------------- #
class DecoderLayer(nn.Module):
    def __init__(self, config: Lfm2Config, layer_idx: int, cache_width: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.is_attention = _layer_types(config)[layer_idx] == "full_attention"
        if self.is_attention:
            self.self_attn = Attention(config, layer_idx)
        else:
            self.conv = ShortConv(config, layer_idx, cache_width)
        self.feed_forward = MLP(dim=hidden_size, hidden_dim=_ff_dim(config))
        self.operator_norm = RMSNormK(hidden_size, config.norm_eps, NORM_K_STREAM)
        self.ffn_norm = RMSNormK(hidden_size, config.norm_eps, NORM_K_STREAM)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        cache: _HybridCacheHandler | None = None,
    ) -> torch.Tensor:
        normed = _down(self.operator_norm(_up(x, "norms")), x)
        if self.is_attention:
            r = self.self_attn(normed, rope_cos, rope_sin, in_step, causal_mask, cache)
        elif "conv" in FP32_PARTS:
            r = _down(self.conv(normed.float(), in_step, causal_mask.float(), cache), x)
        else:
            r = self.conv(normed, in_step, causal_mask, cache)
        h = x + r
        n2 = _down(self.ffn_norm(_up(h, "norms")), h)
        if "mlp" in FP32_PARTS:
            r = _down(self.feed_forward(n2.float()), h)
        else:
            r = self.feed_forward(n2)
        return h + r


class Lfm2Model(nn.Module):
    def __init__(self, config: Lfm2Config, cache_width: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(config, i, cache_width) for i in range(config.num_hidden_layers)]
        )
        # HF names the final norm embedding_norm
        self.embedding_norm = RMSNormK(config.hidden_size, config.norm_eps, NORM_K_STREAM)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        cache: _HybridCacheHandler | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            token_embeddings = layer(token_embeddings, rope_cos, rope_sin, in_step, causal_mask, cache)
        return self.embedding_norm(token_embeddings)


class Lfm2Extend(nn.Module):
    def __init__(self, config: Lfm2Config) -> None:
        super().__init__()
        self.cache_width = _cache_width(config)
        self.model = Lfm2Model(config, self.cache_width)
        self.emb_zero_point = nn.Parameter(torch.zeros([], dtype=torch.int8), requires_grad=False)
        self.emb_scale = nn.Parameter(torch.ones([], dtype=torch.float16), requires_grad=False)

        self.prefill_mode = False

        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = None

        d = config.hidden_size
        split_k = min(self.cache_width, d)
        self.kv_cache = _HybridCacheHandler(
            config.num_hidden_layers, self.cache_width, (_kv_width(config), split_k, d - split_k)
        )
        self.rope = RoPECache(_head_dim(config), config.max_position_embeddings, resolve_rope_theta(config))

    def forward(
        self,
        transformer_input: torch.Tensor,
        position_ids: torch.IntTensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        embedding_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.kv_cache.register_kv_cache(key_cache, value_cache)
        rope_cos, rope_sin = self.rope.gather_cos_sin(position_ids)

        batch_size, seq_len, _, hidden_dim = transformer_input.shape
        out = self.model(transformer_input, rope_cos, rope_sin, in_step, causal_mask, self.kv_cache)
        if self.prefill_mode:
            return self.kv_cache.k_cache[0, 0, 0, 0, 0] + self.kv_cache.v_cache[0, 0, 0, 0, 0]

        if self.lm_head is not None:
            return self.lm_head(out.transpose(-2, -3))

        if embedding_table.dtype == torch.int8:
            embedding_table = dequantize_per_tensor(
                embedding_table, self.emb_scale, self.emb_zero_point, out.dtype
            )
        embedding_table = embedding_table.reshape(
            embedding_table.shape[1], embedding_table.shape[0], embedding_table.shape[2]
        )
        out = out.transpose(-3, -1).reshape(batch_size, 1, hidden_dim, seq_len)
        return (embedding_table @ out).transpose(-2, -1)


# --------------------------------------------------------------------------- #
# iOS model: contract overrides for the widened cache
# --------------------------------------------------------------------------- #
class Lfm2ForCausalLMForiOS(BaseForCausalLMForiOS):
    _HF_MODEL_CLASS = HFLfm2ForCausalLM

    def _init_model(self, config: Lfm2Config) -> None:
        self.extend = Lfm2Extend(config)

    @property
    def cache_width(self) -> int:
        return self.extend.cache_width

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        # Eager composition only (never traced for export). The export pipeline's
        # palettization example inputs size the cache from hf_config's kv heads x head_dim;
        # this model's rows are cache_width wide, so substitute a correctly shaped cache.
        if key_cache.shape[2] != self.cache_width:
            shape = (key_cache.shape[0], key_cache.shape[1], self.cache_width, key_cache.shape[3], key_cache.shape[4])
            key_cache = torch.zeros(shape, dtype=key_cache.dtype, device=key_cache.device)
            value_cache = torch.zeros(shape, dtype=value_cache.dtype, device=value_cache.device)
        token_embeddings = self.gather_embeddings(input_ids, self.load_embeddings.embedding_table)
        return self.extend(
            token_embeddings,
            position_ids,
            in_step,
            causal_mask,
            key_cache,
            value_cache,
            self.load_embeddings.embedding_table,
        )

    @override
    def build_reference_inputs(self, config, target_dtype: torch.dtype, spec: TraceSpec):
        refs = super().build_reference_inputs(config, target_dtype, spec)
        ext = refs[EXTEND_FUNCTION_NAME]
        key_cache = torch.zeros(
            config.num_hidden_layers, 1, _cache_width(config), 1, spec.max_context_length, dtype=torch.float16
        )
        ext[KEY_CACHE_INPUT_NAME] = key_cache
        ext[VALUE_CACHE_INPUT_NAME] = key_cache.clone()
        return refs

    @classmethod
    @override
    def export_static_shape_configs(cls, config, max_context_length: int):
        shapes = super().export_static_shape_configs(config, max_context_length)
        width = _cache_width(config)
        for spec in shapes[EXTEND_FUNCTION_NAME].values():
            for name in (KEY_CACHE_INPUT_NAME, VALUE_CACHE_INPUT_NAME):
                n_layers, b, _, one, cache_len = spec[name]
                spec[name] = (n_layers, b, width, one, cache_len)
        return shapes

    def _mutate_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        max_layer = -1
        for k in state_dict:
            if k.startswith("model.layers."):
                max_layer = max(max_layer, int(k.split(".")[2]))
        if max_layer < 0:
            raise ValueError("invalid state_dict")

        mlp_map = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
        for i in range(max_layer + 1):
            prefix = f"model.layers.{i}."
            # attention or conv projections -> Conv2d 1x1
            for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj",
                         "conv.in_proj", "conv.out_proj"):
                key = prefix + name + ".weight"
                if key in state_dict:
                    state_dict[key] = state_dict[key].unsqueeze(-1).unsqueeze(-1)
            # HF w1/w3/w2 -> MLP primitive gate/up/down (Conv2d)
            for src, dst in mlp_map.items():
                key = f"{prefix}feed_forward.{src}.weight"
                if key in state_dict:
                    state_dict[f"{prefix}feed_forward.{dst}.weight"] = state_dict.pop(key).unsqueeze(-1).unsqueeze(-1)
            # conv.conv.weight [d, 1, k] keeps its name and shape (_ConvTaps)

        embedding_table = state_dict["model.embed_tokens.weight"].unsqueeze(1)
        if not self.disable_embedding_quantization:
            embedding_table, scale, zero_point = quantize_per_tensor(embedding_table, nbits=8, symmetric=True)
        else:
            scale = torch.tensor(1.0, dtype=embedding_table.dtype)
            zero_point = torch.tensor(0, dtype=torch.int8)
        state_dict["load_embeddings.embedding_table"] = embedding_table
        state_dict["gather_embeddings.scale"] = scale
        state_dict["gather_embeddings.zero_point"] = zero_point
        state_dict["extend.emb_scale"] = scale
        state_dict["extend.emb_zero_point"] = zero_point
        state_dict.pop("model.embed_tokens.weight")

        new_state_dict = {}
        for k in list(state_dict):
            if k.startswith("model.") and "gather_embeddings" not in k:
                new_state_dict["extend." + k] = state_dict.pop(k)
        state_dict.update(new_state_dict)

        if not self.config.tie_word_embeddings and "lm_head.weight" in state_dict:
            state_dict["extend.lm_head.weight"] = state_dict["lm_head.weight"]
        state_dict.pop("lm_head.weight", None)


def register() -> None:
    """Register this builder for model_type `lfm2` at runtime.

    Apple's models/registry.py stays unmodified in the checkout; the zoo's export
    scripts call this before `coreai.llm.export`'s main().
    """
    from coreai_models.models.registry import ModelEntry, _get_registry

    registry = _get_registry()
    old = registry.get("lfm2")
    registry["lfm2"] = ModelEntry(
        macos_class=old.macos_class if old is not None else None,
        ios_class=Lfm2ForCausalLMForiOS,
    )
