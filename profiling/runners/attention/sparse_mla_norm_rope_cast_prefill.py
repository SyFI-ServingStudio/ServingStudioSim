"""Profile DeepInfer's prefill core attention: FlashMLA ``fused_norm_rope_attn_rope_cast.prefill``.

The timed call is the one DeepInfer's ``prefill_attn_varlen_sparse``
(``deepinfer/model_executor/layers/attention_backend.py``) issues when
``V4AttentionBase._fused_norm_rope_cast_kwargs`` (``models/v4_modules/attn.py``)
returns kwargs: one ``sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::
core_attn::fwd_kernel`` launch with ``enable_q_norm``, GPT-J RoPE
(``is_rope_neox_style=False``, 64 dims), ``n_wv_group``, ``num_per_channels=32``,
``use_tma_aligned_col_major_sf``, ``round_sf`` and ``use_packed_ue8m0`` set.

Operands, as ``V4Attention.forward`` builds them for a prompt step:

- ``q`` ``[T, H, 512]`` bf16, the ``q_b_proj`` output before the Q RMSNorm and
  RoPE, in the layout ``permute_q_b_proj`` gives it (16-dim chunks across
  heads, ``q_to_fused_layout``).
- ``kv`` ``[total_kv, 1, 512]`` bf16, the gathered buffer
  ``[SWA history: requests x window | this step's SWA rows | compressed rows]``
  (``_prepare_prefill_kv``; ``load_paged_kv_cache`` fills the history and
  compressed parts). Each request owns ``max(1, context // ratio)`` compressed
  rows, as ``prepare_prompt_v4`` lays out ``ks``.
- ``indices`` ``[T, 1, window (+ index_topk)]`` int32: the expanded SWA index
  (``kv_swa_full_index``, from ``kv_swa_full_index_pivots``) and, on compressed
  layers, the indexer's top-k rows offset into the compressed part
  (``kv_index_cat``). ``-1`` marks an unused column. ``topk_length`` is ``None``
  (every compressed layer of DeepSeek-V4.1 is sparse).
- ``attn_sink`` ``[H]`` fp32, ``sm_scale = 512 ** -0.5``.

A YoCo-compacted consumer layer (``get_consumer_view``) is a request's last
``query_len`` rows at its full context. Its window is also clipped to the kept
rows (C++ ``yoco_swa_visibility``), so early rows see fewer than ``window`` keys;
the runner keeps the full window. The kernel walks every index column whether
valid or ``-1``: on B200, 4 x (140, 131072) at ratio 1 took 62.97 us clipped and
63.04 us unclipped, and 4 x (4096, 131072) at ratio 2 took 1503 us with 512
valid compressed columns per row and 1601 us with none valid.

Top-k selections are stratified, as in ``compressed_sparse_mla_rope_cast``:
exactly ``min((pos + 1) // ratio, index_topk)`` distinct causal compressed rows
per token, one drawn from each of that many equal strata, ascending (the
indexer's ``sorted_index=True``).

The output (FP8 values and packed UE8M0 scales in ``wo_a``'s permuted layout)
is checked against an independent fp32 Torch composite on sampled rows before
timing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

KIND = "sparse_mla_norm_rope_cast_prefill"
BACKEND = "flashmla_deepinfer"
KERNEL_NAME = "fwd_kernel"
_HEAD_DIM = 512
_ROPE_DIM = 64
_WV_GROUP_HEADS = 8
_SUPPORTED_HEADS = (64, 128)
# DeepSeek-V4.1's max_position_embeddings: the longest context a request has.
_MAX_CONTEXT = 1_048_576
_RMS_EPS = 1e-6
_CHECK_ROWS = 48
_REL_TOL = 0.06


@dataclass(frozen=True)
class Shape:
    pairs: tuple[tuple[int, int], ...]
    window_size: int
    compress_ratio: int
    index_topk: int
    num_heads: int
    n_wv_group: int
    enable_q_norm: bool

    @property
    def num_tokens(self) -> int:
        return sum(query for query, _ in self.pairs)

    @property
    def width(self) -> int:
        return self.window_size + self.index_topk


def validate(
    query_context_pairs: Any,
    window_size: int,
    compress_ratio: int,
    index_topk: int,
    num_heads: int,
    head_dim: int,
    rope_dim: int,
    n_wv_group: int,
    enable_q_norm: bool,
    q_dtype: DType | str,
    kv_dtype: DType | str,
    output_dtype: DType | str,
) -> Shape:
    pairs = tuple((int(q), int(c)) for q, c in query_context_pairs)
    if not pairs:
        raise ValueError(f"{KIND}: query_context_pairs must be non-empty")
    for query, context in pairs:
        if not 0 < query <= context <= _MAX_CONTEXT:
            raise ValueError(f"{KIND}: each pair needs 0 < query <= context <= {_MAX_CONTEXT}")
    if (int(head_dim), int(rope_dim)) != (_HEAD_DIM, _ROPE_DIM):
        raise ValueError(f"{KIND}: FlashMLA takes head_dim 512 and rope_dim 64")
    if int(num_heads) not in _SUPPORTED_HEADS:
        raise ValueError(f"{KIND}: num_heads must be 64 or 128, got {num_heads}")
    if int(n_wv_group) * _WV_GROUP_HEADS != int(num_heads):
        raise ValueError(f"{KIND}: n_wv_group must be num_heads / 8")
    ratio, topk, window = int(compress_ratio), int(index_topk), int(window_size)
    if window < 1 or window % 4:
        raise ValueError(f"{KIND}: window_size must be a positive multiple of 4")
    if ratio < 0 or topk < 0 or (ratio == 0) != (topk == 0) or topk % 4:
        raise ValueError(
            f"{KIND}: compress_ratio and index_topk are both 0 (SWA only) or both positive "
            "(index_topk a multiple of 4)"
        )
    dtypes = tuple(DType.from_value(d) for d in (q_dtype, kv_dtype, output_dtype))
    if dtypes != (DType.BF16, DType.BF16, DType.FP8_E4M3):
        raise ValueError(f"{KIND}: q/kv must be bf16 and the output fp8_e4m3, got {dtypes}")
    return Shape(
        pairs=pairs,
        window_size=window,
        compress_ratio=ratio,
        index_topk=topk,
        num_heads=int(num_heads),
        n_wv_group=int(n_wv_group),
        enable_q_norm=bool(enable_q_norm),
    )


@dataclass
class Workload:
    q: Any  # [T, H, 512] bf16, standard layout
    kv: Any  # [total_kv, 1, 512] bf16
    indices: Any  # [T, W] int32
    positions: Any  # [T] int32
    cos_sin: Any  # [P, 64] fp32
    sink: Any  # [H] fp32
    valid_keys: int


def cos_sin_cache(torch: Any, max_pos: int, device: Any) -> Any:
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, _ROPE_DIM, 2, device=device, dtype=torch.float32) / _ROPE_DIM)
    )
    freqs = torch.outer(torch.arange(max_pos, device=device, dtype=torch.float32), inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()


def swa_indices(torch: Any, shape: Shape, device: Any) -> tuple[Any, Any]:
    """Expanded SWA index ``[T, window]`` and token positions ``[T]``.

    ``prepare_prompt_v4``'s pivots: section 1 is the request's history window
    in the first ``requests x window`` rows, section 2 this step's rows.
    """
    window = shape.window_size
    n_seq = len(shape.pairs)
    request = torch.repeat_interleave(
        torch.arange(n_seq, dtype=torch.int64),
        torch.tensor([q for q, _ in shape.pairs], dtype=torch.int64),
    )
    cu = torch.zeros(n_seq + 1, dtype=torch.int64)
    cu[1:] = torch.cumsum(torch.tensor([q for q, _ in shape.pairs], dtype=torch.int64), 0)
    q_start = torch.tensor([c - q for q, c in shape.pairs], dtype=torch.int64)[request]
    j = torch.arange(shape.num_tokens, dtype=torch.int64) - cu[request]
    pos = q_start + j
    abs_lo = (pos - window + 1).clamp(min=0)
    start_slot = torch.minimum(abs_lo, q_start) - (q_start - window)
    sec1_start = request * window + start_slot
    sec1_len = window - start_slot
    k_lo = (abs_lo - q_start).clamp(min=0)
    sec2_start = n_seq * window + cu[request] + k_lo
    sec2_len = j - k_lo + 1
    col = torch.arange(window, dtype=torch.int64)[None, :]
    out = torch.full((shape.num_tokens, window), -1, dtype=torch.int64)
    out = torch.where(col < sec1_len[:, None], sec1_start[:, None] + col, out)
    in2 = (col >= sec1_len[:, None]) & (col < (sec1_len + sec2_len)[:, None])
    out = torch.where(in2, (sec2_start - sec1_len)[:, None] + col, out)
    return out.to(torch.int32).to(device), pos.to(torch.int32).to(device)


def topk_indices(torch: Any, shape: Shape, positions: Any, generator: Any) -> Any:
    """Stratified causal top-k ``[T, index_topk]`` into the gathered buffer (``-1`` unused)."""
    ratio, topk = shape.compress_ratio, shape.index_topk
    n_seq = len(shape.pairs)
    counts = torch.tensor([q for q, _ in shape.pairs], dtype=torch.int64)
    request = torch.repeat_interleave(torch.arange(n_seq, dtype=torch.int64), counts)
    per_request = torch.tensor([max(1, c // ratio) for _, c in shape.pairs], dtype=torch.int64)
    base = torch.zeros(n_seq, dtype=torch.int64)
    base[1:] = torch.cumsum(per_request, 0)[:-1]
    offset = n_seq * shape.window_size + shape.num_tokens + base[request]
    available = (positions.cpu().long() + 1) // ratio
    count = available.clamp(max=topk)
    columns = torch.arange(topk, dtype=torch.int64)
    denominator = count.clamp(min=1)[:, None]
    lower = columns[None, :] * available[:, None] // denominator
    upper = (columns[None, :] + 1) * available[:, None] // denominator
    jitter = torch.rand((shape.num_tokens, topk), generator=generator)
    picked = lower + (jitter * (upper - lower).clamp(min=1)).long()
    picked = torch.minimum(picked, (upper - 1).clamp(min=0))
    return torch.where(columns[None, :] < count[:, None], picked + offset[:, None], -1).to(
        torch.int32
    )


def build(torch: Any, shape: Shape, device: Any) -> Workload:
    generator = torch.Generator().manual_seed(0)
    swa, positions = swa_indices(torch, shape, device)
    if shape.index_topk:
        extra = topk_indices(torch, shape, positions, generator).to(device)
        indices = torch.cat((swa, extra), dim=1).contiguous()
    else:
        indices = swa
    n_seq = len(shape.pairs)
    compressed = (
        sum(max(1, c // shape.compress_ratio) for _, c in shape.pairs)
        if shape.compress_ratio
        else 0
    )
    total_kv = n_seq * shape.window_size + shape.num_tokens + compressed
    torch.manual_seed(0)
    q = torch.randn(shape.num_tokens, shape.num_heads, _HEAD_DIM, device=device).to(torch.bfloat16)
    kv = (torch.randn(total_kv, 1, _HEAD_DIM, device=device) * 0.5).to(torch.bfloat16)
    sink = torch.randn(shape.num_heads, device=device, dtype=torch.float32)
    max_pos = max(c for _, c in shape.pairs)
    return Workload(
        q=q,
        kv=kv,
        indices=indices,
        positions=positions,
        cos_sin=cos_sin_cache(torch, max_pos, device),
        sink=sink,
        valid_keys=int((indices >= 0).sum().item()),
    )


def kernel_call(shape: Shape, work: Workload, q_fused: Any) -> Any:
    """``prefill_attn_varlen_sparse``'s fused call with DeepInfer's kwargs."""
    from flash_mla import fused_norm_rope_attn_rope_cast

    def run() -> Any:
        return fused_norm_rope_attn_rope_cast.prefill(
            q=q_fused,
            kv=work.kv,
            indices=work.indices.unsqueeze(1),
            sm_scale=_HEAD_DIM**-0.5,
            attn_sink=work.sink,
            topk_length=None,
            enable_q_norm=shape.enable_q_norm,
            rms_norm_eps=_RMS_EPS if shape.enable_q_norm else 0,
            token_positions=work.positions,
            is_rope_neox_style=False,
            rope_dim=_ROPE_DIM,
            cos_sin_cache=work.cos_sin,
            n_wv_group=shape.n_wv_group,
            num_per_channels=32,
            use_tma_aligned_col_major_sf=True,
            round_sf=True,
            use_packed_ue8m0=True,
        )

    return run


def reference(torch: Any, shape: Shape, work: Workload, rows: Any) -> Any:
    """fp32 ``[R, H, 512]`` output before the FP8 cast, for the sampled ``rows``."""
    from profiling.runners.attention.compressed_sparse_mla_rope_cast_reference import _rope

    q = work.q[rows].float()
    if shape.enable_q_norm:
        q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + _RMS_EPS)
    table = work.cos_sin[work.positions[rows].long()]
    cos, sin = table[:, None, : _ROPE_DIM // 2], table[:, None, _ROPE_DIM // 2 :]
    q = _rope(q, cos, sin).to(torch.bfloat16).float()
    idx = work.indices[rows].long()
    mask = idx >= 0
    keys = work.kv[idx.clamp(min=0), 0].float() * mask[..., None]
    scores = torch.einsum("thd,tkd->thk", q, keys) * _HEAD_DIM**-0.5
    scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
    sink = work.sink.float()
    row_max = torch.maximum(scores.amax(-1), sink[None, :])
    weights = torch.exp(scores - row_max[..., None])
    denominator = weights.sum(-1) + torch.exp(sink[None, :] - row_max)
    attended = torch.einsum("thk,tkd->thd", weights, keys) / denominator[..., None]
    return _rope(attended, cos, -sin)


def check(torch: Any, shape: Shape, work: Workload, out: Any) -> dict:
    """Compare sampled rows of the kernel's FP8 output with :func:`reference`."""
    from profiling.runners.attention.compressed_sparse_mla_rope_cast_reference import (
        dequantize_output,
        output_from_fused_layout,
        quantize_output,
    )

    out_fp8, out_sf = out[0], out[1]
    n = shape.num_tokens
    rows = (
        torch.arange(n)
        if n <= _CHECK_ROWS
        else torch.linspace(0, n - 1, _CHECK_ROWS).round().long().unique()
    ).to(work.q.device)
    values, exponents = output_from_fused_layout(out_fp8[rows], out_sf[rows].contiguous())
    got = dequantize_output(values, exponents)
    expected = reference(torch, shape, work, rows)
    _, ref_exponents = quantize_output(expected)
    gap = int((exponents.int() - ref_exponents.int()).abs().max().item())
    rel = (got - expected).norm(dim=-1) / expected.norm(dim=-1).clamp(min=1e-6)
    report = {
        "rows_checked": int(rows.numel()),
        "max_rel_l2": float(rel.max().item()),
        "mean_rel_l2": float(rel.mean().item()),
        "max_exponent_gap": gap,
    }
    if not bool(torch.isfinite(got).all()) or gap > 1 or report["max_rel_l2"] > _REL_TOL:
        raise KernelLaunchFailed(f"{KIND}:{BACKEND} mismatches the Torch reference: {report}")
    return report


def prepare(torch: Any, shape: Shape, device: Any) -> tuple[Workload, Any, dict]:
    """Build, run once (JIT/warm), check, and return ``(work, run, check report)``."""
    from profiling.runners.attention.compressed_sparse_mla_rope_cast_reference import (
        q_to_fused_layout,
    )

    work = build(torch, shape, device)
    q_fused = q_to_fused_layout(work.q).contiguous()
    run = kernel_call(shape, work, q_fused)
    out = run()
    torch.cuda.synchronize(device)
    report = check(torch, shape, work, out)
    del out
    return work, run, report


def logical_work(shape: Shape, work: Workload) -> tuple[float, float]:
    """FLOPs and bytes: scores and values each ``2 * H * 512`` FLOPs per valid key;
    bytes are Q (bf16), one bf16 KV row and one index word per valid key, the
    index rows, and the FP8 output with its UE8M0 scales."""
    t, h = shape.num_tokens, shape.num_heads
    flops = 4.0 * h * _HEAD_DIM * work.valid_keys
    q_bytes = t * h * _HEAD_DIM * 2
    out_bytes = t * h * (_HEAD_DIM + _HEAD_DIM // 32)
    bytes_ = q_bytes + out_bytes + 4 * t * shape.width + work.valid_keys * _HEAD_DIM * 2
    return flops, float(bytes_)


def profile_sparse_mla_norm_rope_cast_prefill_flashmla(**kwargs: Any) -> ComputeMetrics:
    shape = validate(**kwargs)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ProfilerNotImplemented("PyTorch is unavailable") from exc
    try:
        import flash_mla  # noqa: F401
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{KIND}:{BACKEND} needs DeepInfer's FlashMLA build (deepinfer_env)"
        ) from exc
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        work, run, _report = prepare(torch, shape, device)
        time_ms = Timer.cupti(run, kernel_name=KERNEL_NAME)
        energy_j = Energy.perf(run, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{KIND}:{BACKEND} ran out of CUDA memory") from exc
    except (KernelLaunchFailed, ProfilerNotImplemented, TypeError, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{KIND}:{BACKEND} failed: {exc}") from exc
    flops, bytes_ = logical_work(shape, work)
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=flops / seconds / 1e12,
        memory_bandwidth_gbps=bytes_ / seconds / 1e9,
        energy_j=float(energy_j),
    )


__all__ = ["profile_sparse_mla_norm_rope_cast_prefill_flashmla", "swa_indices", "validate"]
