"""MI300X rope-free BF16 sparse-MLA attention (decode) via the vLLM Triton kernel.

This is the ROCm/MI300X backend of the ``dsa_sparse_mla_attention`` kind -- the
decode step of GLM-5.3-Flash's DSA sparse attention, where each query token
attends to its DSA-selected cache tokens. It is the AMD analog of the B200
``flashinfer_trtllm_fp8`` backend, but a different kernel AND a different dtype:
the vLLM Triton ragged kernel ``_sparse_attn_prefill_ragged_kernel`` reached
through ``rocm_sparse_attn_prefill`` at ``head_dim=512, nope_head_dim=512,
rope_head_dim=0`` on a **BF16** latent, not FlashInfer fp8. A decode step is a
ragged batch whose every request contributes a single query row; GLM's rope-free
latent cannot use the fp8/448+64 decode entry point, so both decode and prefill
funnel through ``rocm_sparse_attn_prefill`` (see
``_rocm_triton_mla_sparse_common`` for why).

The kind's ``DsaSparseMlaAttentionArgs`` is reused unchanged and validated by the
B200 runner's ``_validate_args`` -- with ``expected_rope_dim=0`` and the rope-free
cache layout -- so the MI300X and NVIDIA rows sit at the same arg coordinate and
differ only by ``(gpu_name, backend, rope_dim, dtype)``. Timing is kernel-only via
rocprofv3 with the autotune-robust trailing fold; correctness is checked against a
rope-free Torch reference before timing, and the Triton path is asserted from the
capture's kernel-name set (no AITER opus fall-through).
"""

from __future__ import annotations

from typing import Any

import profiling.runners.attention._rocm_triton_mla_sparse_common as common
from profiling.db.args import DType
from profiling.runners.attention.dsa_sparse_mla_attention import (
    _decode_valid_counts,
    _row_indices,
    _validate_args,
)
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_KIND = "dsa_sparse_mla_attention"
_BACKEND = "rocm_triton_mla_sparse"
_FULL = f"{_KIND}:{_BACKEND}"
# Per-call GPU-dispatch count of rocm_sparse_attn_prefill on the rope-free Triton
# path: the ragged attention kernel ``_sparse_attn_prefill_ragged_kernel`` plus
# the one ``__amd_rocclr_copyBuffer`` the path issues to stage its output chunk.
# Pinned from the first real MI300X rocpd capture (job on lease 448076, gfx942);
# the trailing fold sums the last rep*D dispatches, so a one-time device-init /
# ragged-index-build prefix is skipped. None would make the profiler refuse to
# guess a time.
_DISPATCHES_PER_LAUNCH: int | None = 2


def _validate(**kwargs: Any) -> Any:
    # selected_k is gated to the allowed page-table widths here, in the runner:
    # the shared ``_validate_args`` no longer allowlists it (the B200 runner moved
    # operand bounds to the sweep grid's infeasible_mask), but the MI300X Triton
    # path composes the 2176 kpool page-table width / 2048 raw index_topk and
    # accepts only those (see ``common.ALLOWED_SELECTED_K``). This mirrors the
    # prefill runner's own inline selected_k check.
    selected_k = kwargs.get("selected_k")
    if selected_k not in common.ALLOWED_SELECTED_K:
        raise ProfilerNotImplemented(
            f"{_FULL} requires selected_k in "
            f"{sorted(common.ALLOWED_SELECTED_K)}, got {selected_k}"
        )
    return _validate_args(
        expected_rope_dim=common.ROPE_HEAD_DIM,
        expected_cache_layout=common.CACHE_LAYOUT,
        **kwargs,
    )


def _build_batch(torch: Any, validated: Any, *, selected_k: int, num_cache_tokens: int) -> common.RaggedBatch:
    num_queries = validated.num_queries
    dense = torch.full((num_queries, selected_k), -1, dtype=torch.int32)
    lengths = torch.zeros(num_queries, dtype=torch.int32)
    for row, count in enumerate(validated.valid_counts):
        if count:
            idx = _row_indices(
                torch,
                row=row,
                count=count,
                num_cache_tokens=num_cache_tokens,
                distribution=validated.index_distribution,
                device=torch.device("cpu"),
            )
            dense[row, :count].copy_(idx.to(torch.int32))
        lengths[row] = count
    return common.RaggedBatch(
        dense_indices=dense,
        lengths=lengths,
        num_cache_tokens=num_cache_tokens,
        num_queries=num_queries,
        valid_counts=tuple(validated.valid_counts),
    )


def build_dsa_sparse_mla_attention_rocm_triton_kernel(
    *,
    num_queries: int,
    num_cache_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    selected_k: int,
    latent_dim: int,
    rope_dim: int,
    value_dim: int,
    softmax_scale: float,
    q_dtype: DType | str,
    cache_dtype: DType | str,
    index_dtype: str,
    output_dtype: DType | str,
    valid_counts: str,
    index_distribution: str,
    cache_layout: str,
) -> dict[str, Any]:
    """Build the timed call + fixed operands for one decode sparse-MLA batch."""
    validated = _validate(
        num_queries=num_queries,
        num_cache_tokens=num_cache_tokens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        selected_k=selected_k,
        latent_dim=latent_dim,
        rope_dim=rope_dim,
        value_dim=value_dim,
        softmax_scale=softmax_scale,
        q_dtype=q_dtype,
        cache_dtype=cache_dtype,
        index_dtype=index_dtype,
        output_dtype=output_dtype,
        valid_counts=valid_counts,
        index_distribution=index_distribution,
        cache_layout=cache_layout,
    )
    import torch

    common.load_callables(torch)
    device = torch.device("cuda", torch.cuda.current_device())
    batch = _build_batch(
        torch, validated, selected_k=selected_k, num_cache_tokens=num_cache_tokens
    )
    operands = common.build_operands(torch, batch, num_heads=num_heads, device=device)
    kernel = common.make_kernel(torch, operands)
    return {
        "torch": torch,
        "kernel": kernel,
        "operands": operands,
        "batch": batch,
        "num_heads": num_heads,
        "warmup": common.WARMUP,
        "rep": common.REP,
    }


def _spec(**kwargs: Any) -> dict[str, Any]:
    spec = dict(kwargs)
    spec["q_dtype"] = DType.from_value(kwargs["q_dtype"]).value
    spec["cache_dtype"] = DType.from_value(kwargs["cache_dtype"]).value
    spec["output_dtype"] = DType.from_value(kwargs["output_dtype"]).value
    return spec


def profile_dsa_sparse_mla_attention_rocm_triton(
    *,
    num_queries: int,
    num_cache_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    selected_k: int,
    latent_dim: int,
    rope_dim: int,
    value_dim: int,
    softmax_scale: float,
    q_dtype: DType | str,
    cache_dtype: DType | str,
    index_dtype: str,
    output_dtype: DType | str,
    valid_counts: str,
    index_distribution: str,
    cache_layout: str,
) -> ComputeMetrics:
    """Profile one GLM-5.3-Flash sparse-MLA decode call on an MI300X."""
    kwargs = dict(
        num_queries=num_queries,
        num_cache_tokens=num_cache_tokens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        selected_k=selected_k,
        latent_dim=latent_dim,
        rope_dim=rope_dim,
        value_dim=value_dim,
        softmax_scale=softmax_scale,
        q_dtype=q_dtype,
        cache_dtype=cache_dtype,
        index_dtype=index_dtype,
        output_dtype=output_dtype,
        valid_counts=valid_counts,
        index_distribution=index_distribution,
        cache_layout=cache_layout,
    )
    validated = _validate(**kwargs)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_FULL}") from exc
    try:
        common.require_mi300(torch, backend=_FULL)
        from profiling.profilers.rocprof_kernel_profiler import measure_registered_via_rocprofv3

        built = build_dsa_sparse_mla_attention_rocm_triton_kernel(**kwargs)
        common.check_correctness(
            torch, built["operands"], built["batch"], num_heads=num_heads, backend=_FULL
        )

        spec = _spec(**kwargs)
        common.assert_triton_path(kind=_KIND, backend=_BACKEND, spec=spec)

        if _DISPATCHES_PER_LAUNCH is None:
            raise ProfilerNotImplemented(
                f"{_FULL}: per-launch dispatch count D not yet pinned; "
                "read it off the first MI300X rocpd capture before timing"
            )
        time_ms = measure_registered_via_rocprofv3(
            kind=_KIND,
            backend=_BACKEND,
            spec=spec,
            kernel_name_contains=None,
            warmup=common.WARMUP,
            rep=common.REP,
            dispatches_per_launch=_DISPATCHES_PER_LAUNCH,
        )
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError(f"{_FULL} ran out of device memory") from exc
        raise KernelLaunchFailed(f"{_FULL} failed: {exc}") from exc

    elapsed = time_ms / 1000.0
    flops = common.logical_flops(num_queries=num_queries, num_heads=num_heads, selected_k=selected_k)
    byts = common.logical_bytes(
        num_queries=num_queries,
        num_heads=num_heads,
        selected_k=selected_k,
        valid_counts=validated.valid_counts,
    )
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=flops / elapsed / 1e12 if elapsed > 0 else 0.0,
        memory_bandwidth_gbps=byts / elapsed / 1e9 if elapsed > 0 else 0.0,
        energy_j=0.0,
    )


__all__ = [
    "build_dsa_sparse_mla_attention_rocm_triton_kernel",
    "profile_dsa_sparse_mla_attention_rocm_triton",
]
