"""FlashInfer runner for the GDN prefill chunked delta rule.

This file is L1a-only: it allocates tensors, times one kernel, and returns
metrics. DB writes, JIT policy, subprocess selection, and registry routing all
live in L1b.

The measured callable is `flashinfer.gdn_prefill.chunk_gated_delta_rule`, reached
through exactly the operand preparation vLLM performs in
`fi_chunk_gated_delta_rule` (`vllm/model_executor/layers/mamba/gdn/
qwen_gdn_linear_attn.py`): contiguous 3-D q/k/v, `g` passed already exponentiated
in FP32, `beta` in FP32, and an FP32 initial state. `use_qk_l2norm_in_kernel`
stays False because vLLM applies `l2norm_fwd` outside this call — folding it in
would measure a kernel production never launches here.

`cu_seqlens` is int64 and `backend="flashinfer"`, as vLLM passes them. FlashInfer
dispatches by architecture: one CUTLASS `FlatKernel` launch on SM90, one CuTe DSL
`GatedDeltaNetChunkedKernel` persistent launch on SM100 (or several launches if
its context-parallel heuristic fires). CUPTI therefore counts every launch the
call makes; operand construction sits outside the measured window. Reported FLOPs and
bytes are semantic logical counts, not physical CUTLASS instructions or traffic.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention._gdn_common import (
    canonical_sequence_boundaries as _canonical_sequence_boundaries,
)
from profiling.runners.attention._gdn_common import (
    chunk_delta_rule_logical_bytes,
    chunk_delta_rule_semantic_flops,
    load_required_callable,
)
from profiling.runners.attention._gdn_common import (
    exact_int as _exact_int,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "gdn_chunk_delta_rule:flashinfer"
_CALLABLE_MODULE = "flashinfer.gdn_prefill"
_CALLABLE_NAME = "chunk_gated_delta_rule"


@dataclass(frozen=True)
class _Args:
    num_tokens: int
    max_sequence_length: int
    num_key_heads: int
    num_heads: int
    key_head_dim: int
    value_head_dim: int
    dtype: DType


def _validate_args(
    num_tokens: int,
    max_sequence_length: int,
    num_key_heads: int,
    num_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
) -> _Args:
    args = _Args(
        num_tokens=_exact_int("num_tokens", num_tokens),
        max_sequence_length=_exact_int("max_sequence_length", max_sequence_length),
        num_key_heads=_exact_int("num_key_heads", num_key_heads),
        num_heads=_exact_int("num_heads", num_heads),
        key_head_dim=_exact_int("key_head_dim", key_head_dim),
        value_head_dim=_exact_int("value_head_dim", value_head_dim),
        dtype=DType(dtype) if not isinstance(dtype, DType) else dtype,
    )
    if args.max_sequence_length < 1 or args.num_tokens < args.max_sequence_length:
        raise ValueError("num_tokens must be >= max_sequence_length >= 1")
    if args.num_heads % args.num_key_heads != 0:
        raise ValueError("num_heads must be a multiple of num_key_heads")
    if args.dtype is not DType.BF16:
        raise ProfilerNotImplemented(f"{_BACKEND} supports bf16 only, got {args.dtype}")
    return args


def profile_gdn_chunk_delta_rule_flashinfer(
    num_tokens: int,
    max_sequence_length: int,
    num_key_heads: int,
    num_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile FlashInfer's GDN prefill delta rule exactly as vLLM calls it."""
    args = _validate_args(
        num_tokens,
        max_sequence_length,
        num_key_heads,
        num_heads,
        key_head_dim,
        value_head_dim,
        dtype,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_BACKEND}") from exc

    try:
        fused_callable = load_required_callable(
            importlib.import_module,
            backend=_BACKEND,
            module_name=_CALLABLE_MODULE,
            callable_name=_CALLABLE_NAME,
            missing_message=(
                f"{_BACKEND} requires {_CALLABLE_MODULE}.{_CALLABLE_NAME}; "
                "install FlashInfer with GDN prefill support"
            ),
        )
        device = torch.device("cuda", torch.cuda.current_device())

        query = torch.randn(
            args.num_tokens, args.num_key_heads, args.key_head_dim,
            dtype=torch.bfloat16, device=device,
        )
        key = torch.randn(
            args.num_tokens, args.num_key_heads, args.key_head_dim,
            dtype=torch.bfloat16, device=device,
        )
        value = torch.randn(
            args.num_tokens, args.num_heads, args.value_head_dim,
            dtype=torch.bfloat16, device=device,
        )
        # vLLM hands FlashInfer `exp(g)` and `beta`, both already FP32; the decay
        # must be in (0, 1] or the recurrent scan diverges over a long chunk.
        decay = torch.rand(
            args.num_tokens, args.num_heads, dtype=torch.float32, device=device
        )
        beta = torch.rand(
            args.num_tokens, args.num_heads, dtype=torch.float32, device=device
        )
        boundaries = _canonical_sequence_boundaries(
            args.num_tokens, args.max_sequence_length
        )
        initial_state = torch.zeros(
            len(boundaries) - 1, args.num_heads, args.key_head_dim, args.value_head_dim,
            dtype=torch.float32, device=device,
        )
        cu_seqlens = torch.tensor(boundaries, dtype=torch.int64, device=device)

        def kernel() -> Any:
            return fused_callable(
                q=query,
                k=key,
                v=value,
                g=decay,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                backend="flashinfer",
            )

        # `initial_state` is read-only here (`output_final_state=True` returns a
        # fresh state tensor rather than updating in place), so repeated calls
        # need no reset between launches.
        time_ms = Timer.cupti(kernel, warmup=5, kernel_name=None)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed") from exc

    flops = chunk_delta_rule_semantic_flops(args)
    logical_bytes = chunk_delta_rule_logical_bytes(args)
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(flops / (time_ms / 1000.0) / 1e12) if time_ms > 0 else 0.0,
        memory_bandwidth_gbps=(
            float(logical_bytes / (time_ms / 1000.0) / 1e9) if time_ms > 0 else 0.0
        ),
        energy_j=float(energy_j),
    )
