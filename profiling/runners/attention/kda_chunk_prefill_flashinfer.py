"""FlashInfer Blackwell runners for the GLM-5.3-Flash KDA chunked prefill.

This runner only allocates tensors, times one public call, and returns
metrics. DB writes, subprocess selection and registry routing live in L1b.

It times ``flashinfer.kda.recurrent_kda`` with an explicit ``backend=``, made
the way vLLM's ``_flashinfer_kda_prefill`` makes it
(``vllm/models/kimi_k3/nvidia/kda.py``, vLLM PR #55364), on the same operands
as the ``vllm_triton`` runner:

- q/k/v are row-strided ``[1,T,H,K]`` views into one ``[T, 3*H*K]`` buffer.
  Both selected backends require dense q/k/v, so the call's own
  ``.contiguous()`` copies are three real launches inside the time, as they
  are for ``vllm_triton``.
- ``g`` is the raw bf16 gate projection ``[1,T,H,K]``; the gate
  ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`` is computed in-kernel
  (``use_gate_in_kernel=True``).
- ``beta`` is the raw bf16 logit ``[1,T,H]``; its sigmoid is in-kernel
  (``beta_is_logit=True``). The ``vllm_triton`` call instead receives an fp32
  beta that a separate launch outside the call has already passed through a
  sigmoid, so the two rows bound slightly different work.
- ``A_log`` is fp32 ``[H]``, ``dt_bias`` fp32 ``[H*K]``.
- ``initial_state`` is fp32 ``[N,H,V,K]`` (decodes random, prefills zero),
  updated in place; ``output_final_state=False`` as in vLLM, which then reads
  the updated ``initial_state`` back.
- ``cu_seqlens`` is int64, as vLLM's FlashInfer metadata builds it once per
  step (int32 would add a conversion launch per call).
- ``output`` is a preallocated bf16 buffer, as vLLM passes ``core_attn_out``.

vLLM also passes a longest-first ``seq_order`` for mixed batches. Both
backends here reject ``seq_order`` and schedule sequences themselves, so it is
not passed.

Timing is the CUPTI sum over every launch of one call (``kernel_name=None``):
the three copies plus the backend's own kernels. Each eager call also reads
``cu_seqlens`` back to the host to build or check its work plan; that sync is
host time and not in the row. See ``profiling/kernels/kda_chunk_prefill.py``
for the per-backend launch lists.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention import kda_chunk_prefill_vllm_triton as _layout
from profiling.runners.attention._gdn_common import load_required_callable
from profiling.runners.attention.kda_chunk_prefill_vllm_triton import (
    KdaChunkPrefillShape,
    guard_shape,
    logical_bytes,
    semantic_flops,
    sequence_lengths,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_MODULE = "flashinfer.kda"
_CALLABLE = "recurrent_kda"
_ENVIRONMENT = "flashinfer_kda_env"
LOWER_BOUND = _layout.LOWER_BOUND
# Both backends are D128-only and tile heads in groups of eight
# (flashinfer/kda_prefill_tirx.py, flashinfer/kda_prefill_persistent.py).
HEAD_DIM = 128
HEAD_GROUP = 8
# FlashInfer's own KDA validation bound is a 3% relative L2 error; these
# backends round tensor-core operands and residuals to bf16. Measured on B200
# against the per-token oracle: output 4.2e-3..5.3e-3, state 3.2e-4..4.4e-3.
RMSE_RATIO_TOL = 1e-2
# 32-bit index limits both backends assert on (tokens*H*K, sequences*H*K*K).
# With H >= 8 they also cover TIRx's own caps (< 2**21 tokens, < 65536
# sequences), which therefore never bind first.
_INDEX_LIMIT = 2**31


@dataclass(frozen=True)
class _Backend:
    """One FlashInfer ``recurrent_kda`` backend and its shape limits."""

    name: str  # the registry backend string
    flashinfer_backend: str  # recurrent_kda(backend=...)
    packages: tuple[str, ...]  # distributions recorded in row provenance
    heads_up_to_sm_count: bool = False  # TIRx gives each head its own CTA set


TIRX = _Backend(
    name="flashinfer_tirx",
    flashinfer_backend="tirx",
    packages=("flashinfer-python", "apache-tvm", "apache-tvm-ffi", "tirx-kernels"),
    heads_up_to_sm_count=True,
)
CUTE_PERSISTENT = _Backend(
    name="flashinfer_cute_persistent",
    flashinfer_backend="cute-dsl-persistent",
    packages=("flashinfer-python", "nvidia-cutlass-dsl"),
)


@dataclass(frozen=True)
class _Operands:
    qkv: Any
    q: Any
    k: Any
    v: Any
    g: Any
    beta_logit: Any
    a_log: Any
    dt_bias: Any
    initial_state: Any
    cu_seqlens: Any
    output: Any
    boundaries: tuple[int, ...]


def validate_args(
    backend: _Backend,
    num_tokens: int,
    max_sequence_length: int,
    num_decode_sequences: int,
    num_heads: int,
    head_dim: int,
    dtype: DType | str,
) -> KdaChunkPrefillShape:
    """The vllm_triton path contract plus this backend's shape limits."""
    shape = _layout.validate_args(
        num_tokens, max_sequence_length, num_decode_sequences, num_heads, head_dim, dtype
    )
    label = f"kda_chunk_prefill:{backend.name}"
    if shape.head_dim != HEAD_DIM:
        raise ValueError(f"{label} requires head_dim={HEAD_DIM}, got {shape.head_dim}")
    if shape.num_heads % HEAD_GROUP:
        raise ValueError(f"{label} requires num_heads divisible by {HEAD_GROUP}")
    sequences = len(sequence_lengths(shape))
    if shape.num_tokens * shape.num_heads * HEAD_DIM >= _INDEX_LIMIT:
        raise ValueError(f"{label} requires num_tokens*num_heads*128 < 2**31")
    if sequences * shape.num_heads * HEAD_DIM * HEAD_DIM >= _INDEX_LIMIT:
        raise ValueError(f"{label} requires sequences*num_heads*128*128 < 2**31")
    return shape


def build_operands(torch: Any, shape: KdaChunkPrefillShape, *, device: Any) -> _Operands:
    """The vllm_triton operands, with beta as raw bf16 logits and a fixed output."""
    base = _layout.build_operands(torch, shape, device=device)
    generator = torch.Generator(device=device).manual_seed(43)
    beta_logit = torch.randn(
        1, shape.num_tokens, shape.num_heads, generator=generator, device=device
    ).to(torch.bfloat16)
    output = torch.empty(
        1, shape.num_tokens, shape.num_heads, shape.head_dim, dtype=torch.bfloat16, device=device
    )
    return _Operands(
        qkv=base.qkv,
        q=base.q,
        k=base.k,
        v=base.v,
        g=base.raw_g,
        beta_logit=beta_logit,
        a_log=base.a_log.reshape(shape.num_heads),
        dt_bias=base.dt_bias,
        initial_state=base.initial_state,
        cu_seqlens=base.cu_seqlens.to(torch.int64),
        output=output,
        boundaries=base.boundaries,
    )


def invoke(callable_: Any, backend: _Backend, operands: _Operands) -> Any:
    """The keyword set of vLLM's ``_flashinfer_kda_prefill``, backend made explicit."""
    return callable_(
        q=operands.q.contiguous(),
        k=operands.k.contiguous(),
        v=operands.v.contiguous(),
        g=operands.g,
        beta=operands.beta_logit,
        A_log=operands.a_log,
        dt_bias=operands.dt_bias,
        scale=operands.q.shape[-1] ** -0.5,
        initial_state=operands.initial_state,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=LOWER_BOUND,
        cu_seqlens=operands.cu_seqlens,
        output=operands.output,
        beta_is_logit=True,
        backend=backend.flashinfer_backend,
    )


def check_correctness(
    torch: Any,
    callable_: Any,
    backend: _Backend,
    operands: _Operands,
    shape: KdaChunkPrefillShape,
) -> tuple[float, float]:
    """One call against the naive oracle; return (output, state) RMSE ratios."""
    from profiling.runners.attention.kda_chunk_prefill_reference import (
        kda_chunk_prefill_reference,
        rmse_ratio,
    )

    snapshots = {
        name: getattr(operands, name).clone()
        for name in ("qkv", "g", "beta_logit", "a_log", "dt_bias", "initial_state", "cu_seqlens")
    }
    output, _ = invoke(callable_, backend, operands)
    torch.cuda.synchronize()
    if output.data_ptr() != operands.output.data_ptr():
        raise AssertionError("recurrent_kda did not write the supplied output buffer")
    final_state = operands.initial_state
    if not (torch.isfinite(output).all() and torch.isfinite(final_state).all()):
        raise AssertionError("non-finite KDA output or state")
    for name, snapshot in snapshots.items():
        if name != "initial_state" and not torch.equal(getattr(operands, name), snapshot):
            raise AssertionError(f"recurrent_kda mutated {name}")

    expected_o, expected_state = kda_chunk_prefill_reference(
        operands.q.squeeze(0),
        operands.k.squeeze(0),
        operands.v.squeeze(0),
        operands.g.squeeze(0),
        torch.sigmoid(operands.beta_logit.float()).squeeze(0),
        operands.a_log,
        operands.dt_bias,
        snapshots["initial_state"],
        operands.boundaries,
        lower_bound=LOWER_BOUND,
    )
    o_err = rmse_ratio(expected_o, output.squeeze(0))
    state_err = rmse_ratio(expected_state, final_state)
    if not (o_err < RMSE_RATIO_TOL and state_err < RMSE_RATIO_TOL):
        raise AssertionError(
            f"KDA mismatch vs naive oracle: o rmse ratio {o_err:.2e}, "
            f"state rmse ratio {state_err:.2e} (tol {RMSE_RATIO_TOL:.0e})"
        )
    return o_err, state_err


def _profile(
    backend: _Backend,
    num_tokens: int,
    max_sequence_length: int,
    num_decode_sequences: int,
    num_heads: int,
    head_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    shape = validate_args(
        backend, num_tokens, max_sequence_length, num_decode_sequences, num_heads, head_dim, dtype
    )
    label = f"kda_chunk_prefill:{backend.name}"
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {label}") from exc
    try:
        callable_ = load_required_callable(
            importlib.import_module,
            backend=label,
            module_name=_MODULE,
            callable_name=_CALLABLE,
            environment_label=_ENVIRONMENT,
        )
        device = torch.device("cuda", torch.cuda.current_device())
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        if backend.heads_up_to_sm_count and shape.num_heads > sm_count:
            raise ValueError(f"{label} requires num_heads <= SM count ({sm_count})")
        guard = guard_shape(shape)
        check_correctness(
            torch, callable_, backend, build_operands(torch, guard, device=device), guard
        )
        operands = build_operands(torch, shape, device=device)

        def kernel() -> Any:
            return invoke(callable_, backend, operands)

        # The call updates initial_state in place. Repeated calls keep it
        # bounded (decayed delta-rule updates) and the kernels' work does not
        # depend on state values, so no reset is needed between launches.
        time_ms = Timer.cupti(kernel, warmup=5, kernel_name=None)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{label} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{label} failed: {exc}") from exc

    elapsed = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=semantic_flops(shape) / elapsed / 1e12 if elapsed > 0 else 0.0,
        memory_bandwidth_gbps=logical_bytes(shape) / elapsed / 1e9 if elapsed > 0 else 0.0,
        energy_j=float(energy_j),
    )


def profile_kda_chunk_prefill_flashinfer_tirx(
    num_tokens: int,
    max_sequence_length: int,
    num_decode_sequences: int,
    num_heads: int,
    head_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile one ``recurrent_kda(backend="tirx")`` prefill call."""
    return _profile(
        TIRX, num_tokens, max_sequence_length, num_decode_sequences, num_heads, head_dim, dtype
    )


def profile_kda_chunk_prefill_flashinfer_cute_persistent(
    num_tokens: int,
    max_sequence_length: int,
    num_decode_sequences: int,
    num_heads: int,
    head_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile one ``recurrent_kda(backend="cute-dsl-persistent")`` prefill call."""
    return _profile(
        CUTE_PERSISTENT,
        num_tokens,
        max_sequence_length,
        num_decode_sequences,
        num_heads,
        head_dim,
        dtype,
    )


def package_versions(backend: _Backend) -> str:
    """``name version`` for each package, plus FlashInfer's source commit."""
    parts = []
    for package in backend.packages:
        try:
            parts.append(f"{package} {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            parts.append(f"{package} missing")
    try:
        commit = importlib.import_module("flashinfer._build_meta").__git_commit__
    except (ImportError, AttributeError):
        commit = "unknown"
    return f"{', '.join(parts)}; flashinfer commit {commit}"


def row_provenance_tirx(**_spec: Any) -> str:
    """The FlashInfer and TIRx stack that measured a ``flashinfer_tirx`` row."""
    return package_versions(TIRX)


def row_provenance_cute_persistent(**_spec: Any) -> str:
    """The FlashInfer and CuTe DSL stack that measured a ``flashinfer_cute_persistent`` row."""
    return package_versions(CUTE_PERSISTENT)


__all__ = [
    "CUTE_PERSISTENT",
    "TIRX",
    "build_operands",
    "check_correctness",
    "invoke",
    "package_versions",
    "profile_kda_chunk_prefill_flashinfer_cute_persistent",
    "profile_kda_chunk_prefill_flashinfer_tirx",
    "row_provenance_cute_persistent",
    "row_provenance_tirx",
    "validate_args",
]
