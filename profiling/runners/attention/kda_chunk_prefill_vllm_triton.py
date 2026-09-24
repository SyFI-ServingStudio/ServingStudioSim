"""vLLM-fork Triton runner for the GLM-5.3-Flash KDA chunked prefill.

This runner only allocates tensors, times one public call, and returns
metrics. DB writes, subprocess selection and registry routing live in L1b.

It times ``chunk_kda_with_fused_gate`` from
``vllm.models.glm5next.nvidia.ops.third_party.kda``. The operands are laid out
the way ``Glm5NextLinearAttention._forward`` hands them over:

- q/k/v are row-strided ``[1,T,H,K]`` views into one contiguous
  ``[T, 3*H*K]`` short-conv output. The callable's own ``.contiguous()``
  therefore performs three real copies, which the capture records as three
  ``elementwise_kernel`` launches of about 11.6 us each at T=2048.
- ``raw_g`` is the contiguous bf16 ``f_b_proj`` output, ``[1,T,H,K]``.
- ``beta`` has already been passed through the fp32 sigmoid, ``[1,T,H]``.
  The sigmoid is a separate launch outside the call.
- ``A_log`` is ``[1,1,H,1]`` and ``dt_bias`` is ``[H*K]``, both fp32.
- ``initial_state`` is fp32 ``[N,H,V,K]``, as ``gather_initial_states``
  returns it: nonzero rows for decodes, zero rows for fresh prefills.
- ``cu_seqlens`` is int32. ``safe_gate=True``, ``lower_bound=-5.0``,
  ``use_qk_l2norm_in_kernel=True`` and ``output_final_state=True``.

Timing is the CUPTI sum over every launch of one call (``kernel_name=None``);
the callable is a fixed 15-launch chain. The same ``cu_seqlens`` tensor is
reused, so FLA's ``@tensor_cache`` chunk-index helpers hit their cache, as they
do on KDA layers 1..33 of a real iteration. Only layer 0 of each iteration pays
the ~23 us of small index launches plus the host sync that a cache miss costs.
That cost is not part of this row.

Autotune state. Seven of the FLA/KDA Triton kernels are ``@autotune``d on
keys that hold H/K/BT but not the token count, so the first shape a process
tunes picks the configs for every later row (b2-stab). The registry row
therefore disables autotune persistence (``TRITON_CACHE_AUTOTUNING=0``), and
before a worker's first row of a given (num_heads, head_dim, dtype) the runner
tunes once at the documented anchor: the capture layout T=2048, L=2019, D=29
(one 2019-token prefill with 29 co-scheduled decodes), at the spec's own heads,
head dim and dtype. The token layout never changes with H; only the tuning-key
axes come from the spec. The anchor and a digest of the selected configs go
into the row's ``backend_version`` through ``row_provenance``.

FLOPs and bytes are semantic logical counts, not physical Triton work.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from itertools import accumulate
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention._gdn_common import (
    exact_int as _exact_int,
)
from profiling.runners.attention._gdn_common import (
    load_required_callable,
    require_exact_gpu,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.triton_autotune_pin import AutotunePin

_BACKEND = "kda_chunk_prefill:vllm_triton"
_MODULE = "vllm.models.glm5next.nvidia.ops.third_party.kda"
_CALLABLE = "chunk_kda_with_fused_gate"
_GPU = "NVIDIA B200"
_HEAD_DIM = 128
# FLA_CHUNK_SIZE: the callable hard-codes it. Here it only sizes the semantic
# FLOP estimate and `num_chunks`; it never reaches the call.
CHUNK_SIZE = 64
# GLM-5.3-Flash `linear_attn_config.gate_lower_bound`.
LOWER_BOUND = -5.0
# The fork's own chunk-KDA-vs-naive tolerance (tests/models/kimi_k3/test_kda.py).
RMSE_RATIO_TOL = 5e-3
# Autotune anchor (num_tokens, max_sequence_length, num_decode_sequences): the
# GLM-5.3-Flash capture shape. Heads, head dim and dtype come from the spec.
AUTOTUNE_ANCHOR = (2048, 2019, 29)
# Modules whose @autotune kernels the call reaches: the vendored KDA kernels and
# the FLA ops they import (chunk_delta_h, solve_tril, l2norm, cumsum).
_AUTOTUNE_MODULES = (_MODULE, "vllm.third_party.flash_linear_attention")
_PIN = AutotunePin(_AUTOTUNE_MODULES)
# Correctness witness bounds: the per-token oracle loop is O(T) launches.
_GUARD_MAX_DECODES = 8
_GUARD_MAX_LENGTH = 1024


@dataclass(frozen=True)
class KdaChunkPrefillShape:
    num_tokens: int
    max_sequence_length: int
    num_decode_sequences: int
    num_heads: int
    head_dim: int
    dtype: DType

    @property
    def num_prefill_tokens(self) -> int:
        return self.num_tokens - self.num_decode_sequences


@dataclass(frozen=True)
class _Operands:
    qkv: Any
    q: Any
    k: Any
    v: Any
    raw_g: Any
    beta: Any
    a_log: Any
    dt_bias: Any
    initial_state: Any
    cu_seqlens: Any
    boundaries: tuple[int, ...]


def validate_args(
    num_tokens: int,
    max_sequence_length: int,
    num_decode_sequences: int,
    num_heads: int,
    head_dim: int,
    dtype: DType | str,
) -> KdaChunkPrefillShape:
    shape = KdaChunkPrefillShape(
        num_tokens=_exact_int("num_tokens", num_tokens),
        max_sequence_length=_exact_int("max_sequence_length", max_sequence_length),
        num_decode_sequences=_exact_int("num_decode_sequences", num_decode_sequences),
        num_heads=_exact_int("num_heads", num_heads),
        head_dim=_exact_int("head_dim", head_dim),
        dtype=DType.from_value(dtype),
    )
    if shape.num_decode_sequences < 0:
        raise ValueError("num_decode_sequences must be >= 0")
    if shape.max_sequence_length < 1:
        raise ValueError("max_sequence_length must be >= 1")
    if shape.max_sequence_length == 1:
        # vLLM's GDN/KDA metadata splits with decode_threshold=1, so a
        # query-length-1 sequence is always a decode and never reaches this
        # prefill call. (At T=1 the q/k/v views are also already contiguous, so
        # the call would write through into the caller's qkv buffer.)
        raise ValueError(
            "max_sequence_length=1 is unreachable on the KDA prefill path: every "
            "query-length-1 sequence is a decode (decode_threshold=1); use "
            "num_decode_sequences, or kda_recurrent_decode for a pure-decode batch"
        )
    if shape.num_prefill_tokens < shape.max_sequence_length:
        # The prefill branch runs only when there is at least one prefill; a
        # pure-decode batch takes fused_recurrent_kda instead (another kind).
        raise ValueError(
            "num_tokens - num_decode_sequences must be >= max_sequence_length "
            "(at least one full-length prefill sequence)"
        )
    if shape.num_heads < 1:
        raise ValueError("num_heads must be >= 1")
    if shape.head_dim != _HEAD_DIM:
        raise ValueError(f"{_BACKEND} requires head_dim={_HEAD_DIM}")
    if shape.dtype is not DType.BF16:
        raise ValueError(f"{_BACKEND} requires dtype=bf16, got {shape.dtype}")
    return shape


def sequence_lengths(shape: KdaChunkPrefillShape) -> tuple[int, ...]:
    """Decodes first (one token each), then full-length prefills plus remainder."""
    full, remainder = divmod(shape.num_prefill_tokens, shape.max_sequence_length)
    lengths = (1,) * shape.num_decode_sequences + (shape.max_sequence_length,) * full
    return lengths + ((remainder,) if remainder else ())


def sequence_boundaries(shape: KdaChunkPrefillShape) -> tuple[int, ...]:
    return (0, *accumulate(sequence_lengths(shape)))


def num_chunks(shape: KdaChunkPrefillShape) -> int:
    """Chunk programs the chunk-parallel kernels launch (every sequence >= 1)."""
    return sum(-(-length // CHUNK_SIZE) for length in sequence_lengths(shape))


def guard_shape(shape: KdaChunkPrefillShape) -> KdaChunkPrefillShape:
    """A bounded witness that keeps decodes, a multi-chunk prefill and a tail."""
    length = min(shape.max_sequence_length, _GUARD_MAX_LENGTH)
    decodes = min(shape.num_decode_sequences, _GUARD_MAX_DECODES)
    prefill = min(shape.num_prefill_tokens, 2 * length - 1)
    guard = replace(
        shape,
        num_tokens=decodes + prefill,
        max_sequence_length=length,
        num_decode_sequences=decodes,
    )
    return shape if guard == shape else guard


def build_operands(torch: Any, shape: KdaChunkPrefillShape, *, device: Any) -> _Operands:
    tokens, heads, dim = shape.num_tokens, shape.num_heads, shape.head_dim
    lengths = sequence_lengths(shape)
    boundaries = sequence_boundaries(shape)
    generator = torch.Generator(device=device).manual_seed(42)

    def normal(*size: int, dtype: Any) -> Any:
        return torch.randn(*size, generator=generator, dtype=torch.float32, device=device).to(dtype)

    projection = heads * dim
    qkv = normal(tokens, 3 * projection, dtype=torch.bfloat16)
    q, k, v = (part.reshape(1, tokens, heads, dim) for part in qkv.split(projection, dim=-1))
    raw_g = normal(1, tokens, heads, dim, dtype=torch.bfloat16)
    beta = torch.sigmoid(normal(tokens, heads, dtype=torch.float32)).unsqueeze(0)
    # Mamba/Kimi-style A init: exp(A_log) uniform in [1, 16].
    a_log = (
        torch.empty(1, 1, heads, 1, dtype=torch.float32, device=device)
        .uniform_(1.0, 16.0, generator=generator)
        .log()
    )
    dt_bias = normal(projection, dtype=torch.float32) * 0.1
    initial_state = normal(len(lengths), heads, dim, dim, dtype=torch.float32)
    # gather_initial_states zero-fills rows without prior state (fresh prefills).
    initial_state[shape.num_decode_sequences :] = 0
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device=device)
    return _Operands(
        qkv=qkv,
        q=q,
        k=k,
        v=v,
        raw_g=raw_g,
        beta=beta,
        a_log=a_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        boundaries=boundaries,
    )


def invoke(callable_: Any, operands: _Operands) -> Any:
    """The exact keyword set kda.py passes on the prefill branch."""
    return callable_(
        q=operands.q,
        k=operands.k,
        v=operands.v,
        raw_g=operands.raw_g,
        beta=operands.beta,
        A_log=operands.a_log,
        g_bias=operands.dt_bias,
        initial_state=operands.initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=operands.cu_seqlens,
        safe_gate=True,
        lower_bound=LOWER_BOUND,
    )


def check_correctness(
    torch: Any, callable_: Any, operands: _Operands, shape: KdaChunkPrefillShape
) -> tuple[float, float]:
    """Compare one call against the naive oracle; return (o, state) RMSE ratios."""
    from profiling.runners.attention.kda_chunk_prefill_reference import (
        kda_chunk_prefill_reference,
        rmse_ratio,
    )

    snapshots = {
        name: getattr(operands, name).clone()
        for name in ("qkv", "raw_g", "beta", "a_log", "dt_bias", "initial_state", "cu_seqlens")
    }
    output, final_state = invoke(callable_, operands)
    torch.cuda.synchronize()
    tokens, heads, dim = shape.num_tokens, shape.num_heads, shape.head_dim
    sequences = len(operands.boundaries) - 1
    if tuple(output.shape) != (1, tokens, heads, dim) or output.dtype is not torch.bfloat16:
        raise AssertionError(f"unexpected output {tuple(output.shape)} {output.dtype}")
    if (
        tuple(final_state.shape) != (sequences, heads, dim, dim)
        or final_state.dtype is not torch.float32
    ):
        raise AssertionError(f"unexpected final state {tuple(final_state.shape)}")
    if not (torch.isfinite(output).all() and torch.isfinite(final_state).all()):
        raise AssertionError("non-finite KDA output or state")
    for name, snapshot in snapshots.items():
        if not torch.equal(getattr(operands, name), snapshot):
            raise AssertionError(f"chunk_kda_with_fused_gate mutated {name}")

    expected_o, expected_state = kda_chunk_prefill_reference(
        operands.q.squeeze(0),
        operands.k.squeeze(0),
        operands.v.squeeze(0),
        operands.raw_g.squeeze(0),
        operands.beta.squeeze(0),
        operands.a_log,
        operands.dt_bias,
        operands.initial_state,
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


def profile_kda_chunk_prefill_vllm_triton(
    num_tokens: int,
    max_sequence_length: int,
    num_decode_sequences: int,
    num_heads: int,
    head_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile one GLM-5.3-Flash KDA chunked-prefill call on a B200."""
    shape = validate_args(
        num_tokens, max_sequence_length, num_decode_sequences, num_heads, head_dim, dtype
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_BACKEND}") from exc
    try:
        require_exact_gpu(torch, backend=_BACKEND, required_gpu=_GPU)
        callable_ = load_required_callable(
            importlib.import_module,
            backend=_BACKEND,
            module_name=_MODULE,
            callable_name=_CALLABLE,
            environment_label="the vllm_fork_env (alignment vLLM fork)",
        )
        device = torch.device("cuda", torch.cuda.current_device())
        pin_autotune(torch, callable_, shape, device=device)
        guard = guard_shape(shape)
        check_correctness(torch, callable_, build_operands(torch, guard, device=device), guard)
        operands = build_operands(torch, shape, device=device)

        def kernel() -> Any:
            return invoke(callable_, operands)

        # Every call allocates fresh contiguous copies, intermediates, output
        # and final state; nothing it reads is written, so no reset is needed.
        time_ms = Timer.cupti(kernel, warmup=5, kernel_name=None)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc

    elapsed = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=semantic_flops(shape) / elapsed / 1e12 if elapsed > 0 else 0.0,
        memory_bandwidth_gbps=logical_bytes(shape) / elapsed / 1e9 if elapsed > 0 else 0.0,
        energy_j=float(energy_j),
    )


def anchor_shape(shape: KdaChunkPrefillShape) -> KdaChunkPrefillShape:
    tokens, length, decodes = AUTOTUNE_ANCHOR
    return replace(
        shape, num_tokens=tokens, max_sequence_length=length, num_decode_sequences=decodes
    )


def _autotune_state(shape: KdaChunkPrefillShape) -> tuple[int, int, str]:
    return (shape.num_heads, shape.head_dim, shape.dtype.value)


def pin_autotune(
    torch: Any, callable_: Any, shape: KdaChunkPrefillShape, *, device: Any
) -> str:
    """Tune once per worker and state key at the anchor, before any other call."""
    anchor = anchor_shape(shape)
    label = (
        f"T{anchor.num_tokens}/L{anchor.max_sequence_length}/"
        f"D{anchor.num_decode_sequences}/H{anchor.num_heads}"
    )

    def tune() -> None:
        invoke(callable_, build_operands(torch, anchor, device=device))
        torch.cuda.synchronize()

    return _PIN.ensure(_autotune_state(shape), label, tune)


def row_provenance(
    num_tokens: int,
    max_sequence_length: int,
    num_decode_sequences: int,
    num_heads: int,
    head_dim: int,
    dtype: DType | str,
) -> str | None:
    """The autotune note for a row this worker just measured (registry hook)."""
    del num_tokens, max_sequence_length, num_decode_sequences
    return _PIN.note((num_heads, head_dim, DType.from_value(dtype).value))


def semantic_flops(shape: KdaChunkPrefillShape) -> float:
    """Logical 2*MAC count per token and head, at chunk width C=64.

    The state terms are the K x V inter-chunk work: the w.h and k^T.v_new
    updates plus the q.h read-out, i.e. 3 K x V MACs per token. The
    intra-chunk terms grow with C: the two K.Kt/Q.Kt products (2*C*K) and
    the w/u recompute plus the Aqk.v output (2*C*V).
    """
    k = v = shape.head_dim
    per_token_head = 2.0 * (3 * k * v + CHUNK_SIZE * (2 * k + 2 * v))
    return float(shape.num_tokens * shape.num_heads) * per_token_head


def logical_bytes(shape: KdaChunkPrefillShape) -> float:
    """Minimum traffic: q/k/v/raw_g and beta in, o out, the fp32 state in and out."""
    elements = shape.num_tokens * shape.num_heads * shape.head_dim
    sequences = len(sequence_lengths(shape))
    state = sequences * shape.num_heads * shape.head_dim * shape.head_dim * 4
    return float(
        4 * elements * 2 + shape.num_tokens * shape.num_heads * 4 + elements * 2 + 2 * state
    )


__all__ = [
    "AUTOTUNE_ANCHOR",
    "CHUNK_SIZE",
    "KdaChunkPrefillShape",
    "build_operands",
    "check_correctness",
    "guard_shape",
    "invoke",
    "logical_bytes",
    "num_chunks",
    "anchor_shape",
    "pin_autotune",
    "profile_kda_chunk_prefill_vllm_triton",
    "row_provenance",
    "semantic_flops",
    "sequence_boundaries",
    "sequence_lengths",
    "validate_args",
]
