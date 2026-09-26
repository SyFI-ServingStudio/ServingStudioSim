"""vLLM-fork Triton runner for the GLM-5.3-Flash KDA recurrent decode.

This runner only allocates tensors, times one public call, and returns
metrics. DB writes, subprocess selection and registry routing live in L1b.

It times ``fused_recurrent_kda`` from
``vllm.models.glm5next.nvidia.ops.third_party.kda``. The operands are laid out
the way ``Glm5NextLinearAttention.forward`` / ``_forward`` hand them over on a
plain decode step:

- ``projected`` is the merged ``in_proj_qkvbfg_a`` output
  ``[B, 3*H*K + H + 2*K]`` (q | k | v | beta | f_a | g_a). ``causal_conv1d_update``
  overwrites its q|k|v slice in place (``out = x``), so q/k/v are row-strided
  ``[1,B,H,K]`` views of that buffer. The raw bf16 beta is the strided
  ``[1,B,H]`` column slice next to them. The callable's four ``.contiguous()``
  calls are therefore four real copies, the four ``elementwise_kernel``
  launches the capture shows before the recurrent kernel.
- ``g`` is the contiguous bf16 ``f_b_proj`` output ``[1,B,H,K]``. It holds raw
  logits, and the kernel computes the gate from them (``compute_gate=True``).
- ``A_log`` is ``[1,1,H,1]`` and ``dt_bias`` is ``[H*K]``, both fp32.
- ``initial_state`` is the whole fp32 ``[slots,H,V,K]`` pool, updated in place.
  Slot 0 is the NULL block. ``ssm_state_indices`` is int32 ``[B]``, a fixed
  permutation of slots ``1..B``. ``cu_seqlens`` is ``arange(B+1)`` int32.
- ``out`` is the contiguous ``[1,B,H,V]`` layer output buffer.
  ``sigmoid_beta=True``, ``lower_bound=-5.0``, ``use_qk_l2norm_in_kernel=True``.

Timing is the CUPTI sum over the five launches of one call
(``kernel_name=None``). Every call advances the states in place, as
production does. The delta rule with unit-norm k and beta in (0, 1) is
contractive, so repeated timing calls keep the state finite.

Autotune state. The recurrent kernel on this path is not ``@autotune``d in
the pinned fork, but the call lives in the same module as the autotuned chunk
kernels. To keep a future autotuned config from depending on grid order, the
registry row disables autotune persistence and the runner makes one anchor call
at batch 32 (the capture's decode batch) per worker and (num_heads, head_dim,
dtype) before any row, like the chunk-prefill runner. The note in
``backend_version`` records the anchor and the selected-config digest
(``configs=0`` while nothing on the path is autotuned).

FLOPs and bytes are semantic logical counts, not physical Triton work.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
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

_BACKEND = "kda_recurrent_decode:vllm_triton"
_MODULE = "vllm.models.glm5next.nvidia.ops.third_party.kda"
_CALLABLE = "fused_recurrent_kda"
_GPU = "NVIDIA B200"
_HEAD_DIM = 128
# GLM-5.3-Flash `linear_attn_config.gate_lower_bound`.
LOWER_BOUND = -5.0
# The fork's fused_recurrent_kda-vs-naive tolerances (RMSE ratio), from
# tests/models/kimi_k3/test_kda.py::test_kda_spec_decode_correctness.
OUTPUT_RMSE_RATIO_TOL = 1e-3
STATE_RMSE_RATIO_TOL = 3e-3
# Autotune anchor batch: the capture's decode batch. Heads, head dim and dtype
# come from the spec.
AUTOTUNE_ANCHOR_BATCH = 32
_PIN = AutotunePin((_MODULE, "vllm.third_party.flash_linear_attention"))


@dataclass(frozen=True)
class KdaRecurrentDecodeShape:
    batch_size: int
    num_heads: int
    head_dim: int
    dtype: DType

    @property
    def projection(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def projected_width(self) -> int:
        """Merged in_proj width: q, k, v, beta (H), f_a (K), g_a (K)."""
        return 3 * self.projection + self.num_heads + 2 * self.head_dim


@dataclass(frozen=True)
class _Operands:
    projected: Any
    q: Any
    k: Any
    v: Any
    g: Any
    beta: Any
    a_log: Any
    dt_bias: Any
    state_pool: Any
    state_indices: Any
    cu_seqlens: Any
    out: Any


def validate_args(
    batch_size: int,
    num_heads: int,
    head_dim: int,
    dtype: DType | str,
) -> KdaRecurrentDecodeShape:
    shape = KdaRecurrentDecodeShape(
        batch_size=_exact_int("batch_size", batch_size),
        num_heads=_exact_int("num_heads", num_heads),
        head_dim=_exact_int("head_dim", head_dim),
        dtype=DType.from_value(dtype),
    )
    if shape.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if shape.num_heads < 1:
        raise ValueError("num_heads must be >= 1")
    if shape.head_dim != _HEAD_DIM:
        raise ValueError(f"{_BACKEND} requires head_dim={_HEAD_DIM}")
    if shape.dtype is not DType.BF16:
        raise ValueError(f"{_BACKEND} requires dtype=bf16, got {shape.dtype}")
    return shape


def state_slots(batch_size: int) -> tuple[int, ...]:
    """Distinct non-NULL slots in a fixed scattered order (slot 0 is NULL)."""
    stride = 7 if batch_size % 7 else 1
    return tuple(1 + (i * stride) % batch_size for i in range(batch_size))


def build_operands(torch: Any, shape: KdaRecurrentDecodeShape, *, device: Any) -> _Operands:
    batch, heads, dim, projection = (
        shape.batch_size,
        shape.num_heads,
        shape.head_dim,
        shape.projection,
    )
    generator = torch.Generator(device=device).manual_seed(42)

    def normal(*size: int, dtype: Any) -> Any:
        return torch.randn(*size, generator=generator, dtype=torch.float32, device=device).to(dtype)

    projected = normal(batch, shape.projected_width, dtype=torch.bfloat16)
    qkv = projected[:, : 3 * projection]
    q, k, v = (part.reshape(1, batch, heads, dim) for part in qkv.split(projection, dim=-1))
    beta = projected[:, 3 * projection : 3 * projection + heads].unsqueeze(0)
    g = normal(batch, projection, dtype=torch.bfloat16).reshape(1, batch, heads, dim)
    # Mamba/Kimi-style A init: exp(A_log) uniform in [1, 16].
    a_log = (
        torch.empty(1, 1, heads, 1, dtype=torch.float32, device=device)
        .uniform_(1.0, 16.0, generator=generator)
        .log()
    )
    dt_bias = normal(projection, dtype=torch.float32) * 0.1
    state_pool = normal(batch + 1, heads, dim, dim, dtype=torch.float32)
    state_indices = torch.tensor(state_slots(batch), dtype=torch.int32, device=device)
    cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device=device)
    out = torch.empty(1, batch, heads, dim, dtype=torch.bfloat16, device=device)
    return _Operands(
        projected=projected,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        a_log=a_log,
        dt_bias=dt_bias,
        state_pool=state_pool,
        state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        out=out,
    )


def invoke(callable_: Any, operands: _Operands) -> Any:
    """The exact keyword set kda.py passes on the plain-decode branch."""
    return callable_(
        q=operands.q,
        k=operands.k,
        v=operands.v,
        g=operands.g,
        beta=operands.beta,
        initial_state=operands.state_pool,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=operands.cu_seqlens,
        ssm_state_indices=operands.state_indices,
        out=operands.out,
        sigmoid_beta=True,
        a_log=operands.a_log,
        g_bias=operands.dt_bias,
        compute_gate=True,
        lower_bound=LOWER_BOUND,
    )


def check_correctness(
    torch: Any, callable_: Any, operands: _Operands, shape: KdaRecurrentDecodeShape
) -> tuple[float, float]:
    """Compare one call against the naive oracle, restore the pool, and
    return the (o, state) RMSE ratios."""
    from profiling.runners.attention.kda_chunk_prefill_reference import rmse_ratio
    from profiling.runners.attention.kda_recurrent_decode_reference import (
        kda_recurrent_decode_reference,
    )

    pool_snapshot = operands.state_pool.clone()
    immutable = {
        name: getattr(operands, name).clone()
        for name in ("projected", "g", "a_log", "dt_bias", "state_indices", "cu_seqlens")
    }
    expected_o, expected_pool = kda_recurrent_decode_reference(
        operands.q.squeeze(0),
        operands.k.squeeze(0),
        operands.v.squeeze(0),
        operands.g.squeeze(0),
        operands.beta.squeeze(0),
        operands.a_log,
        operands.dt_bias,
        pool_snapshot,
        operands.state_indices,
        lower_bound=LOWER_BOUND,
    )
    try:
        output, final_state = invoke(callable_, operands)
        torch.cuda.synchronize()
        if output is not operands.out or final_state is not operands.state_pool:
            raise AssertionError("fused_recurrent_kda must write out/state in place")
        batch, heads, dim = shape.batch_size, shape.num_heads, shape.head_dim
        if tuple(output.shape) != (1, batch, heads, dim) or output.dtype is not torch.bfloat16:
            raise AssertionError(f"unexpected output {tuple(output.shape)} {output.dtype}")
        slots = operands.state_indices.long()
        actual_state = operands.state_pool[slots]
        if not (torch.isfinite(output).all() and torch.isfinite(actual_state).all()):
            raise AssertionError("non-finite KDA output or state")
        if not torch.equal(operands.state_pool[0], pool_snapshot[0]):
            raise AssertionError("NULL state slot 0 was written")
        for name, snapshot in immutable.items():
            if not torch.equal(getattr(operands, name), snapshot):
                raise AssertionError(f"fused_recurrent_kda mutated {name}")
        o_err = rmse_ratio(expected_o, output.squeeze(0))
        state_err = rmse_ratio(expected_pool[slots], actual_state)
        if not (o_err < OUTPUT_RMSE_RATIO_TOL and state_err < STATE_RMSE_RATIO_TOL):
            raise AssertionError(
                f"KDA decode mismatch vs naive oracle: o rmse ratio {o_err:.2e} "
                f"(tol {OUTPUT_RMSE_RATIO_TOL:.0e}), state rmse ratio {state_err:.2e} "
                f"(tol {STATE_RMSE_RATIO_TOL:.0e})"
            )
    finally:
        operands.state_pool.copy_(pool_snapshot)
    return o_err, state_err


def profile_kda_recurrent_decode_vllm_triton(
    batch_size: int,
    num_heads: int,
    head_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile one GLM-5.3-Flash KDA recurrent-decode call on a B200."""
    shape = validate_args(batch_size, num_heads, head_dim, dtype)
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
        operands = build_operands(torch, shape, device=device)
        check_correctness(torch, callable_, operands, shape)

        def kernel() -> Any:
            return invoke(callable_, operands)

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


def pin_autotune(
    torch: Any, callable_: Any, shape: KdaRecurrentDecodeShape, *, device: Any
) -> str:
    """Tune once per worker and state key at the anchor, before any other call."""
    anchor = replace(shape, batch_size=AUTOTUNE_ANCHOR_BATCH)

    def tune() -> None:
        invoke(callable_, build_operands(torch, anchor, device=device))
        torch.cuda.synchronize()

    state = (shape.num_heads, shape.head_dim, shape.dtype.value)
    return _PIN.ensure(state, f"B{anchor.batch_size}/H{anchor.num_heads}", tune)


def row_provenance(
    batch_size: int, num_heads: int, head_dim: int, dtype: DType | str
) -> str | None:
    """The autotune note for a row this worker just measured (registry hook)."""
    del batch_size
    return _PIN.note((num_heads, head_dim, DType.from_value(dtype).value))


def semantic_flops(shape: KdaRecurrentDecodeShape) -> float:
    """Logical FLOPs of one step per (sequence, head): the K x V decay, the
    k^T.S read, the rank-1 update and the q.S read-out (1 + 3 * 2 = 7 K*V)."""
    k = v = shape.head_dim
    return float(shape.batch_size * shape.num_heads * 7 * k * v)


def logical_bytes(shape: KdaRecurrentDecodeShape) -> float:
    """Minimum traffic: q/k/v/g and beta in, o out, the fp32 state in and out."""
    elements = shape.batch_size * shape.projection
    state = shape.batch_size * shape.num_heads * shape.head_dim * shape.head_dim * 4
    return float(
        4 * elements * 2 + shape.batch_size * shape.num_heads * 2 + elements * 2 + 2 * state
    )


__all__ = [
    "KdaRecurrentDecodeShape",
    "build_operands",
    "check_correctness",
    "invoke",
    "logical_bytes",
    "pin_autotune",
    "profile_kda_recurrent_decode_vllm_triton",
    "row_provenance",
    "semantic_flops",
    "state_slots",
    "validate_args",
]
