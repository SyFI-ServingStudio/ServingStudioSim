"""FlashKDA runner for the GLM-5.3-Flash KDA chunked prefill.

This runner only allocates tensors, times one public call, and returns
metrics. DB writes, subprocess selection and registry routing live in L1b.

Upstream vLLM runs this prefill with FlashKDA by default on SM90/SM10x/SM12x
(``vllm/models/glm5next/common/kda.py``: ``_resolve_kda_prefill_backend`` and
``Glm5NextLinearAttention._flashkda_prefill``). vLLM vendors the kernel from
vllm-project/FlashKDA at ``17a037d98da546deb4591e967cf961a43c034d8b``
(``cmake/external_projects/flashkda.cmake``) and registers it as
``torch.ops._flashkda_C.fwd``. ``flashkda_env`` builds the same commit with its
own ``setup.py``, which registers the same C++ ``fwd`` with the same schema as
``torch.ops.flash_kda.fwd``. The runner calls that op positionally, exactly as
``_flashkda_prefill`` does:

- q/k/v are row-strided ``[1,T,H,K]`` views into one ``[T, 3*H*K]`` short-conv
  output, as in the vllm_triton runner, so their ``.contiguous()`` calls are
  three real copies. ``g`` is the contiguous bf16 ``f_b_proj`` output, so its
  ``.contiguous()`` is a no-op, as in production.
- ``beta`` is the raw bf16 logit column of the merged ``in_proj_qkvbfg_a``
  output, ``[1,T,H]`` with the merged row stride. FlashKDA transposes it to
  ``[H,T]`` (one copy launch inside ``fwd``) and applies the sigmoid in-kernel.
  The vllm_triton path instead takes an fp32 sigmoid that runs outside its call.
- ``A_log`` is ``[H]`` and ``dt_bias`` is ``[H,K]``, both fp32 views.
- ``initial_state`` is fp32 ``[N,H,V,K]`` as ``gather_initial_states`` returns
  it (random rows for decodes, zero rows for fresh prefills); ``final_state``,
  ``out`` and the uint8 workspace are preallocated, as vLLM's workspace
  manager hands them over. No checkpoint state is passed.
- ``cu_seqlens`` is int32, scale is ``head_dim**-0.5``, lower bound ``-5.0``.

Timing is the CUPTI sum over every launch of one call (``kernel_name=None``):
the three q/k/v copies, the beta transpose, FlashKDA's prepare kernel (K1) and
its recurrence kernel (K2). FlashKDA has no autotuning; K2's V-split choice is
a fixed function of ``H``, ``N`` and the SM count.

FLOPs and bytes are the kind's semantic logical counts, shared with the
vllm_triton row so the two backends' TFLOPS compare directly.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention._gdn_common import exact_int as _exact_int
from profiling.runners.attention.kda_chunk_prefill_vllm_triton import (
    LOWER_BOUND,
    KdaChunkPrefillShape,
    guard_shape,
    logical_bytes,
    semantic_flops,
    sequence_boundaries,
    sequence_lengths,
)
from profiling.runners.attention.kda_chunk_prefill_vllm_triton import (
    validate_args as _validate_topology,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "kda_chunk_prefill:flashkda"
# Standalone build of the kernel vLLM vendors; importing it registers
# torch.ops.flash_kda (vLLM's build registers the same op as torch.ops._flashkda_C).
_MODULE = "flash_kda"
_PACKAGE = "flash_kda"
# flash_kda.cpp: STD_TORCH_CHECK(D == 128), and K = V.
HEAD_DIM = 128
# FlashKDA's own end-to-end tolerance is RMSE-ratio based too. Its recurrent
# state is fp32 between tiles but narrowed to bf16 at the chunk-boundary load
# and store (FlashKDA 17a037d commit message: a ~1.6e-3 floor; 3.1e-3 final
# state and 4.7e-3 output on real GLM-5.3-Flash activations), so the fork's
# 5e-3 Triton bound is too tight. Measured on B200 against the per-token oracle
# with these random inputs, H 8-64, up to 32832 tokens: output 4.1e-3 to
# 5.7e-3, state 2.4e-3 to 4.7e-3.
RMSE_RATIO_TOL = 1e-2


@dataclass(frozen=True)
class _Operands:
    qkv: Any
    projected: Any
    q: Any
    k: Any
    v: Any
    g: Any
    beta: Any
    a_log: Any
    dt_bias: Any
    initial_state: Any
    final_state: Any
    out: Any
    workspace: Any
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
    """FlashKDA's own limits first, then the kind's shared topology rules."""
    if DType.from_value(dtype) is not DType.BF16:
        raise ValueError(f"{_BACKEND} requires dtype=bf16, got {dtype}")
    if _exact_int("head_dim", head_dim) != HEAD_DIM:
        raise ValueError(f"{_BACKEND} requires head_dim={HEAD_DIM} (K = V = 128), got {head_dim}")
    return _validate_topology(
        num_tokens, max_sequence_length, num_decode_sequences, num_heads, head_dim, dtype
    )


def load_fwd(torch: Any) -> Any:
    try:
        importlib.import_module(_MODULE)
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{_BACKEND} requires the {_MODULE!r} package (run under flashkda_env)"
        ) from exc
    return torch.ops.flash_kda.fwd, torch.ops.flash_kda.get_workspace_size


def build_operands(
    torch: Any, shape: KdaChunkPrefillShape, get_workspace_size: Any, *, device: Any
) -> _Operands:
    tokens, heads, dim = shape.num_tokens, shape.num_heads, shape.head_dim
    lengths = sequence_lengths(shape)
    boundaries = sequence_boundaries(shape)
    generator = torch.Generator(device=device).manual_seed(42)

    def normal(*size: int, dtype: Any) -> Any:
        return torch.randn(*size, generator=generator, dtype=torch.float32, device=device).to(dtype)

    projection = heads * dim
    qkv = normal(tokens, 3 * projection, dtype=torch.bfloat16)
    q, k, v = (part.reshape(1, tokens, heads, dim) for part in qkv.split(projection, dim=-1))
    g = normal(1, tokens, heads, dim, dtype=torch.bfloat16)
    # in_proj_qkvbfg_a output, split [3*H*K, H, K, K]: beta is the H-wide
    # column block, a row-strided view.
    projected = normal(tokens, 3 * projection + heads + 2 * dim, dtype=torch.bfloat16)
    beta = projected[:, 3 * projection : 3 * projection + heads].unsqueeze(0)
    # Mamba/Kimi-style A init: exp(A_log) uniform in [1, 16].
    a_log = (
        torch.empty(heads, dtype=torch.float32, device=device)
        .uniform_(1.0, 16.0, generator=generator)
        .log()
    )
    dt_bias = (normal(projection, dtype=torch.float32) * 0.1).view(heads, dim)
    sequences = len(lengths)
    initial_state = normal(sequences, heads, dim, dim, dtype=torch.float32)
    # gather_initial_states zero-fills rows without prior state (fresh prefills).
    initial_state[shape.num_decode_sequences :] = 0
    workspace = torch.empty(
        int(get_workspace_size(tokens, heads, sequences)), dtype=torch.uint8, device=device
    )
    return _Operands(
        qkv=qkv,
        projected=projected,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        a_log=a_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        final_state=torch.empty_like(initial_state),
        out=torch.empty(1, tokens, heads, dim, dtype=torch.bfloat16, device=device),
        workspace=workspace,
        cu_seqlens=torch.tensor(boundaries, dtype=torch.int32, device=device),
        boundaries=boundaries,
    )


def invoke(fwd: Any, operands: _Operands, head_dim: int) -> None:
    """The positional call ``_flashkda_prefill`` makes, checkpointing off."""
    fwd(
        operands.q.contiguous(),
        operands.k.contiguous(),
        operands.v.contiguous(),
        operands.g.contiguous(),
        operands.beta,
        head_dim**-0.5,
        operands.out,
        operands.workspace,
        operands.a_log,
        operands.dt_bias,
        LOWER_BOUND,
        operands.initial_state.contiguous(),
        operands.final_state,
        operands.cu_seqlens.contiguous(),
        None,
        None,
    )


def check_correctness(
    torch: Any, fwd: Any, operands: _Operands, shape: KdaChunkPrefillShape
) -> tuple[float, float]:
    """Compare one call against the naive oracle; return (o, state) RMSE ratios."""
    from profiling.runners.attention.kda_chunk_prefill_reference import (
        kda_chunk_prefill_reference,
        rmse_ratio,
    )

    snapshots = {
        name: getattr(operands, name).clone()
        for name in ("qkv", "projected", "g", "a_log", "dt_bias", "initial_state", "cu_seqlens")
    }
    operands.out.fill_(float("nan"))
    operands.final_state.fill_(float("nan"))
    invoke(fwd, operands, shape.head_dim)
    torch.cuda.synchronize()
    output, final_state = operands.out, operands.final_state
    if not (torch.isfinite(output).all() and torch.isfinite(final_state).all()):
        raise AssertionError("non-finite or unwritten FlashKDA output or state")
    for name, snapshot in snapshots.items():
        if not torch.equal(getattr(operands, name), snapshot):
            raise AssertionError(f"flash_kda.fwd mutated {name}")

    expected_o, expected_state = kda_chunk_prefill_reference(
        operands.q.squeeze(0),
        operands.k.squeeze(0),
        operands.v.squeeze(0),
        operands.g.squeeze(0),
        # The oracle takes post-sigmoid beta; FlashKDA applies it in-kernel.
        torch.sigmoid(operands.beta.squeeze(0).float()),
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
            f"FlashKDA mismatch vs naive oracle: o rmse ratio {o_err:.2e}, "
            f"state rmse ratio {state_err:.2e} (tol {RMSE_RATIO_TOL:.0e})"
        )
    return o_err, state_err


def profile_kda_chunk_prefill_flashkda(
    num_tokens: int,
    max_sequence_length: int,
    num_decode_sequences: int,
    num_heads: int,
    head_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile one GLM-5.3-Flash KDA chunked-prefill call through FlashKDA."""
    shape = validate_args(
        num_tokens, max_sequence_length, num_decode_sequences, num_heads, head_dim, dtype
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_BACKEND}") from exc
    fwd, get_workspace_size = load_fwd(torch)
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        guard = guard_shape(shape)
        check_correctness(
            torch, fwd, build_operands(torch, guard, get_workspace_size, device=device), guard
        )
        operands = build_operands(torch, shape, get_workspace_size, device=device)

        def kernel() -> None:
            invoke(fwd, operands, shape.head_dim)

        # Each call makes fresh q/k/v copies and overwrites out, final_state and
        # the workspace; nothing it reads is written, so no reset is needed.
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


def row_provenance(
    num_tokens: int,
    max_sequence_length: int,
    num_decode_sequences: int,
    num_heads: int,
    head_dim: int,
    dtype: DType | str,
) -> str | None:
    """The FlashKDA build that measured the row (registry hook)."""
    del num_tokens, max_sequence_length, num_decode_sequences, num_heads, head_dim, dtype
    try:
        return f"flash_kda {importlib.metadata.version(_PACKAGE)}"
    except importlib.metadata.PackageNotFoundError:
        return None


__all__ = [
    "HEAD_DIM",
    "RMSE_RATIO_TOL",
    "build_operands",
    "check_correctness",
    "invoke",
    "load_fwd",
    "profile_kda_chunk_prefill_flashkda",
    "row_provenance",
    "validate_args",
]
