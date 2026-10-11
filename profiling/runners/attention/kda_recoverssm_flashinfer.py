"""FlashInfer runner for the KDA RecoverSSM verify kind (backend ``flashinfer_cute``).

This runner only allocates tensors, times one public call, and returns
metrics. DB writes, subprocess selection and registry routing live in L1b.

It times FlashInfer's drop-in for the vLLM fork's ``kda_recoverssm_verify``:
``flashinfer.kda_decode.recurrent_kda`` with ``disable_state_update=True`` and
the RecoverSSM ``correction_cache`` / ``kg_cache``. That routes to
``kda_decode_wy_output_only.kda_recoverssm_verify``, which evaluates each
request's window in the chunk-parallel WY form with tensor-core MMAs, reads the
committed state once, never writes it, and writes the per-token fp32
correction and raw key/gate records the RecoverSSM commit consumes.

The operands are the ``vllm_triton`` runner's (the same strided q/k/v views of
``spec_conv_out``, the row-strided ``raw_beta`` slice, the same scattered
slots and the same layer page), with the checkpoint in bf16, the only pool
dtype the kernel takes. The call is made as a vLLM layer would make it in
place of the fork op: packed ``cu_seqlens`` (``query_start_loc``), slot
indices, the raw gate with ``lower_bound`` applied in-kernel
(``use_gate_in_kernel``), the raw beta logit (``beta_is_logit``) and q/k L2
norms in-kernel.

Timing is the CUPTI sum over the call's launches (``kernel_name=None``).
Correctness is checked once against the same Torch oracle as ``vllm_triton``,
outside timing, on the first ``CHECK_REQUESTS`` requests. The WY form rounds
its intermediate operands to bf16 for the tensor-core MMAs, so the output and
correction tolerances are looser than the fork's register recurrence.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention._gdn_common import load_required_callable
from profiling.runners.attention.kda_recoverssm_vllm_triton import (
    LOWER_BOUND,
    KdaRecoverSSMShape,
    VerifyOperands,
    build_verify_operands,
    check_verify,
    validate_verify_args,
    verify_bytes,
    verify_flops,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_MODULE = "flashinfer.kda_decode"
_ENV = "flashinfer_kda_env"
_BACKEND = "kda_recoverssm_verify:flashinfer_cute"
PACKAGES = ("flashinfer-python", "nvidia-cutlass-dsl")
# RMSE ratios against the fp32 oracle; the WY MMAs take bf16 operands.
OUTPUT_RMSE_RATIO_TOL = 2e-2
CORRECTION_RMSE_RATIO_TOL = 2e-2


def validate_args(
    batch_size: int,
    num_heads: int,
    head_dim: int,
    spec_query_len: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> KdaRecoverSSMShape:
    shape = validate_verify_args(
        batch_size, num_heads, head_dim, spec_query_len, dtype, state_dtype
    )
    if shape.state_dtype is not DType.BF16:
        raise ValueError(f"{_BACKEND} needs a bf16 state pool, got {shape.state_dtype}")
    if shape.head_dim != 128:
        raise ValueError(f"{_BACKEND} needs head_dim 128, got {shape.head_dim}")
    # The WY kernel's window is at most 16 tokens.
    if shape.spec_query_len > 16:
        raise ValueError(f"{_BACKEND} verifies at most 16 tokens, got {shape.spec_query_len}")
    return shape


def invoke(callable_: Any, operands: VerifyOperands, spec_query_len: int) -> Any:
    page = operands.page
    output, _ = callable_(
        q=operands.q,
        k=operands.k,
        v=operands.v,
        g=operands.raw_g,
        beta=operands.raw_beta,
        A_log=page.a_log,
        dt_bias=page.dt_bias,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=LOWER_BOUND,
        cu_seqlens=operands.query_start_loc,
        num_spec_tokens=spec_query_len - 1,
        output=operands.out,
        initial_state_source=page.checkpoint,
        initial_state_indices=operands.state_indices,
        beta_is_logit=True,
        disable_state_update=True,
        correction_cache=page.correction,
        kg_cache=page.kg,
    )
    return output


def profile_kda_recoverssm_verify_flashinfer_cute(
    batch_size: int,
    num_heads: int,
    head_dim: int,
    spec_query_len: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> ComputeMetrics:
    """Profile one FlashInfer frozen KDA verify call with RecoverSSM records."""
    shape = validate_args(batch_size, num_heads, head_dim, spec_query_len, dtype, state_dtype)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_BACKEND}") from exc
    try:
        callable_ = load_required_callable(
            importlib.import_module,
            backend=_BACKEND,
            module_name=_MODULE,
            callable_name="recurrent_kda",
            environment_label=_ENV,
        )
        device = torch.device("cuda", torch.cuda.current_device())
        operands = build_verify_operands(torch, shape, device=device)
        check_verify(
            torch,
            callable_,
            operands,
            shape,
            invoke=invoke,
            output_tol=OUTPUT_RMSE_RATIO_TOL,
            correction_tol=CORRECTION_RMSE_RATIO_TOL,
        )

        def kernel() -> Any:
            return invoke(callable_, operands, shape.spec_query_len)

        time_ms = Timer.cupti(kernel, warmup=5, kernel_name=None)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc
    elapsed = time_ms / 1000.0
    flops, nbytes = verify_flops(shape), verify_bytes(shape)
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=flops / elapsed / 1e12 if elapsed > 0 else 0.0,
        memory_bandwidth_gbps=nbytes / elapsed / 1e9 if elapsed > 0 else 0.0,
        energy_j=float(energy_j),
    )


def row_provenance_flashinfer_cute(**_spec: Any) -> str:
    """The FlashInfer and CuTe DSL stack that measured a ``flashinfer_cute`` row."""
    parts = []
    for package in PACKAGES:
        try:
            parts.append(f"{package} {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            parts.append(f"{package} missing")
    try:
        commit = importlib.import_module("flashinfer._build_meta").__git_commit__
    except (ImportError, AttributeError):
        commit = "unknown"
    return f"{', '.join(parts)}; flashinfer commit {commit}"


__all__ = [
    "invoke",
    "profile_kda_recoverssm_verify_flashinfer_cute",
    "row_provenance_flashinfer_cute",
    "validate_args",
]
