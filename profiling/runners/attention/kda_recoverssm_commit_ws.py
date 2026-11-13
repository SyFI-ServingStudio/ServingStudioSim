"""Persistent warp-specialized CUDA runners for ``kda_recoverssm_commit``
(backends ``cuda_persistent_ws`` and ``cuda_persistent_ws_tf32_mma``).

The kernels are the MoE session's commits "v4" and "v7", vendored read-only in
``kda_commit_ws_vendor/kcommit4.cu`` / ``kcommit7.cu`` (see their headers).
``cuda_persistent_ws`` (v4) folds on packed FP32 FMAs, one per accepted row,
and is byte-identical to the fork's cute commit. ``cuda_persistent_ws_tf32_mma``
(v7) keeps the structure, decodes each item once in the producer, computes the
gate chain for the whole window without branches and folds as one 3xTF32
tensor-core MMA over the window, so its time does not depend on commit_len;
its state differs from cute by rounding only (the Torch oracle tolerance
gates it, as it gates cute), its conv window is byte-identical. One call is what the
fork's ``KDARecoverSSMCommitContext.commit`` does without align metadata, with
the conv compaction fused into the fold:

- the fork's ``_prepare_commit_plan_kernel`` (Triton, one program per request)
  clamps each accepted count and plans the destination slots, as ``commit``
  launches it;
- ``commit_fold`` (one persistent launch over every layer of the group): a TMA
  producer warp streams each (layer, request, head) item's records, 32 KB state
  page and conv history into a 2- to 4-stage shared-memory ring, coefficient
  warps build the decayed keys, fold warps add the accepted rows (packed FP32
  FMAs or 3xTF32 MMA), store the state in place and write the compacted conv
  window.

The operands are the CuTe runner's (``build_cute_commit_operands``): the
Triton runner's pages, bf16 checkpoints, records by request row, and the
fork's default SD conv layout, which the kernel requires (conv channel
stride 1: both copy each conv history row as one contiguous per-head channel
slice; neither reads the dim stride). Before timing, one call is checked
against the fork's cute commit on identical pages: v4 must leave every page
byte-identical; v7 must leave the conv windows and every untouched slot
byte-identical, and its state's error against cute is logged. Then the Torch
oracle check of the other backends runs on the same pages. Both ring depths
are checked and timed; the faster one is reported. Timing is the CUPTI sum of
the plan and fold launches.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention._gdn_common import load_required_callable
from profiling.runners.attention.kda_recoverssm_vllm_cute import build_cute_commit_operands
from profiling.runners.attention.kda_recoverssm_vllm_triton import (
    _import_torch,
    _metrics,
    check_commit,
    commit_bytes,
    commit_flops,
    invoke_commit,
    validate_commit_args,
)
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented

_ENV = "vllm_env"
_OPS = "vllm.models.kimi_k3.nvidia.ops.recoverssm"
_KERNELS = "vllm.models.kimi_k3.nvidia.recoverssm.kernels"
_VENDOR = Path(__file__).resolve().parent / "kda_commit_ws_vendor"
_HEAD_DIM = 128  # KD in the kernel
_MAX_WINDOW = 8  # QMAX
_FOLD_RUNG = 3  # the full call: records, fold and the fused conv compaction


@dataclass(frozen=True)
class Variant:
    backend: str
    source: Path
    module: str
    ring_stages: tuple[int, ...]
    byte_exact: bool  # state byte-identical to the fork's cute commit


FFMA = Variant("cuda_persistent_ws", _VENDOR / "kcommit4.cu", "kda_recoverssm_commit_ws", (3, 4), True)
TF32_MMA = Variant(
    "cuda_persistent_ws_tf32_mma", _VENDOR / "kcommit7.cu", "kda_recoverssm_commit_ws_tf32_mma", (2, 3), False
)
_SOURCE = FFMA.source
RING_STAGES = FFMA.ring_stages
_exts: dict[str, Any] = {}


def _extension(variant: Variant = FFMA) -> Any:
    """Build a vendored kernel once per process (cached under HOME)."""
    if variant.module not in _exts:
        from torch.utils.cpp_extension import load

        _exts[variant.module] = load(
            name=variant.module,
            sources=[str(variant.source)],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "-std=c++20"],
            verbose=False,
        )
    return _exts[variant.module]


def _plan(ops_module: Any, operands: Any, spec_query_len: int) -> None:
    """The plan launch of ``KDARecoverSSMCommitContext.commit`` (non-align)."""
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    c, na = operands.context, operands.num_accepted_tokens
    si, q = operands.state_indices, operands.query_start_loc
    ops_module._prepare_commit_plan_kernel[(si.shape[0],)](
        na, None, si, q, None, None,
        c.commit_lens, c.final_state_indices, c.boundary_state_indices, c.boundary_recovery_lens,
        NULL_BLOCK_ID, 1, 1,
        na.stride(0), 0, si.stride(0), q.stride(0), 0, 0, 0,
        SPEC_QUERY_LEN=spec_query_len,
        num_warps=1,
    )


def _fold(ext: Any, operands: Any, shape: Any, stages: int) -> Any:
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    c = operands.context
    args = (
        c.state_base_addrs, c.state_block_strides, c.correction_cache_base_addrs,
        c.correction_cache_block_strides, c.kg_cache_base_addrs, c.kg_cache_block_strides,
        c.A_log, c.dt_bias, operands.state_indices, c.commit_lens, c.final_state_indices,
        c.conv_state_base_addrs, c.conv_state_block_strides, c.conv_state_dim_strides,
        c.conv_state_token_strides, shape.num_heads, shape.spec_query_len, c.conv_states[0].shape[1],
        c.conv_history_len, float(c.lower_bound), NULL_BLOCK_ID, shape.batch_size,
    )
    return lambda: ext.commit_fold(_FOLD_RUNG, stages, *args)


def profile_kda_recoverssm_commit_cuda_persistent_ws(
    batch_size: int,
    commit_len: int,
    num_heads: int,
    head_dim: int,
    spec_query_len: int,
    num_layers: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> Any:
    """Profile one plan + persistent FFMA fold commit over ``num_layers`` layers."""
    return _profile(FFMA, batch_size, commit_len, num_heads, head_dim, spec_query_len, num_layers, dtype, state_dtype)


def profile_kda_recoverssm_commit_cuda_persistent_ws_tf32_mma(
    batch_size: int,
    commit_len: int,
    num_heads: int,
    head_dim: int,
    spec_query_len: int,
    num_layers: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> Any:
    """Profile one plan + persistent 3xTF32-MMA fold commit over ``num_layers`` layers."""
    return _profile(TF32_MMA, batch_size, commit_len, num_heads, head_dim, spec_query_len, num_layers, dtype, state_dtype)


def _profile(
    variant: Variant,
    batch_size: int,
    commit_len: int,
    num_heads: int,
    head_dim: int,
    spec_query_len: int,
    num_layers: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> Any:
    backend = f"kda_recoverssm_commit:{variant.backend}"
    shape = validate_commit_args(
        batch_size, commit_len, num_heads, head_dim, spec_query_len, num_layers, dtype, state_dtype
    )
    if shape.state_dtype is not DType.BF16 or shape.head_dim != _HEAD_DIM:
        raise ValueError("the persistent commit folds bf16 checkpoints with head_dim 128 only")
    if shape.spec_query_len > _MAX_WINDOW:
        raise ValueError(f"the persistent commit holds at most {_MAX_WINDOW} records per request")
    torch = _import_torch(backend)
    try:
        ops_module = importlib.import_module(_OPS)
        compile_state_commit = load_required_callable(
            importlib.import_module,
            backend=backend,
            module_name=_KERNELS,
            callable_name="compile_state_commit",
            environment_label=_ENV,
        )
        ext = _extension(variant)
        device = torch.device("cuda", torch.cuda.current_device())
        compile_state_commit(shape.num_heads, shape.spec_query_len, device, align=False)
        operands = build_cute_commit_operands(
            torch, ops_module.KDARecoverSSMCommitContext, shape, device=device
        )
        if not bool((operands.context.conv_state_dim_strides == 1).all()):
            raise ValueError("the persistent commit needs channel-contiguous (SD) conv pages")
        slots = operands.state_indices.long()
        base = [page.raw.clone() for page in operands.pages]

        def restore() -> None:
            for page, saved in zip(operands.pages, base):
                page.raw.copy_(saved)

        # Reference: the fork's cute commit on the same pages.
        invoke_commit(operands)
        torch.cuda.synchronize()
        want = [page.raw.clone() for page in operands.pages]
        want_state = [page.checkpoint[slots].float() for page in operands.pages]
        calls = {}
        rel_err = 0.0
        for stages in variant.ring_stages:
            fold = _fold(ext, operands, shape, stages)

            def call(fold: Any = fold) -> None:
                _plan(ops_module, operands, shape.spec_query_len)
                fold()

            restore()
            call()
            torch.cuda.synchronize()
            if variant.byte_exact:
                if not all(torch.equal(page.raw, w) for page, w in zip(operands.pages, want)):
                    raise AssertionError(
                        f"{backend} (ring {stages}) is not byte-identical to the fork's cute commit"
                    )
            else:
                for page, w, ref in zip(operands.pages, want, want_state):
                    blocks = page.checkpoint.shape[0]
                    w = w.view(page.raw.shape)
                    untouched = torch.ones(blocks, dtype=torch.bool, device=device)
                    untouched[slots] = False
                    if not torch.equal(page.raw.view(blocks, -1)[untouched], w.view(blocks, -1)[untouched]):
                        raise AssertionError(f"{backend} (ring {stages}) wrote outside the committed slots")
                    got = page.checkpoint[slots].float()
                    rel_err = max(rel_err, ((got - ref).norm() / ref.norm().clamp_min(1e-30)).item())
                # Conv windows: byte-identical to cute's.
                restore()
                invoke_commit(operands)
                cute_conv = [page.conv[slots].clone() for page in operands.pages]
                restore()
                call()
                torch.cuda.synchronize()
                ok_conv = all(torch.equal(page.conv[slots], c) for page, c in zip(operands.pages, cute_conv))
                del cute_conv
                if not ok_conv:
                    raise AssertionError(f"{backend} (ring {stages}) conv windows differ from the fork's cute commit")
            calls[stages] = call
        del want, want_state
        restore()
        del base
        check_commit(torch, operands, shape, invoke=calls[variant.ring_stages[-1]])
        if not variant.byte_exact:
            print(
                f"[{backend}] state rel error vs cute (worst layer, ||d||/||cute||): {rel_err:.3e} "
                f"B{shape.batch_size} cl{shape.commit_len} L{shape.num_layers} H{shape.num_heads} q{shape.spec_query_len}",
                file=sys.stderr,
                flush=True,
            )
        times = {s: Timer.cupti(fn, warmup=5, kernel_name=None) for s, fn in calls.items()}
        best = min(times, key=times.__getitem__)
        time_ms = times[best]
        energy_j = Energy.perf(calls[best], warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{backend} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{backend} failed: {exc}") from exc
    return _metrics(time_ms, energy_j, commit_flops(shape), commit_bytes(shape))


__all__ = [
    "FFMA",
    "RING_STAGES",
    "TF32_MMA",
    "profile_kda_recoverssm_commit_cuda_persistent_ws",
    "profile_kda_recoverssm_commit_cuda_persistent_ws_tf32_mma",
]
