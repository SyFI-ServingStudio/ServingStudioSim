"""Single-launch vLLM Triton ``fused_moe_kernel`` backend.

The timed callable is exactly one ``invoke_fused_moe_triton_kernel`` call, so
CUPTI needs no name filter -- nothing else runs inside it. Everything vLLM does
*around* that launch (routing, alignment, quantization, the finalize reduce) has
its own kernel kind and must not be folded in here.

Two production behaviors are reproduced rather than reimplemented, because both
change the launch shape and both are easy to get subtly wrong:

* ``try_get_optimal_moe_config`` picks ``BLOCK_SIZE_M/N/K`` from vLLM's tuned
  config tables. Hard-coding a block size would measure a kernel production
  never launches.
* ``_prepare_expert_assignment`` decides between ``moe_align_block_size`` and the
  naive no-sort path (taken when ``num_tokens * top_k * 4 <= num_experts``). The
  two produce different grids for the same shape.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.kernels.vllm_fused_moe import LAUNCH_ROLE_DOWN, LAUNCH_ROLE_GATE_UP
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "vllm_fused_moe:vllm_triton"
_FUSED_MOE_MODULE = "vllm.model_executor.layers.fused_moe.fused_moe"
_GPU = "NVIDIA H200"


@dataclass(frozen=True)
class _ValidatedArgs:
    n: int
    k: int
    dtype: DType
    num_local_experts: int
    num_tokens: int
    experts_per_token: int
    launch_role: str
    block_size: int
    per_group_batches: tuple[int, ...]

    @property
    def assignments(self) -> int:
        """Router assignments this launch's alignment is built from."""
        return self.num_tokens * self.experts_per_token

    @property
    def kernel_top_k(self) -> int:
        """The ``top_k`` value vLLM passes for this launch role."""
        return self.experts_per_token if self.launch_role == LAUNCH_ROLE_GATE_UP else 1

    @property
    def mul_routed_weight(self) -> bool:
        return self.launch_role == LAUNCH_ROLE_DOWN

    @property
    def a_rows(self) -> int:
        """Rows of the activation matrix this launch reads.

        w13 reads the hidden states once per token; w2 reads the intermediate
        activation, which physically exists once per (token, selected expert).
        """
        return self.num_tokens if self.launch_role == LAUNCH_ROLE_GATE_UP else self.assignments


def _validate_args(
    n: int,
    k: int,
    dtype: DType | str,
    num_local_experts: int,
    num_tokens: int,
    experts_per_token: int,
    launch_role: str,
    block_size: int,
    per_group_batches: tuple[int, ...] | list[int],
) -> _ValidatedArgs:
    resolved_dtype = DType.from_value(dtype)
    if resolved_dtype is not DType.FP8_E4M3:
        raise ProfilerNotImplemented(f"{_BACKEND} implements only the FP8 E4M3 w8a8 recipe")
    if launch_role not in (LAUNCH_ROLE_GATE_UP, LAUNCH_ROLE_DOWN):
        raise ValueError(
            f"{_BACKEND} launch_role must be {LAUNCH_ROLE_GATE_UP!r} or {LAUNCH_ROLE_DOWN!r}"
        )
    for name, value in (
        ("n", n),
        ("k", k),
        ("num_local_experts", num_local_experts),
        ("num_tokens", num_tokens),
        ("experts_per_token", experts_per_token),
        ("block_size", block_size),
    ):
        if int(value) <= 0:
            raise ValueError(f"{_BACKEND} requires positive {name}, got {value}")
    batches = tuple(int(value) for value in per_group_batches)
    if len(batches) != int(num_local_experts):
        raise ValueError(
            f"{_BACKEND} per_group_batches must hold one count per local expert "
            f"({len(batches)} != {num_local_experts})"
        )
    if any(value < 0 for value in batches):
        raise ValueError(f"{_BACKEND} per_group_batches must be non-negative")
    if sum(batches) != int(num_tokens) * int(experts_per_token):
        raise ValueError(
            f"{_BACKEND} per_group_batches must sum to num_tokens*experts_per_token "
            f"({sum(batches)} != {int(num_tokens) * int(experts_per_token)})"
        )
    if int(experts_per_token) > int(num_local_experts):
        raise ValueError(f"{_BACKEND} requires experts_per_token <= num_local_experts")
    for name, value in (("n", n), ("k", k)):
        if int(value) % int(block_size):
            raise ValueError(
                f"{_BACKEND} requires {name} divisible by the FP8 block size {block_size}"
            )
    return _ValidatedArgs(
        n=int(n),
        k=int(k),
        dtype=resolved_dtype,
        num_local_experts=int(num_local_experts),
        num_tokens=int(num_tokens),
        experts_per_token=int(experts_per_token),
        launch_role=str(launch_role),
        block_size=int(block_size),
        per_group_batches=batches,
    )


def _require_h200(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"CUDA is required for {_BACKEND}")
    name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if name != _GPU:
        raise ProfilerNotImplemented(f"{_BACKEND} is verified only on {_GPU}, got {name}")


def _load_vllm() -> tuple[Any, Any, Any, Any]:
    try:
        fused_moe = importlib.import_module(_FUSED_MOE_MODULE)
        triton_language = importlib.import_module("triton.language")
    except Exception as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the repository vllm_env") from exc
    for name in (
        "invoke_fused_moe_triton_kernel",
        "try_get_optimal_moe_config",
        "_prepare_expert_assignment",
    ):
        if not hasattr(fused_moe, name):
            raise ProfilerNotImplemented(f"{_FUSED_MOE_MODULE} has no {name}; vLLM API drifted")
    return (
        fused_moe.invoke_fused_moe_triton_kernel,
        fused_moe.try_get_optimal_moe_config,
        fused_moe._prepare_expert_assignment,
        triton_language,
    )


def _topk_ids(torch: Any, args: _ValidatedArgs, *, device: Any) -> Any:
    """Materialize a ``topk_ids`` matrix realizing ``per_group_batches`` exactly.

    Only the per-expert multiset matters for cost -- it is what alignment pads
    into blocks -- so the assignment order is free. Dealing the expanded expert
    list column-major keeps each token row's experts distinct whenever the
    distribution allows it, which is what a real router produces; a distribution
    concentrated enough to force a repeat is accepted rather than rejected,
    because the block padding is identical either way.
    """
    expanded: list[int] = []
    for expert, count in enumerate(args.per_group_batches):
        expanded.extend([expert] * count)
    ids = torch.tensor(expanded, dtype=torch.int32, device=device)
    # Column-major fill: consecutive entries land in different token rows.
    return ids.reshape(args.experts_per_token, args.num_tokens).t().contiguous()


@dataclass(frozen=True)
class _Launch:
    invoke: Any
    kwargs: dict[str, Any]
    positional: tuple[Any, ...]

    def run_once(self) -> None:
        self.invoke(*self.positional, **self.kwargs)


def _build_launch(torch: Any, args: _ValidatedArgs, *, device: Any) -> _Launch:
    invoke, optimal_config, prepare_assignment, triton_language = _load_vllm()

    fp8 = torch.float8_e4m3fn
    scale_blocks = lambda size: math.ceil(size / args.block_size)  # noqa: E731

    activation = torch.empty((args.a_rows, args.k), dtype=fp8, device=device)
    activation_scale = torch.ones(
        (args.a_rows, scale_blocks(args.k)), dtype=torch.float32, device=device
    )
    weight = torch.empty(
        (args.num_local_experts, args.n, args.k), dtype=fp8, device=device
    )
    weight_scale = torch.ones(
        (args.num_local_experts, scale_blocks(args.n), scale_blocks(args.k)),
        dtype=torch.float32,
        device=device,
    )
    # C is [tokens, router_top_k, n] for BOTH roles: vLLM sizes the output by the
    # router's top_k even on the w2 launch, where the kernel's own top_k is 1.
    output = torch.empty(
        (args.num_tokens, args.experts_per_token, args.n),
        dtype=torch.bfloat16,
        device=device,
    )
    topk_weights = torch.full(
        (args.num_tokens, args.experts_per_token),
        1.0 / args.experts_per_token,
        dtype=torch.float32,
        device=device,
    )
    topk_ids = _topk_ids(torch, args, device=device)

    block_shape = [args.block_size, args.block_size]
    config = optimal_config(
        (args.num_local_experts, args.n, args.k),
        (args.num_local_experts, args.k, args.n),
        args.experts_per_token,
        "fp8_w8a8",
        args.num_tokens,
        block_shape=block_shape,
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = prepare_assignment(
        topk_ids,
        config,
        args.num_tokens,
        args.experts_per_token,
        args.num_local_experts,
        None,
        block_shape=block_shape,
    )

    return _Launch(
        invoke=invoke,
        positional=(
            activation,
            weight,
            output,
            activation_scale,
            weight_scale,
            topk_weights if args.mul_routed_weight else None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            args.mul_routed_weight,
            args.kernel_top_k,
            config,
        ),
        kwargs={
            "compute_type": triton_language.bfloat16,
            "use_fp8_w8a8": True,
            "use_int8_w8a8": False,
            "use_int8_w8a16": False,
            "use_int4_w4a16": False,
            "per_channel_quant": False,
            "block_shape": block_shape,
        },
    )


def _semantic_flops(args: _ValidatedArgs) -> int:
    """Multiply-accumulate FLOPs of the routed rows, ignoring block padding.

    Deliberately the *logical* count: padded-block waste is exactly what this
    kernel kind exists to capture, so folding it into the denominator would hide
    the effect in the reported TFLOPS.
    """
    return 2 * sum(args.per_group_batches) * args.n * args.k


def _logical_bytes(args: _ValidatedArgs) -> int:
    rows = sum(args.per_group_batches)
    active_experts = sum(1 for count in args.per_group_batches if count > 0)
    scale_blocks = lambda size: math.ceil(size / args.block_size)  # noqa: E731
    activation_bytes = rows * args.k
    activation_scale_bytes = rows * scale_blocks(args.k) * 4
    weight_bytes = active_experts * args.n * args.k
    weight_scale_bytes = active_experts * scale_blocks(args.n) * scale_blocks(args.k) * 4
    output_bytes = rows * args.n * 2
    return (
        activation_bytes
        + activation_scale_bytes
        + weight_bytes
        + weight_scale_bytes
        + output_bytes
    )


def profile_vllm_fused_moe_triton(
    n: int,
    k: int,
    dtype: DType | str,
    num_local_experts: int,
    num_tokens: int,
    experts_per_token: int,
    launch_role: str,
    block_size: int,
    per_group_batches: tuple[int, ...] | list[int],
) -> ComputeMetrics:
    args = _validate_args(
        n,
        k,
        dtype,
        num_local_experts,
        num_tokens,
        experts_per_token,
        launch_role,
        block_size,
        per_group_batches,
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_BACKEND}") from exc
    try:
        _require_h200(torch)
        device = torch.device("cuda", torch.cuda.current_device())
        launch = _build_launch(torch, args, device=device)
        # Triton autotune/JIT compilation stays outside the timed window.
        launch.run_once()
        torch.cuda.synchronize()

        time_ms = Timer.cupti(launch.run_once, kernel_name=None)
        energy_j = Energy.perf(launch.run_once, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed") from exc

    elapsed = time_ms / 1000
    return ComputeMetrics(
        time_ms=float(time_ms),
        energy_j=float(energy_j),
        tflops=float(_semantic_flops(args) / elapsed / 1e12 if elapsed else 0),
        memory_bandwidth_gbps=float(_logical_bytes(args) / elapsed / 1e9 if elapsed else 0),
    )
