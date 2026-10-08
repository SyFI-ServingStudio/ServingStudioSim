"""Profile Inkling's gate select (``inkling_gate_select``) from the alignment fork.

``vllm_triton`` times one call of
``vllm.models.inkling.nvidia.moe.inkling_gate_select`` with the arguments
``InklingGate.select_experts`` passes: the router GEMM's fp32 logits
``[T, logits_row_stride]`` (routed columns, then the sink columns, then GEMM
padding the kernel must not read), ``n_gate_experts = R + S``, the selection
bias, ``route_scale`` and the ``[1]`` global scale. The call allocates its two
``[T, K + S]`` outputs and launches ``_inkling_gate_select_kernel`` once with one
program per token; CUPTI prices that one kernel.

Inputs: logits ~ N(0, 2^2) (a sigmoid router's range), a small random bias,
route_scale 8 (Inkling's config) and global scale 1.25; the padding columns hold
a large poison value so a kernel that read them would select them. The work does
not depend on the values: each program runs K full-width argmax passes.

Every spec checks the call's structure before timing (sink ids, distinct
in-range routed ids, weights summing to route_scale * global_scale). The full
comparison with the independent Torch reference in
``moe_sink_gate_select_reference`` runs on anchor shapes through ``oracle``
(``python -m profiling.runners.moe.moe_sink_gate_select_vllm_triton --oracle``),
including inputs with many exact ties.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.moe.moe_sink_gate_select_reference import GateShape, compare

KIND = "moe_sink_gate_select"
_BACKEND = f"{KIND}:vllm_triton"
_KERNEL_NAME = "_inkling_gate_select_kernel"
_MAX_TOKENS = 65_536
# Inkling's route_scale (config.route_scale); the global scale is a checkpoint
# value. Neither changes the work.
_ROUTE_SCALE = 8.0
_GLOBAL_SCALE = 1.25
_LOGIT_STD = 2.0
_PAD_POISON = 1.0e4
_SEED = 0x9A7E
_SUM_RTOL = 1e-5


def _validate_args(
    num_tokens: int,
    num_routed_experts: int,
    num_shared_experts: int,
    top_k: int,
    logits_row_stride: int,
    has_bias: bool,
    has_global_scale: bool,
    logits_dtype: DType | str,
) -> GateShape:
    if type(num_tokens) is not int or not 1 <= num_tokens <= _MAX_TOKENS:
        raise ValueError(f"num_tokens must be an integer in [1, {_MAX_TOKENS}]")
    for name, value in (
        ("num_routed_experts", num_routed_experts),
        ("top_k", top_k),
        ("logits_row_stride", logits_row_stride),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(num_shared_experts) is not int or num_shared_experts < 0:
        raise ValueError("num_shared_experts must be a non-negative integer")
    if top_k > num_routed_experts:
        raise ValueError("top_k must not exceed num_routed_experts")
    if logits_row_stride < num_routed_experts + num_shared_experts:
        raise ValueError("logits_row_stride must cover the routed and sink columns")
    for name, value in (("has_bias", has_bias), ("has_global_scale", has_global_scale)):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a bool")
    active = top_k + num_shared_experts
    # The kernel spans the selected set with tl.arange(0, K + S), which Triton
    # only builds for a power of two (moe.py, _inkling_gate_select_kernel).
    if active & (active - 1):
        raise ProfilerNotImplemented(
            f"{_BACKEND} needs top_k + num_shared_experts to be a power of two, got {active}"
        )
    # inkling_gate_select asserts fp32 logits.
    if DType.from_value(logits_dtype) is not DType.FP32:
        raise ProfilerNotImplemented(f"{_BACKEND} requires logits_dtype=fp32")
    return GateShape(num_routed_experts, num_shared_experts, top_k, logits_row_stride)


@dataclass
class GateInputs:
    shape: GateShape
    logits: Any  # [T, logits_row_stride] fp32
    bias: Any | None  # [R] fp32
    global_scale: Any | None  # [1] fp32
    route_scale: float


def build_inputs(
    torch: Any,
    device: Any,
    shape: GateShape,
    num_tokens: int,
    *,
    has_bias: bool,
    has_global_scale: bool,
    ties: bool = False,
    seed: int = _SEED,
) -> GateInputs:
    """Seeded router logits, bias and global scale for one call.

    ``ties`` draws logits from a grid of 0.5 and the bias from {0, 0.25}, so
    many experts share a score exactly and the lowest-id tie rule decides the
    selection (oracle use only).
    """

    gen = torch.Generator().manual_seed(seed)
    G = shape.num_gate_experts
    logits = torch.full((num_tokens, shape.logits_row_stride), _PAD_POISON, dtype=torch.float32)
    values = _LOGIT_STD * torch.randn((num_tokens, G), generator=gen)
    if ties:
        values = torch.round(values * 2) / 2
    logits[:, :G] = values
    bias = None
    if has_bias:
        if ties:
            bias = 0.25 * torch.randint(0, 2, (shape.num_routed_experts,), generator=gen).float()
        else:
            bias = 0.05 * torch.randn(shape.num_routed_experts, generator=gen)
        bias = bias.to(device)
    global_scale = (
        torch.tensor([_GLOBAL_SCALE], dtype=torch.float32, device=device)
        if has_global_scale
        else None
    )
    return GateInputs(shape, logits.to(device), bias, global_scale, _ROUTE_SCALE)


def call_select(inkling_gate_select: Any, x: GateInputs) -> tuple[Any, Any]:
    """One ``inkling_gate_select`` call with ``InklingGate.select_experts``' arguments."""
    s = x.shape
    return inkling_gate_select(
        x.logits,
        s.num_gate_experts,
        s.num_routed_experts,
        s.top_k,
        s.num_shared_experts,
        x.bias,
        x.route_scale,
        x.global_scale,
    )


def check_structure(torch: Any, x: GateInputs, weights: Any, ids: Any) -> None:
    """Cheap per-spec guard: shapes, sink ids, distinct in-range routed ids, weight sums."""
    s = x.shape
    T = x.logits.shape[0]
    if tuple(ids.shape) != (T, s.active) or tuple(weights.shape) != (T, s.active):
        raise KernelLaunchFailed(f"{_BACKEND} returned shapes {ids.shape} / {weights.shape}")
    routed = ids[:, : s.top_k].long()
    sinks = torch.arange(
        s.num_routed_experts, s.num_gate_experts, device=ids.device, dtype=torch.long
    )
    scale = x.route_scale * (_GLOBAL_SCALE if x.global_scale is not None else 1.0)
    sums = weights.double().sum(dim=1)
    ok = (
        bool(((routed >= 0) & (routed < s.num_routed_experts)).all().item())
        and (s.top_k == 1 or bool((routed.sort(dim=1).values.diff(dim=1) != 0).all().item()))
        and bool((ids[:, s.top_k :].long() == sinks).all().item())
        and bool((weights > 0).all().item())
        and bool(((sums - scale).abs() <= _SUM_RTOL * scale).all().item())
    )
    if not ok:
        raise KernelLaunchFailed(f"{_BACKEND} produced a malformed selection")


def logical_bytes(shape: GateShape, num_tokens: int) -> int:
    """Read the routed and sink logits; write fp32 weights and int32 ids."""
    return 4 * num_tokens * (shape.num_gate_experts + 2 * shape.active)


def _import_callable() -> tuple[Any, Any]:
    try:
        import torch
        from vllm.models.inkling.nvidia.moe import inkling_gate_select
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{_BACKEND} requires the alignment vLLM fork with Inkling (vllm_upstream_fork_env)"
        ) from exc
    return torch, inkling_gate_select


def profile_moe_sink_gate_select_vllm_triton(
    num_tokens: int,
    num_routed_experts: int,
    num_shared_experts: int,
    top_k: int,
    logits_row_stride: int,
    has_bias: bool,
    has_global_scale: bool,
    logits_dtype: DType | str,
) -> ComputeMetrics:
    """Time one ``inkling_gate_select`` call (its single Triton kernel, CUPTI)."""
    shape = _validate_args(
        num_tokens,
        num_routed_experts,
        num_shared_experts,
        top_k,
        logits_row_stride,
        has_bias,
        has_global_scale,
        logits_dtype,
    )
    torch, inkling_gate_select = _import_callable()
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        x = build_inputs(
            torch,
            device,
            shape,
            num_tokens,
            has_bias=has_bias,
            has_global_scale=has_global_scale,
        )
        weights, ids = call_select(inkling_gate_select, x)
        torch.cuda.synchronize(device)
        check_structure(torch, x, weights, ids)

        def launch() -> None:
            call_select(inkling_gate_select, x)

        time_ms = Timer.cupti(launch, warmup=5, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(launch, warmup=5, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except (KernelLaunchFailed, ProfilerNotImplemented):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(logical_bytes(shape, num_tokens) / (time_ms / 1000.0) / 1e9),
        energy_j=float(energy_j),
    )


# Anchor shapes for the full oracle: Inkling-Small (256 routed, 2 sinks, top-6,
# stride 264) at decode and mixed-batch sizes, plus the flag variants.
_INKLING = dict(
    num_routed_experts=256,
    num_shared_experts=2,
    top_k=6,
    logits_row_stride=264,
    logits_dtype="fp32",
)
ORACLE_ANCHORS: tuple[dict[str, Any], ...] = (
    *(
        dict(_INKLING, num_tokens=t, has_bias=True, has_global_scale=True)
        for t in (1, 32, 256, 2048)
    ),
    dict(_INKLING, num_tokens=32, has_bias=False, has_global_scale=True),
    dict(_INKLING, num_tokens=32, has_bias=True, has_global_scale=False),
    dict(_INKLING, num_tokens=4096, has_bias=False, has_global_scale=False),
)


def oracle(anchors: tuple[dict[str, Any], ...] = ORACLE_ANCHORS) -> list[dict[str, Any]]:
    """Compare the production call with the Torch reference on each anchor.

    Each anchor runs twice: on continuous random logits and on tie-heavy
    logits (exact ties decided by the lowest-id rule).
    """
    torch, inkling_gate_select = _import_callable()
    device = torch.device("cuda", torch.cuda.current_device())
    reports = []
    for spec in anchors:
        shape = _validate_args(**spec)
        for ties in (False, True):
            x = build_inputs(
                torch,
                device,
                shape,
                spec["num_tokens"],
                has_bias=spec["has_bias"],
                has_global_scale=spec["has_global_scale"],
                ties=ties,
            )
            weights, ids = call_select(inkling_gate_select, x)
            torch.cuda.synchronize(device)
            report = compare(
                torch, shape, x.logits, x.bias, x.route_scale, x.global_scale, ids, weights
            )
            reports.append({**spec, "ties": ties, **report})
    return reports


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Full Torch oracle for inkling_gate_select anchors.")
    ap.add_argument("--oracle", action="store_true", required=True)
    ap.add_argument("--out", default=None, help="Write the JSON report here as well.")
    args = ap.parse_args(argv)
    reports = oracle()
    text = json.dumps(reports, indent=1)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    return 0 if all(r["ok"] for r in reports) else 1


__all__ = [
    "ORACLE_ANCHORS",
    "GateInputs",
    "build_inputs",
    "call_select",
    "check_structure",
    "logical_bytes",
    "oracle",
    "profile_moe_sink_gate_select_vllm_triton",
]


if __name__ == "__main__":
    sys.exit(main())
