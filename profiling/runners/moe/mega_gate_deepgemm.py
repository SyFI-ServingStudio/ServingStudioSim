"""DeepInfer's MoE router on SM100 -- the ``deepgemm_sm100`` backend of ``mega_gate``.

The timed call is the one ``DeepseekV4MoE.select_experts`` issues on GPUs
(``deepinfer/model_executor/models/v4_modules/mlp.py``, ``use_mega_gate``):
``deep_gemm.bf16_mega_gate(hidden_states, gate.weight, num_topk, ...)`` with
the score-correction bias, ``use_shared_as_routed=False``, the per-token valid
mask (all valid) and force-random flags (all off), and the EPLB layer's
``logical_to_physical_map`` / ``logical_count``. The slot map is DeepInfer's
initial layout (``get_initial_global_physical_to_logical_map``): the routed
experts in order, then ``num_physical_experts - num_experts`` redundant copies
of experts ``0, 1, ...``. One ``sm100_bf16_mega_gate_impl`` launch.

Check: the chosen logical experts must match TileKernels' unfused
``moe_topk_gate`` on fp32 logits of the same operands (DeepInfer's non-GPU
path) for nearly every token; GEMM accumulation order can flip near ties.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BACKEND = "deepgemm_sm100"
# DeepSeek-V4.1's routed_scaling_factor; it scales the weights only.
ROUTED_SCALING_FACTOR = 1.5
CHECK_TOKENS = 4096
MIN_MATCH = 0.99


def _validate(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    num_physical_experts: int,
    top_k: int,
    scoring_func: str,
    dtype: DType | str,
) -> tuple[int, int, int, int, int, str]:
    if DType.from_value(dtype) is not DType.BF16:
        raise ValueError(f"{BACKEND} mega_gate requires dtype=bf16, got {dtype!r}")
    values = [int(v) for v in (num_tokens, hidden_size, num_experts, num_physical_experts, top_k)]
    num_tokens, hidden_size, num_experts, num_physical_experts, top_k = values
    if min(values) < 1:
        raise ValueError(f"{BACKEND} mega_gate needs positive sizes, got {values}")
    if num_physical_experts < num_experts:
        raise ValueError(f"num_physical_experts {num_physical_experts} < num_experts {num_experts}")
    if top_k > num_experts:
        raise ValueError(f"top_k {top_k} > num_experts {num_experts}")
    return num_tokens, hidden_size, num_experts, num_physical_experts, top_k, str(scoring_func)


def slot_map(
    torch: Any, num_experts: int, num_physical_experts: int, device: Any
) -> tuple[Any, Any, Any]:
    """DeepInfer's initial EPLB maps for one layer.

    Returns ``(physical_to_logical, logical_to_physical [E, R+1] padded with -1,
    logical_count [E])`` exactly as ``ExpertLoadBalancer.__init__`` builds them.
    """
    extra = num_physical_experts - num_experts
    physical_to_logical = [*range(num_experts), *(i % num_experts for i in range(extra))]
    logical_to_physical = torch.full((num_experts, extra + 1), -1, dtype=torch.int32)
    count = torch.zeros(num_experts, dtype=torch.int32)
    for physical, logical in enumerate(physical_to_logical):
        logical_to_physical[logical, count[logical]] = physical
        count[logical] += 1
    p2l = torch.tensor(physical_to_logical, dtype=torch.int64, device=device)
    return p2l, logical_to_physical.to(device), count.to(device)


def profile_mega_gate_deepgemm(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    num_physical_experts: int,
    top_k: int,
    scoring_func: str,
    dtype: DType | str,
) -> ComputeMetrics:
    num_tokens, hidden_size, num_experts, num_physical_experts, top_k, scoring_func = _validate(
        num_tokens, hidden_size, num_experts, num_physical_experts, top_k, scoring_func, dtype
    )
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"torch is required for {BACKEND}") from exc
    try:
        import deep_gemm
        from tile_kernels.moe import moe_topk_gate
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{BACKEND} mega_gate requires DeepInfer's DeepGEMM and TileKernels (deepinfer_env)"
        ) from exc
    try:
        torch.manual_seed(0)
        device = torch.device("cuda")
        x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
        weight = torch.randn(num_experts, hidden_size, device=device, dtype=torch.bfloat16) / 64
        bias = torch.randn(num_experts, device=device, dtype=torch.float32) / 16
        p2l, l2p, count = slot_map(torch, num_experts, num_physical_experts, device)
        kwargs = dict(
            num_topk=top_k,
            use_shared_as_routed=False,
            num_shared_experts=0,
            routed_scaling_factor=ROUTED_SCALING_FACTOR,
            ep_rank=0,
            mask=torch.ones(num_tokens, dtype=torch.bool, device=device),
            bias=bias,
            force_random=torch.zeros(num_tokens, dtype=torch.bool, device=device),
            to_physical_map=l2p,
            logical_count=count,
            scoring_func=scoring_func,
        )

        def run_once() -> Any:
            return deep_gemm.bf16_mega_gate(x, weight, **kwargs)

        idx, _ = run_once()  # JIT compile, outside timing.
        torch.cuda.synchronize()
        rows = slice(0, min(num_tokens, CHECK_TOKENS))
        ref_kwargs = {
            **kwargs,
            "mask": kwargs["mask"][rows],
            "force_random": kwargs["force_random"][rows],
        }
        ref_idx, _ = moe_topk_gate(x[rows].float() @ weight.float().t(), **ref_kwargs)
        got = p2l[idx[rows].long()].sort(dim=-1).values
        want = p2l[ref_idx.long()].sort(dim=-1).values
        match = (got == want).all(dim=-1).float().mean().item()
        if match < MIN_MATCH:
            raise KernelLaunchFailed(
                f"{BACKEND} mega_gate: only {match:.3f} of tokens chose the reference experts"
            )

        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError(f"{BACKEND} mega_gate ran out of memory") from exc
        raise KernelLaunchFailed(f"{BACKEND} mega_gate failed: {exc}") from exc

    flops = 2 * num_tokens * num_experts * hidden_size
    bytes_accessed = (
        2 * num_tokens * hidden_size + 2 * num_experts * hidden_size + 8 * num_tokens * top_k
    )
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=flops / elapsed_s / 1e12,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


__all__ = ["profile_mega_gate_deepgemm", "slot_map"]
