"""FlashInfer TRT-LLM fused all-reduce + residual + RMSNorm runner.

The call mirrors vLLM's ``allreduce_rms`` compilation pass: Pattern 1,
residual input/output, RMS gamma, PDL enabled, completion at kernel end, and
FP32 accumulation. A torch.distributed process group only bootstraps the
FlashInfer IPC workspace; the timed device work is FlashInfer's fused kernel.
"""

from __future__ import annotations

from collections import defaultdict

from profiling.db.args import DType
from profiling.runners.comm._batch import all_error, run_comm_batch
from profiling.runners.comm._launcher import TorchMpLauncher
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import RunnerResult


def profile_all_reduce_residual_rms_norm_batch(
    kwargs_list: list[dict],
) -> list[RunnerResult]:
    if not kwargs_list:
        return []
    num_gpus = int(kwargs_list[0]["num_gpus"])
    if num_gpus not in {2, 4, 8}:
        return all_error(
            len(kwargs_list),
            f"FlashInfer TRT-LLM fused all-reduce supports TP 2/4/8, got {num_gpus}",
        )
    return run_comm_batch(
        TorchMpLauncher(num_gpus, backend="nccl"),
        _fused_per_rank_batch,
        kwargs_list,
    )


def _fused_per_rank_batch(
    *,
    rank: int,
    world_size: int,
    specs: list[dict],
    warmup: int,
    rep: int,
) -> list[dict] | None:
    try:
        import flashinfer.comm as flashinfer_comm
        import torch
        import torch.distributed as dist
        from flashinfer.comm.mnnvl import TorchDistBackend
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "the vLLM profiling environment with FlashInfer comm is required"
        ) from exc

    indexed_by_workspace: dict[tuple[int, DType], list[tuple[int, dict]]] = defaultdict(list)
    for spec_index, spec in enumerate(specs):
        dtype = DType.from_value(spec["dtype"])
        indexed_by_workspace[(int(spec["hidden_dim"]), dtype)].append((spec_index, spec))

    ordered_results: list[dict | None] = [None] * len(specs)
    try:
        for (hidden_dim, dtype), indexed_specs in indexed_by_workspace.items():
            torch_dtype = dtype.torch()
            max_num_tokens = max(int(spec["num_tokens"]) for _, spec in indexed_specs)
            workspace = flashinfer_comm.create_allreduce_fusion_workspace(
                backend="trtllm",
                world_size=world_size,
                rank=rank,
                max_token_num=max_num_tokens,
                hidden_dim=hidden_dim,
                dtype=torch_dtype,
                comm_backend=TorchDistBackend(group=dist.group.WORLD),
            )
            try:
                for spec_index, spec in indexed_specs:
                    ordered_results[spec_index] = _profile_one_shape(
                        flashinfer_comm=flashinfer_comm,
                        torch=torch,
                        dist=dist,
                        workspace=workspace,
                        world_size=world_size,
                        spec=spec,
                        torch_dtype=torch_dtype,
                        warmup=warmup,
                        rep=rep,
                    )
            finally:
                workspace.destroy()
    except (RuntimeError, ValueError, AssertionError) as exc:
        raise KernelLaunchFailed(str(exc)) from exc

    if rank != 0:
        return None
    return [
        result if result is not None else _missing_result(spec_index)
        for spec_index, result in enumerate(ordered_results)
    ]


def _profile_one_shape(
    *,
    flashinfer_comm,
    torch,
    dist,
    workspace,
    world_size: int,
    spec: dict,
    torch_dtype,
    warmup: int,
    rep: int,
) -> dict:
    num_tokens = int(spec["num_tokens"])
    hidden_dim = int(spec["hidden_dim"])
    strategy = str(spec["strategy"])
    if strategy not in {"auto", "oneshot", "twoshot"}:
        raise ValueError(f"unknown FlashInfer strategy {strategy!r}")
    use_oneshot = None if strategy == "auto" else strategy == "oneshot"

    input_tensor = torch.randn(num_tokens, hidden_dim, dtype=torch_dtype, device="cuda")
    residual_in = torch.randn_like(input_tensor)
    residual_out = torch.empty_like(input_tensor)
    norm_out = torch.empty_like(input_tensor)
    rms_gamma = torch.ones(hidden_dim, dtype=torch_dtype, device="cuda")

    def launch() -> None:
        flashinfer_comm.allreduce_fusion(
            input=input_tensor,
            workspace=workspace,
            pattern=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
            residual_in=residual_in,
            residual_out=residual_out,
            norm_out=norm_out,
            rms_gamma=rms_gamma,
            rms_eps=1e-6,
            use_oneshot=use_oneshot,
            launch_with_pdl=bool(spec["launch_with_pdl"]),
            trigger_completion_at_end=bool(spec["trigger_completion_at_end"]),
            fp32_acc=bool(spec["fp32_acc"]),
        )

    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        launch()
    end.record()
    torch.cuda.synchronize()
    time_ms = start.elapsed_time(end) / rep

    element_size = torch.tensor([], dtype=torch_dtype).element_size()
    message_size_bytes = num_tokens * hidden_dim * element_size
    latency_s = time_ms / 1000.0
    algbw_gbps = (message_size_bytes / latency_s) / 1e9 if latency_s > 0 else 0.0
    busbw_gbps = algbw_gbps * 2 * (world_size - 1) / world_size
    return {
        "time_ms": time_ms,
        "algbw_gbps": algbw_gbps,
        "busbw_gbps": busbw_gbps,
    }


def _missing_result(spec_index: int) -> dict:
    raise RuntimeError(f"missing rank-0 fused result at spec index {spec_index}")
