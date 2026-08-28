"""FlashInfer TRT-LLM shape-aware all-reduce runners.

The standalone call mirrors vLLM's ``FlashInferAllReduce.all_reduce`` path:
``kAllReduce``, PDL enabled, and completion at kernel end only above 16 tokens.
The fused call mirrors vLLM's ``allreduce_rms`` compilation pass. A
torch.distributed process group only bootstraps the FlashInfer IPC workspace;
the timed device work is FlashInfer's kernel.
"""

from __future__ import annotations

import time
from collections import defaultdict

from profiling.db.args import DType
from profiling.runners.comm._batch import all_error, run_comm_batch
from profiling.runners.comm._launcher import TorchMpLauncher
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import RunnerResult


def profile_all_reduce_fusion_batch(kwargs_list: list[dict]) -> list[RunnerResult]:
    if not kwargs_list:
        return []
    num_gpus = int(kwargs_list[0]["num_gpus"])
    if num_gpus not in {2, 4, 8}:
        return all_error(
            len(kwargs_list),
            f"FlashInfer TRT-LLM all-reduce supports TP 2/4/8, got {num_gpus}",
        )
    return run_comm_batch(
        TorchMpLauncher(num_gpus, backend="nccl"),
        _all_reduce_per_rank_batch,
        kwargs_list,
    )


def _all_reduce_per_rank_batch(
    *,
    rank: int,
    world_size: int,
    specs: list[dict],
    warmup: int,
    rep: int,
) -> list[dict] | None:
    flashinfer_comm, torch, dist, torch_dist_backend = _imports()
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
                comm_backend=torch_dist_backend(group=dist.group.WORLD),
            )
            try:
                for spec_index, spec in indexed_specs:
                    ordered_results[spec_index] = _profile_one_all_reduce_shape(
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


def _imports():
    try:
        import flashinfer.comm as flashinfer_comm
        import torch
        import torch.distributed as dist
        from flashinfer.comm.mnnvl import TorchDistBackend
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "the profiling environment with FlashInfer comm is required"
        ) from exc
    return flashinfer_comm, torch, dist, TorchDistBackend


def _profile_one_all_reduce_shape(
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
    input_tensor = torch.randn(num_tokens, hidden_dim, dtype=torch_dtype, device="cuda")

    def launch() -> None:
        flashinfer_comm.allreduce_fusion(
            input=input_tensor,
            workspace=workspace,
            pattern=flashinfer_comm.AllReduceFusionPattern.kAllReduce,
            launch_with_pdl=True,
            trigger_completion_at_end=num_tokens > 16,
        )

    time_ms = _time_cuda_graph(
        torch=torch,
        dist=dist,
        launch=launch,
        input_tensor=input_tensor,
        warmup=warmup,
        rep=rep,
        operations_per_graph=1,
    )
    return _comm_payload(
        time_ms=time_ms,
        message_size_bytes=input_tensor.numel() * input_tensor.element_size(),
        world_size=world_size,
    )


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
    flashinfer_comm, torch, dist, torch_dist_backend = _imports()

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
                comm_backend=torch_dist_backend(group=dist.group.WORLD),
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

    time_ms = _time_cuda_graph(
        torch=torch,
        dist=dist,
        launch=launch,
        input_tensor=input_tensor,
        warmup=warmup,
        rep=rep,
        operations_per_graph=10,
    )
    return _comm_payload(
        time_ms=time_ms,
        message_size_bytes=input_tensor.numel() * input_tensor.element_size(),
        world_size=world_size,
    )


def _time_cuda_graph(
    *,
    torch,
    dist,
    launch,
    input_tensor,
    warmup: int,
    rep: int,
    operations_per_graph: int,
) -> float:
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    # Match vLLM's serving path: the PDL-enabled collective runs inside a CUDA
    # graph. Per-launch events alter its dependency behavior, so measure graph
    # replay throughput and reduce the slowest rank.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(operations_per_graph):
            launch()
    torch.cuda.synchronize()

    for _ in range(max(1, warmup // operations_per_graph)):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    start_time = time.perf_counter()
    for _ in range(rep // operations_per_graph):
        graph.replay()
    torch.cuda.synchronize()
    local_time_ms = ((time.perf_counter() - start_time) / rep) * 1000.0
    slowest_rank_time_ms = torch.tensor(
        local_time_ms,
        dtype=torch.float32,
        device=input_tensor.device,
    )
    dist.all_reduce(slowest_rank_time_ms, op=dist.ReduceOp.MAX)
    return float(slowest_rank_time_ms.item())


def _comm_payload(*, time_ms: float, message_size_bytes: int, world_size: int) -> dict:
    latency_s = time_ms / 1000.0
    algbw_gbps = (message_size_bytes / latency_s) / 1e9 if latency_s > 0 else 0.0
    busbw_gbps = algbw_gbps * 2 * (world_size - 1) / world_size
    return {"time_ms": time_ms, "algbw_gbps": algbw_gbps, "busbw_gbps": busbw_gbps}


def _missing_result(spec_index: int) -> dict:
    raise RuntimeError(f"missing rank-0 fused result at spec index {spec_index}")
