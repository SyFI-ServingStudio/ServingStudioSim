"""Production PyNCCL collectives used by vLLM's naive DP/EP MoE path."""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.runners.comm._batch import all_error, run_comm_batch
from profiling.runners.comm._launcher import VllmLauncher
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import RunnerResult

_GPU_NAMES = frozenset({"NVIDIA H200", "NVIDIA B200"})
_NVFP4_SCALE_GROUP = 16
_FP8_BLOCK_SCALE_GROUP = 128
# A DP/EP group gathers every rank's scheduler batch: 8 ranks x 8192 tokens.
_MAX_TOTAL_TOKENS = 65536


def profile_moe_ep_all_gather_batch(kwargs_list: list[dict]) -> list[RunnerResult]:
    return _profile_batch(kwargs_list, _all_gather_per_rank_batch, _validate_all_gather)


def profile_moe_ep_quantized_all_gather_batch(kwargs_list: list[dict]) -> list[RunnerResult]:
    return _profile_batch(
        kwargs_list, _quantized_all_gather_per_rank_batch, _validate_quantized_all_gather
    )


def profile_moe_ep_reduce_scatter_batch(kwargs_list: list[dict]) -> list[RunnerResult]:
    return _profile_batch(kwargs_list, _reduce_scatter_per_rank_batch, _validate_reduce_scatter)


def _profile_batch(kwargs_list: list[dict], per_rank_fn, validator) -> list[RunnerResult]:
    if not kwargs_list:
        return []
    try:
        validated = [validator(**spec) for spec in kwargs_list]
        num_gpus = validated[0][0]
        if any(values[0] != num_gpus for values in validated):
            raise ProfilerNotImplemented("all batch specs must use the same num_gpus")
    except Exception as exc:  # noqa: BLE001 - validation maps one bad group to one batch error
        return all_error(len(kwargs_list), str(exc))
    return run_comm_batch(
        VllmLauncher(num_gpus, required_gpu_names=_GPU_NAMES),
        per_rank_fn,
        kwargs_list,
    )


def _validate_topology(
    num_gpus: int,
    per_rank_tokens: tuple[int, ...] | list[int],
    hidden_size: int,
    fabric: str,
) -> tuple[int, tuple[int, ...], int]:
    # PyNcclCommunicator takes any world size but disables itself at one rank,
    # where these collectives would not run.
    if not isinstance(num_gpus, int) or isinstance(num_gpus, bool) or num_gpus < 2:
        raise ProfilerNotImplemented(f"vllm_pynccl needs num_gpus >= 2, got {num_gpus}")
    token_counts = tuple(per_rank_tokens)
    if len(token_counts) != num_gpus:
        raise ValueError(
            f"per_rank_tokens must contain num_gpus={num_gpus} entries, got {len(token_counts)}"
        )
    if any(
        not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in token_counts
    ):
        raise ValueError("per_rank_tokens must contain non-negative integers")
    if not 0 < sum(token_counts) <= _MAX_TOTAL_TOKENS:
        raise ProfilerNotImplemented(
            f"total tokens must be in 1..={_MAX_TOTAL_TOKENS}, got {sum(token_counts)}"
        )
    if not isinstance(hidden_size, int) or isinstance(hidden_size, bool) or hidden_size <= 0:
        raise ValueError("hidden_size must be a positive integer")
    if fabric != "nvlink":
        raise ProfilerNotImplemented(f"vllm_pynccl supports fabric='nvlink', got {fabric!r}")
    return num_gpus, token_counts, hidden_size


def _validate_all_gather(
    num_gpus: int,
    per_rank_tokens: tuple[int, ...] | list[int],
    hidden_size: int,
    num_experts: int,
    hidden_dtype: DType | str,
    router_dtype: DType | str,
    fabric: str,
) -> tuple[int, tuple[int, ...], int, int]:
    num_gpus, token_counts, hidden_size = _validate_topology(
        num_gpus, per_rank_tokens, hidden_size, fabric
    )
    if not isinstance(num_experts, int) or isinstance(num_experts, bool) or num_experts <= 0:
        raise ValueError("num_experts must be a positive integer")
    hidden_dtype = DType.from_value(hidden_dtype)
    if hidden_dtype not in (DType.BF16, DType.FP8_E4M3):
        raise ProfilerNotImplemented(
            "grouped MoE all-gather requires hidden_dtype=bf16 or fp8_e4m3"
        )
    if hidden_dtype is DType.FP8_E4M3 and hidden_size % _FP8_BLOCK_SCALE_GROUP:
        raise ValueError(f"fp8 hidden_size must be a multiple of {_FP8_BLOCK_SCALE_GROUP}")
    if DType.from_value(router_dtype) is not DType.FP32:
        raise ProfilerNotImplemented("grouped MoE all-gather requires router_dtype=fp32")
    return num_gpus, token_counts, hidden_size, num_experts


def all_gather_columns(
    hidden_size: int, num_experts: int, hidden_dtype: DType | str
) -> tuple[tuple[int, str, int], ...]:
    """(columns, torch dtype name, bytes per element) of each gathered tensor, in
    `AgRsAll2AllManager.dispatch_router_logits` order: hidden states, router
    logits, then the extra tensors. Block-FP8 activations carry one fp32 scale
    per 128 elements as the single extra tensor (`naive_dp_ep.py` prepare)."""
    columns = [(hidden_size, "bfloat16", 2), (num_experts, "float32", 4)]
    if DType.from_value(hidden_dtype) is DType.FP8_E4M3:
        columns[0] = (hidden_size, "float8_e4m3fn", 1)
        columns.append((hidden_size // _FP8_BLOCK_SCALE_GROUP, "float32", 4))
    return tuple(columns)


def _validate_quantized_all_gather(
    num_gpus: int,
    per_rank_tokens: tuple[int, ...] | list[int],
    hidden_size: int,
    top_k: int,
    activation_dtype: DType | str,
    fabric: str,
) -> tuple[int, tuple[int, ...], int, int]:
    num_gpus, token_counts, hidden_size = _validate_topology(
        num_gpus, per_rank_tokens, hidden_size, fabric
    )
    if hidden_size % _NVFP4_SCALE_GROUP:
        raise ValueError(f"hidden_size must be a multiple of {_NVFP4_SCALE_GROUP}")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if DType.from_value(activation_dtype) is not DType.NVFP4_E2M1:
        raise ProfilerNotImplemented(
            "quantized MoE all-gather requires activation_dtype=nvfp4_e2m1"
        )
    return num_gpus, token_counts, hidden_size, top_k


def quantized_all_gather_token_bytes(hidden_size: int, top_k: int) -> int:
    """Bytes one token adds to the gather: packed e2m1 pairs, one e4m3 scale per
    16 elements, then fp32 weights and int32 IDs for each selected expert."""
    return hidden_size // 2 + hidden_size // _NVFP4_SCALE_GROUP + top_k * (4 + 4)


def _validate_reduce_scatter(
    num_gpus: int,
    per_rank_tokens: tuple[int, ...] | list[int],
    hidden_size: int,
    dtype: DType | str,
    fabric: str,
) -> tuple[int, tuple[int, ...], int]:
    values = _validate_topology(num_gpus, per_rank_tokens, hidden_size, fabric)
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented("MoE reduce-scatter requires dtype=bf16")
    return values


def _rank_tensor(torch_module: Any, rows: int, columns: int, rank: int, dtype: Any):
    row_ids = torch_module.arange(rows, device="cuda", dtype=torch_module.int32).view(-1, 1)
    column_ids = torch_module.arange(columns, device="cuda", dtype=torch_module.int32).view(1, -1)
    return ((row_ids * 3 + column_ids * 5 + rank * 7) % 17 - 8).to(dtype).contiguous()


def _all_gather_per_rank_batch(
    *,
    rank: int,
    world_size: int,
    communicator: Any,
    specs: list[dict],
    warmup: int,
    rep: int,
) -> list[dict] | None:
    import torch
    import torch.distributed as dist

    results = []
    for spec in specs:
        _, token_counts, hidden_size, num_experts = _validate_all_gather(**spec)
        columns = all_gather_columns(hidden_size, num_experts, spec["hidden_dtype"])
        total_tokens = sum(token_counts)

        def rank_inputs(source_rank: int, rows: int) -> list[Any]:
            # fp8 test values are small integers, so the e4m3 cast is exact.
            return [
                _rank_tensor(torch, rows, width, source_rank, getattr(torch, dtype))
                for width, dtype, _ in columns
            ]

        inputs = rank_inputs(rank, token_counts[rank])
        outputs = [
            torch.empty((total_tokens, width), dtype=getattr(torch, dtype), device="cuda")
            for width, dtype, _ in columns
        ]

        def launch() -> None:
            communicator.group_start()
            for output, tensor in zip(outputs, inputs, strict=True):
                if len(set(token_counts)) == 1:
                    communicator.all_gather(output, tensor)
                else:
                    communicator.all_gatherv(output, tensor, sizes=list(token_counts))
            communicator.group_end()

        launch()
        torch.cuda.synchronize()
        expected = [
            torch.cat(parts)
            for parts in zip(
                *(rank_inputs(source, count) for source, count in enumerate(token_counts)),
                strict=True,
            )
        ]
        # Compare bytes: torch.equal has no float8 kernel.
        if not all(
            torch.equal(got.view(torch.uint8), want.view(torch.uint8))
            for got, want in zip(outputs, expected, strict=True)
        ):
            raise KernelLaunchFailed("grouped PyNCCL all-gather failed the Torch reference")

        rank_time_ms = _time_launch(torch, dist, launch, warmup, rep)
        rank_times = _gather_rank_times(dist, rank, world_size, rank_time_ms)
        if rank == 0:
            token_bytes = sum(width * size for width, _, size in columns)
            local_payloads = [count * token_bytes for count in token_counts]
            results.append(
                _metric_payload(
                    rank_times,
                    sum(local_payloads) / world_size,
                    sum(local_payloads) - min(local_payloads),
                )
            )
    return results if rank == 0 else None


def _quantized_all_gather_per_rank_batch(
    *,
    rank: int,
    world_size: int,
    communicator: Any,
    specs: list[dict],
    warmup: int,
    rep: int,
) -> list[dict] | None:
    import torch
    import torch.distributed as dist

    results = []
    for spec in specs:
        _, token_counts, hidden_size, top_k = _validate_quantized_all_gather(**spec)
        total_tokens = sum(token_counts)
        # vLLM's AgRsAll2AllManager.dispatch gathers [hidden_states, topk_weights,
        # topk_ids] followed by the extra block-scale tensor, in this order.
        columns_and_dtypes = (
            (hidden_size // 2, torch.uint8),
            (top_k, torch.float32),
            (top_k, torch.int32),
            (hidden_size // _NVFP4_SCALE_GROUP, torch.uint8),
        )

        def rank_inputs(source_rank: int, rows: int) -> list[Any]:
            tensors = [
                _rank_tensor(
                    torch,
                    rows,
                    columns,
                    source_rank,
                    torch.int32 if dtype is torch.uint8 else dtype,
                )
                for columns, dtype in columns_and_dtypes
            ]
            # Map signed test values onto bytes for the packed operands.
            tensors[0] = (tensors[0] + 8).to(torch.uint8)
            tensors[3] = (tensors[3] + 8).to(torch.uint8)
            return tensors

        inputs = rank_inputs(rank, token_counts[rank])
        # The block scales travel as float8_e4m3fn; the byte view is what NCCL moves.
        inputs[3] = inputs[3].view(torch.float8_e4m3fn)
        outputs = [
            torch.empty((total_tokens, columns), dtype=dtype, device="cuda")
            for columns, dtype in columns_and_dtypes
        ]
        outputs[3] = outputs[3].view(torch.float8_e4m3fn)

        def launch() -> None:
            communicator.group_start()
            for output, tensor in zip(outputs, inputs, strict=True):
                if len(set(token_counts)) == 1:
                    communicator.all_gather(output, tensor)
                else:
                    communicator.all_gatherv(output, tensor, sizes=list(token_counts))
            communicator.group_end()

        launch()
        torch.cuda.synchronize()
        expected = [
            torch.cat(parts)
            for parts in zip(
                *(rank_inputs(source, count) for source, count in enumerate(token_counts)),
                strict=True,
            )
        ]
        actual = list(outputs)
        actual[3] = actual[3].view(torch.uint8)
        if not all(torch.equal(got, want) for got, want in zip(actual, expected, strict=True)):
            raise KernelLaunchFailed(
                "grouped PyNCCL quantized all-gather failed the Torch reference"
            )

        rank_time_ms = _time_launch(torch, dist, launch, warmup, rep)
        rank_times = _gather_rank_times(dist, rank, world_size, rank_time_ms)
        if rank == 0:
            token_bytes = quantized_all_gather_token_bytes(hidden_size, top_k)
            local_payloads = [count * token_bytes for count in token_counts]
            results.append(
                _metric_payload(
                    rank_times,
                    sum(local_payloads) / world_size,
                    sum(local_payloads) - min(local_payloads),
                )
            )
    return results if rank == 0 else None


def _reduce_scatter_per_rank_batch(
    *,
    rank: int,
    world_size: int,
    communicator: Any,
    specs: list[dict],
    warmup: int,
    rep: int,
) -> list[dict] | None:
    import torch
    import torch.distributed as dist

    results = []
    for spec in specs:
        _, token_counts, hidden_size = _validate_reduce_scatter(**spec)
        total_tokens = sum(token_counts)
        input_tensor = _rank_tensor(torch, total_tokens, hidden_size, rank, torch.bfloat16)
        output = torch.empty((token_counts[rank], hidden_size), dtype=torch.bfloat16, device="cuda")

        def launch() -> None:
            if len(set(token_counts)) == 1:
                communicator.reduce_scatter(output, input_tensor, dist.ReduceOp.SUM)
            else:
                communicator.reduce_scatterv(
                    output, input_tensor, sizes=list(token_counts), op=dist.ReduceOp.SUM
                )

        launch()
        torch.cuda.synchronize()
        reduced = sum(
            _rank_tensor(torch, total_tokens, hidden_size, source_rank, torch.bfloat16)
            for source_rank in range(world_size)
        )
        start_row = sum(token_counts[:rank])
        expected = reduced[start_row : start_row + token_counts[rank]]
        if not torch.equal(output, expected):
            raise KernelLaunchFailed("PyNCCL reduce-scatter failed the Torch reference")

        rank_time_ms = _time_launch(torch, dist, launch, warmup, rep)
        rank_times = _gather_rank_times(dist, rank, world_size, rank_time_ms)
        if rank == 0:
            largest_shard_bytes = max(token_counts) * hidden_size * 2
            results.append(
                _metric_payload(
                    rank_times,
                    largest_shard_bytes,
                    (world_size - 1) * largest_shard_bytes,
                )
            )
    return results if rank == 0 else None


def _time_launch(torch_module: Any, dist_module: Any, launch, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        launch()
    torch_module.cuda.synchronize()
    dist_module.barrier()
    start = torch_module.cuda.Event(enable_timing=True)
    end = torch_module.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        launch()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / rep)


def _gather_rank_times(
    dist_module: Any, rank: int, world_size: int, rank_time_ms: float
) -> list[float] | None:
    rank_times = [None] * world_size if rank == 0 else None
    dist_module.gather_object(rank_time_ms, rank_times, dst=0)
    return rank_times


def _metric_payload(
    rank_times_ms: list[float] | None,
    algorithm_bytes: float,
    wire_bytes: float,
) -> dict[str, float]:
    if not rank_times_ms:
        raise KernelLaunchFailed("PyNCCL timing returned no rank durations")
    time_ms = max(float(value) for value in rank_times_ms)
    elapsed_seconds = time_ms / 1000.0
    return {
        "time_ms": time_ms,
        "algbw_gbps": algorithm_bytes / elapsed_seconds / 1e9,
        "busbw_gbps": wire_bytes / elapsed_seconds / 1e9,
        "energy_j": 0.0,
    }
