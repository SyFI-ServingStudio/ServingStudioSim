"""Profile vLLM's NCCL all-gather and vocabulary layout conversion together."""

from types import SimpleNamespace

from profiling.db.args import DType
from profiling.profilers.timer import Timer
from profiling.runners.comm._batch import all_error, run_comm_batch
from profiling.runners.comm._launcher import TorchMpLauncher
from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.logits._common import logits_dtype, positive_int, require_b200
from profiling.runners.logits.reference import vocab_parallel_all_gather_reference
from profiling.runners.metrics import RunnerResult


def validate_args(
    num_gpus: int,
    num_rows: int,
    vocab_size_per_rank: int,
    dtype: DType | str,
    fabric: str = "nvlink",
) -> None:
    positive_int("num_gpus", num_gpus)
    positive_int("num_rows", num_rows)
    positive_int("vocab_size_per_rank", vocab_size_per_rank)
    logits_dtype(dtype)
    if num_gpus < 2:
        raise ValueError("vocabulary all-gather requires at least two GPUs")
    if fabric != "nvlink":
        raise ProfilerNotImplemented("vocabulary all-gather currently supports fabric='nvlink'")


def profile_vocab_parallel_all_gather_batch(kwargs_list: list[dict]) -> list[RunnerResult]:
    if not kwargs_list:
        return []
    try:
        for spec in kwargs_list:
            validate_args(**spec)
        num_gpus = kwargs_list[0]["num_gpus"]
        if any(spec["num_gpus"] != num_gpus for spec in kwargs_list):
            raise ValueError("all vocabulary all-gather batch specs must use the same num_gpus")
    except (ValueError, TypeError, ProfilerNotImplemented) as exc:
        return all_error(len(kwargs_list), str(exc))
    return run_comm_batch(TorchMpLauncher(num_gpus, backend="nccl"), _per_rank_batch, kwargs_list)


def _per_rank_batch(
    *, rank: int, world_size: int, specs: list[dict], warmup: int, rep: int
) -> list[dict] | None:
    import torch
    import torch.distributed as dist
    from vllm.distributed.device_communicators.base_device_communicator import (
        BaseDeviceCommunicator,
    )

    require_b200(torch)
    # The public method only needs these two fields; no model or device
    # communicator initialization is necessary for the already-live NCCL group.
    communicator = SimpleNamespace(world_size=world_size, device_group=dist.group.WORLD)
    results = []
    for spec in specs:
        validate_args(**spec)
        rows, width = spec["num_rows"], spec["vocab_size_per_rank"]
        dtype = logits_dtype(spec["dtype"]).torch()
        generator = torch.Generator(device="cuda").manual_seed(rank)
        shard = torch.randn((rows, width), dtype=dtype, device="cuda", generator=generator)

        def launch():
            return BaseDeviceCommunicator.all_gather(communicator, shard, dim=1)

        shards = [torch.empty_like(shard) for _ in range(world_size)]
        dist.all_gather(shards, shard)
        expected = vocab_parallel_all_gather_reference(torch, shards)
        actual = launch()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if not actual.is_contiguous():
            raise ValueError("vocabulary all-gather output must be contiguous")
        del shards, expected, actual
        dist.barrier()
        # Fixed counts keep every collective rank in lockstep. CUPTI excludes
        # CPU launch gaps and includes both NCCL and the ensuing layout copy.
        rank_ms = Timer.cupti(launch, warmup=warmup, rep=rep, kernel_name=None)
        maximum = torch.tensor(rank_ms, dtype=torch.float64, device="cuda")
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        time_ms = float(maximum.item())
        output_bytes = rows * width * world_size * shard.element_size()
        seconds = time_ms / 1000
        results.append(
            {
                "time_ms": time_ms,
                "algbw_gbps": output_bytes / seconds / 1e9 if seconds else 0.0,
                "busbw_gbps": output_bytes * (world_size - 1) / world_size / seconds / 1e9
                if seconds
                else 0.0,
                # Energy.perf uses adaptive rank-local loop counts and cannot safely
                # drive collectives. Match the established comm unknown-energy field.
                "energy_j": 0.0,
            }
        )
    return results if rank == 0 else None
