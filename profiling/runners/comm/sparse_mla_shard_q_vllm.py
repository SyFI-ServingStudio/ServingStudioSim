"""vLLM ShardQ exchange (sparse-MLA prefill token sharding) over symmetric memory."""

from __future__ import annotations

from collections import defaultdict

from profiling.db.args import DType
from profiling.runners.comm._batch import all_error, run_comm_batch
from profiling.runners.comm._launcher import VllmTpGroupLauncher
from profiling.runners.comm._multinode import MultiNodeVllmTpGroupLauncher, spans_nodes
from profiling.runners.comm.moe_ep_collectives_vllm_pynccl import (
    _gather_rank_times,
    _metric_payload,
    _time_launch,
)
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import RunnerResult

_DIRECTIONS = ("shard_q", "unshard_out")


def profile_sparse_mla_shard_q_batch(kwargs_list: list[dict]) -> list[RunnerResult]:
    if not kwargs_list:
        return []
    num_gpus = int(kwargs_list[0]["num_gpus"])
    try:
        for spec in kwargs_list:
            _check_spec(spec, num_gpus)
    except ValueError as exc:
        return all_error(len(kwargs_list), str(exc))
    # A TP16 group on GB200 NVL72 spans four trays of one MNNVL domain.
    launcher = (
        MultiNodeVllmTpGroupLauncher(num_gpus)
        if spans_nodes(num_gpus, "nvlink")
        else VllmTpGroupLauncher(num_gpus)
    )
    return run_comm_batch(launcher, _per_rank_batch, kwargs_list)


def _check_spec(spec: dict, num_gpus: int) -> None:
    if int(spec["num_gpus"]) != num_gpus:
        raise ValueError("a ShardQ chunk must share one num_gpus")
    if num_gpus < 2:
        raise ValueError("ShardQ needs a TP group of at least 2 GPUs")
    if spec["direction"] not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}, got {spec['direction']}")
    if int(spec["num_tokens"]) < 2 * num_gpus:
        raise ValueError("every GPU must own at least one zigzag chunk")
    dtype = DType.from_value(spec["dtype"])
    if spec["direction"] == "shard_q" and dtype not in (DType.BF16, DType.FP8_E4M3):
        raise ValueError(f"shard_q moves a bf16 or fp8 query, got {dtype}")
    if spec["direction"] == "unshard_out" and dtype is not DType.BF16:
        raise ValueError(f"unshard_out moves a bf16 output, got {dtype}")


def _imports():
    try:
        import numpy as np
        import torch
        import torch.distributed as dist
        from vllm.distributed import get_tp_group
        from vllm.v1.attention.ops import shardq
        from vllm.v1.attention.ops.cp_common import direct_cp_enabled
    except ImportError as exc:
        raise ProfilerNotImplemented("the alignment vLLM checkout (vllm_env) is required") from exc
    return np, torch, dist, get_tp_group, shardq, direct_cp_enabled


def _torch_dtype(torch, value) -> object:
    dtype = DType.from_value(value)
    return torch.float8_e4m3fn if dtype is DType.FP8_E4M3 else torch.bfloat16


def _per_rank_batch(
    *, rank: int, world_size: int, specs: list[dict], warmup: int, rep: int
) -> list[dict] | None:
    np, torch, dist, get_tp_group, shardq, direct_cp_enabled = _imports()
    group = get_tp_group()
    if group.world_size != world_size or group.rank_in_group != rank:
        raise KernelLaunchFailed("vLLM TP group does not match the spawned ranks")
    if not direct_cp_enabled(group, torch.bfloat16, None):
        raise KernelLaunchFailed("the TP group has no symmetric-memory fabric")
    device = torch.device("cuda", torch.cuda.current_device())

    # One workspace per (heads, q dim, q dtype, out dim) family sized for its largest step.
    families: dict[tuple, list[int]] = defaultdict(list)
    for index, spec in enumerate(specs):
        families[(int(spec["num_heads"]), int(spec["head_dim"]), spec["dtype"],
                  spec["direction"])].append(index)

    results: list[dict | None] = [None] * len(specs)
    for (heads, dim, dtype_value, direction), indices in families.items():
        dtype = _torch_dtype(torch, dtype_value)
        max_tokens = max(int(specs[i]["num_tokens"]) for i in indices)
        max_local = -(-max_tokens // world_size) + 2
        q_shape = (max_local, world_size * heads, dim)
        out_shape = (max_tokens, heads, dim)
        shardq._workspace = shardq.ShardQWorkspace(
            group.device_group, device, 1, q=(q_shape, dtype), out=(out_shape, dtype)
        )
        for index in indices:
            tokens = int(specs[index]["num_tokens"])
            token_ids, per_rank = shardq.zigzag_split(np.array([tokens]), world_size)
            metadata = shardq.ShardQMetadata(
                num_decode_tokens=0,
                num_tokens_per_rank=per_rank.tolist(),
                token_ids=torch.from_numpy(token_ids.astype(np.int64)).to(device),
                rank_offsets=torch.from_numpy(
                    np.r_[0, np.cumsum(per_rank)].astype(np.int64)
                ).to(device),
            )
            num_local = int(per_rank[rank])
            if direction == "shard_q":
                q = torch.full((tokens, heads, dim), float(rank + 1), device=device).to(dtype)
                _, local = shardq.shard_q(q, metadata)
                torch.cuda.synchronize()
                expect = torch.arange(1, world_size + 1, device=device, dtype=torch.float32)
                expect = expect.repeat_interleave(heads)[None, :, None].expand(num_local, -1, dim)
                if not torch.equal(local.float(), expect):
                    raise KernelLaunchFailed("ShardQ shard_q failed the rank-tag check")

                def launch(q=q, metadata=metadata):
                    shardq.shard_q(q, metadata)

                moved = (tokens - int(per_rank[rank])) * heads * dim * q.element_size()
            else:
                out = torch.full(
                    (num_local, world_size * heads, dim), float(rank + 1), device=device
                ).to(dtype)
                restored = shardq.unshard_out(out, metadata, None)
                torch.cuda.synchronize()
                owner = torch.empty(tokens, device=device, dtype=torch.float32)
                for peer in range(world_size):
                    start = int(metadata.rank_offsets[peer])
                    owner[metadata.token_ids[start : start + int(per_rank[peer])]] = peer + 1
                if not torch.equal(restored.float(), owner[:, None, None].expand(-1, heads, dim)):
                    raise KernelLaunchFailed("ShardQ unshard_out failed the rank-tag check")

                def launch(out=out, metadata=metadata):
                    shardq.unshard_out(out, metadata, None)

                moved = num_local * (world_size - 1) * heads * dim * out.element_size()
            launch()
            torch.cuda.synchronize()
            rank_time_ms = _time_launch(torch, dist, launch, warmup, rep)
            rank_times = _gather_rank_times(dist, rank, world_size, rank_time_ms)
            if rank == 0:
                results[index] = _metric_payload(rank_times, moved, moved)
        torch.cuda.synchronize()
        group.barrier()
        shardq._workspace = None
    if rank != 0:
        return None
    return list(results)  # type: ignore[arg-type]
