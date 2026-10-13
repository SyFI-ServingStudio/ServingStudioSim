"""Profile the alignment vLLM fork's ``expand_qsa_block_indices``."""

from __future__ import annotations

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention.sparse_selection import random_group_selection
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "sparse_index_expand:vllm_triton"


def profile_sparse_index_expand_vllm_triton(
    num_rows: int, token_topk: int, compress_ratio: int
) -> ComputeMetrics:
    for name, value in (
        ("num_rows", num_rows),
        ("token_topk", token_topk),
        ("compress_ratio", compress_ratio),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive int, got {value!r}")
    if token_topk % compress_ratio:
        raise ValueError("token_topk must be a multiple of compress_ratio")
    try:
        import torch
        from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import expand_qsa_block_indices
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the alignment vLLM fork") from exc

    try:
        device = torch.device("cuda", torch.cuda.current_device())
        # Full selections with an open tail of ratio - 1 tokens.
        context_len = 4 * token_topk + compress_ratio - 1
        blocks, positions, visible = random_group_selection(
            torch,
            num_rows=num_rows,
            context_len=context_len,
            token_topk=token_topk,
            compress_ratio=compress_ratio,
            device=device,
        )
        width = token_topk + compress_ratio - 1
        out = torch.empty((num_rows, width + 1), dtype=torch.int32, device=device)

        def launch() -> None:
            expand_qsa_block_indices(blocks, positions, visible, compress_ratio, token_topk, out)

        launch()
        torch.cuda.synchronize()
        row = num_rows - 1
        offsets = torch.arange(compress_ratio, device=device)
        expanded = (blocks[row].long()[:, None] * compress_ratio + offsets).reshape(-1)
        tail_start = (context_len // compress_ratio) * compress_ratio
        tail = torch.arange(tail_start, context_len, device=device)
        want = torch.cat((expanded, tail)).to(torch.int32)
        if not torch.equal(out[row, : want.numel()], want) or int(out[row, width]) != want.numel():
            raise KernelLaunchFailed(f"{_BACKEND} expansion disagrees with the reference")

        time_ms = Timer.cupti(launch, warmup=3, kernel_name=None)
        energy_j = Energy.perf(launch, warmup=3, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except (KernelLaunchFailed, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc

    traffic = num_rows * (4 * (token_topk // compress_ratio) + 4 * (width + 1) + 12)
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(traffic / (time_ms / 1000.0) / 1e9),
        energy_j=float(energy_j),
    )


__all__ = ["profile_sparse_index_expand_vllm_triton"]
