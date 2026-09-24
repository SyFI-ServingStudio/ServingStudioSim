"""FlashInfer MNNVL standalone all-reduce runner.

Mirrors vLLM's ``FlashInferAllReduce.all_reduce`` on B200 when
``VLLM_FLASHINFER_ALLREDUCE_BACKEND=auto`` resolves to ``mnnvl``
(``vllm/distributed/device_communicators/flashinfer_all_reduce.py``
``_resolve_fi_ar_backend`` and ``all_reduce``):

- the workspace is ``create_allreduce_fusion_workspace(backend="mnnvl", ...)``
  over the TP group's gloo CPU group, sized by vLLM's per-world-size byte budget;
- the call is ``allreduce_fusion(pattern=kAllReduce, launch_with_pdl=True,
  trigger_completion_at_end=num_tokens > 16)`` with no ``use_oneshot``, so
  FlashInfer's AUTO rule picks one-shot iff
  ``num_tokens * hidden_dim * num_gpus * elem_size <= 1 MiB``
  (``flashinfer/comm/trtllm_mnnvl_ar.py`` ``MNNVL_ONE_SHOT_THRESHOLD``).

The timed boundary is the same CUDA-graph replay throughput as the TRT-LLM
backend, reduced to the slowest rank. The reduced output is checked against the
fp32 sum of every rank's input before timing.
"""

from __future__ import annotations

from collections import defaultdict

from profiling.db.args import DType
from profiling.runners.comm._batch import run_comm_batch
from profiling.runners.comm._launcher import TorchMpLauncher
from profiling.runners.comm.flashinfer_trtllm import _comm_payload, _time_cuda_graph
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import RunnerResult

# vLLM's standalone FlashInfer all-reduce byte budget on SM100, keyed by TP size
# (``FI_ALLREDUCE_FUSION_MAX_SIZE_MB[100]`` in
# ``vllm/compilation/passes/fusion/allreduce_rms_fusion.py``). SM100 has no
# ``FI_MNNVL_ALLREDUCE_MAX_SIZE_MB`` override, so this budget both sizes the
# workspace and bounds the tensors vLLM dispatches to FlashInfer; larger
# tensors fall through to another all-reduce backend.
SM100_MAX_SIZE_MB: dict[int, float] = {2: 64, 4: 32, 8: 1}
MIB = 1024 * 1024
# vLLM ``PDL_ADVANCE_LAUNCH_TOKENS``; FlashInfer ignores it on MNNVL but the
# call keeps production's arguments.
PDL_ADVANCE_LAUNCH_TOKENS = 16
# FlashInfer ``MNNVL_ONE_SHOT_THRESHOLD`` (bytes of num_tokens*hidden*tp*elem).
ONE_SHOT_THRESHOLD_BYTES = 64 * 1024 * 8 * 2
# Graph replay amortizes host launch cost over several back-to-back collectives.
OPERATIONS_PER_GRAPH = 10
# The family default (100 timed ops, about 1 ms at small T) leaves microsecond
# rows at +-30% run to run; time at least this many ops per shape instead.
MIN_TIMED_OPERATIONS = 1000

_ELEM_SIZE = {DType.BF16: 2, DType.FP16: 2, DType.FP32: 4}


def max_workspace_bytes(num_gpus: int) -> int:
    return int(SM100_MAX_SIZE_MB[num_gpus] * MIB)


def uses_oneshot(num_gpus: int, num_tokens: int, hidden_dim: int, dtype: DType) -> bool:
    """FlashInfer MNNVL AUTO strategy (documentation and tests for the Rust side)."""
    return num_tokens * hidden_dim * num_gpus * _ELEM_SIZE[dtype] <= ONE_SHOT_THRESHOLD_BYTES


def spec_error(spec: dict) -> str | None:
    """Reason vLLM would not route ``spec`` to the MNNVL all-reduce, else None."""
    num_gpus = int(spec["num_gpus"])
    if num_gpus not in SM100_MAX_SIZE_MB:
        return f"vLLM FlashInfer all-reduce supports TP 2/4/8 on SM100, got {num_gpus}"
    if spec["fabric"] != "nvlink":
        return f"FlashInfer MNNVL all-reduce needs NVLink multicast, got fabric={spec['fabric']!r}"
    dtype = DType.from_value(spec["dtype"])
    if dtype not in _ELEM_SIZE:
        return f"FlashInfer all-reduce does not support dtype {dtype.value}"
    num_tokens = int(spec["num_tokens"])
    hidden_dim = int(spec["hidden_dim"])
    if num_tokens < 1 or hidden_dim < 1:
        return f"num_tokens and hidden_dim must be positive, got {num_tokens}, {hidden_dim}"
    nbytes = num_tokens * hidden_dim * _ELEM_SIZE[dtype]
    budget = max_workspace_bytes(num_gpus)
    if nbytes > budget:
        return (
            f"{nbytes} bytes exceeds vLLM's TP{num_gpus} FlashInfer all-reduce "
            f"budget of {budget} bytes; production dispatches another backend"
        )
    return None


def profile_all_reduce_fusion_batch(kwargs_list: list[dict]) -> list[RunnerResult]:
    if not kwargs_list:
        return []
    results: list[RunnerResult | None] = [None] * len(kwargs_list)
    runnable: list[int] = []
    for index, spec in enumerate(kwargs_list):
        error = spec_error(spec)
        if error is None:
            runnable.append(index)
        else:
            results[index] = RunnerResult(error=error)
    if runnable:
        num_gpus = int(kwargs_list[runnable[0]]["num_gpus"])
        measured = run_comm_batch(
            TorchMpLauncher(num_gpus, backend="cpu:gloo,cuda:nccl"),
            _all_reduce_per_rank_batch,
            [kwargs_list[index] for index in runnable],
        )
        for index, result in zip(runnable, measured, strict=True):
            results[index] = result
    return [result for result in results if result is not None]


def _all_reduce_per_rank_batch(
    *,
    rank: int,
    world_size: int,
    specs: list[dict],
    warmup: int,
    rep: int,
) -> list[dict] | None:
    flashinfer_comm, torch, dist, torch_dist_backend = _imports()
    # vLLM passes the TP group's gloo CPU group to the workspace.
    cpu_group = dist.new_group(backend="gloo")
    by_workspace: dict[tuple[int, DType], list[tuple[int, dict]]] = defaultdict(list)
    for spec_index, spec in enumerate(specs):
        dtype = DType.from_value(spec["dtype"])
        by_workspace[(int(spec["hidden_dim"]), dtype)].append((spec_index, spec))

    ordered: list[dict | None] = [None] * len(specs)
    try:
        for (hidden_dim, dtype), indexed_specs in by_workspace.items():
            torch_dtype = dtype.torch()
            workspace = flashinfer_comm.create_allreduce_fusion_workspace(
                backend="mnnvl",
                world_size=world_size,
                rank=rank,
                max_token_num=max_workspace_bytes(world_size) // (hidden_dim * _ELEM_SIZE[dtype]),
                hidden_dim=hidden_dim,
                dtype=torch_dtype,
                comm_backend=torch_dist_backend(group=cpu_group),
                group=cpu_group,
            )
            try:
                if not getattr(workspace, "mc_ptr", 0):
                    raise RuntimeError("FlashInfer MNNVL multicast is unavailable")
                for spec_index, spec in indexed_specs:
                    ordered[spec_index] = _profile_one_shape(
                        flashinfer_comm=flashinfer_comm,
                        torch=torch,
                        dist=dist,
                        workspace=workspace,
                        rank=rank,
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
    if any(result is None for result in ordered):
        raise KernelLaunchFailed("missing rank-0 MNNVL all-reduce result")
    return ordered


def _imports():
    try:
        import flashinfer.comm as flashinfer_comm
        import torch
        import torch.distributed as dist
        from flashinfer.comm.mnnvl import TorchDistBackend
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "the vLLM fork environment with FlashInfer comm is required"
        ) from exc
    return flashinfer_comm, torch, dist, TorchDistBackend


def _profile_one_shape(
    *,
    flashinfer_comm,
    torch,
    dist,
    workspace,
    rank: int,
    world_size: int,
    spec: dict,
    torch_dtype,
    warmup: int,
    rep: int,
) -> dict:
    num_tokens = int(spec["num_tokens"])
    hidden_dim = int(spec["hidden_dim"])
    if not workspace.is_buffer_size_sufficient(
        tp_size=world_size,
        num_tokens=num_tokens,
        hidden_dim=hidden_dim,
        dtype=torch_dtype,
    ):
        raise ValueError(
            f"MNNVL workspace cannot hold num_tokens={num_tokens} hidden_dim={hidden_dim}"
        )
    generator = torch.Generator(device="cuda").manual_seed(1234 + rank)
    input_tensor = torch.randn(
        num_tokens, hidden_dim, dtype=torch_dtype, device="cuda", generator=generator
    )

    def launch():
        return flashinfer_comm.allreduce_fusion(
            input=input_tensor,
            workspace=workspace,
            pattern=flashinfer_comm.AllReduceFusionPattern.kAllReduce,
            launch_with_pdl=True,
            trigger_completion_at_end=num_tokens > PDL_ADVANCE_LAUNCH_TOKENS,
        )

    output_tensor = launch()
    torch.cuda.synchronize()
    _check_against_torch_sum(
        torch=torch,
        dist=dist,
        input_tensor=input_tensor,
        output_tensor=output_tensor,
        world_size=world_size,
    )

    time_ms = _time_cuda_graph(
        torch=torch,
        dist=dist,
        launch=launch,
        input_tensor=input_tensor,
        warmup=warmup,
        rep=max(rep, MIN_TIMED_OPERATIONS),
        operations_per_graph=OPERATIONS_PER_GRAPH,
    )
    return _comm_payload(
        time_ms=time_ms,
        message_size_bytes=input_tensor.numel() * input_tensor.element_size(),
        world_size=world_size,
    )


def _check_against_torch_sum(*, torch, dist, input_tensor, output_tensor, world_size: int) -> None:
    gathered = [torch.empty_like(input_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, input_tensor)
    reference = torch.stack(gathered).float().sum(dim=0)
    # One bf16 rounding of the fp32 sum, plus slack for a low-precision
    # accumulation order inside the kernel.
    tolerance = 2e-2 * reference.abs() + 3e-2
    excess = ((output_tensor.float() - reference).abs() - tolerance).max()
    worst = excess.reshape(1)
    dist.all_reduce(worst, op=dist.ReduceOp.MAX)
    if worst.item() > 0:
        raise AssertionError(
            f"MNNVL all-reduce differs from the torch rank sum by {worst.item():.4g} "
            "beyond tolerance"
        )
