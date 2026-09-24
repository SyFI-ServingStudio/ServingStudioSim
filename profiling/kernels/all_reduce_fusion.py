"""FlashInfer shape-aware standalone all-reduce kernel kind.

Backends: ``flashinfer_trtllm`` (TRT-LLM IPC workspace) and ``flashinfer_mnnvl``
(NVLink-multicast workspace, vLLM's default on B200).

Unlike the generic byte-keyed ``all_reduce`` kind, vLLM's FlashInfer path
requires a contiguous ``[num_tokens, hidden_dim]`` tensor and changes its PDL
completion policy at 16 tokens; on MNNVL, FlashInfer also switches from
one-shot to two-shot above ``num_tokens * hidden_dim * num_gpus * 2 = 1 MiB``.
Those two dimensions therefore belong in this kernel's cache identity.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "all_reduce_fusion"


@dataclass(frozen=True)
class AllReduceFusionArgs(KernelArgs):
    num_gpus: int
    num_tokens: int
    hidden_dim: int
    dtype: DType
    fabric: str


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_trtllm",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.comm.flashinfer_trtllm",
            function_name="profile_all_reduce_fusion_batch",
        ),
        table_name=KIND,
        args_schema=AllReduceFusionArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="flashinfer_pip_env",
        gpu_count_fn=lambda spec: int(spec["num_gpus"]),
        list_native=True,
    )
)

# vLLM's default FlashInfer all-reduce backend on single-node B200
# (``VLLM_FLASHINFER_ALLREDUCE_BACKEND=auto`` -> mnnvl). It runs in the vLLM
# fork venv because production pins FlashInfer 0.6.18 there; the project venv
# ships 0.6.11, older than the mnnvl CUDA-graph fix vLLM relies on (>= 0.6.12).
# One-shot vs two-shot is FlashInfer's AUTO rule inside the call, not an arg.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_mnnvl",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.comm.flashinfer_mnnvl",
            function_name="profile_all_reduce_fusion_batch",
        ),
        table_name=KIND,
        args_schema=AllReduceFusionArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_fork_env",
        gpu_count_fn=lambda spec: int(spec["num_gpus"]),
        list_native=True,
    )
)
