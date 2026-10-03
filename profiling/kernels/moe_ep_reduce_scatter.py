"""Variable-shard reduce-scatter used to combine naive DP/EP MoE outputs."""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND = "moe_ep_reduce_scatter"


@dataclass(frozen=True)
class MoeEpReduceScatterArgs(KernelArgs):
    num_gpus: int = arg(unit="GPUs", doc="GPUs in the data-parallel group.")
    per_rank_tokens: tuple[int, ...] = arg(
        unit="tokens", doc="Output token counts for each GPU, in rank order."
    )
    hidden_size: int = arg(unit="elements", doc="Elements in each token's hidden state.")
    dtype: DType = arg(doc="Element type of the hidden states; bf16 is measured.")
    fabric: str = arg(doc="Interconnect label; this backend requires nvlink.")


DOC = KernelDoc(
    title="MoE output reduce-scatter",
    summary="Sum expert outputs across GPUs and return each GPU's token shard.",
    description=(
        "Under vLLM's default all-to-all backend, allgather_reducescatter, a MoE "
        "layer with data and expert parallelism ends with a reduce-scatter. Every "
        "GPU holds its experts' partial outputs for all gathered tokens, and GPU r "
        "gets back the sum for its own per_rank_tokens[r] tokens. Equal shards use "
        "reduce_scatter, unequal shards reduce_scatterv."
    ),
    category="Communication",
    formula=(
        "largest shard bytes = max(per_rank_tokens) · hidden_size · 2",
        "algbw = largest shard bytes / time",
        "busbw = (num_gpus − 1) · largest shard bytes / time",
    ),
    default_metric="time_ms",
    method=(
        "CUDA-event time around 100 repeated PyNCCL reduce-scatter calls after "
        "50 warm-up calls and a barrier. Each GPU times its own stream; the "
        "largest mean per-call time across ranks is kept. Input tensors are "
        "built before timing."
    ),
    caveats=(
        "The reported bandwidth uses the largest output shard, rather than "
        "the total input tensor size.",
    ),
    # The runner checks against a Torch sum but has no separate reference module.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_pynccl",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200", "NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.comm.moe_ep_collectives_vllm_pynccl",
            function_name="profile_moe_ep_reduce_scatter_batch",
        ),
        table_name=KIND,
        args_schema=MoeEpReduceScatterArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        gpu_count_fn=lambda spec: int(spec["num_gpus"]),
        list_native=True,
        doc=BackendDoc(
            summary=(
                "vLLM's PyNCCL wrapper: reduce_scatter, or reduce_scatterv for unequal shards."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/pynccl.py",
        ),
    )
)
