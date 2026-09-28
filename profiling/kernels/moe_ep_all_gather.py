"""Grouped hidden-state and router-logit all-gatherv used by naive DP/EP MoE."""

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

KIND = "moe_ep_all_gather"


@dataclass(frozen=True)
class MoeEpAllGatherArgs(KernelArgs):
    num_gpus: int = arg(unit="GPUs", doc="GPUs in the data-parallel group.")
    per_rank_tokens: tuple[int, ...] = arg(
        unit="tokens", doc="Token counts for each GPU, in rank order."
    )
    hidden_size: int = arg(unit="elements", doc="Elements in each token's hidden state.")
    num_experts: int = arg(unit="experts", doc="Router logits per token, one per expert.")
    hidden_dtype: DType = arg(doc="Element type of the hidden states; bf16 is measured.")
    router_dtype: DType = arg(doc="Element type of the router logits; fp32 is measured.")
    fabric: str = arg(doc="Interconnect label; this backend requires nvlink.")


DOC = KernelDoc(
    title="MoE all-gather of hidden states and router logits",
    summary="Gather hidden states and router logits from every data-parallel GPU.",
    description=(
        "Under vLLM's default all-to-all backend, allgather_reducescatter, a MoE "
        "layer with data and expert parallelism starts by gathering every GPU's "
        "hidden states and router logits, so each GPU can route all tokens. GPU r "
        "contributes per_rank_tokens[r] tokens and receives both full tensors. "
        "The two gathers run in one NCCL group: all_gather when the counts are "
        "equal, all_gatherv otherwise."
    ),
    category="Communication",
    formula=(
        "total bytes = sum(per_rank_tokens) · (2·hidden_size + 4·num_experts)",
        "algbw = (total bytes / num_gpus) / time",
        "busbw = (total bytes − min(per_rank_tokens)·(2·hidden_size + 4·num_experts)) / time",
    ),
    default_metric="time_ms",
    method=(
        "CUDA-event time around 100 repeated groups of two PyNCCL calls after "
        "50 warm-up groups and a barrier. Each GPU times its own stream; the "
        "largest mean per-call time across ranks is kept. Input tensors are "
        "built before timing."
    ),
    caveats=(
        "The measurement gathers only hidden states and router logits; "
        "the production call can include extra tensors.",
        "The two gathers are timed together, so their individual costs are not reported.",
    ),
    # The runner verifies outputs against Torch tensors but has no separate reference module.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_pynccl",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.comm.moe_ep_collectives_vllm_pynccl",
            function_name="profile_moe_ep_all_gather_batch",
        ),
        table_name=KIND,
        args_schema=MoeEpAllGatherArgs,
        metric_family=MetricFamily.COMM,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        gpu_count_fn=lambda spec: int(spec["num_gpus"]),
        list_native=True,
        doc=BackendDoc(
            summary=(
                "vLLM's PyNCCL wrapper: both gathers in one NCCL group, with "
                "all_gatherv for unequal token counts."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/pynccl.py",
        ),
    )
)
