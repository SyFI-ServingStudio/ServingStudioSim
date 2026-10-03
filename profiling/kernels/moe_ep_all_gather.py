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
    hidden_dtype: DType = arg(
        doc=(
            "Element type of the gathered hidden states: bf16, or fp8_e4m3 with one "
            "fp32 scale per 128 elements gathered alongside."
        )
    )
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
        "With block-FP8 experts the GPUs first quantize their tokens, so the hidden "
        "states travel as fp8_e4m3 together with one fp32 scale per 128 elements. "
        "The gathers run in one NCCL group: all_gather when the counts are "
        "equal, all_gatherv otherwise."
    ),
    category="Communication",
    formula=(
        "token bytes = 2·hidden_size + 4·num_experts (bf16), "
        "hidden_size + 4·hidden_size/128 + 4·num_experts (fp8_e4m3)",
        "total bytes = sum(per_rank_tokens) · token bytes",
        "algbw = (total bytes / num_gpus) / time",
        "busbw = (total bytes − min(per_rank_tokens)·token bytes) / time",
    ),
    default_metric="time_ms",
    method=(
        "CUDA-event time around 100 repeated groups of PyNCCL calls after "
        "50 warm-up groups and a barrier. Each GPU times its own stream; the "
        "largest mean per-call time across ranks is kept. Input tensors are "
        "built before timing."
    ),
    caveats=(
        "Measured with NCCL_NVLS_ENABLE=0: NVLS multicast is unavailable on the profiling host.",
        "Only the tensors named above are gathered; a production call that adds "
        "other extra tensors moves more bytes.",
        "The gathers are timed together, so their individual costs are not reported.",
    ),
    # The runner verifies outputs against Torch tensors but has no separate reference module.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_pynccl",
        supports=BackendSupport(
            compute=frozenset({DType.BF16, DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA H200", "NVIDIA B200"}),
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
        # NCCL cannot bind NVLS multicast memory on the B200 host (CUDA error
        # 401 at communicator setup, NCCL 2.29.7; logs/20261003_4_dp_attn_ep/
        # debug/3416.out), and the collective then fails as "unhandled cuda
        # error". With NVLS off NCCL runs its NVLink ring/tree algorithms.
        # all_gatherv is grouped broadcasts, which never use NVLS anyway.
        worker_env=(("NCCL_NVLS_ENABLE", "0"),),
        gpu_count_fn=lambda spec: int(spec["num_gpus"]),
        list_native=True,
        doc=BackendDoc(
            summary=(
                "vLLM's PyNCCL wrapper: all gathers in one NCCL group, with "
                "all_gatherv for unequal token counts."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/pynccl.py",
        ),
    )
)
