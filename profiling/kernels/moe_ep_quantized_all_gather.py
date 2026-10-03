"""Grouped all-gatherv of NVFP4 activations and top-k routing used by naive DP/EP MoE."""

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

KIND = "moe_ep_quantized_all_gather"


@dataclass(frozen=True)
class MoeEpQuantizedAllGatherArgs(KernelArgs):
    num_gpus: int = arg(unit="GPUs", doc="GPUs in the data-parallel group.")
    per_rank_tokens: tuple[int, ...] = arg(
        unit="tokens", doc="Token counts for each GPU, in rank order."
    )
    hidden_size: int = arg(unit="elements", doc="Elements in each token's hidden state.")
    top_k: int = arg(unit="experts", doc="Experts selected per token.")
    activation_dtype: DType = arg(
        doc="Element type of the gathered activations; nvfp4_e2m1 is measured."
    )
    fabric: str = arg(doc="Interconnect label; this backend requires nvlink.")


DOC = KernelDoc(
    title="MoE all-gather of quantized activations and top-k routing",
    summary="Gather NVFP4 activations, scales and top-k routing from every data-parallel GPU.",
    description=(
        "With data and expert parallelism, vLLM's allgather_reducescatter backend "
        "can quantize activations and select experts before dispatch. Each GPU then "
        "gathers four tensors from every GPU: the packed NVFP4 activations "
        "(hidden_size/2 bytes per token), their fp8 e4m3 block scales, one per 16 "
        "elements, the fp32 top-k weights and the int32 top-k expert IDs. GPU r "
        "contributes per_rank_tokens[r] tokens. The four gathers run in one NCCL "
        "group: all_gather when the counts are equal, all_gatherv otherwise."
    ),
    category="Communication",
    formula=(
        "token bytes = hidden_size/2 + hidden_size/16 + 8·top_k",
        "total bytes = sum(per_rank_tokens) · token bytes",
        "algbw = (total bytes / num_gpus) / time",
        "busbw = (total bytes − min(per_rank_tokens) · token bytes) / time",
    ),
    default_metric="time_ms",
    method=(
        "CUDA-event time around 100 repeated groups of four PyNCCL calls after "
        "50 warm-up groups and a barrier. Each GPU times its own stream; the "
        "largest mean per-call time across ranks is kept. Input tensors are "
        "built before timing."
    ),
    caveats=(
        "Measured with NCCL_NVLS_ENABLE=0: NVLS multicast is unavailable on the profiling host.",
        "The scales are gathered in the linear layout; the swizzle happens after "
        "the gather and is not part of this measurement.",
        "The four gathers are timed together, so their individual costs are not reported.",
    ),
    # The runner verifies outputs against Torch tensors but has no separate reference module.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_pynccl",
        supports=BackendSupport(
            compute=frozenset({DType.NVFP4_E2M1}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.comm.moe_ep_collectives_vllm_pynccl",
            function_name="profile_moe_ep_quantized_all_gather_batch",
        ),
        table_name=KIND,
        args_schema=MoeEpQuantizedAllGatherArgs,
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
                "vLLM's PyNCCL wrapper: the four gathers in one NCCL group, with "
                "all_gatherv for unequal token counts."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/all2all.py",
        ),
    )
)
