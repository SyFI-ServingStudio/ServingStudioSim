"""Whole FlashInfer TRT-LLM BF16 MoE callable used by vLLM on SM100."""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import CUPTI_METHOD, EP_RANKS_BY_LOAD, BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND = "bf16_fused_moe"


@dataclass(frozen=True)
class Bf16FusedMoeArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Tokens entering the routed MoE layer.")
    hidden_size: int = arg(unit="elements", doc="Width of each token's hidden state.")
    intermediate_size: int = arg(unit="elements", doc="Width of each expert's intermediate state.")
    num_experts: int = arg(unit="experts", doc="Routed experts across all GPUs.")
    num_local_experts: int = arg(
        unit="experts", doc="Routed experts whose weights reside on this GPU."
    )
    top_k: int = arg(unit="experts", doc="Experts selected for each token.")
    dtype: DType = arg(doc="Element type of the hidden states and expert weights.")
    routing_method: str = arg(
        doc=(
            "Router rule, as FlashInfer's RoutingMethodType names it; the runner "
            "builds minimax2: sigmoid scores plus a selection bias, weights "
            "renormalized."
        )
    )
    n_group: int = arg(unit="groups", doc="Expert groups considered by the router.")
    topk_group: int = arg(unit="groups", doc="Expert groups retained before expert selection.")
    routed_scaling_numerator: int = arg(
        unit="parts", doc="Numerator of the routed output scaling factor."
    )
    routed_scaling_denominator: int = arg(
        unit="parts", doc="Denominator of the routed output scaling factor."
    )
    per_expert_batches: tuple[int, ...] = arg(
        unit="tokens", doc="Selected token count for each global expert, in expert order."
    )


DOC = KernelDoc(
    title="BF16 fused MoE",
    summary="Route tokens through BF16 experts and combine their gated MLP outputs.",
    description=(
        "An MoE layer runs its routed experts as this one FlashInfer call: "
        "routing, the gate-up projection, SwiGLU, the down "
        "projection and the weighted combine. per_expert_batches gives the tokens "
        "routed to every expert across all GPUs, and the first num_local_experts "
        "are this GPU's. The router logits are built so that routing selects "
        "exactly those counts."
    ),
    category="MoE",
    subcategory="Expert compute",
    formula=(
        "local_rows = sum(per_expert_batches[:num_local_experts])",
        "active_experts = count(per_expert_batches[:num_local_experts] > 0)",
        "TFLOPS = 2·local_rows·(hidden_size·2·intermediate_size "
        "+ intermediate_size·hidden_size) / time",
        "logical_bytes = 2·num_tokens·num_experts + 2·num_experts "
        "+ 2·local_rows·hidden_size + active_experts·(4·intermediate_size·hidden_size "
        "+ 2·hidden_size·intermediate_size) + 2·num_tokens·hidden_size",
        "GB/s = logical_bytes / time",
    ),
    default_metric="time_ms",
    method=(
        f"{CUPTI_METHOD} "
        "Overlapping launches count once, by the time the GPU is busy. Weight "
        "preparation and one validated call run before timing."
    ),
    caveats=(
        "GB/s counts logical traffic: routing inputs, the local expert rows, "
        "the weights of experts that have rows, and the output.",
        "The measured call always finalizes the output; vLLM can defer the "
        "combine to a later kernel.",
    ),
    # The PyTorch check is private to this measured runner, not a separate reference module.
    reference=None,
    view=EP_RANKS_BY_LOAD,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_trtllm_sm100",
        # TRT-LLM-gen cubins target the SM10x family; vLLM enables this kernel on any SM10x device
        # (TrtLlmBf16ExpertsBase: is_device_capability_family(100)).
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            sm_targets=frozenset({"sm_100f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.bf16_fused_moe",
            function_name="profile_bf16_fused_moe_sm100",
        ),
        table_name=KIND,
        args_schema=Bf16FusedMoeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "FlashInfer trtllm_bf16_moe on B200, with vLLM's weight layout: "
                "routing and both expert projections in one call."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/experts/trtllm_bf16_moe.py",
        ),
        subprocess_env="vllm_env",
    )
)
