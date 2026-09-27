"""Whole FlashInfer TRT-LLM NVFP4 MoE callable used on B200.

The operation starts with already-quantized activations and contains routing,
both expert GEMMs, SwiGLU, and finalize routing.  It is intentionally one L1
kind: SM100 uses PDL between several physical launches, so summing independently
profiled stages would double-count overlap and could select different tactics.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import CUPTI_METHOD, BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND = "nvfp4_fused_moe"


@dataclass(frozen=True)
class Nvfp4FusedMoeArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Tokens entering the routed MoE layer.")
    hidden_size: int = arg(unit="elements", doc="Width of each token's hidden state.")
    intermediate_size: int = arg(unit="elements", doc="Width of each expert's intermediate state.")
    num_experts: int = arg(unit="experts", doc="Routed experts across all GPUs.")
    num_local_experts: int = arg(
        unit="experts", doc="Routed experts whose weights reside on this GPU."
    )
    top_k: int = arg(unit="experts", doc="Experts selected for each token.")
    input_dtype: DType = arg(doc="Element type before the hidden states are quantized to NVFP4.")
    weight_format: DType = arg(
        doc=(
            "Packed expert weight format, and the tensor-core precision: the call"
            " quantizes activations to it, so both GEMM operands are NVFP4."
            " This backend requires nvfp4_e2m1."
        )
    )
    group_size: int = arg(unit="elements", doc="Elements sharing one NVFP4 scale.")
    routing_method: str = arg(doc="Router selection rule; this backend accepts minimax2.")
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
    title="NVFP4 fused MoE",
    summary="Route NVFP4 tokens through packed expert weights and produce their MoE outputs.",
    description=(
        "A routed MoE layer runs as this one FlashInfer call on hidden states"
        " already quantized to NVFP4: routing, the gate-up projection, SwiGLU "
        "and the down projection. per_expert_batches gives the tokens routed to"
        " every expert across all GPUs, and the first num_local_experts are "
        "this GPU's. The vLLM backend also combines the expert outputs; the "
        "SGLang backend leaves the combine to a later kernel, "
        "moe_finalize_fuse_shared."
    ),
    category="MoE",
    subcategory="Expert compute",
    formula=(
        "local_rows = sum(per_expert_batches[:num_local_experts])",
        "active_experts = count(per_expert_batches[:num_local_experts] > 0)",
        "TFLOPS = 2·local_rows·(hidden_size·2·intermediate_size "
        "+ intermediate_size·hidden_size) / time",
        "output_rows = num_tokens for vLLM; local_rows for SGLang",
        "routing_bytes = 2·num_tokens·num_experts + 2·num_experts",
        "activation_bytes = local_rows·(hidden_size/2 + hidden_size/group_size)",
        "weight_bytes = active_experts·(intermediate_size·hidden_size "
        "+ intermediate_size·hidden_size/8 + hidden_size·intermediate_size/2 "
        "+ hidden_size·intermediate_size/group_size)",
        "GB/s = (routing_bytes + activation_bytes + weight_bytes "
        "+ 12·active_experts + 2·hidden_size·output_rows) / time",
    ),
    default_metric="time_ms",
    method=(
        f"{CUPTI_METHOD} "
        "Overlapping launches count once, by the time the GPU is busy. The vLLM"
        " backend autotunes the call first, as vLLM does; the SGLang backend "
        "does not, because SGLang tunes a different signature. Three warm-up "
        "calls follow."
    ),
    caveats=(
        "The two backends end at different points, so their times cover "
        "different work: vLLM includes the combine, SGLang does not.",
        "Activation quantization is outside the call, and the router logits are synthetic.",
        "GB/s counts logical traffic for the local rows and the experts that "
        "have rows; small side outputs are left out.",
    ),
    # No separate PyTorch reference module exists for this fused call.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_trtllm_sm100",
        supports=BackendSupport(
            compute=frozenset({DType.NVFP4_E2M1}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.nvfp4_fused_moe",
            function_name="profile_nvfp4_fused_moe_sm100",
        ),
        table_name=KIND,
        args_schema=Nvfp4FusedMoeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "FlashInfer trtllm_fp4_block_scale_moe through vLLM's NVFP4 path, "
                "including final expert combination."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py",
        ),
        subprocess_env="vllm_env",
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_trtllm_sm100_deferred_finalize",
        supports=BackendSupport(
            compute=frozenset({DType.NVFP4_E2M1}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.moe.nvfp4_fused_moe",
            function_name="profile_nvfp4_fused_moe_deferred_finalize_sm100",
        ),
        table_name=KIND,
        args_schema=Nvfp4FusedMoeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "FlashInfer trtllm_fp4_block_scale_moe through SGLang, leaving "
                "expert combination to a later call."
            ),
            url="https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py",
        ),
        subprocess_env="sglang_env",
    )
)
