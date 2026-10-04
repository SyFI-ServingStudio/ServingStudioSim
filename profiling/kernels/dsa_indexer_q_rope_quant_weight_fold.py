"""Fused DSA indexer-query RoPE and FP8 quantization, folding the scales into the head weights."""

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

KIND = "dsa_indexer_q_rope_quant_weight_fold"


@dataclass(frozen=True)
class DsaIndexerQRopeQuantWeightFoldArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Indexer query rows processed together.")
    num_heads: int = arg(unit="heads", doc="Indexer query heads per token.")
    head_dim: int = arg(unit="elements", doc="Elements in each indexer query head.")
    rope_dim: int = arg(unit="elements", doc="Trailing query elements rotated by RoPE.")
    max_model_len: int = arg(unit="tokens", doc="Maximum position used to size the RoPE table.")
    max_num_batched_tokens: int = arg(
        unit="tokens", doc="Maximum query rows admitted to one batch."
    )
    index_weights_softmax_scale: float = arg(
        unit="multiplier", doc="Scale folded into each output head weight for query width."
    )
    index_weights_head_scale: float = arg(
        unit="multiplier", doc="Scale folded into each output head weight for head count."
    )
    fp8_max: float = arg(
        unit="unitless", doc="Largest magnitude used when quantizing each query head."
    )
    scale_epsilon: float = arg(
        unit="unitless",
        doc="Minimum head magnitude used to calculate the FP8 scale.",
    )
    positions_dtype: str = arg(doc="Element type of token positions.")
    q_dtype: DType = arg(doc="Element type of the unrotated indexer queries.")
    rope_dtype: DType = arg(doc="Element type of the RoPE table.")
    weight_dtype: DType = arg(doc="Element type of input indexer head weights.")
    q_output_dtype: DType = arg(doc="Element type of rotated, quantized indexer queries.")
    weight_output_dtype: DType = arg(doc="Element type of output indexer head weights.")
    rope_style: str = arg(doc="Position and feature layout used by RoPE.")
    quant_mode: str = arg(doc="FP8 scaling scheme, including how scales enter the output weights.")


DOC = KernelDoc(
    title="Indexer query RoPE and FP8 quantization, scales in weights",
    summary="Rotate and quantize indexer queries while folding their FP8 scales into head weights.",
    description=(
        "A first step of the DSA indexer: each query head has RoPE applied to "
        "its trailing rope_dim elements and is "
        "quantized to FP8 with a power-of-two scale per token and head. That "
        "scale, together with both indexer scaling factors, is folded into the "
        "head's output weight, so the logits come out right without "
        "dequantizing. Positions are spread across max_model_len; queries and "
        "weights are deterministic."
    ),
    category="Attention",
    subcategory="DSA",
    formula=(
        "q_rot = RoPE(q), over the trailing rope_dim elements",
        "scale = 2^⌈log₂(max(max|q_rot|, scale_epsilon) / fp8_max)⌉, per token and head",
        "q_fp8 = q_rot / scale; weight_out = weight · scale · "
        "index_weights_softmax_scale · index_weights_head_scale",
        "GB/s = (8 · num_tokens + 2 · num_tokens · 64 · 128 + "
        "4 · num_tokens · 64 + 2 · num_tokens · 64 + "
        "num_tokens · 64 · 128 + 4 · num_tokens · 64) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "The output is checked against a PyTorch computation before timing; "
        "every launch of the CuTe DSL call is counted."
    ),
    caveats=(
        "GB/s counts logical query, position, weight and output bytes, not RoPE-table reads.",
    ),
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cutedsl_fp8",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_indexer_q_rope_quant_weight_fold_cutedsl",
            function_name="profile_dsa_indexer_q_rope_quant_weight_fold_cutedsl",
        ),
        table_name=KIND,
        args_schema=DsaIndexerQRopeQuantWeightFoldArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 9),
        ),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's fused_indexer_q_rope_quant_fp8_cutedsl call combines "
                "RoPE, per-head FP8 quantization and weight scaling."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py",
        ),
    )
)

__all__ = ["DsaIndexerQRopeQuantWeightFoldArgs", "KIND"]
