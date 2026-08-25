"""DeepSeek V4 fused indexer-Q RoPE and FP8 quantization."""

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

KIND = "deepseek_v4_indexer_q_rope_quant"


@dataclass(frozen=True)
class DeepseekV4IndexerQRopeQuantArgs(KernelArgs):
    num_tokens: int
    num_heads: int
    head_dim: int
    rope_dim: int
    max_model_len: int
    max_num_batched_tokens: int
    index_weights_softmax_scale: float
    index_weights_head_scale: float
    fp8_max: float
    scale_epsilon: float
    positions_dtype: str
    q_dtype: DType
    rope_dtype: DType
    weight_dtype: DType
    q_output_dtype: DType
    weight_output_dtype: DType
    rope_style: str
    quant_mode: str


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cutedsl_fp8",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.deepseek_v4_indexer_q_rope_quant_cutedsl",
            function_name="profile_deepseek_v4_indexer_q_rope_quant_cutedsl",
        ),
        table_name=KIND,
        args_schema=DeepseekV4IndexerQRopeQuantArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        subprocess_env="vllm_env",
    )
)

__all__ = ["DeepseekV4IndexerQRopeQuantArgs", "KIND"]
