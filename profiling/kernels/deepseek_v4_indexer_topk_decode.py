"""DeepSeek V4 indexer decode persistent top-k operation."""

from profiling.db.args import DType
from profiling.db.doc import CUPTI_METHOD, BackendDoc, KernelDoc
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)
from profiling.kernels.dsa_persistent_topk_decode import DsaPersistentTopkDecodeArgs

KIND = "deepseek_v4_indexer_topk_decode"


DOC = KernelDoc(
    title="DeepSeek V4 decode indexer top-k",
    summary="Select up to 512 compressed-key positions per decode request from indexer logits.",
    description=(
        "After scoring, DeepSeek V4's sparse-attention indexer keeps the 512 "
        "highest-scoring compressed keys per decode request; sparse attention "
        "reads only those. Request b has max(0, context_len − b) valid logits. "
        "Every FP32 row is filled from one seeded random template, and the "
        "output and workspace are reused across calls."
    ),
    category="Attention",
    subcategory="DSA",
    formula=(
        "length(b) = max(0, context_len − b); valid_indices(b) = "
        "top-min(512, length(b))(logits(b, :length(b)))",
        "GB/s = (4 · Σ_b length(b) + 4 · batch_size + 4 · batch_size · 512) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "The output is checked before timing; every launch of vLLM's "
        "persistent_topk call is counted."
    ),
    caveats=(
        "A row shorter than 512 has only min(length, 512) valid indices, though"
        " the output still holds 512 per request.",
        "GB/s counts logical logits, lengths and output indices, not workspace "
        "traffic.",
    ),
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        runner_ref=RunnerRef(
            module_name=(
                "profiling.runners.attention.deepseek_v4_indexer_topk_decode_cuda"
            ),
            function_name="profile_deepseek_v4_indexer_topk_decode_cuda",
        ),
        table_name=KIND,
        args_schema=DsaPersistentTopkDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.FP32}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        subprocess_env="vllm_env",
        doc=BackendDoc(summary=(
                "vLLM's persistent_topk CUDA call selects up to 512 positions per "
                "request with a reusable workspace."
            ), url="https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/persistent_topk.cuh"),
    )
)

__all__ = ["KIND"]
