"""DeepSeek V4 fused Q normalization/RoPE and packed KV insert."""

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

KIND = "deepseek_v4_qnorm_rope_kv_insert"


@dataclass(frozen=True)
class DeepseekV4QnormRopeKvInsertArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Query and KV rows processed together.")
    num_insert_tokens: int = arg(unit="tokens", doc="Rows inserted into the packed KV cache.")
    num_heads: int = arg(unit="heads", doc="Query heads per token.")
    padded_heads: int = arg(unit="heads", doc="Query heads in the padded output.")
    head_dim: int = arg(unit="elements", doc="Elements per query head and KV row.")
    rope_dim: int = arg(unit="elements", doc="Trailing elements rotated by RoPE.")
    block_size: int = arg(unit="tokens", doc="Token capacity of each packed cache block.")
    rms_eps: float = arg(
        unit="unitless",
        doc="Constant added to each query mean square before RMS normalization.",
    )
    input_dtype: DType = arg(doc="Element type of the query and KV inputs.")
    cache_dtype: str = arg(doc="Format of the packed MLA cache.")
    cache_layout: str = arg(doc="Ordering of data and scales within each cache block.")
    scale_format: str = arg(doc="Format of the cache quantization scales.")


DOC = KernelDoc(
    title="DeepSeek V4 query norm, RoPE and KV insert",
    summary=(
        "Normalize and rotate queries while rotating and inserting mixed "
        "FP8/bf16 KV rows into a packed MLA cache."
    ),
    description=(
        "Between DeepSeek V4's projections and attention, one CUDA call "
        "prepares both sides. Each query head is RMS-normalized, RoPE is "
        "applied to its trailing rope_dim elements, and it is written into a "
        "head-padded output. Each KV row gets RoPE, its first 448 elements are "
        "quantized to FP8 while the rotary part stays BF16, and only the last "
        "num_insert_tokens rows are written to the packed cache. Inputs are "
        "seeded BF16 values with consecutive positions and cache slots."
    ),
    category="Attention",
    subcategory="MLA",
    formula=(
        "q_out = RoPE(q / √(mean(q²) + rms_eps))",
        "KV cache receives 448 FP8 bytes, 128 bf16 bytes and 8 scale bytes per inserted row",
        "GB/s = (2 · num_tokens · (num_heads · head_dim + head_dim + "
        "padded_heads · head_dim) + 8 · (num_tokens + num_insert_tokens) + "
        "4 · num_tokens · rope_dim + 584 · num_insert_tokens) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "Three warm-up calls run first; the call rewrites the same preallocated"
        " cache each time."
    ),
    caveats=(
        "GB/s counts logical input, output, position, RoPE-table and "
        "inserted-cache bytes, not physical memory transactions.",
    ),
    # The runner checks sampled outputs with PyTorch, but has no separate full reference.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.deepseek_v4_qnorm_rope_kv_insert_vllm_cuda",
            function_name="profile_deepseek_v4_qnorm_rope_kv_insert_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=DeepseekV4QnormRopeKvInsertArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert CUDA "
                "call writes mixed FP8/bf16 MLA cache blocks."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu",
        ),
    )
)

__all__ = ["DeepseekV4QnormRopeKvInsertArgs", "KIND"]
