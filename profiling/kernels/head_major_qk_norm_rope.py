"""Per-head QK RMSNorm + full RoPE that also splits q, k and v into head-major tensors."""

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

KIND = "head_major_qk_norm_rope"


@dataclass(frozen=True)
class HeadMajorQkNormRopeArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Projection rows prepared.")
    num_q_heads: int = arg(
        unit="heads",
        doc="Query heads per row; 0 when the row is a key/value projection only.",
    )
    num_kv_heads: int = arg(unit="heads", doc="Key heads per row, and as many value heads.")
    head_dim: int = arg(unit="elements", doc="Width of each head; all of it is rotated.")
    dtype: DType = arg(doc="Element type of the projection and the outputs.")


DOC = KernelDoc(
    title="Head-major QK RMSNorm + RoPE",
    summary=(
        "RMS-normalize and rotate every query and key head of a [q | k | v] projection row "
        "and write q, k and v as head-major [heads, rows, head_dim] tensors."
    ),
    description=(
        "The DFlash2 drafter's attention reads its keys and values head-major, so one "
        "kernel takes the fused qkv projection row [q_0 .. q_H-1 | k_0 .. | v_0 ..], "
        "RMS-normalizes each query and key head over head_dim with its own weight, applies "
        "rotate-half RoPE to the whole head from per-row cos/sin tables, and copies the "
        "value heads untouched, writing all three as [heads, rows, head_dim]. The context "
        "path runs it on the key/value projection alone (no query heads). Unlike "
        "qk_rms_norm_rope_gate and gated_qk_rms_norm_rope there is no gate, the value heads "
        "are moved, and the outputs change from row-major to head-major."
    ),
    category="Attention",
    subcategory="MHA / GQA",
    formula=(
        "q, k ← RoPE(x / √(mean(x²) + ε) · weight) per head; v ← v",
        "GB/s = 4 · num_tokens · (num_q_heads + 2 · num_kv_heads) · head_dim / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} A guard call is compared with an independent PyTorch reference "
        "before capture; only the _prep_kernel launch is counted."
    ),
    caveats=(
        "GB/s counts the projection read and the outputs written; cos/sin tables are not counted.",
    ),
    reference="profiling.runners.attention.head_major_qk_norm_rope_reference",
)


# TensorFold perf-study 6469730a, tensorfold/families/qwen3_5/cuda/dflash2.py
# `DFlash2._prep(qkv, layer, cos, sin, heads)`: the Triton `_prep_kernel` on grid
# (rows, heads + 2 * kv_heads), one warp a program, tl.arange over head_dim / 2
# (a power of two). bf16 casts; plain Triton, sm_80.
TENSORFOLD_TRITON = "tensorfold_triton"

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend=TENSORFOLD_TRITON,
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 0),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.head_major_qk_norm_rope_tensorfold_triton",
            function_name="profile_head_major_qk_norm_rope_tensorfold_triton",
        ),
        table_name=KIND,
        args_schema=HeadMajorQkNormRopeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="tensorfold_env",
        doc=BackendDoc(
            summary=(
                "TensorFold's Triton _prep_kernel (DFlash2._prep): one launch, a program per "
                "row and head."
            ),
        ),
    )
)

__all__ = ["HeadMajorQkNormRopeArgs", "KIND"]
