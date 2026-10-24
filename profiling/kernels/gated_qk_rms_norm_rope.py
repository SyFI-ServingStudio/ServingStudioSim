"""Per-head QK RMSNorm + partial RoPE of a gated attention whose gate stays in place."""

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

KIND = "gated_qk_rms_norm_rope"


@dataclass(frozen=True)
class GatedQkRmsNormRopeArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Tokens whose query and key heads are prepared.")
    num_q_heads: int = arg(
        unit="heads",
        doc="Query heads per token; each sits beside its gate head in the projection row.",
    )
    num_kv_heads: int = arg(unit="heads", doc="Key heads per token.")
    head_dim: int = arg(unit="elements", doc="Width of each head.")
    rotary_dim: int = arg(unit="elements", doc="Leading elements of each head that are rotated.")
    mrope: bool = arg(
        doc=(
            "Whether positions are multimodal (three rows: temporal, height, width) "
            "rather than one row; each rotated pair then loads its section's position."
        )
    )
    dtype: DType = arg(doc="Element type of the projections and outputs.")


DOC = KernelDoc(
    title="Gated QK RMSNorm + RoPE",
    summary=(
        "RMS-normalize every query and key head of a gated attention and rotate its "
        "leading rotary_dim elements, leaving the output gate where the projection wrote it."
    ),
    description=(
        "A gated attention projects each query head together with an output-gate head, "
        "so a projection row is [q_0 | gate_0 | q_1 | gate_1 | ...]. This kernel reads "
        "the query halves of those rows and a separate key projection, RMS-normalizes "
        "each head over head_dim, applies NeoX rotate-half RoPE to the first rotary_dim "
        "elements, and writes contiguous query and key tensors. The gate halves are "
        "neither read nor copied: the later output-gate multiply (attn_output_gate) reads "
        "them from the projection row. That is the difference from "
        "qk_rms_norm_rope_gate, which also writes a contiguous gate tensor. One program "
        "handles one head of one token. Value heads are not touched."
    ),
    category="Attention",
    subcategory="MHA / GQA",
    formula=(
        "q, k ← x / √(mean(x²) + ε) · weight per head; RoPE on [0, rotary_dim)",
        "GB/s = 4 · num_tokens · head_dim · (num_q_heads + num_kv_heads) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} A guard call is compared with an independent PyTorch reference "
        "before capture; only the _attn_prep launch is counted."
    ),
    caveats=(
        "GB/s counts the query and key heads read and written once; the rotated halves' "
        "reload and the position and inverse-frequency loads hit cache.",
        "tensorfold_triton computes cos and sin in fp32 from the position and an fp32 "
        "inverse-frequency vector inside the kernel; there is no cos/sin table.",
    ),
    reference="profiling.runners.attention.gated_qk_rms_norm_rope_reference",
)


# TensorFold perf-study 6469730a, tensorfold/families/qwen3_5/cuda/glue.py
# `attn_prep`: the Triton `_attn_prep` on grid (tokens, q heads + kv heads),
# two warps a program. It casts to tl.bfloat16, so bf16 only, and is plain
# Triton (no TensorFold CUDA extension); Triton's bf16 needs sm_80.
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
            module_name="profiling.runners.attention.gated_qk_rms_norm_rope_tensorfold_triton",
            function_name="profile_gated_qk_rms_norm_rope_tensorfold_triton",
        ),
        table_name=KIND,
        args_schema=GatedQkRmsNormRopeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="tensorfold_env",
        doc=BackendDoc(
            summary=(
                "TensorFold's Triton _attn_prep: one launch, a program per token and "
                "query-or-key head, reading queries from the [q | gate] projection rows."
            ),
        ),
    )
)

__all__ = ["KIND", "GatedQkRmsNormRopeArgs"]
