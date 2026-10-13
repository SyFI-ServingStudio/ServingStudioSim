"""Fused per-head QK RMSNorm + partial RoPE + output-gate split of a gated attention."""

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

KIND = "qk_rms_norm_rope_gate"


@dataclass(frozen=True)
class QkRmsNormRopeGateArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Tokens whose query and key heads are prepared.")
    num_q_heads: int = arg(unit="heads", doc="Query heads per token (each with a gate head).")
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
    title="QK RMSNorm + RoPE + gate split",
    summary=(
        "RMS-normalize every query and key head, rotate its leading rotary_dim "
        "elements, and split the query projection's output gate off, in one launch."
    ),
    description=(
        "A gated attention projects each query head together with an output-gate "
        "head. vLLM's fused Triton kernel reads the query-and-gate and key slices of "
        "the QKV projection output in place, RMS-normalizes each head over head_dim, "
        "applies NeoX rotary embedding to the first rotary_dim elements, and writes "
        "contiguous query, key and gate tensors. One program handles one head of one "
        "token. Value heads are not touched."
    ),
    category="Attention",
    subcategory="MHA / GQA",
    formula=(
        "q, k ← x / √(mean(x²) + ε) · weight per head; RoPE on [0, rotary_dim)",
        "GB/s = 2 · num_tokens · head_dim · (4·num_q_heads + 2·num_kv_heads) / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} Three warm-up calls and a comparison with an independent "
        "PyTorch reference precede capture; every launch of the call is counted."
    ),
    caveats=(
        "GB/s counts the query, gate and key heads read and written once; the "
        "cos/sin rows and the rotated halves' reload hit cache.",
        "The cos/sin table length and the position values do not change the time "
        "(measured: under 1% between 4K and 1M rows, sequential or random positions).",
    ),
    reference="profiling.runners.attention.qk_rms_norm_rope_gate_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        # A portable Triton kernel; Triton lowers bf16 only on SM80+.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 0),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.qk_rms_norm_rope_gate_vllm_triton",
            function_name="profile_qk_rms_norm_rope_gate_vllm_triton",
        ),
        table_name=KIND,
        args_schema=QkRmsNormRopeGateArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary="vLLM's fused_qk_rmsnorm_rope_gate Triton kernel.",
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_qk_norm_rope.py",
        ),
    )
)

__all__ = ["KIND", "QkRmsNormRopeGateArgs"]
