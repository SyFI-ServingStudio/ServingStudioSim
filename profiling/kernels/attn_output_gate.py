"""Gated attention's output gate: attention output times sigmoid of the projection's gate heads."""

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

KIND = "attn_output_gate"


@dataclass(frozen=True)
class AttnOutputGateArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Tokens whose attention output is gated.")
    num_heads: int = arg(
        unit="heads",
        doc="Query heads per token; each has a gate head beside its query in the projection row.",
    )
    head_dim: int = arg(unit="elements", doc="Width of each head.")
    dtype: DType = arg(doc="Element type of the attention output, the gate and the result.")


DOC = KernelDoc(
    title="Attention output gate",
    summary=(
        "Multiply each attention output head by the sigmoid of its gate head, read in place "
        "from the [q | gate] projection rows."
    ),
    description=(
        "A gated attention (Qwen3.5/3.6 full-attention layers) projects each query head "
        "together with an output-gate head, so the projection row is "
        "[q_0 | gate_0 | q_1 | gate_1 | ...]. After attention, every output head is "
        "multiplied elementwise by sigmoid of its gate head, and the result is the o "
        "projection's input. Here that is its own launch reading the gate straight from the "
        "projection row (a head_dim-wide slice every 2 · head_dim elements); vLLM instead "
        "fuses the same multiply into its FP8 quantization (static_fp8_quant, gated)."
    ),
    category="Attention",
    subcategory="MHA / GQA",
    formula=(
        "y = o · σ(gate), computed in fp32, per head",
        "GB/s = 6 · num_tokens · num_heads · head_dim / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} A guard call is compared with an independent PyTorch reference "
        "before capture; only the _gate_mul launch is counted."
    ),
    caveats=(
        "GB/s counts the output, the gate and the result once each (bf16).",
        "tensorfold_triton also writes fp32 sums of the result over each 64-element group "
        "(head_dim / 64 floats a head, about 3% more bytes than the result), which GB/s "
        "does not count.",
    ),
    reference="profiling.runners.attention.attn_output_gate_reference",
)


# TensorFold perf-study 6469730a, tensorfold/families/qwen3_5/cuda/glue.py
# `gate_mul`: the Triton `_gate_mul` on grid (tokens, heads), two warps a
# program. It casts to tl.bfloat16, so bf16 only, and is plain Triton (no
# TensorFold CUDA extension); Triton's bf16 needs sm_80.
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
            module_name="profiling.runners.attention.attn_output_gate_tensorfold_triton",
            function_name="profile_attn_output_gate_tensorfold_triton",
        ),
        table_name=KIND,
        args_schema=AttnOutputGateArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="tensorfold_env",
        doc=BackendDoc(
            summary=(
                "TensorFold's Triton _gate_mul: one launch, a program per token and head, "
                "that also writes 64-element group sums of the result."
            ),
        ),
    )
)

__all__ = ["KIND", "AttnOutputGateArgs"]
