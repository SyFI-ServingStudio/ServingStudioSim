"""Sigmoid-gated RMSNorm kernel kind: y = rmsnorm(x) * weight * sigmoid(g), per row.

Unlike ``gdn_gated_rms_norm`` (a SiLU gate, z * sigmoid(z)), the gate here is
sigmoid(g) alone. The ``vllm_fla_triton`` backend is FLA's
``rms_norm_gated(activation="sigmoid")`` from vLLM's third-party FLA ops (the
``FusedRMSNormGated.forward_cuda`` path), one ``layer_norm_gated_fwd_kernel``
launch.
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

KIND: str = "sigmoid_gated_rms_norm"


@dataclass(frozen=True)
class SigmoidGatedRmsNormArgs(KernelArgs):
    m: int = arg(unit="rows", doc="Rows normalized, e.g. tokens times heads for a per-head norm.")
    hidden: int = arg(unit="elements", doc="Features normalized within each row.")
    dtype: DType = arg(doc="Element type of the input, gate, weight and output.")


DOC = KernelDoc(
    title="Sigmoid-gated RMSNorm",
    summary="RMS-normalize each row, scale by a learned weight and multiply by sigmoid of a gate row.",
    description=(
        "Each row of x is RMS-normalized, scaled by a learned per-feature weight and "
        "multiplied elementwise by sigmoid(g), where g is a gate tensor of the same shape. "
        "Linear-attention output norms use it per head: m counts tokens times heads and "
        "hidden is the head dimension. Statistics and gating are computed in FP32 and the "
        "output is rounded to the input dtype."
    ),
    category="Normalization",
    subcategory="RMSNorm",
    formula=(
        "y = (x / √(mean(x²) + 1e−5) · weight) · sigmoid(g)",
        "FLOPs = 6·m·hidden + 2·m",
        "bytes = (3·m·hidden + hidden)·bytes(dtype)",
        "TFLOPS = FLOPs / time",
        "GB/s = bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "vllm_fla_triton counts the layer_norm_gated_fwd_kernel launch. Allocation and the "
        "output check against an FP32 reference run before timing."
    ),
    caveats=(
        "The FLA wrapper writes the output into x (out_dtype None), as the production call does.",
        "GB/s counts logical input and output bytes.",
    ),
    reference="profiling.runners.norm.sigmoid_gated_rms_norm_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_fla_triton",
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.norm.sigmoid_gated_rms_norm_vllm_fla_triton",
            function_name="profile_sigmoid_gated_rms_norm_vllm_fla_triton",
        ),
        table_name=KIND,
        args_schema=SigmoidGatedRmsNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "FLA's rms_norm_gated with activation='sigmoid' (vLLM's FusedRMSNormGated.forward_cuda): "
                "one Triton launch."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/third_party/flash_linear_attention/ops/kda.py",
        ),
        subprocess_env="vllm_env",
    )
)
