"""Residual-add RMSNorm kernel kind.

This module owns the ``residual_rms_norm`` wire string, its Args schema, the
multi-launch Torch semantic backend, the production-aligned fused vLLM CUDA
backend, and TensorFold's Triton kernel.

Both runners are referenced lazily so importing the registry does not import
Torch, vLLM, or CUDA runtime code.
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

KIND: str = "residual_rms_norm"


@dataclass(frozen=True)
class ResidualRmsNormArgs(KernelArgs):
    m: int = arg(unit="tokens", doc="Tokens in the batch.")
    hidden: int = arg(unit="elements", doc="Hidden size of the model.")
    dtype: DType = arg(doc="Element type of x, residual and weight.")


DOC = KernelDoc(
    title="Residual add + RMSNorm",
    summary=("Add the residual stream, then normalize: the fused step between transformer blocks."),
    description=(
        "Before each attention and MLP block, the model adds the previous block's "
        "output to the residual stream and RMS-normalizes the sum. vLLM fuses both "
        "into one kernel that writes the normalized output over the input and the "
        "new residual over the old one. The torch backend runs the same math as "
        "separate launches."
    ),
    category="Normalization",
    subcategory="RMSNorm",
    formula=(
        "s = x + residual",
        "y = s / √(mean(s²) + ε) · weight, with ε = 1e-5",
        "returns (y, s)",
        "GB/s = (4·m·hidden + hidden) · bytes per element / time",
    ),
    default_metric="memory_bandwidth_gbps",
    caveats=(
        "The reference, which the torch backend times, computes the sum and the RMS "
        "in fp32 and rounds to the input dtype before the weight multiply.",
        "GB/s counts logical traffic: x, residual and weight read, y and s written. "
        "The torch backend's fp32 temporaries are not counted.",
        "vllm_cuda works in place, so it is timed on zero inputs with a weight of "
        "ones: repeated calls then cannot overflow fp16, and the launch and memory "
        "access are unchanged.",
        "tensorfold_triton is out of place (h and y are new tensors) and also writes "
        "fp32 sums of y over each 64-element group, hidden/64 floats a row (about 3% "
        "more bytes than one bf16 row), which GB/s does not count. It is timed with "
        "TensorFold's eps of 1e-6; eps does not change the launch.",
    ),
    method=(
        f"{CUPTI_METHOD} vllm_cuda counts only its fused_add_rms_norm_kernel launch and "
        "tensorfold_triton only its _add_rmsnorm launch; torch counts every launch of the "
        "reference."
    ),
    reference="profiling.runners.norm.residual_rms_norm_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(compute=frozenset({DType.BF16, DType.FP16})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.norm.residual_rms_norm_torch",
            function_name="profile_residual_rms_norm",
        ),
        table_name=KIND,
        args_schema=ResidualRmsNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(summary="The PyTorch reference, timed as its separate launches."),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        # No capability rule: vLLM's fused_add_rms_norm is a generic CUDA kernel built
        # for every arch vLLM ships.
        supports=BackendSupport(
            compute=frozenset({DType.BF16, DType.FP16}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.norm.residual_rms_norm_vllm_cuda",
            function_name="profile_residual_rms_norm_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=ResidualRmsNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary="vLLM's fused_add_rms_norm CUDA kernel: one in-place launch.",
            url="https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/layernorm_kernels.cu",
        ),
        subprocess_env="vllm_env",
    )
)

# TensorFold perf-study 6469730a, tensorfold/families/qwen3_5/cuda/glue.py
# `add_rmsnorm(x, r, w, eps)`: the Triton `_add_rmsnorm`, one program a row,
# h = bf16(x + r), y = bf16(rmsnorm(h) * w), plus fp32 64-group sums of y. The
# kernel casts to tl.bfloat16, so bf16 only. It is plain Triton (no TensorFold
# CUDA extension and its 8.9 floor); Triton's bf16 needs sm_80, so that is the
# device floor. It runs in TensorFold's own venv.
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
            module_name="profiling.runners.norm.residual_rms_norm_tensorfold_triton",
            function_name="profile_residual_rms_norm_tensorfold_triton",
        ),
        table_name=KIND,
        args_schema=ResidualRmsNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="tensorfold_env",
        doc=BackendDoc(
            summary=(
                "TensorFold's Triton _add_rmsnorm: one out-of-place launch that also "
                "writes 64-element group sums of the output."
            ),
        ),
    )
)
