"""SiLU-and-mul activation of a dense gated MLP."""

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

KIND = "silu_and_mul"


@dataclass(frozen=True)
class SiluAndMulArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Rows entering the activation.")
    hidden_dim: int = arg(
        unit="elements", doc="Width of the gate half and of the up half of each row."
    )
    dtype: DType = arg(doc="Element type of the input and output.")


DOC = KernelDoc(
    title="SiLU and mul",
    summary="Multiply SiLU of the gate half of each row by its up half.",
    description=(
        "A gated MLP applies this activation to the output of its fused gate-up "
        "projection: each input row holds a gate half and an up half, each "
        "hidden_dim wide, and the output row is hidden_dim wide. Under "
        "torch.compile vLLM's native SiluAndMul becomes one Inductor pointwise "
        "kernel; a following quantization of the down projection's input stays a "
        "separate kernel when it is a custom op. Inputs are seeded random bf16 rows."
    ),
    category="Other",
    formula=(
        "y = SiLU(gate) · up, computed in fp32",
        "GB/s = 6 · num_tokens · hidden_dim / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} The block is compiled once with a symbolic token count, "
        "and its first call, at 8192 tokens or at the row's token count when larger, "
        "picks the launch config the row then uses, as vLLM's profile run at "
        "max_num_batched_tokens does. Every launch of the compiled block is counted. "
        "vllm_cuda: the CUDA custom op writes into a preallocated output and is one launch. "
        "tensorfold_triton counts its one _swiglu launch."
    ),
    caveats=(
        "GB/s counts the two input halves and the output once each.",
        "tensorfold_triton reads both halves in place from the stacked [gate | up] "
        "projection output and also writes fp32 sums of the output over each 64-element "
        "group (hidden_dim / 64 floats a row, about 3% more bytes than the output row), "
        "which GB/s does not count.",
    ),
    reference="profiling.runners.elementwise.silu_and_mul_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_inductor",
        # Inductor emits a portable Triton pointwise kernel; Triton lowers bf16 only on SM80+.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 0),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.elementwise.silu_and_mul_vllm_inductor",
            function_name="profile_silu_and_mul_vllm_inductor",
        ),
        table_name=KIND,
        args_schema=SiluAndMulArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary="vLLM's SiluAndMul.forward_native compiled by Inductor into one Triton kernel.",
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/activation.py",
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        # csrc/libtorch_stable/activation_kernels.cu act_and_mul_kernel: a plain vectorized
        # elementwise CUDA kernel with no architecture-specific path; vLLM itself
        # serves bf16 only on SM80+ (platforms/cuda.py check_if_supports_dtype).
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 0),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.elementwise.silu_and_mul_vllm_cuda",
            function_name="profile_silu_and_mul_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=SiluAndMulArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary="vLLM's SiluAndMul.forward_cuda: the torch.ops._C.silu_and_mul custom op.",
            url="https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/activation_kernels.cu",
        ),
    )
)

# TensorFold perf-study 6469730a, tensorfold/families/qwen3_5/cuda/glue.py
# `swiglu_stacked(gu)`: the Triton `_swiglu` on grid (tokens, cdiv(hidden_dim,
# 1024)), four warps a program, reading gate and up from one [gate | up] row
# (LD = 2 * hidden_dim). It casts to tl.bfloat16, so bf16 only; plain Triton,
# so Triton's sm_80 bf16 floor. It runs in TensorFold's own venv.
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
            module_name="profiling.runners.elementwise.silu_and_mul_tensorfold_triton",
            function_name="profile_silu_and_mul_tensorfold_triton",
        ),
        table_name=KIND,
        args_schema=SiluAndMulArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="tensorfold_env",
        doc=BackendDoc(
            summary=(
                "TensorFold's Triton _swiglu on the stacked [gate | up] projection output "
                "(swiglu_stacked), which also writes 64-element group sums of the output."
            ),
        ),
    )
)

__all__ = ["KIND", "SiluAndMulArgs"]
