"""BF16 or FP32 matrix multiplication with FP32 output."""

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

KIND = "gemm_fp32_output"


@dataclass(frozen=True)
class GemmFp32OutputArgs(KernelArgs):
    m: int = arg(unit="tokens", doc="Token rows in the activation.")
    n: int = arg(unit="elements", doc="Output features in the weight.")
    k: int = arg(unit="elements", doc="Input features, the reduction dimension.")
    input_dtype: DType = arg(doc="Element type of the activation and weight; output is fp32.")


DOC = KernelDoc(
    title="FP32-output GEMM",
    summary="Multiply bf16 or fp32 activations and weights while writing an fp32 result.",
    description=(
        "Some call sites need an fp32 result. vLLM computes attention "
        "KV-compressor projections and MoE router logits from bf16 inputs with "
        "torch.mm and out_dtype=float32, and the DSA indexer's head weights "
        "from fp32 inputs with a plain torch.mm; both are measured by "
        "torch_cublas. SGLang's MoE router computes router logits in fp32, "
        "measured by sglang_router_auto. m is the tokens; n and k are the "
        "output and input features."
    ),
    category="GEMM",
    formula=(
        "C[m, n] = A[m, k] · B[n, k]ᵀ, with fp32 output",
        "TFLOPS = 2·m·n·k / time",
        "GB/s = [bytes(input_dtype)·(m·k + n·k) + 4·m·n] / time",
    ),
    default_metric="tflops",
    method=(
        f"{CUPTI_METHOD} torch_cublas counts every launch of the call, one kernel "
        "or a split-K pair. sglang_router_auto counts overlapping launches once, by "
        "the time the GPU is busy."
    ),
    caveats=(
        "The fp32 form runs at float32 matmul precision highest; the cast of "
        "the activation to fp32 is a separate launch, not timed here.",
        "Outside its dedicated shapes, sglang_router_auto calls linear_bf16_fp32, "
        "whose kernel (cuBLAS, HPC-Ops or DeepGEMM) is chosen by "
        "SGLANG_OPT_BF16_FP32_GEMM_ALGO in the profiling environment.",
    ),
    # The measured library calls have no separate PyTorch reference module.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch_cublas",
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.gemm_fp32_output_torch_cublas",
            function_name="profile_gemm_fp32_output_torch_cublas",
        ),
        table_name=KIND,
        args_schema=GemmFp32OutputArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        # No capability rule: torch.mm/cuBLAS runs on any CUDA GPU.
        supports=BackendSupport(
            compute=frozenset({DType.BF16, DType.FP32}),
        ),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "torch.mm with out_dtype=torch.float32 on bf16 inputs, as vLLM calls "
                "it for KV-compressor projections and router logits, or plain "
                "torch.mm on fp32 inputs."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v4/attention.py",
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_router_auto",
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.gemm_fp32_output_sglang_router",
            function_name="profile_gemm_fp32_output_sglang_router",
        ),
        table_name=KIND,
        args_schema=GemmFp32OutputArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
        ),
        subprocess_env="sglang_env",
        doc=BackendDoc(
            summary=(
                "SGLang's router dispatch: dsv3_router_gemm when m ≤ 16 (m ≤ 4 on "
                "SM100/103), k % 1024 == 0 and n is 256 or 384; otherwise "
                "linear_bf16_fp32."
            ),
            # SGLang main replaced dsv3_router_gemm (#34693); pin the last upstream revision.
            url="https://github.com/sgl-project/sglang/blob/ee462b5899c02db4e9d250f43c4c54d81253c4c6/python/sglang/kernels/ops/gemm/dsv3_router_gemm.py",
        ),
    )
)

__all__ = ["KIND"]


# MI300X elementwise byte-placeholder floor (GLM-5.3-Flash port, decision #37
# "mechanism B"): this kind carries a negligible predicted share of iteration
# time and has no MI300X-native backend yet, so instead of leaving it pinned to
# an NVIDIA-only backend -- which a real MI300X ``timing-predict`` rejects at
# ``BackendSupport.allows`` -- its MI300X cost is a closed-form analytic memory
# roofline (decision #49): the runner derives this kind's memory-bound byte
# footprint from its shape args and converts it to a time arithmetically,
# t = launch_latency + (read+write bytes) / effective MI300X HBM bandwidth
# (``profiling.runners.elementwise.floor``) -- no allocation, no rocprofv3, so a
# cell whose footprint exceeds 192 GB HBM yields a finite time instead of OOM.
# MI300X-gated and compute-agnostic,
# so every NVIDIA target -- B200 included -- stays byte-identical.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="elementwise_floor",
        supports=BackendSupport(compute=None, arch_targets=frozenset({"CDNA3"})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.elementwise.floor",
            function_name="profile_gemm_fp32_output_floor",
        ),
        table_name=KIND,
        args_schema=GemmFp32OutputArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_rocm_env",
        doc=BackendDoc(
            summary=(
                "Analytic memory-roofline floor: t = launch_latency + "
                "(read+write bytes) / effective MI300X HBM bandwidth, over this "
                "kind's shape-derived footprint. A derived negligible-share floor "
                "for the GLM-5.3-Flash MI300X port, not a measured native kernel."
            ),
        ),
    )
)
