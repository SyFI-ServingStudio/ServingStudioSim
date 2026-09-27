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
    summary="Multiply bf16 activations and weights while writing an fp32 result.",
    description=(
        "Two call sites need an fp32 result from bf16 inputs. vLLM computes "
        "attention KV-compressor projections with torch.mm and "
        "out_dtype=float32, measured by torch_cublas. SGLang's MoE router computes "
        "router logits in fp32, measured by sglang_router_auto. m is the tokens; "
        "n and k are the output and input features."
    ),
    category="GEMM",
    formula=(
        "C[m, n] = A[m, k] · B[n, k]ᵀ, with fp32 output",
        "TFLOPS = 2·m·n·k / time",
        "GB/s = [2·m·k + 2·n·k + 4·m·n] / time",
    ),
    default_metric="tflops",
    method=(
        f"{CUPTI_METHOD} torch_cublas counts every launch of the call, one kernel "
        "or a split-K pair. sglang_router_auto counts overlapping launches once, by "
        "the time the GPU is busy."
    ),
    caveats=(
        "torch_cublas is measured only at k = 4096 and n in {256, 512, 1024, 2048}, on H200.",
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
        supports=BackendSupport(
            compute=frozenset({DType.BF16, DType.FP32}),
            gpus=frozenset({"NVIDIA H200", "NVIDIA B200"}),
        ),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "torch.mm with out_dtype=torch.float32, the call vLLM makes for "
                "attention KV-compressor projections."
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
            gpus=frozenset({"NVIDIA B200"}),
        ),
        subprocess_env="sglang_env",
        doc=BackendDoc(
            summary=(
                "SGLang's router dispatch: dsv3_router_gemm when m ≤ 16 (m ≤ 4 on "
                "SM100/103), k % 1024 == 0 and n is 256 or 384; otherwise "
                "linear_bf16_fp32."
            ),
            url="https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/gemm/dsv3_router_gemm.py",
        ),
    )
)

__all__ = ["KIND"]
