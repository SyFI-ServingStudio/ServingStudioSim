"""GEMM that accumulates a bf16 product into an existing fp32 matrix.

Wire string: ``"gemm_fp32_accumulate"`` -- matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/gemm_fp32_accumulate.rs``.

``C[m, n] += A[m, k] · B[n, k]ᵀ`` with bf16 A and B and an fp32 C that is read
and written. DeepInfer's DFlash head adds its Markov low-rank correction to
each draft step's logits this way (``models/mtp/dflash.py``:
``self.markov_w2(w1, out_acc=logits)``, a ``ReplicatedLinear`` that calls
``deep_gemm.bf16_gemm_nt(x, w, d=logits, c=logits)``); DeepGEMM sends it to
cuBLASLt with beta = 1 (the capture's ``nvjet_..._badd`` kernel).

Why not a backend of ``gemm_fp32_output``: that kind writes a fresh output;
here the output is also an input (beta = 1 reads all of C). Measured on B200 at
n=129280, k=256: 91.1 us accumulate vs 48.9 us plain store at m=384, 219.9 vs
109.9 us at m=1024 (equal at m=64), with cuBLASLt picking the ``badd`` twin of
the same tile (logs/20261120_2_ab).

Key: ``m`` (rows: requests per Markov step), ``n``, ``k``, ``input_dtype``.
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

KIND = "gemm_fp32_accumulate"


@dataclass(frozen=True)
class GemmFp32AccumulateArgs(KernelArgs):
    m: int = arg(unit="tokens", doc="Rows of the activation and of the accumulated matrix.")
    n: int = arg(unit="elements", doc="Output features in the weight.")
    k: int = arg(unit="elements", doc="Input features, the reduction dimension.")
    input_dtype: DType = arg(
        doc="Element type of the activation and weight; the accumulator is fp32."
    )


DOC = KernelDoc(
    title="FP32-accumulate GEMM",
    summary="Add a bf16 matrix product into an existing fp32 matrix in place.",
    description=(
        "DeepInfer's DFlash draft head adds a rank-k Markov correction to each draft "
        "step's fp32 logits: C += A·Bᵀ with the step's token embeddings as A and the "
        "vocabulary projection as B. m is the rows (requests), n the vocabulary, k the "
        "Markov rank."
    ),
    category="GEMM",
    formula=(
        "C[m, n] += A[m, k] · B[n, k]ᵀ, C fp32",
        "TFLOPS = 2·m·n·k / time",
        "GB/s = [bytes(input_dtype)·(m·k + n·k) + 8·m·n] / time",
    ),
    default_metric="tflops",
    method=CUPTI_METHOD + " Every launch of the call is counted.",
    caveats=(
        "C is contiguous; DeepInfer accumulates into every fifth row of the draft "
        "logits (a strided view), measured within 4% of contiguous.",
    ),
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="deepgemm_cublaslt",
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.gemm_fp32_accumulate_deepgemm",
            function_name="profile_gemm_fp32_accumulate_deepgemm",
        ),
        table_name=KIND,
        args_schema=GemmFp32AccumulateArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        # No capability rule: the plain-store/accumulate path goes to cuBLASLt on
        # any architecture (gemm.hpp bf16_gemm_nt checks cuBLASLt before arch).
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(
            summary=(
                "DeepGEMM bf16_gemm_nt(a, b, d, c=d) into an fp32 d, as DeepInfer's "
                "ReplicatedLinear(out_acc=...) calls it; cuBLASLt with beta = 1."
            ),
            url="https://github.com/deepseek-ai/DeepGEMM",
        ),
    )
)

__all__ = ["KIND", "GemmFp32AccumulateArgs"]
