"""Fused gate|up GEMM + SwiGLU + NVFP4 quantization of the result.

Wire string ``"gemm_swiglu_quant"``: matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/gemm_swiglu_quant.rs`` and the generated
``get_gemm_swiglu_quant_times`` / ``count_missing_gemm_swiglu_quant`` facade.

One launch reads NVFP4 activation rows (m x k), multiplies them by the gate and
the up weight (each n x k, NVFP4), applies SiLU(gate) * up in fp32 on the
accumulators and writes the product as NVFP4 rows (m x n codes plus one e4m3
scale per 16 elements, under the next projection's static input scale): the
input of the down projection. TensorFold serves a checkpoint-math NVFP4 dense
MLP's prompt rows this way (``tensorfold.cuda.nvfp4.checkpoint.mlp_prompt`` ->
``gemm_gu_ck``; ``gemm_kernel`` with EPI 2 in ``cuda/nvfp4/gemm_ck.cu`` and
``swiglu4.cuh``).

Reuse verdict. ``single_gemm`` is one weight and a bf16 (or fp32) output: here
two weights share one launch and the output is not the GEMM result but its
SwiGLU, quantized, so neither its args (``n`` would have to mean 2n in and n
out) nor its output contract fits, and the epilogue's work (SiLU, per-block
amax, e4m3/e2m1 rounding in-register) is absent from every single_gemm row.
``silu_and_mul`` and ``clamped_swiglu`` take a bf16 (gate|up) tensor from memory
and write bf16; here the activation never exists in memory, so the fused cost is
not GEMM + activation. ``nvfp4_quant`` quantizes bf16 rows read from memory; here
quantization happens in the GEMM's epilogue. Pricing the launch as
single_gemm(2n) + silu_and_mul + nvfp4_quant would count two bf16 round trips
and two launches that never happen and miss the epilogue that does, so this is a
new kind. The quantization of its input (``nvfp4_quant``-like ``quant4``) and the
down projection (``single_gemm``) stay their own kinds.
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

KIND = "gemm_swiglu_quant"


@dataclass(frozen=True)
class GemmSwigluQuantArgs(KernelArgs):
    m: int = arg(unit="tokens", doc="Rows of the activation: tokens in the step.")
    n: int = arg(
        unit="elements",
        doc=(
            "Intermediate size: output features of the gate weight and of the up "
            "weight, each; also the width of the quantized output."
        ),
    )
    k: int = arg(unit="elements", doc="Input features (hidden size), the reduction dimension.")
    dtype: DType = arg(
        doc=(
            "Element type of the activation, both weights and the output. "
            "nvfp4_e2m1: e2m1 codes with one e4m3 scale per 16 elements."
        )
    )


DOC = KernelDoc(
    title="Fused gate-up GEMM, SwiGLU and quantization",
    summary=(
        "One launch: gate and up projections of NVFP4 rows, SiLU(gate)·up, written "
        "as NVFP4 rows for the down projection."
    ),
    description=(
        "A dense gated MLP whose projections are all NVFP4 can keep the "
        "intermediate activation out of memory: one GEMM computes the gate and up "
        "projections of the same quantized rows tile by tile, applies SiLU(gate)·up "
        "to the fp32 accumulators and quantizes the product to e2m1 codes with an "
        "e4m3 scale per 16 elements under the down projection's static input "
        "scale. TensorFold runs a checkpoint-math NVFP4 MLP's prompt rows this "
        "way. The input rows are quantized before it (a separate launch) and the "
        "down projection runs after it as a plain GEMM."
    ),
    category="GEMM",
    formula=(
        "out = quant_nvfp4(SiLU(x · Wg^T) · (x · Wu^T)), x[m, k], Wg, Wu[n, k], out[m, n]",
        "TFLOPS = 2·m·(2n)·k / time",
        "GB/s = ((m·k + 2·n·k + m·n)·(1/2 + 1/16)) / time",
    ),
    default_metric="tflops",
    method=(
        f"{CUPTI_METHOD} The timed call is the single gemm_gu_ck launch on "
        "already-quantized rows and already-packed weights; it is filtered by "
        "kernel name. The input quantization, weight packing and a check against "
        "an independent PyTorch reference run before timing."
    ),
    caveats=(
        "Weights are random e2m1 codes with random e4m3 block scales; the "
        "activation is seeded random bf16 rows quantized under a calibrated "
        "input scale.",
        "GB/s counts the packed codes and e4m3 scales of the input, both weights "
        "and the output once each; tile padding and global scales are excluded.",
    ),
    reference="profiling.runners.gemm.gemm_swiglu_quant_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="tensorfold_ck",
        # gemm_ck.cu's FP4 path is mma.sync kind::mxf4nvf4 (mma4.cuh), compiled
        # only under __CUDA_ARCH_FEAT_SM120_ALL / SM121_ALL; TensorFold builds the
        # extension for the device's arch-specific target when
        # precision.own_math reports NVFP4 (capability major 12).
        supports=BackendSupport(
            compute=frozenset({DType.NVFP4_E2M1}),
            sm_targets=frozenset({"sm_120f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.tensorfold_gemm_swiglu_quant",
            function_name="profile_gemm_swiglu_quant_tensorfold_ck",
        ),
        table_name=KIND,
        args_schema=GemmSwigluQuantArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="tensorfold_env",
        doc=BackendDoc(
            summary=(
                "TensorFold's checkpoint-math prompt MLP: _ext().gemm_gu_ck, gemm_kernel "
                "with the fp32 SwiGLU epilogue (EPI 2) on 128 x 256 blocks, as "
                "checkpoint.mlp_prompt issues it."
            ),
            url="https://github.com/efeslab/TensorFold",
        ),
    )
)

__all__ = ["KIND", "GemmSwigluQuantArgs"]
