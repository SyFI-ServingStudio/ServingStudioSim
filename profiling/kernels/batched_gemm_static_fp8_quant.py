"""Batched GEMM whose bf16 result is quantized to FP8 with one static scale, in one launch.

MLA's query absorption multiplies each head's query by W_UK, and the FP8
sparse-attention kernel then reads that latent query as FP8 E4M3 under one
static per-tensor scale. Unfused, that is a bf16 ``torch.bmm`` followed by a
static ``scaled_fp8_quant``; this kind is the pair as one kernel that quantizes
in its epilogue. It is not a ``batched_gemm`` backend: that kind's output is the
bf16 product, this one's is the FP8 tensor the next kernel reads.
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

KIND: str = "batched_gemm_static_fp8_quant"


@dataclass(frozen=True)
class BatchedGemmStaticFp8QuantArgs(KernelArgs):
    num_batches: int = arg(
        unit="heads", doc="Attention heads multiplied independently on this GPU."
    )
    m: int = arg(unit="tokens", doc="Token rows per attention head.")
    n: int = arg(unit="elements", doc="Output features per attention head.")
    k: int = arg(unit="elements", doc="Input features per attention head.")
    dtype: DType = arg(doc="Element type of both inputs (bf16). The output is always FP8 E4M3.")


DOC = KernelDoc(
    title="Batched GEMM with static FP8 output",
    summary=(
        "Multiply one query and weight pair per attention head and write the product as FP8 "
        "under one static scale, in one launch."
    ),
    description=(
        "MLA query absorption maps each head's query into the compressed KV space "
        "(C[b] = A[b] · W_UK[b]); an FP8 sparse-attention kernel then reads that latent "
        "query as FP8 E4M3 with one static per-tensor scale. This kind computes the "
        "product with FP32 accumulation, rounds it to bf16 as the unfused torch.bmm "
        "output would be stored, multiplies by 1/scale, clamps to ±448 and writes FP8 "
        "E4M3, token-major ([m, num_batches, n]). It replaces a bf16 batched GEMM plus a "
        "static FP8 quant launch, and is bit-identical to that pair."
    ),
    category="GEMM",
    formula=(
        "C[t, b, n] = fp8(clamp(bf16(Σ_k A[b, t, k] · W[b, k, n]) · (1/scale), ±448))",
        "TFLOPS = 2·num_batches·m·n·k / time",
        "GB/s = num_batches·(2·m·k + 2·k·n + m·n) / time",
    ),
    default_metric="tflops",
    method=(
        f"{CUPTI_METHOD} One launch per call; warm-up calls run first. Before timing, "
        "the FP8 output is checked byte for byte against vLLM's unfused pair (torch.bmm "
        "into the bf16 latent query, then the static scaled_fp8_quant), and against a "
        "Torch float32 reference within one FP8 code."
    ),
    caveats=(
        "The cuda_tcgen05 backend has its head geometry compiled in (64 heads, k = 256, "
        "n = 512) and refuses any other. It reads the weight pre-transposed to a K-major "
        "[num_batches·n, k] copy, a layout change made once when the weights load.",
        "The cuda_tcgen05 backend has three tile configurations; each row is the fastest "
        "of them at that m, as an autotuned launch would pick.",
    ),
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="cuda_tcgen05",
        # tcgen05 MMA and TMEM, built for sm_100a: exactly SM100.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            sm_targets=frozenset({"sm_100a"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.batched_gemm_static_fp8_quant_tcgen05",
            function_name="profile_batched_gemm_static_fp8_quant_cuda_tcgen05",
        ),
        table_name=KIND,
        args_schema=BatchedGemmStaticFp8QuantArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "A repo-vendored CUDA kernel: TMA loads, tcgen05 MMA into TMEM (one CTA per "
                "128-token tile, head and n chunk, two CTAs per SM), and an epilogue that "
                "rounds to bf16, scales, clamps and stores FP8."
            ),
            url=None,
        ),
    )
)

__all__ = ["KIND", "BatchedGemmStaticFp8QuantArgs"]
