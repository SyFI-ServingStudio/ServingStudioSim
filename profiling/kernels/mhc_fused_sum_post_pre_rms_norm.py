"""Sum of partial block outputs, then the fused mHC post/pre block with RMSNorm."""

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

KIND = "mhc_fused_sum_post_pre_rms_norm"


@dataclass(frozen=True)
class MhcFusedSumArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Token rows processed by the MHC block.")
    hidden_size: int = arg(unit="elements", doc="Features in each residual stream.")
    hc_mult: int = arg(unit="streams", doc="Parallel residual streams per token.")
    hidden_dtype: DType = arg(doc="Element type of the residual streams and the partial outputs.")
    num_partials: int = arg(
        unit="tensors",
        doc="Partial block outputs summed into the block output (one bf16 rounding).",
    )
    shared_addend: bool = arg(
        doc="Whether one more block-output tensor is added after the sum (a shared expert)."
    )


DOC = KernelDoc(
    title="Partial-output sum, mHC post-block, pre-block and RMSNorm",
    summary=(
        "Sum a block's partial outputs, mix the result into the mHC residual "
        "streams, then compute the next block's mixing weights and normalized "
        "input, in one call."
    ),
    description=(
        "The boundary after an expert-parallel MoE layer when the combine leaves "
        "each rank's partial sums in the token owner's memory: the owner adds the "
        "num_partials bf16 partial outputs (fp32 accumulate, one bf16 rounding), "
        "adds the shared expert's output when shared_addend is set (a second bf16 "
        "rounding, as an unfused residual add does), and then does what "
        "mhc_fused_post_pre_rms_norm does with that block output: the mHC post "
        "mix into the residual streams, the next pre's mixes and the RMS-normalized "
        "input. With num_partials = 1 and no addend it is mhc_fused_post_pre_rms_norm."
    ),
    category="Normalization",
    subcategory="Hyper-connections",
    formula=(
        "x = bf16(bf16(Σₚ partialₚ) + shared)",
        "streamⱼ ← Σᵢ combᵢⱼ · streamᵢ + postⱼ · x",
        "then the mhc_pre_rms_norm computation on the updated streams",
    ),
    default_metric="time_ms",
    method=(
        f"{CUPTI_METHOD} "
        "Three warm-up calls run first, and every launch of the call is counted. "
        "The outputs are checked against the unfused PyTorch path (sum, add, "
        "vLLM's Torch post and pre, RMSNorm) before timing."
    ),
    caveats=(
        "The partial outputs are local tensors; a combine that leaves them in "
        "peer memory (MNNVL) reads them over NVLink instead, which this does not "
        "measure.",
        "The runner uses ε = 1e-6; another ε runs the same launches.",
        "GB/s counts one pass over the call's tensors, each partial and the addend "
        "once; vllm_tilelang also counts its add (two reads, one write) and its "
        "own re-reads of the updated streams. TFLOPS is not computed.",
    ),
    reference=None,
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_tilelang",
        runner_ref=RunnerRef(
            module_name="profiling.runners.mhc.mhc_fused_post_pre_rms_norm_vllm_tilelang",
            function_name="profile_mhc_fused_sum_post_pre_rms_norm_vllm_tilelang",
        ),
        table_name=KIND,
        args_schema=MhcFusedSumArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's path: the shared expert's output added to the reduced routed "
                "output (one bf16 tensor add, when shared_addend is set), then the "
                "mhc_fused_post_pre_rms_norm vllm_tilelang call. Only num_partials = 1: "
                "vLLM's combine hands over the routed output already reduced."
            ),
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_tilelang_post_addend",
        runner_ref=RunnerRef(
            module_name="profiling.runners.mhc.mhc_fused_sum_post_addend_tilelang",
            function_name="profile_mhc_fused_sum_post_pre_rms_norm_vllm_tilelang_post_addend",
        ),
        table_name=KIND,
        args_schema=MhcFusedSumArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(compute=frozenset({DType.BF16})),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's path with the shared expert's add read by the post kernel: above 32 "
                "tokens a copy of vLLM's mhc_post_tilelang that takes both block outputs "
                "(x = bf16(shared + routed) in-register), then vLLM's prenorm GEMM and pre "
                "big-fuse unchanged; at 32 tokens or fewer vLLM's add and fused call "
                "unchanged. Only num_partials = 1 with shared_addend. Outputs bit-exact "
                "with vllm_tilelang."
            ),
        ),
    )
)

__all__ = ["KIND", "MhcFusedSumArgs"]
