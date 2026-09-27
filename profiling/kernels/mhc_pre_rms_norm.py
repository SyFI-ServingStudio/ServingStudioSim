"""Standalone DeepSeek MHC pre block with fused RMSNorm."""

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

KIND = "mhc_pre_rms_norm"


@dataclass(frozen=True)
class MhcRmsNormArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Token rows processed by the MHC block.")
    hidden_size: int = arg(unit="elements", doc="Features in each residual stream.")
    hc_mult: int = arg(unit="streams", doc="Parallel residual streams per token.")
    hidden_dtype: DType = arg(doc="Element type of the residual streams.")


DOC = KernelDoc(
    title="mHC pre-block with RMSNorm",
    summary=(
        "From a token's parallel residual streams, compute the mHC mixing "
        "weights and the RMS-normalized input of the next block."
    ),
    description=(
        "DeepSeek V4 keeps hc_mult residual streams per token (mHC) instead of "
        "one. Before each block, one vLLM TileLang call projects the streams to"
        " three sets of mixing weights, sums the streams with the pre-mix "
        "weights into the block input, and RMS-normalizes it. The measurement "
        "uses 4 bf16 streams of 4,096 features and random residuals."
    ),
    category="Normalization",
    formula=(
        "x = the hc_mult streams concatenated; mixes = x · fnᵀ / √(mean(x²) + 1e-6)",
        "pre = σ(s₀·mixes_pre + b_pre) + 1e-6; post = 2·σ(s₁·mixes_post + b_post)",
        "comb = Sinkhorn₂₀(softmax(s₂·mixes_comb + b_comb)), an hc_mult × hc_mult matrix",
        "input = RMSNorm(Σⱼ preⱼ · streamⱼ), ε = 1e-6; returns (post, comb, input)",
    ),
    default_metric="time_ms",
    method=(
        f"{CUPTI_METHOD} "
        "Three warm-up calls run first, and every launch of the TileLang call "
        "is counted. The outputs are checked against vLLM's PyTorch "
        "implementation before timing."
    ),
    caveats=(
        "Only hidden_size = 4096, hc_mult = 4 in bf16 on H200 is measured.",
        "TFLOPS is not computed. GB/s counts the streams and weights read "
        "once and the mixes and block input written once.",
    ),
    reference="profiling.runners.mhc._deepseek_v4",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_tilelang",
        runner_ref=RunnerRef(
            module_name="profiling.runners.mhc.mhc_pre_rms_norm_vllm_tilelang",
            function_name="profile_mhc_pre_rms_norm_vllm_tilelang",
        ),
        table_name=KIND,
        args_schema=MhcRmsNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary="vLLM's mhc_pre_tilelang with the RMSNorm fused in.",
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/kernels/mhc/tilelang.py",
        ),
    )
)

__all__ = ["KIND"]
