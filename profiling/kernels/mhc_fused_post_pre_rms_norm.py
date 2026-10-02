"""Fused mHC post/pre block with fused RMSNorm."""

from profiling.db.args import DType
from profiling.db.doc import CUPTI_METHOD, BackendDoc, KernelDoc
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)
from profiling.kernels.mhc_pre_rms_norm import MhcRmsNormArgs

KIND = "mhc_fused_post_pre_rms_norm"

DOC = KernelDoc(
    title="mHC post-block, pre-block and RMSNorm",
    summary=(
        "Mix a block's output into the mHC residual streams, then compute the "
        "next block's mixing weights and normalized input, in one call."
    ),
    description=(
        "Between consecutive blocks, vLLM fuses two mHC steps into one TileLang"
        " call. The post step mixes the finished block's output x into the "
        "residual streams with the previous post and comb weights; the pre step"
        " then derives new mixing weights from the updated streams and forms "
        "the next block's RMS-normalized input. The measurement uses 4 bf16 "
        "streams of 4,096 features and random activations."
    ),
    category="Normalization",
    subcategory="Hyper-connections",
    formula=(
        "streamⱼ ← Σᵢ combᵢⱼ · streamᵢ + postⱼ · x",
        "then the mhc_pre_rms_norm computation on the updated streams: new (post, comb, input)",
    ),
    default_metric="time_ms",
    method=(
        f"{CUPTI_METHOD} "
        "Three warm-up calls run first, and every launch of the TileLang call "
        "is counted. The outputs are checked against vLLM's PyTorch "
        "implementation before timing."
    ),
    caveats=(
        "Only hidden_size = 4096, hc_mult = 4 in bf16 on H200 and B200 is "
        "measured, with ε = 1e-6; another ε runs the same launches.",
        "The previous post and comb weights come from the pre step on the same random streams.",
        "TFLOPS is not computed. GB/s counts the layer output, streams, "
        "previous mixes and weights read once and the updated streams, next "
        "mixes and next block input written once.",
    ),
    # The check composes PyTorch post-mix and pre-mix; no separate whole-call reference exists.
    reference=None,
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_tilelang",
        runner_ref=RunnerRef(
            module_name="profiling.runners.mhc.mhc_fused_post_pre_rms_norm_vllm_tilelang",
            function_name="profile_mhc_fused_post_pre_rms_norm_vllm_tilelang",
        ),
        table_name=KIND,
        args_schema=MhcRmsNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200", "NVIDIA B200"}),
        ),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary="vLLM's mhc_fused_post_pre_tilelang with the RMSNorm fused in.",
            url="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/kernels/mhc/tilelang.py",
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
        supports=BackendSupport(compute=None, gpus=frozenset({"MI300X"})),
        runner_ref=RunnerRef(
            module_name="profiling.runners.elementwise.floor",
            function_name="profile_mhc_fused_post_pre_rms_norm_floor",
        ),
        table_name=KIND,
        args_schema=MhcRmsNormArgs,
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
