"""DeepSeek fused MHC post/pre block with fused RMSNorm.

One call maps the previous sublayer's output back into the ``hc_mult``
residual streams (mHC post), computes the next sublayer's mixes from them
(TF32 prenorm GEMM, sigmoid gates, Sinkhorn), collapses the streams into the
sublayer input and applies RMSNorm to it.

Backends:

- ``vllm_tilelang``: DeepSeek V4 (hidden 4096) ``mhc_fused_post_pre_tilelang``
  on H200; the collapse uses this call's own pre-mix.
- ``deepgemm_mega``: DeepSeek-V4.1 (hidden 5120) fork
  ``mhc_shifted_post_pre_deep_gemm``, one DeepGEMM ``mega_mhc`` launch on
  B200. The collapse uses the pre-mix carried from the previous sublayer
  ("shifted"), and this call's pre-mix is returned for the next one. The
  per-token work and the args fields are the same.
"""

from profiling.db.args import DType
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
            gpus=frozenset({"NVIDIA H200"}),
        ),
        subprocess_env="vllm_env",
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="deepgemm_mega",
        runner_ref=RunnerRef(
            module_name="profiling.runners.mhc.mhc_fused_post_pre_rms_norm_deepgemm_mega",
            function_name="profile_mhc_fused_post_pre_rms_norm_deepgemm_mega",
        ),
        table_name=KIND,
        args_schema=MhcRmsNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        subprocess_env="vllm_fork_env",
    )
)

__all__ = ["KIND"]
