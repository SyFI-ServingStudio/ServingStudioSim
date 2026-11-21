"""DeepInfer's standalone mHC post (TileKernels ``mhc_post_fwd``).

Wire string: ``"mhc_post"`` -- matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/mhc_post.rs``.

One ``mhc_post_fwd_cuda_kernel`` launch: the finished sub-layer's output is
mixed into the residual streams, ``streamⱼ ← postⱼ · x + Σᵢ combᵢⱼ · streamᵢ``.
DeepInfer runs it alone where the next pre cannot fuse with it
(``v4_modules/hc.py`` ``hc_post``): before an Engram layer's hook and after the
last layer (``deepseek.py``).

Why a new kind: no existing kind is the post alone; ``mhc_fused_post_pre_rms_norm``
and ``mega_mhc`` include the next pre, ``hc_combine`` is the gated (not comb-matrix)
combine of another model.
"""

from __future__ import annotations

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

KIND = "mhc_post"

DOC = KernelDoc(
    title="mHC post",
    summary=(
        "Mix a sub-layer's output into the mHC residual streams with the post gates and the comb "
        "matrix."
    ),
    description=(
        "TileKernels' mhc_post_fwd as DeepInfer calls it outside the fused boundary: each new "
        "stream is the post gate times the sub-layer output plus the comb-weighted sum of the old "
        "streams. One launch, bf16 streams, fp32 mixes."
    ),
    category="Normalization",
    subcategory="Hyper-connections",
    formula=("streamⱼ ← postⱼ · x + Σᵢ combᵢⱼ · streamᵢ",),
    default_metric="time_ms",
    method=(
        f"{CUPTI_METHOD} One launch. The output is checked against an fp32 Torch "
        "transcription before timing."
    ),
    caveats=("GB/s counts x, the streams and the mixes read and the streams written once.",),
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="tile_kernels",
        # TileKernels JIT-compiles its TileLang kernels for the device; no
        # capability gate in tile_kernels.mhc.post_kernel (CUDA path).
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.mhc.mhc_unfused_tile_kernels",
            function_name="profile_mhc_post_tile_kernels",
        ),
        table_name=KIND,
        args_schema=MhcRmsNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(
            summary="TileKernels mhc_post_fwd, as DeepInfer's HCLayerMixin.hc_post calls it.",
            url="https://github.com/deepseek-ai/TileKernels",
        ),
    )
)

__all__ = ["KIND"]
