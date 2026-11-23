"""DeepInfer's unfused shifted mHC pre (``_hc_pre`` with ``hc_shifted``, before the layer norm).

Wire string: ``"mhc_shifted_pre"`` -- matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/mhc_shifted_pre.rs``.

Five launches (``v4_modules/mhc_ops.py``): ``mhc_pre_apply_mix_fwd`` (collapse
the streams with the pre mix carried from the previous sub-layer), DeepGEMM
``tf32_hc_prenorm_gemm`` (16 K-splits) and ``mhc_reduce_partials_and_rmsnorm_fwd``
(the mixes and their RMS scale), ``mhc_pre_split_mixes_fwd`` (pre/post gates,
comb logits) and ``mhc_sinkhorn_fwd`` (20 iterations). DeepInfer runs it where
``mega_mhc`` cannot fuse: layer 0's attention entry (after ``expand_to_mhc``) and
the Engram layers' attention entries (after the Engram hook). The layer's
RMSNorm follows as a separate call.

Why a new kind: ``mhc_pre_rms_norm`` is vLLM's non-shifted pre with the RMSNorm
fused; this one collapses with the carried mix and stops before the norm.
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

KIND = "mhc_shifted_pre"

DOC = KernelDoc(
    title="Shifted mHC pre (unfused)",
    summary=(
        "Collapse the mHC streams with the carried pre mix and compute this sub-layer's gates and "
        "Sinkhorn comb."
    ),
    description=(
        "DeepInfer's unfused shifted pre: the layer input is the streams weighted by the pre mix "
        "the previous sub-layer computed; the TF32 prenorm GEMM, the RMS scale, the sigmoid gates "
        "and the 20-step Sinkhorn then produce the mixes the next sub-layer and the post use. Five"
        " launches, all timed."
    ),
    category="Normalization",
    subcategory="Hyper-connections",
    formula=(
        "layer_input = Σₕ prev_preₕ · streamₕ",
        "mixes = RMS-scaled (flatten(streams) @ fnᵀ) · scale + base; gates and Sinkhorn comb",
    ),
    default_metric="time_ms",
    method=(
        f"{CUPTI_METHOD} Every launch of the sequence is counted. Outputs are checked "
        "against an fp32 Torch transcription before timing."
    ),
    caveats=(
        "GB/s counts the streams read twice, fn, and the layer input and mixes written once.",
    ),
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="tile_kernels",
        # tf32_hc_prenorm_gemm is DeepGEMM's sm100 kernel (csrc/apis: arch_major 10 only).
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            sm_targets=frozenset({"sm_100f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.mhc.mhc_unfused_tile_kernels",
            function_name="profile_mhc_shifted_pre_tile_kernels",
        ),
        table_name=KIND,
        args_schema=MhcRmsNormArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(
            summary=(
                "DeepInfer's mhc_ops shifted pre: TileKernels apply-mix, split-mixes, Sinkhorn; "
                "DeepGEMM tf32_hc_prenorm_gemm."
            ),
            url="https://github.com/deepseek-ai/TileKernels",
        ),
    )
)

__all__ = ["KIND"]
