"""DeepGEMM's fused mHC boundary with the FP8 cast, as DeepInfer calls it (``deep_gemm.mega_mhc``).

Wire string: ``"mega_mhc"`` -- matches Rust ``KernelSpec::KIND`` in
``simulator/src/timing/kernels/mega_mhc.rs``.

One ``sm100_mega_mhc_impl`` launch per sub-layer entry and GPU: the previous
sub-layer's mHC post (``post_mix * x + comb_res_mixᵀ @ residual``), this
sub-layer's TF32 prenorm GEMM over the updated streams, the sigmoid pre/post
gates and the Sinkhorn comb, the shifted collapse with the pre mix carried
from the previous sub-layer, the RMSNorm, and the cast of the normalized row
to FP8 E4M3 with UE8M0 scales per 32 channels. DeepInfer calls it from
``v4_modules/hc.py`` ``HCLayerMixin._mega_mhc``.

Why a new kind: ``mhc_fused_post_pre_rms_norm`` ends at a BF16 layer input; its
consumer quantizes. This call also writes the FP8 rows and their scales in the
consumer's layout, so its output contract (dtype and storage layout) differs.

Key: geometry and dtype plus ``sf_layout``, the consumer's scale layout, which
selects a different kernel specialization and a different set of outputs.
``"gemm"`` (attention entry): TMA-aligned column-major scales for the MXFP8
input projections. ``"mega_moe"`` (FFN entry): the FP8 rows and row-major
scales straight into the mega MoE symmetric buffer plus a second copy of the
scales in the fused shared expert's tiled layout.
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

KIND = "mega_mhc"


@dataclass(frozen=True)
class MegaMhcArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Rows entering the sub-layer on this GPU.")
    hidden_size: int = arg(unit="elements", doc="Width of each residual stream.")
    hc_mult: int = arg(unit="streams", doc="Residual streams per token (mHC expansion).")
    hidden_dtype: DType = arg(doc="Element type of the streams and the layer output (bf16).")
    sf_layout: str = arg(
        doc=(
            "Consumer of the FP8 output: 'gemm' (attention entry, column-major scales for "
            "the MXFP8 GEMM) or 'mega_moe' (FFN entry, mega MoE buffer scales plus the "
            "shared expert's tiled copy)."
        )
    )


DOC = KernelDoc(
    title="mHC post/pre, RMSNorm and FP8 cast",
    summary=(
        "Mix a sub-layer's output into the mHC residual streams, compute the next "
        "sub-layer's mixes and RMS-normalized input, and cast it to FP8, in one kernel."
    ),
    description=(
        "DeepInfer runs every sub-layer entry that follows another sub-layer as one "
        "persistent DeepGEMM mega_mhc launch: the previous sub-layer's post mix into the "
        "residual streams, the next pre's TF32 GEMM and RMS statistic split over K, the "
        "sigmoid gates and Sinkhorn comb, the collapse with the pre mix carried from the "
        "previous sub-layer (shifted mHC), the RMSNorm, and the FP8 E4M3 cast with UE8M0 "
        "scales per 32 channels. It writes the BF16 and the FP8 layer input."
    ),
    category="Normalization",
    subcategory="Hyper-connections",
    formula=(
        "streamⱼ ← Σᵢ combᵢⱼ · streamᵢ + postⱼ · x",
        "mixes = RMS-scaled (flatten(streams) @ fnᵀ); y = RMSNorm(Σₕ prev_preₕ · streamₕ)",
        "y_fp8, scales = cast_e4m3(y, 32 channels per UE8M0 scale)",
    ),
    default_metric="time_ms",
    method=(
        f"{CUPTI_METHOD} One launch per call. The outputs are checked against "
        "DeepInfer's unfused shifted path (TileKernels post, apply-mix, split-mixes, "
        "Sinkhorn; DeepGEMM prenorm GEMM) on the same operands before timing."
    ),
    caveats=(
        "DeepGEMM picks the K-split count from num_tokens and B200's 148 SMs; the time "
        "steps where it changes and where the 16-split work spills into another wave.",
        "For sf_layout='mega_moe' the shared-scale tile height is DeepInfer's "
        "get_block_m_for_mega_moe for DP4/EP4, 416 slots, top-6, fp8 x fp4.",
        "TFLOPS counts the prenorm GEMM; GB/s one pass over the call's tensors.",
    ),
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="deepgemm_sm100",
        # csrc/apis/mega_mhc.hpp: DG_HOST_ASSERT(jit->device.get_arch_major() == 10);
        # DeepInfer's use_mega_mhc additionally needs hc_mult == 4 and
        # hidden % 1024 == 0 (the runner rejects other shapes).
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            sm_targets=frozenset({"sm_100f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.mhc.mega_mhc_deepgemm",
            function_name="profile_mega_mhc_deepgemm",
        ),
        table_name=KIND,
        args_schema=MegaMhcArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(
            summary=(
                "DeepGEMM mega_mhc as DeepInfer's _mega_mhc calls it: shifted, BF16 and "
                "FP8 outputs, scales in the attention GEMM or mega MoE layout."
            ),
            url="https://github.com/deepseek-ai/DeepGEMM",
        ),
    )
)

__all__ = ["KIND", "MegaMhcArgs"]
