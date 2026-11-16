"""DeepSeek-V4 mHC post-block (rebuild the residual streams), as SGLang dispatches it.

Source (SGLang v0.5.23): ``srt/models/deepseek_v4_mhc.py`` ``post`` (line 525), reached from
``_post_fusion`` (line 662) when neither the SM100 mega boundary nor the >= 4096-row
``mhc_post_combine_norm_prefill`` seam applies. It computes
``R'[i] = post[i]·x + Σ_j comb[j][i]·R[j]``. On Blackwell with ``hc_pre_from_prev_sublayer``
(V4.1), hc_mult 4 and hidden 5120 it is the Triton ``_mhc_post_split_h_kernel``
(``kernels/ops/layernorm/mhc_post_split_h.py``) up to 384 rows, else the TileLang
``mhc_post_tilelang`` (``SGLANG_OPT_USE_TILELANG_MHC_POST``, default on; ``mhc.py`` line 1331).

One call after each sublayer (attention and FFN) on every TP rank over all scheduled tokens, unless
a fused all-reduce + post kernel (custom all-reduce over NVLink) took the seam; on PCIe SM12x
(NCCL all-reduce) the plain post runs.
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

KIND = "mhc_post"


@dataclass(frozen=True)
class MhcPostArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Token rows.")
    hidden_size: int = arg(unit="elements", doc="Features in each residual stream.")
    hc_mult: int = arg(unit="streams", doc="Parallel residual streams per token.")
    hidden_dtype: DType = arg(doc="Element type of the sublayer output and the streams.")


DOC = KernelDoc(
    title="mHC post-block",
    summary=(
        "Mix a finished sublayer's output into the hc_mult residual streams with the post and comb "
        "coefficients."
    ),
    description=(
        "R'[i] = post[i]·x + Σ_j comb[j][i]·R[j] for every stream i, written as new bf16 streams. "
        "SGLang's dispatch for this geometry is a Triton kernel with independent hidden-dimension "
        "CTAs up to 384 rows and a TileLang kernel above; the row is the dispatch, so one row per "
        "token count is whichever kernel SGLang would launch. Inputs are seeded random bf16 rows "
        "and fp32 coefficients."
    ),
    category="Normalization",
    subcategory="Hyper-connections",
    formula=(
        "R'[i] = post[i]·x + Σ_j comb[j][i]·R[j]",
        "GB/s = num_tokens · (2·hc_mult + 1) · hidden_size · 2 / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} Compared with SGLang's own PyTorch post (_mhc_post_torch) before timing."
    ),
    caveats=(
        "Uses SGLang's dispatch on the profiling device; off Blackwell or for other geometries it "
        "measures whatever kernel SGLang picks there.",
    ),
    reference=None,
)


# Triton (any SM80+) or TileLang (compiled for the device) by row count; no arch-specific build.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_auto",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 0),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.mhc.mhc_sglang_triton",
            function_name="profile_mhc_post_sglang_auto",
        ),
        table_name=KIND,
        args_schema=MhcPostArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_env",
        doc=BackendDoc(
            summary=(
                "SGLang deepseek_v4_mhc.post: _mhc_post_split_h_kernel (<= 384 rows) or "
                "mhc_post_tilelang."
            ),
            url="https://github.com/sgl-project/sglang",
        ),
    )
)

__all__ = ["KIND", "MhcPostArgs"]
