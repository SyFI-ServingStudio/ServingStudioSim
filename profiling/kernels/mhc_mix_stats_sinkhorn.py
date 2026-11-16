"""DeepSeek-V4 mHC mixing coefficients: split-K mix GEMM + RMS stats + Sinkhorn, as SGLang runs them.

Source (SGLang v0.5.23): ``srt/models/deepseek_v4_mhc.py`` ``_mix_stats_impl`` (line 332)
-> ``kernels/ops/layernorm/mhc.py`` ``hc_mix_stats_sinkhorn`` (line 2465): two Triton launches,
``_hc_mix_stats_partial_kernel`` (split-K ``x_flat @ hc_fn.T`` partial sums and partial sum of
squares) and ``_hc_mix_reduce_sinkhorn_kernel`` (reduce, RMS scale, sigmoid split into pre/post and
the Sinkhorn-normalized hc_mult x hc_mult comb). On SM12x this is the path for every row count:
``use_fused_stats`` holds on Blackwell, and the DeepGEMM / BF16x3 projections need
``SGLANG_OPT_DEEPGEMM_HC_PRENORM`` on SM90/SM100 only. With ``hc_pre_from_prev_sublayer`` (V4.1),
decode and target-verify issue it on the stats side stream beside the sublayer
(``use_stats_stream``, line 304); the worklet decides the overlap, the row is the plain call.

One call per mHC sublayer boundary (pre-attention and pre-FFN of every layer), on every TP rank over
all scheduled tokens (the residual streams are replicated across TP).
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

KIND = "mhc_mix_stats_sinkhorn"


@dataclass(frozen=True)
class MhcMixStatsSinkhornArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Token rows.")
    hidden_size: int = arg(unit="elements", doc="Features in each residual stream.")
    hc_mult: int = arg(unit="streams", doc="Parallel residual streams per token.")
    sinkhorn_iters: int = arg(unit="iterations", doc="Sinkhorn normalization rounds (model config).")
    hidden_dtype: DType = arg(doc="Element type of the residual streams.")


DOC = KernelDoc(
    title="mHC mixing coefficients (mix GEMM + Sinkhorn)",
    summary=(
        "Project a token's hc_mult residual streams to the mHC mixing logits, RMS-scale them, and "
        "split them into pre, post and Sinkhorn-normalized comb coefficients."
    ),
    description=(
        "DeepSeek-V4 manifold hyper-connections keep hc_mult residual streams of hidden_size "
        "features per token. Before each sublayer the flattened streams x (hc_mult·hidden_size "
        "wide) are projected by an fp32 weight of (2 + hc_mult)·hc_mult rows; the result is scaled "
        "by rsqrt(mean(x²) + ε) and split into pre (hc_mult), post (hc_mult) and comb "
        "(hc_mult × hc_mult, Sinkhorn-normalized for sinkhorn_iters rounds). SGLang runs it as a "
        "split-K Triton partial GEMM plus a fused reduce + Sinkhorn kernel. Inputs are seeded random "
        "bf16 streams and fp32 weights."
    ),
    category="Normalization",
    subcategory="Hyper-connections",
    formula=(
        "m = (x · W_fnᵀ) · rsqrt(mean(x²) + ε); pre, post = σ(m·scale + base); comb = Sinkhorn(m)",
        "GB/s = num_tokens · hc_mult · hidden_size · 2 / time (stream read; weight re-read per tile "
        "not counted)",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} Both launches of the call are counted. Before timing, pre/post/comb are "
        "compared with SGLang's unfused path (a torch fp32 mix GEMM + RMS, then the Triton "
        "_hc_split_sinkhorn port)."
    ),
    caveats=(
        "Times the call on the caller's stream; the decode-time side-stream overlap is the "
        "worklet's to model.",
    ),
    reference=None,
)


# Triton only (bf16 storage, fp32 tl.dot with tf32/ieee precision): runs wherever Triton lowers
# bf16 (SM80+). It is SGLang's path on SM12x; on SM90/SM100 SGLang prefers DeepGEMM/BF16x3 variants
# for >= 32 rows, which this backend does not model -- the arch chooses the backend per GPU.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 0),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.mhc.mhc_sglang_triton",
            function_name="profile_mhc_mix_stats_sinkhorn_sglang_triton",
        ),
        table_name=KIND,
        args_schema=MhcMixStatsSinkhornArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_env",
        doc=BackendDoc(
            summary=(
                "SGLang hc_mix_stats_sinkhorn (kernels/ops/layernorm/mhc.py:2465): "
                "_hc_mix_stats_partial_kernel + _hc_mix_reduce_sinkhorn_kernel."
            ),
            url="https://github.com/sgl-project/sglang",
        ),
    )
)

__all__ = ["KIND", "MhcMixStatsSinkhornArgs"]
