"""GDN fused causal convolution, q/k normalization and gates kernel kind.

All Python-side per-kernel knowledge for ``gdn_conv_norm_gates`` lives here:
the wire string ``KIND``, the ``GdnConvNormGatesArgs`` schema, and the
``register(...)`` call. The Rust ``KernelSpec`` is
``simulator/src/timing/kernels/gdn_conv_norm_gates.rs``.

The kind is one launch that does what vLLM splits into a causal convolution
(``gdn_causal_conv_prefill`` / ``gdn_causal_conv_decode``) and the
post-convolution prep (``gdn_prefill_post_conv``): every token's depthwise
causal convolution over its own window of earlier rows, SiLU, normalized
queries and keys, values, the decay and beta. Neither existing kind fits: the
convolution kinds stop at the convolution output, and the post-convolution
kind starts from it (and writes the log decay, not the decay).
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

KIND: str = "gdn_conv_norm_gates"


@dataclass(frozen=True)
class GdnConvNormGatesArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Rows in the launch, across every stream.")
    num_qk_heads: int = arg(unit="heads", doc="Query and key heads per token.")
    num_value_heads: int = arg(unit="heads", doc="Value heads and gate values per token.")
    key_head_dim: int = arg(unit="elements", doc="Features in each query or key head.")
    value_head_dim: int = arg(unit="elements", doc="Features in each value head.")
    kernel_size: int = arg(unit="tokens", doc="Width of each causal convolution filter.")
    dtype: DType = arg(doc="Element type of the packed projection, state, filter and q/k/v outputs.")


DOC = KernelDoc(
    title="GDN convolution, q/k norm and gates",
    summary=(
        "Causal convolution of each token's window, SiLU, normalized queries and keys, "
        "values, decay and beta, in one launch."
    ),
    description=(
        "The step of a Gated DeltaNet layer between the input projections and the "
        "recurrence. Each token's packed q|k|v row is convolved depthwise with the "
        "kernel_size − 1 rows before it on its own path (a prompt's previous rows, "
        "or a draft tree's ancestors, read from the stream's convolution state when "
        "the path starts before this launch), then passed through SiLU. Queries and "
        "keys are L2-normalized per head (queries also scaled by 1/√key_head_dim), "
        "values are kept, and two raw gates per value head become the decay "
        "g = exp(−exp(A_log)·softplus(a + dt_bias)) and beta = sigmoid(b)."
    ),
    category="Attention",
    subcategory="Gated DeltaNet",
    formula=(
        "C = 2·num_qk_heads·key_head_dim + num_value_heads·value_head_dim",
        "x = SiLU(Σ_j window_j · w_j), j over kernel_size rows",
        "q = x_q / ‖x_q‖ / √key_head_dim; k = x_k / ‖x_k‖; v = x_v",
        "g = exp(−exp(A_log)·softplus(a + dt_bias)); beta = sigmoid(b)",
        "FLOPs = num_tokens·(2·kernel_size·C + 4·C)",
        (
            "bytes = num_tokens·(kernel_size·C + C + 2·num_value_heads)·bytes(dtype) + "
            "num_tokens·2·num_value_heads·4 + C·kernel_size·bytes(dtype)"
        ),
        "TFLOPS = FLOPs / time",
        "GB/s = bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} tensorfold_triton counts only _gdn_pre. Input construction, "
        "the Triton compile and the output check run before timing."
    ),
    caveats=(
        "GB/s counts each window row as read once per token (kernel_size reads of a row), "
        "as the kernel issues them; most repeats hit L2.",
        "The convolution, statistics and gates run in FP32; q, k and v are rounded to "
        "BF16 and g and beta stay FP32.",
    ),
    reference="profiling.runners.attention.gdn_conv_norm_gates_reference",
)


# TensorFold perf-study 6469730a, tensorfold/families/qwen3_5/cuda/glue.py
# `gdn_pre`: the Triton `_gdn_pre`, one 2-warp program per (row, head) over
# 2·num_qk_heads + num_value_heads heads. forward.py runs it in every verify
# forward (tree_forward and multi_tree_forward, MULTI mode: each row's state
# rows at its stream's offset) and prefill.py in every prefill chunk
# (prefill_chunk, one stream) and batched prefill step (prefill_rows, MULTI).
# The kernel casts to tl.bfloat16, so bf16 only. It is plain Triton (no
# TensorFold CUDA extension); Triton's bf16 needs sm_80. It runs in
# TensorFold's own venv.
TENSORFOLD_TRITON = "tensorfold_triton"

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend=TENSORFOLD_TRITON,
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 0),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.gdn_conv_norm_gates_tensorfold_triton",
            function_name="profile_gdn_conv_norm_gates_tensorfold_triton",
        ),
        table_name=KIND,
        args_schema=GdnConvNormGatesArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="tensorfold_env",
        doc=BackendDoc(
            summary=(
                "TensorFold's Triton _gdn_pre (glue.gdn_pre): one launch, a program per "
                "token and head, reading each token's convolution window through a row-index "
                "table and per-stream state offsets."
            ),
            url="https://github.com/efeslab/TensorFold",
        ),
    )
)
