"""DeepGEMM's FP4 dense prefill indexer logits over one query chunk.

Wire string: ``"mqa_logits_prefill_chunk"`` -- matches Rust ``KernelSpec::KIND``
in ``simulator/src/timing/kernels/mqa_logits_prefill_chunk.rs``.

``deep_gemm.fp8_fp4_mqa_logits`` as DeepInfer's prefill indexer calls it
(``v4_modules/index_attn.py`` ``_compute_prefill_p_sum``, dense branch): FP4
E2M1 index Q and K with packed UE8M0 scales per 32 elements, bf16 head
weights, bf16 logits, one ``sm100_mqa_logits<..., float_e2m1_t, ...>``
launch per query chunk of at most ``compute_chunk_size`` (8192) rows.

Why not a backend of ``dsa_mqa_logits_prefill``: that kind's call is vLLM's,
one FP8 call per request with fp32 scales/weights/logits and a one-key-per-row
causal span (``single_causal_tail``, ``num_sequences = 1``). DeepInfer's
call scores a chunk that packs several requests (``ks``/``ke`` offsets into
the concatenated keys, chunking across request boundaries), on compressed
keys whose span grows one key per ``compress_ratio`` rows, and in FP4 with a
bf16 output. Reusing it would change the span rule, the call boundary and
the operand layout at once.

Key: one request's causal-tail chunk -- ``num_queries`` rows ending a
``context_len``-token context -- plus ratio and head geometry. A chunk that
packs several requests is priced at the single request with the same rows and
the same mean key window (see the Rust kernel).
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

KIND = "mqa_logits_prefill_chunk"


@dataclass(frozen=True)
class MqaLogitsPrefillChunkArgs(KernelArgs):
    num_queries: int = arg(unit="tokens", doc="Query rows scored in the call.")
    context_len: int = arg(
        unit="tokens", doc="Context of the request, the rows included; the rows are its last tokens."
    )
    compress_ratio: int = arg(unit="tokens", doc="Context tokens per compressed index key.")
    num_heads: int = arg(unit="heads", doc="Indexer query heads.")
    head_dim: int = arg(unit="elements", doc="Width of each query head and key.")
    dtype: DType = arg(doc="Element type of index Q and K (FP4 E2M1 with UE8M0 scales per 32).")


DOC = KernelDoc(
    title="Prefill indexer logits (FP4, chunked)",
    summary="Score a prefill query chunk against its compressed index keys in FP4.",
    description=(
        "For each row and each key in its window the kernel forms the ReLU of every "
        "head's query-key dot product, weights it by the row's bf16 head weight and "
        "sums over heads into one bf16 logit. Row i of the chunk sees "
        "(context_len − num_queries + i + 1) // compress_ratio keys."
    ),
    category="Attention",
    subcategory="DSA",
    formula=(
        "P = Σ_i (context_len − num_queries + i + 1) // compress_ratio",
        "FLOPs = 2 · P · num_heads · head_dim",
        "bytes = num_queries · num_heads · (head_dim/2 + 4) + keys · (head_dim/2 + 4) "
        "+ 2 · num_queries · num_heads + 2 · P",
    ),
    default_metric="time_ms",
    method=CUPTI_METHOD + " Four rows are checked against fp32 math on the dequantized operands first.",
    caveats=(
        "max_seqlen_k is the request's key count; DeepInfer passes "
        "runtime_max_model_len // ratio, which only sizes the uninitialized logits buffer.",
    ),
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="deepgemm_sm100",
        # SM100 is the only arch with the FP4 MQA-logits kernel (attention.hpp:
        # arch_major == 10 -> sm100_mqa_logits; SM90 asserts FP8).
        supports=BackendSupport(
            compute=frozenset({DType.MXFP4_E2M1}),
            sm_targets=frozenset({"sm_100f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.mqa_logits_prefill_chunk_deepgemm",
            function_name="profile_mqa_logits_prefill_chunk_deepgemm",
        ),
        table_name=KIND,
        args_schema=MqaLogitsPrefillChunkArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="deepinfer_env",
        doc=BackendDoc(
            summary="DeepGEMM fp8_fp4_mqa_logits with FP4 Q/K, as DeepInfer's prefill indexer calls it.",
            url="https://github.com/deepseek-ai/DeepGEMM",
        ),
    )
)
