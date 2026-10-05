"""GDN prefill chunked delta rule as ONE fused launch.

FLA's Triton realization splits the chunked gated delta rule into six launches
(local cumsum, scaled dot K·Kᵀ, triangular solve, w/u recompute, inter-chunk
state scan, output). This kind models the realization vLLM actually selects on
Hopper and Blackwell instead: FlashInfer's `flashinfer.gdn_prefill.
chunk_gated_delta_rule`, a single CUTLASS TMA warp-specialized kernel
(`flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule`) that keeps every
intermediate on chip. The simulator has no kinds for the six Triton launches.

The choice is not a tuning detail. `ChunkGatedDeltaRule._resolve_gdn_prefill_backend`
picks `flashinfer` for the default `auto` request on any SM90 part with no
further constraints, so a measured H200 vLLM run never launches the six Triton
kernels at all. Costing the six-launch path against it over-predicted the
operation by 110% on a Qwen3.6-35B-A3B-FP8 capture, because the split path
round-trips h/w/u/A through HBM five extra times.

Wire string: ``"gdn_chunk_delta_rule"`` — matches the Rust ``KernelSpec::KIND``
in ``simulator/src/timing/kernels/gdn_chunk_delta_rule.rs`` and the facade stem
``get_gdn_chunk_delta_rule_times`` / ``count_missing_gdn_chunk_delta_rule``.

Chunking is deliberately NOT in the args schema: FlashInfer owns its own chunk
size internally, so unlike FLA's Triton path there is no `num_chunks` the
caller can choose.

The second axis is `max_sequence_length`, NOT a sequence count. The inter-chunk
recurrence is sequential *within* a sequence and independent *across* sequences,
so one CTA per (sequence, value head) walks its own chunks and the launch is
work-conserving rather than wave-quantized. The profiled rows show it on an H200
(16/32 heads, 128/128 dims): 4096x1 198.5 us, 4096x2 198.7 us, 4096x4 212.6 us —
flat while 32*N CTAs still fit the 132 SMs — then 4096x8 425.3 us and 4096x16
868.2 us, i.e. linear in total work. A ragged batch of one 8192-token sequence
plus sixteen 64-token ones measures 1.005x its own session's 8192x1 baseline,
which rules out a `ceil(32*N/132)` wave model: the short sequences retire
immediately.

`(num_tokens, max_sequence_length)` therefore spans both terms — critical path
from the longest sequence, aggregate work from the token total — while a
`(num_tokens, num_sequences)` pair does not: a balanced 4092+4093 split of the
same 8185 tokens measures 198.7 us against 390.7 us for the 8163+22 split vLLM
actually produces, and 390.7 us is what the in-situ capture records (389.9 us).
The canonical realization is `num_tokens // max` full-length sequences plus one
remainder, which *is* the chunked-prefill shape (a capture iteration of 8185
tokens is exactly 8163+22).
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

KIND: str = "gdn_chunk_delta_rule"


@dataclass(frozen=True)
class GdnChunkDeltaRuleArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Tokens across the input sequences.")
    max_sequence_length: int = arg(unit="tokens", doc="Tokens in the longest input sequence.")
    num_key_heads: int = arg(unit="heads", doc="Key and query heads shared by the output heads.")
    num_heads: int = arg(unit="heads", doc="Output heads with separate values and states.")
    key_head_dim: int = arg(unit="elements", doc="Features in each key and query head.")
    value_head_dim: int = arg(unit="elements", doc="Features in each value and output head.")
    dtype: DType = arg(doc="Element type of query, key and value inputs.")


DOC = KernelDoc(
    title="Fused chunked delta rule",
    summary="Compute Gated DeltaNet prefill output and final recurrent states in one launch.",
    description=(
        "The chunked gated delta rule of Gated DeltaNet prefill, "
        "which vLLM on H200 runs as one FlashInfer CUTLASS kernel: it writes "
        "every token's output and each sequence's final state. The inputs are "
        "prepared as vLLM prepares them: queries and keys already L2-normalized"
        " outside the call, the decay passed already exponentiated, beta in "
        "FP32. The measurement uses num_tokens / max_sequence_length full "
        "sequences plus one shorter remainder; FlashInfer picks its chunk "
        "width."
    ),
    category="Attention",
    subcategory="Gated DeltaNet",
    formula=(
        "state = 2·key_head_dim·value_head_dim·num_heads",
        "intra = 2·64·(num_key_heads·key_head_dim + num_heads·value_head_dim)",
        "FLOPs = num_tokens·2·(state + intra)",
        "N = ⌈num_tokens / max_sequence_length⌉",
        "bytes = 4·num_tokens·num_key_heads·key_head_dim + "
        "4·num_tokens·num_heads·value_head_dim + 8·num_tokens·num_heads + "
        "8·N·num_heads·key_head_dim·value_head_dim",
        "TFLOPS = FLOPs / time",
        "GB/s = bytes / time",
    ),
    default_metric="tflops",
    method=(
        f"{CUPTI_METHOD} "
        "Only the FlatKernel launch is counted. Inputs, gates, sequence "
        "boundaries and outputs are allocated before the capture."
    ),
    caveats=(
        "Each call starts from an all-zero state.",
        "The 64-token width appears only in the FLOP estimate; bytes count "
        "minimum logical traffic, not physical traffic.",
    ),
    # No separate PyTorch reference exists for this fused kind.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer",
        # FlashInfer builds the timed CUTLASS kernel only for sm_90a (gen_gdn_prefill_sm90_module,
        # sm90a_nvcc_flags); on SM100 chunk_gated_delta_rule dispatches a different CuTe DSL kernel
        # (chunk_gated_delta_rule_sm100) that the runner does not time.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            sm_targets=frozenset({"sm_90a"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.gdn_chunk_delta_rule_flashinfer",
            function_name="profile_gdn_chunk_delta_rule_flashinfer",
        ),
        table_name=KIND,
        args_schema=GdnChunkDeltaRuleArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "FlashInfer's chunk_gated_delta_rule CUTLASS call computes "
                "output and final state in one launch."
            ),
            url="https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/gdn_prefill.py",
        ),
        subprocess_env="vllm_env",
    )
)
