"""Qwen GDN prefill chunked delta rule as ONE fused launch.

Distinct kernel kind from the six `gdn_chunk_*` kinds on purpose. Those model
FLA's Triton realization, which splits the chunked gated delta rule into six
launches (local cumsum, scaled dot K·Kᵀ, triangular solve, w/u recompute,
inter-chunk state scan, output). This kind models the realization vLLM actually
selects on Hopper and Blackwell: FlashInfer's `flashinfer.gdn_prefill.
chunk_gated_delta_rule`, a single CUTLASS TMA warp-specialized kernel
(`flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule`) that keeps every
intermediate on chip.

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
size internally, so unlike the FLA kinds there is no `num_chunks` the caller
can choose.

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
    num_tokens: int
    max_sequence_length: int
    num_key_heads: int
    num_heads: int
    key_head_dim: int
    value_head_dim: int
    dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.gdn_chunk_delta_rule_flashinfer",
            function_name="profile_gdn_chunk_delta_rule_flashinfer",
        ),
        table_name=KIND,
        args_schema=GdnChunkDeltaRuleArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
    )
)
