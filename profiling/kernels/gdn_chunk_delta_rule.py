"""GDN prefill chunked delta rule: one call computing output and final states.

The operation is the chunked gated delta rule of Gated DeltaNet prefill, one
call per layer and step: queries and keys already L2-normalized, grouped key
heads, FP32 gate and beta, FP32 initial states, varlen sequences. It has two
backends, which differ in implementation and in the form of the gate operand:

- ``flashinfer`` -- `flashinfer.gdn_prefill.chunk_gated_delta_rule`, a single
  kernel that keeps every intermediate on chip: a CUTLASS TMA warp-specialized
  kernel (`flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule`) on SM90 and a
  persistent CuTe DSL kernel (`GatedDeltaNetChunkedKernel`) on SM100. It takes
  the decay already exponentiated, so vLLM's `fi_chunk_gated_delta_rule`
  launches a separate `torch.exp(g)` before it; that launch is not in this
  kind and the arch costs it as its own elementwise leaf.
- ``fla_triton`` -- vLLM's vendored FLA `chunk_gated_delta_rule`
  (`vllm.third_party.flash_linear_attention.ops`), a fixed chain of seven
  launches: local cumsum, scaled K·Kᵀ, a BF16 zero fill and the 64x64
  triangular solve, w/u recompute, inter-chunk state scan, output. It takes
  the decay in log space (`chunk_local_cumsum` consumes it), so this path has
  no exp(g) launch at all and an arch selecting it omits that leaf.

The gate form is a backend difference, not a new kind: both backends compute
the same recurrence over the same q/k/v/beta/state tensors from the same
upstream `fused_post_conv_prep` output, and the args below mean the same thing
for both. Only where `exp` is applied to g moves.

`ChunkGatedDeltaRule._resolve_gdn_prefill_backend` picks `flashinfer` for the
default `auto` request on any SM90 part, and on SM100/SM120 when the head dim
is 128 and CUDA is 13+; every other CUDA device (A100, L40S, ...) runs
`fla_triton`. Costing the split Triton chain on an H200 run over-predicted the
operation by 110% on a Qwen3.6-35B-A3B-FP8 capture, because the split path
round-trips h/w/u/A through HBM five extra times, so the backend must follow
the deployment.

Wire string: ``"gdn_chunk_delta_rule"`` — matches the Rust ``KernelSpec::KIND``
in ``simulator/src/timing/kernels/gdn_chunk_delta_rule.rs`` and the facade stem
``get_gdn_chunk_delta_rule_times`` / ``count_missing_gdn_chunk_delta_rule``.

Chunking is deliberately NOT in the args schema: FlashInfer owns its own chunk
size internally and FLA hard-codes 64, so there is no chunk count a caller can
choose.

The second axis is `max_sequence_length`, NOT a sequence count. The inter-chunk
recurrence is sequential *within* a sequence and independent *across* sequences,
so one CTA per (sequence, value head) walks its own chunks (FlashInfer's kernel,
and FLA's `chunk_gated_delta_rule_fwd_kernel_h`, whose grid is (value blocks,
sequences x heads); FLA's other launches are chunk-parallel over tokens) and the launch is
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
        "The chunked gated delta rule of Gated DeltaNet prefill: one call "
        "writes every token's output and each sequence's final state. vLLM "
        "runs it as one FlashInfer kernel on H200 and B200 (flashinfer) and "
        "as FLA's seven Triton launches elsewhere, e.g. on A100 and L40S "
        "(fla_triton). The inputs are prepared as vLLM prepares them: queries "
        "and keys already L2-normalized outside the call, beta in FP32, the "
        "FP32 decay already exponentiated for flashinfer and in log space for "
        "fla_triton. The measurement uses num_tokens / max_sequence_length "
        "full sequences plus one shorter remainder; the chunk width is the "
        "backend's own (64 in FLA)."
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
        "Every launch of the call is counted: FlatKernel on SM90, "
        "GatedDeltaNetChunkedKernel on SM100, and for fla_triton the seven "
        "FLA launches including solve_tril's BF16 zero fill. Inputs, gates, "
        "sequence boundaries and FLA's chunk indices are built before the "
        "capture, as vLLM's metadata builder builds them once per step. "
        "fla_triton's Triton autotuning runs once per worker at vLLM's own "
        "warm-up shape, one 64-token sequence; its keys omit the token count, "
        "so those configs serve every later shape. Before timing, fla_triton "
        "checks the shape cut to 1024-token sequences, from random initial "
        "states, against a per-token PyTorch reference."
    ),
    caveats=(
        "Each timed call starts from an all-zero state.",
        "The gate is exp(g) for flashinfer and g for fla_triton; the exp "
        "launch vLLM issues before the FlashInfer call is not in this kind.",
        "The state gather, the zeroing of fresh states and the final-state "
        "scatter run outside the call.",
        "The 64-token width appears only in the FLOP estimate; bytes count "
        "minimum logical traffic, not physical traffic.",
    ),
    reference="profiling.runners.attention.gdn_chunk_delta_rule_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer",
        # chunk_gated_delta_rule dispatches by architecture: a CUTLASS kernel on
        # SM90, the CuTe DSL chunk_gated_delta_rule_sm100 kernel on SM100/SM103
        # (CUDA 13+, head dim 128). The runner times whichever the call launches.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            sm_targets=frozenset({"sm_90a", "sm_100f"}),
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
                "FlashInfer's chunk_gated_delta_rule call computes output and "
                "final state in one launch (CUTLASS on SM90, CuTe DSL on SM100)."
            ),
            url="https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/gdn_prefill.py",
        ),
        subprocess_env="vllm_env",
    )
)


# vLLM's FLA path: any CUDA device the FlashInfer gate above does not take.
# The chain is plain Triton with BF16 tl.dot (mma.sync, SM80+); TMA is only
# used on Hopper with FLA_USE_TMA=1, which vLLM leaves off.
register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="fla_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            min_compute_capability=(8, 0),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.gdn_chunk_delta_rule_fla_triton",
            function_name="profile_gdn_chunk_delta_rule_fla_triton",
        ),
        table_name=KIND,
        args_schema=GdnChunkDeltaRuleArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's FLA chunk_gated_delta_rule: seven Triton launches (cumsum, "
                "K·Kᵀ, zero fill + triangular solve, w/u, state scan, output), "
                "decay taken in log space."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/third_party/flash_linear_attention/ops/chunk.py",
        ),
        # vLLM defaults TRITON_CACHE_AUTOTUNING=1, which would let the first
        # shape tuned in a TRITON_CACHE_DIR fix the configs of every later row
        # and process. The runner tunes at vLLM's warm-up shape per worker and
        # records it through row_provenance.
        worker_env=(("TRITON_CACHE_AUTOTUNING", "0"),),
        row_provenance_ref=RunnerRef(
            module_name="profiling.runners.attention.gdn_chunk_delta_rule_fla_triton",
            function_name="row_provenance",
        ),
    )
)
