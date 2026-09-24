"""GLM-5.3-Flash KDA (Kimi Delta Attention) chunked prefill as one public call.

Wire string: ``"kda_chunk_prefill"`` -- the facade stem is
``get_kda_chunk_prefill_times`` / ``count_missing_kda_chunk_prefill``.

The measured boundary is one call of the vendored
``vllm.models.glm5next.nvidia.ops.third_party.kda.chunk_kda_with_fused_gate``,
made exactly as ``Glm5NextLinearAttention._forward`` makes it on the prefill
branch. That callable launches a fixed chain of FLA Triton kernels: the q/k/v
``.contiguous()`` copies, two l2norms, the fused safe-gate + chunk cumsum,
scaled K.Kt (inter + intra), the 64x64 triangular solve, w/u recompute, the
inter-chunk state scan, and the output kernel. The gather/scatter of recurrent
states and the fp32 beta sigmoid are separate launches outside the call and are
costed elsewhere.

This is a separate kind from ``gdn_chunk_delta_rule`` and the FLA
``gdn_chunk_*`` kinds, not a new backend of them. GDN has a scalar per-head
gate ``g[T,H]`` that the caller exponentiates; KDA has a per-channel gate
``gk[T,H,K]`` that the callable derives from the raw projection with
``lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias))``. It is also a
different callable (FlashInfer CUTLASS for gdn_chunk_delta_rule), and its
operand set includes raw_g, A_log and dt_bias.

The schema keeps ``num_decode_sequences`` separate. In a mixed iteration the
decode tokens are length-1 sequences in the same varlen call. Every one of them
costs a whole 64-token chunk in the chunk-parallel kernels and one more
(sequence, head) program in the state scan. The capture's 2019 + 29 batch
therefore has 61 chunks, not the 33 that a 2048-token prefill-only batch with
the same longest sequence would have.

Canonical realization of ``(num_tokens, max_sequence_length,
num_decode_sequences)``: first the decode sequences, one token each (vLLM
places decodes first), then ``P = num_tokens - num_decode_sequences`` prefill
tokens as ``P // max_sequence_length`` full sequences plus one remainder, which
is the chunked-prefill shape. Chunk size (64), fp32 state in ``[N,H,V,K]``
layout, ``safe_gate=True`` and ``lower_bound=-5.0`` are fixed by the
production path. They are not args.
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

KIND: str = "kda_chunk_prefill"


@dataclass(frozen=True)
class KdaChunkPrefillArgs(KernelArgs):
    num_tokens: int
    max_sequence_length: int
    num_decode_sequences: int
    num_heads: int
    head_dim: int
    dtype: DType


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.kda_chunk_prefill_vllm_triton",
            function_name="profile_kda_chunk_prefill_vllm_triton",
        ),
        table_name=KIND,
        args_schema=KdaChunkPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_fork_env",
        # vLLM defaults TRITON_CACHE_AUTOTUNING=1, which would let the first
        # shape tuned in a TRITON_CACHE_DIR fix the configs of every later row
        # and process. The runner instead tunes at a documented anchor per
        # worker and records it through row_provenance.
        worker_env=(("TRITON_CACHE_AUTOTUNING", "0"),),
        row_provenance_ref=RunnerRef(
            module_name="profiling.runners.attention.kda_chunk_prefill_vllm_triton",
            function_name="row_provenance",
        ),
    )
)
