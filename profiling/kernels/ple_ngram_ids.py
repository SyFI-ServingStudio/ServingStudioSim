"""PLE n-gram hash ids (one Triton launch per step).

Source: alignment fork, ``vllm/models/qwen4_exp/nvidia/ops/ple.py``
``ple_ngram_ids`` -> ``_ple_ngram_ids_kernel`` (grid ``ceil(tokens / 8)``),
called by ``Qwen4ExpNGramEmbedding.compute_ngram_ids`` before the pinned
lookup is started. For every token and n-gram head ``h`` of order ``n``
(2..ngram_context_len+1): XOR of ``token[t - i] * multiplier[i]`` over the last
``n`` tokens (earlier steps' tokens come from the per-request
``ngram_context``, an EOS cuts the history), then
``id = offset[h] + (hash mod size[h])``.

Args:

- ``num_tokens``: scheduled tokens.
- ``num_reqs``: requests the tokens belong to (each token binary-searches
  ``query_start_loc``).
- ``ngram_context_len``: predecessor tokens kept per request
  (``ngram_size - 1``).
- ``heads_per_ngram``: hash heads per n-gram order; ids per token are
  ``ngram_context_len * heads_per_ngram``.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import KernelArgs
from profiling.db.doc import CUPTI_METHOD, BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)
from profiling.kernels.neuron_common import neuron_spec

KIND = "ple_ngram_ids"


@dataclass(frozen=True)
class PleNgramIdsArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Scheduled tokens.")
    num_reqs: int = arg(unit="requests", doc="Requests the tokens belong to.")
    ngram_context_len: int = arg(unit="tokens", doc="Predecessor tokens per request (ngram_size − 1).")
    heads_per_ngram: int = arg(unit="heads", doc="Hash heads per n-gram order.")


DOC = KernelDoc(
    title="PLE n-gram hash ids",
    summary="Hash each token's preceding n-grams into per-head embedding-table ids.",
    description=(
        "For every scheduled token, the kernel finds its request, walks back "
        "ngram_context_len predecessor tokens (from this step or the request's "
        "saved context, stopping at an end-of-sequence token) and, for each "
        "n-gram order and hash head, mixes the tokens with per-position "
        "multipliers by XOR and maps the hash into that head's table range."
    ),
    category="Other",
    formula=(
        "id[t, h] = offset[h] + (XORᵢ<n(h) token[t − i]·mult[i]) mod size[h]",
        "bytes = num_tokens · (4 + 8 · ngram_context_len · heads_per_ngram) + small context reads",
    ),
    default_metric="time_ms",
    method=(
        f"{CUPTI_METHOD} Only _ple_ngram_ids_kernel is counted. Tokens are split "
        "into num_reqs near-equal requests; ids are checked bit for bit against "
        "the fork's own eager PyTorch path (compute_ngram_ids on CPU tensors)."
    ),
    caveats=("TFLOPS is not computed.",),
    reference="profiling.runners.ple.ple_ngram_ids",
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        runner_ref=RunnerRef(
            module_name="profiling.runners.ple.ple_ngram_ids",
            function_name="profile_ple_ngram_ids_vllm_triton",
        ),
        table_name=KIND,
        args_schema=PleNgramIdsArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        # Integer-only Triton kernel; PDL is gated on the device inside.
        supports=BackendSupport(compute=None),
        subprocess_env="vllm_env",
        doc=BackendDoc(summary="The fork's ple_ngram_ids (_ple_ngram_ids_kernel)."),
    )
)


# The Flash-Next Trainium engine's exact int32-limb n-gram hash
# (`flashnext/neuron/ops/ple.py::ple_ngram_ids_torch`), equal-length requests
# [num_reqs, num_tokens / num_reqs], one region, NRT device time.
register(
    neuron_spec(
        KIND,
        "torch",
        PleNgramIdsArgs,
        "The Flash-Next Trainium engine's int32-limb n-gram hash, plain torch, one region.",
        compute=None,
    )
)

__all__ = ["KIND", "PleNgramIdsArgs"]
