"""Shared pieces of the Neuron Gated DeltaNet kinds (Flash-Next on Trainium2).

Not a kernel kind: the ``gdn_*`` Neuron kind modules (``gdn_prefill_mixer``,
``gdn_decode_mixer``, ``gdn_in_proj``, ...) declare their args from these, so
the common args (after each kind's shape args, in the order of
``doc/pr/trn2-flashnext/L1_CONTRACT.md``) read the same everywhere:
``hidden_size, num_key_heads, num_value_heads, head_dim, conv_kernel_size, dtype``.

Shapes are per rank: tensor parallelism splits the heads, so a rank at TP size
``tp`` holds ``16 / tp`` key heads and ``48 / tp`` value heads (tp 1, 2, 4, 8, 16).
"""

from __future__ import annotations

from typing import Any

from profiling.db.doc import arg

CATEGORY = "Attention"
SUBCATEGORY = "Gated DeltaNet"

#: How every Neuron GDN kind is timed (``flashnext.neuron.sim_l1.measure``).
NEURON_METHOD = (
    "NRT system-trace device time of one compiled region on one Trainium2 LNC2 "
    "logical core: the median over 30 calls, summed over the NEFFs one call "
    "launches (flashnext.neuron.runtime.device_bench). Compilation, input "
    "construction and warm-up stay outside the trace. Weights are the real "
    "layer-0 GDN weights of the rank-0 TP shard; activations and pool contents "
    "are seeded random, built as the project's bench_gdn.py builds them. Host "
    "launch overhead is the separate kind region_launch_overhead."
)

#: Op module the Neuron GDN backends time.
OPS_MODULE = "flashnext/neuron/ops/gdn.py"


def num_tokens() -> Any:
    return arg(unit="tokens", doc="Tokens in the call (one request chunk in prefill).")


def batch_size() -> Any:
    return arg(unit="requests", doc="Decode requests in the batch.")


def query_len() -> Any:
    return arg(
        unit="tokens",
        doc="Tokens per request: 1 in decode, 1 + k when verifying k MTP draft tokens.",
    )


def hidden_size() -> Any:
    return arg(unit="elements", doc="Model hidden size: the mixer's input and output width.")


def num_key_heads() -> Any:
    return arg(unit="heads", doc="Query and key heads on this rank (16 / tp for Flash-Next).")


def num_value_heads() -> Any:
    return arg(unit="heads", doc="Value heads and recurrent states on this rank (48 / tp).")


def head_dim() -> Any:
    return arg(unit="elements", doc="Features in each query, key and value head.")


def conv_kernel_size() -> Any:
    return arg(unit="tokens", doc="Taps of the depthwise causal conv1d.")


def dtype() -> Any:
    return arg(doc="Element type of activations and weights; the recurrent state is fp32.")
