"""GDN decode mixer on the Flash-Next Trainium engine.

One Gated DeltaNet layer for a decode or MTP-verify batch, with state-pool I/O, as one region.

Wire string ``'gdn_decode_mixer'`` (Rust ``KernelSpec::KIND``); args and their order follow
``doc/pr/trn2-flashnext/L1_CONTRACT.md``. Every backend is ``neuron_<impl>``, ``<impl>`` the
impl key the Flash-Next engine's ``gdn`` ops use; the region is built by
``flashnext.neuron.sim_l1.gdn`` (inputs as in ``bench/bench_gdn.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import KernelDoc
from profiling.db.registry import register
from profiling.kernels import gdn_neuron_common as c
from profiling.kernels.neuron_common import neuron_spec

KIND = "gdn_decode_mixer"


@dataclass(frozen=True)
class GdnDecodeMixerArgs(KernelArgs):
    batch_size: int = c.batch_size()
    query_len: int = c.query_len()
    hidden_size: int = c.hidden_size()
    num_key_heads: int = c.num_key_heads()
    num_value_heads: int = c.num_value_heads()
    head_dim: int = c.head_dim()
    conv_kernel_size: int = c.conv_kernel_size()
    dtype: DType = c.dtype()


DOC = KernelDoc(
    title="GDN decode mixer (Neuron)",
    summary=(
        "One Gated DeltaNet layer for a decode or MTP-verify batch, with state-pool I/O, "
        "as one region."
    ),
    description=(
        "The Flash-Next Trainium engine's whole GDN mixer (ops/gdn.py gdn_decode) for "
        "batch_size requests of query_len tokens each: states gathered by slot from the "
        "pools, fused in_proj, conv, recurrence, gated norm and out_proj, and the new "
        "states scattered back. query_len 1 is decode; query_len 1 + k verifies k MTP "
        "draft tokens and writes the state after every position to its own slot. The "
        "output is the rank's TP partial sum."
    ),
    category=c.CATEGORY,
    subcategory=c.SUBCATEGORY,
    formula=(
        "y = out_proj(norm(o) · σ(z)), (o, S) = recurrence(conv(in_proj(x)), S_pool[slots])",
        "pool[out_slots[:, j]] ← state after token j",
    ),
    default_metric="time_ms",
    method=c.NEURON_METHOD,
    # The engine's ops are checked against flashnext/reference (CPU fp32) by the project's tests.
    reference=None,
)

register(
    neuron_spec(
        KIND,
        "torch",
        GdnDecodeMixerArgs,
        (
            "Whole mixer as one region in plain torch (gather and barrier choice from "
            "decode_region_policy)."
        ),
    )
)
register(
    neuron_spec(
        KIND,
        "nki",
        GdnDecodeMixerArgs,
        (
            "Whole mixer as one region with nkilib gdn_conv1d_decode and gdn_tkg for the "
            "conv and the recurrence."
        ),
    )
)

__all__ = ["KIND", "GdnDecodeMixerArgs"]
