"""GDN prefill mixer on the Flash-Next Trainium engine.

One Gated DeltaNet layer over one prefill chunk, in_proj to out_proj, as one compiled region.

Wire string ``'gdn_prefill_mixer'`` (Rust ``KernelSpec::KIND``); args and their order follow
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

KIND = "gdn_prefill_mixer"


@dataclass(frozen=True)
class GdnPrefillMixerArgs(KernelArgs):
    num_tokens: int = c.num_tokens()
    hidden_size: int = c.hidden_size()
    num_key_heads: int = c.num_key_heads()
    num_value_heads: int = c.num_value_heads()
    head_dim: int = c.head_dim()
    conv_kernel_size: int = c.conv_kernel_size()
    dtype: DType = c.dtype()


DOC = KernelDoc(
    title="GDN prefill mixer (Neuron)",
    summary=(
        "One Gated DeltaNet layer over one prefill chunk, in_proj to out_proj, as one "
        "compiled region."
    ),
    description=(
        "The Flash-Next Trainium engine's whole GDN mixer (ops/gdn.py gdn_prefill) for "
        "one request chunk of num_tokens tokens: fused in_proj, causal conv1d, split and "
        "gating, the chunked gated delta rule, sigmoid-gated RMSNorm and out_proj, with "
        "the request's conv and SSM states gathered from and scattered back to the state "
        "pools inside the region. The output is the rank's TP partial sum; the all-reduce"
        " is not included."
    ),
    category=c.CATEGORY,
    subcategory=c.SUBCATEGORY,
    formula=(
        "y = out_proj(norm(o) · σ(z)), (o, S) = GDN(conv(in_proj(x)), S_pool[slot])",
        "conv_pool[slot], ssm_pool[slot] ← new states (in place)",
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
        GdnPrefillMixerArgs,
        (
            "Whole mixer as one region with the conv and the chunked (WY, chunk 64) core "
            "in plain torch, fusion barriers around the core."
        ),
    )
)
register(
    neuron_spec(
        KIND,
        "nki",
        GdnPrefillMixerArgs,
        (
            "Whole mixer as one region with nkilib gdn_conv1d_prefill and gdn_cte (schur,"
            " chunk 128) for the conv and the core."
        ),
    )
)

__all__ = ["KIND", "GdnPrefillMixerArgs"]
