"""GDN decode conv1d on the Flash-Next Trainium engine.

Depthwise causal conv1d + SiLU for a decode or verify batch, updating each request's conv state.

Wire string ``'gdn_conv1d_decode'`` (Rust ``KernelSpec::KIND``); args and their order follow
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

KIND = "gdn_conv1d_decode"


@dataclass(frozen=True)
class GdnConv1dDecodeArgs(KernelArgs):
    batch_size: int = c.batch_size()
    query_len: int = c.query_len()
    hidden_size: int = c.hidden_size()
    num_key_heads: int = c.num_key_heads()
    num_value_heads: int = c.num_value_heads()
    head_dim: int = c.head_dim()
    conv_kernel_size: int = c.conv_kernel_size()
    dtype: DType = c.dtype()


DOC = KernelDoc(
    title="GDN decode conv1d (Neuron)",
    summary=(
        "Depthwise causal conv1d + SiLU for a decode or verify batch, updating each "
        "request's conv state."
    ),
    description=(
        "The short convolution for batch_size requests of query_len tokens from each "
        "request's conv state. With query_len > 1 (MTP verify) the state after every "
        "position is returned, as the verify mixer needs."
    ),
    category=c.CATEGORY,
    subcategory=c.SUBCATEGORY,
    formula=(
        "C = (2·num_key_heads + num_value_heads)·head_dim",
        "y[b, t, c] = SiLU(Σ_j w[c, j] · x[b, t - K + 1 + j, c])",
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
        GdnConv1dDecodeArgs,
        ("Decode conv in plain torch over the [B, C, K - 1] state window."),
    )
)
register(
    neuron_spec(
        KIND,
        "nki",
        GdnConv1dDecodeArgs,
        ("nkilib gdn_conv1d_decode, one call per token position."),
    )
)

__all__ = ["KIND", "GdnConv1dDecodeArgs"]
