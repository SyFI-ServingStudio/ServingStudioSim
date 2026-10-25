"""GDN prefill conv1d on the Flash-Next Trainium engine.

Depthwise causal conv1d + SiLU over one prefill chunk, continuing the request's conv state.

Wire string ``'gdn_conv1d_prefill'`` (Rust ``KernelSpec::KIND``); args and their order follow
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

KIND = "gdn_conv1d_prefill"


@dataclass(frozen=True)
class GdnConv1dPrefillArgs(KernelArgs):
    num_tokens: int = c.num_tokens()
    hidden_size: int = c.hidden_size()
    num_key_heads: int = c.num_key_heads()
    num_value_heads: int = c.num_value_heads()
    head_dim: int = c.head_dim()
    conv_kernel_size: int = c.conv_kernel_size()
    dtype: DType = c.dtype()


DOC = KernelDoc(
    title="GDN prefill conv1d (Neuron)",
    summary=(
        "Depthwise causal conv1d + SiLU over one prefill chunk, continuing the request's "
        "conv state."
    ),
    description=(
        "The short convolution over the mixed qkv channels of one request chunk, starting"
        " from the previous chunk's last conv_kernel_size - 1 raw inputs, and the new "
        "conv state taken at the last real token."
    ),
    category=c.CATEGORY,
    subcategory=c.SUBCATEGORY,
    formula=(
        "C = (2·num_key_heads + num_value_heads)·head_dim",
        "y[t, c] = SiLU(Σ_j w[c, j] · x[t - K + 1 + j, c])",
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
        GdnConv1dPrefillArgs,
        ("Depthwise causal conv1d + SiLU in plain torch with the incoming conv state prepended."),
    )
)
register(
    neuron_spec(
        KIND,
        "nki",
        GdnConv1dPrefillArgs,
        ("nkilib gdn_conv1d_prefill on [1, C, 3 + T] (the incoming state prepended)."),
    )
)

__all__ = ["KIND", "GdnConv1dPrefillArgs"]
