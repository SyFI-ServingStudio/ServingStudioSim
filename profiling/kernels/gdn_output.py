"""GDN output on the Flash-Next Trainium engine.

Sigmoid-gated RMSNorm of the core output followed by the row-parallel out_proj.

Wire string ``'gdn_output'`` (Rust ``KernelSpec::KIND``); args and their order follow
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

KIND = "gdn_output"


@dataclass(frozen=True)
class GdnOutputArgs(KernelArgs):
    num_tokens: int = c.num_tokens()
    hidden_size: int = c.hidden_size()
    num_key_heads: int = c.num_key_heads()
    num_value_heads: int = c.num_value_heads()
    head_dim: int = c.head_dim()
    conv_kernel_size: int = c.conv_kernel_size()
    dtype: DType = c.dtype()


DOC = KernelDoc(
    title="GDN output (Neuron)",
    summary=("Sigmoid-gated RMSNorm of the core output followed by the row-parallel out_proj."),
    description=(
        "RMSNorm(o)·w·σ(z) over the rank's value heads, then out_proj back to "
        "hidden_size. The result is the rank's TP partial sum; the all-reduce is not "
        "included."
    ),
    category=c.CATEGORY,
    subcategory=c.SUBCATEGORY,
    formula=(
        "y = out_proj(RMSNorm(o)·w·σ(z))",
        "FLOPs = 2·num_tokens·num_value_heads·head_dim·hidden_size",
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
        GdnOutputArgs,
        ("Sigmoid-gated RMSNorm and the row-parallel out_proj GEMM in plain torch."),
    )
)

__all__ = ["KIND", "GdnOutputArgs"]
