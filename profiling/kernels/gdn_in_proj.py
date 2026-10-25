"""GDN input projection on the Flash-Next Trainium engine.

The GDN in_proj GEMM producing qkv, z, b and a for the rank's heads.

Wire string ``'gdn_in_proj'`` (Rust ``KernelSpec::KIND``); args and their order follow
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

KIND = "gdn_in_proj"


@dataclass(frozen=True)
class GdnInProjArgs(KernelArgs):
    num_tokens: int = c.num_tokens()
    hidden_size: int = c.hidden_size()
    num_key_heads: int = c.num_key_heads()
    num_value_heads: int = c.num_value_heads()
    head_dim: int = c.head_dim()
    conv_kernel_size: int = c.conv_kernel_size()
    dtype: DType = c.dtype()


DOC = KernelDoc(
    title="GDN input projection (Neuron)",
    summary=("The GDN in_proj GEMM producing qkv, z, b and a for the rank's heads."),
    description=(
        "x [num_tokens, hidden_size] times the rank's column-parallel in_proj weight: "
        "conv channels 2·num_key_heads·head_dim + num_value_heads·head_dim, the z gate, "
        "and the b and a gating inputs, as one GEMM."
    ),
    category=c.CATEGORY,
    subcategory=c.SUBCATEGORY,
    formula=(
        "N = (2·num_key_heads + 2·num_value_heads)·head_dim + 2·num_value_heads",
        "FLOPs = 2·num_tokens·hidden_size·N",
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
        GdnInProjArgs,
        (
            "One fused GEMM over the concatenated [qkv; z; b; a] weight "
            "(gdn_in_proj_fused), as the mixers run it."
        ),
    )
)

__all__ = ["KIND", "GdnInProjArgs"]
