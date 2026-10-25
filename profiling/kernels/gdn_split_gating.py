"""GDN split and gating on the Flash-Next Trainium engine.

Split the conv output into l2-normalized q, k and v, and compute the decay g and beta.

Wire string ``'gdn_split_gating'`` (Rust ``KernelSpec::KIND``); args and their order follow
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

KIND = "gdn_split_gating"


@dataclass(frozen=True)
class GdnSplitGatingArgs(KernelArgs):
    num_tokens: int = c.num_tokens()
    hidden_size: int = c.hidden_size()
    num_key_heads: int = c.num_key_heads()
    num_value_heads: int = c.num_value_heads()
    head_dim: int = c.head_dim()
    conv_kernel_size: int = c.conv_kernel_size()
    dtype: DType = c.dtype()


DOC = KernelDoc(
    title="GDN split and gating (Neuron)",
    summary=(
        "Split the conv output into l2-normalized q, k and v, and compute the decay g and beta."
    ),
    description=(
        "Between the conv and the chunked core in prefill: q and k are l2-normalized in "
        "fp32 (GQA not expanded), g = -exp(A_log)·softplus(a + dt_bias) and beta = "
        "sigmoid(b), with padding tokens masked to g = beta = 0."
    ),
    category=c.CATEGORY,
    subcategory=c.SUBCATEGORY,
    formula=("q, k = l2norm(split(y)), g = -exp(A_log)·softplus(a + dt_bias), β = σ(b)",),
    default_metric="time_ms",
    method=c.NEURON_METHOD,
    # The engine's ops are checked against flashnext/reference (CPU fp32) by the project's tests.
    reference=None,
)

register(
    neuron_spec(
        KIND,
        "torch",
        GdnSplitGatingArgs,
        ("Split + l2norm of q and k, and the decay and beta gating, in plain torch."),
    )
)

__all__ = ["KIND", "GdnSplitGatingArgs"]
