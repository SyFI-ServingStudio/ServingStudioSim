"""GDN decode state I/O on the Flash-Next Trainium engine.

Gather a decode batch's conv and SSM states from the pools and scatter them back.

Wire string ``'gdn_state_io'`` (Rust ``KernelSpec::KIND``); args and their order follow
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

KIND = "gdn_state_io"


@dataclass(frozen=True)
class GdnStateIoArgs(KernelArgs):
    batch_size: int = c.batch_size()
    hidden_size: int = c.hidden_size()
    num_key_heads: int = c.num_key_heads()
    num_value_heads: int = c.num_value_heads()
    head_dim: int = c.head_dim()
    conv_kernel_size: int = c.conv_kernel_size()
    dtype: DType = c.dtype()


DOC = KernelDoc(
    title="GDN decode state I/O (Neuron)",
    summary=("Gather a decode batch's conv and SSM states from the pools and scatter them back."),
    description=(
        "The state-pool traffic of one decode step on its own: batch_size conv states [C,"
        " conv_kernel_size - 1] bf16 and SSM states [num_value_heads, head_dim, head_dim]"
        " fp32 read from their slots and written to other slots."
    ),
    category=c.CATEGORY,
    subcategory=c.SUBCATEGORY,
    formula=("bytes = 2·batch_size·(2·C·(K - 1) + 4·num_value_heads·head_dim²)",),
    default_metric="time_ms",
    method=c.NEURON_METHOD,
    # The engine's ops are checked against flashnext/reference (CPU fp32) by the project's tests.
    reference=None,
)

register(
    neuron_spec(
        KIND,
        "torch",
        GdnStateIoArgs,
        ("pool[slots] gather and pool[out_slots] = state scatter (index_put_) in plain torch."),
    )
)

__all__ = ["KIND", "GdnStateIoArgs"]
