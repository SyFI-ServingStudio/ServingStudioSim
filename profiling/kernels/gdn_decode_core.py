"""GDN decode core on the Flash-Next Trainium engine.

Decode-step split, gating, recurrence and gated RMSNorm for a batch of requests.

Wire string ``'gdn_decode_core'`` (Rust ``KernelSpec::KIND``); args and their order follow
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

KIND = "gdn_decode_core"


@dataclass(frozen=True)
class GdnDecodeCoreArgs(KernelArgs):
    batch_size: int = c.batch_size()
    query_len: int = c.query_len()
    hidden_size: int = c.hidden_size()
    num_key_heads: int = c.num_key_heads()
    num_value_heads: int = c.num_value_heads()
    head_dim: int = c.head_dim()
    conv_kernel_size: int = c.conv_kernel_size()
    dtype: DType = c.dtype()


DOC = KernelDoc(
    title="GDN decode core (Neuron)",
    summary=("Decode-step split, gating, recurrence and gated RMSNorm for a batch of requests."),
    description=(
        "Everything between the decode conv and out_proj for batch_size requests of "
        "query_len tokens: split and l2norm of q and k, decay and beta, the per-token "
        "delta-rule state update and the sigmoid-gated RMSNorm. With query_len > 1 the "
        "state after every position is returned."
    ),
    category=c.CATEGORY,
    subcategory=c.SUBCATEGORY,
    formula=(
        "S_t = exp(g_t)·S_{t-1} + β_t·k_t ⊗ (v_t - (exp(g_t)·S_{t-1})ᵀk_t)",
        "y_t = RMSNorm(scale · S_tᵀ q_t) · w · σ(z_t)",
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
        GdnDecodeCoreArgs,
        (
            "Split, l2norm, gating, the recurrence ([k; q] @ S per step) and the gated "
            "RMSNorm in plain torch."
        ),
    )
)
register(
    neuron_spec(
        KIND,
        "nki",
        GdnDecodeCoreArgs,
        (
            "nkilib gdn_tkg (l2norm, gating, recurrence and RMSNorm in the kernel), "
            "sigmoid gate in torch."
        ),
    )
)

__all__ = ["KIND", "GdnDecodeCoreArgs"]
