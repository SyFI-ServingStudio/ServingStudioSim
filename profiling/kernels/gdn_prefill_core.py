"""GDN prefill core on the Flash-Next Trainium engine.

The chunked gated delta rule over one prefill chunk, from and to the request's recurrent state.

Wire string ``'gdn_prefill_core'`` (Rust ``KernelSpec::KIND``); args and their order follow
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

KIND = "gdn_prefill_core"


@dataclass(frozen=True)
class GdnPrefillCoreArgs(KernelArgs):
    num_tokens: int = c.num_tokens()
    hidden_size: int = c.hidden_size()
    num_key_heads: int = c.num_key_heads()
    num_value_heads: int = c.num_value_heads()
    head_dim: int = c.head_dim()
    conv_kernel_size: int = c.conv_kernel_size()
    dtype: DType = c.dtype()


DOC = KernelDoc(
    title="GDN prefill core (Neuron)",
    summary=(
        "The chunked gated delta rule over one prefill chunk, from and to the request's "
        "recurrent state."
    ),
    description=(
        "The gated delta rule recurrence of one request chunk computed chunk-parallel: "
        "every token's output and the final state, starting from the request's incoming "
        "fp32 state."
    ),
    category=c.CATEGORY,
    subcategory=c.SUBCATEGORY,
    formula=(
        "S_t = exp(g_t)·S_{t-1} + β_t·k_t ⊗ (v_t - (exp(g_t)·S_{t-1})ᵀk_t)",
        "o_t = scale · S_tᵀ q_t",
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
        GdnPrefillCoreArgs,
        (
            "Static chunk-parallel WY form (chunk 64) in fp32 plain torch, the inter-"
            "chunk loop unrolled."
        ),
    )
)
register(
    neuron_spec(
        KIND,
        "nki",
        GdnPrefillCoreArgs,
        ("nkilib gdn_cte (schur, chunk 128) with native 3:1 GQA, T padded to a multiple of 128."),
    )
)

__all__ = ["KIND", "GdnPrefillCoreArgs"]
