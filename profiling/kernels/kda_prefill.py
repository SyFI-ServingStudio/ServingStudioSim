"""KDA prefill (Neuron).

The whole KDA block of one rank over one request chunk (projections, short conv, chunked delta rule, gated norm, o_proj partial).

Wire string ``'kda_prefill'`` (Rust ``KernelSpec::KIND``); args and their order follow
``doc/pr/trn2-glm53-flash/L1_CONTRACT.md``. Every backend is ``neuron_<impl>``; rows are imported from the glm5-trn2
bench (bench_kda.py kind kda_prefill) by ``tools/neuron-bench-import`` and are never profiled live here.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import arg
from profiling.db.registry import register
from profiling.kernels.neuron_common import GLM_IMPLS, neuron_spec

KIND = "kda_prefill"


@dataclass(frozen=True)
class KdaPrefillArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Tokens of the one request chunk (bucket-padded).")
    hidden_size: int = arg(unit="elements", doc="Model hidden width.")
    num_heads: int = arg(unit="elements", doc="KDA heads on this rank (64 / attn_tp).")
    head_dim: int = arg(unit="elements", doc="KDA head dimension.")
    conv_kernel_size: int = arg(unit="elements", doc="Short-conv kernel width.")
    dtype: DType = arg(doc="Activation dtype.")
    weight_dtype: DType = arg(doc="Weight storage dtype (attention stays BF16 in the FP8 checkpoint).")


for _impl in GLM_IMPLS[KIND]:
    register(neuron_spec(KIND, _impl, KdaPrefillArgs, "KDA prefill (Neuron), impl " + _impl + "."))

__all__ = ["KIND", "KdaPrefillArgs"]
