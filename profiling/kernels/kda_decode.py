"""KDA decode (Neuron).

The whole KDA block of one rank over a decode batch (one token per request, recurrent state update).

Wire string ``'kda_decode'`` (Rust ``KernelSpec::KIND``); args and their order follow
``doc/pr/trn2-glm53-flash/L1_CONTRACT.md``. Every backend is ``neuron_<impl>``; rows are imported from the glm5-trn2
bench (bench_kda.py kind kda_decode) by ``tools/neuron-bench-import`` and are never profiled live here.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import arg
from profiling.db.registry import register
from profiling.kernels.neuron_common import GLM_IMPLS, neuron_spec

KIND = "kda_decode"


@dataclass(frozen=True)
class KdaDecodeArgs(KernelArgs):
    batch_size: int = arg(unit="tokens", doc="Decode rows (requests, bucket-padded).")
    hidden_size: int = arg(unit="elements", doc="Model hidden width.")
    num_heads: int = arg(unit="elements", doc="KDA heads on this rank (64 / attn_tp).")
    head_dim: int = arg(unit="elements", doc="KDA head dimension.")
    conv_kernel_size: int = arg(unit="elements", doc="Short-conv kernel width.")
    dtype: DType = arg(doc="Activation dtype.")
    weight_dtype: DType = arg(doc="Weight storage dtype.")


for _impl in GLM_IMPLS[KIND]:
    register(neuron_spec(KIND, _impl, KdaDecodeArgs, "KDA decode (Neuron), impl " + _impl + "."))

__all__ = ["KIND", "KdaDecodeArgs"]
