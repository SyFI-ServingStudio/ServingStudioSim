"""Dense FFN partial (Neuron).

One rank's 1/ep column shard of a dense FFN layer (layers 0-2) over every gathered row.

Wire string ``'dense_ffn_partial'`` (Rust ``KernelSpec::KIND``); args and their order follow
``doc/pr/trn2-glm53-flash/L1_CONTRACT.md``. Every backend is ``neuron_<impl>``; rows are imported from the glm5-trn2
bench (bench_moe.py kind dense_ffn_partial) by ``tools/neuron-bench-import`` and are never profiled live here.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import arg
from profiling.db.registry import register
from profiling.kernels.neuron_common import GLM_IMPLS, neuron_spec

KIND = "dense_ffn_partial"


@dataclass(frozen=True)
class DenseFfnPartialArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Rows gathered over the world.")
    hidden_size: int = arg(unit="elements", doc="Model hidden width.")
    intermediate_size: int = arg(unit="elements", doc="Intermediate columns on this rank (12288 / ep).")
    dtype: DType = arg(doc="Activation dtype.")
    weight_dtype: DType = arg(doc="Weight storage dtype (FP8, upcast in-graph).")


for _impl in GLM_IMPLS[KIND]:
    register(neuron_spec(KIND, _impl, DenseFfnPartialArgs, "Dense FFN partial (Neuron), impl " + _impl + "."))

__all__ = ["KIND", "DenseFfnPartialArgs"]
