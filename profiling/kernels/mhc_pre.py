"""mHC pre (Neuron).

Hyper-connection pre: stream mixing weights (Sinkhorn), the collapse to one row and the module's RMSNorm, on the rank's own rows.

Wire string ``'mhc_pre'`` (Rust ``KernelSpec::KIND``); args and their order follow
``doc/pr/trn2-glm53-flash/L1_CONTRACT.md``. Every backend is ``neuron_<impl>``; rows are imported from the glm5-trn2
bench (bench_mhc_head.py kind mhc_pre) by ``tools/neuron-bench-import`` and are never profiled live here.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import arg
from profiling.db.registry import register
from profiling.kernels.neuron_common import GLM_IMPLS, neuron_spec

KIND = "mhc_pre"


@dataclass(frozen=True)
class MhcPreArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Rows on this rank (its sequence-parallel slice).")
    hidden_size: int = arg(unit="elements", doc="Model hidden width.")
    hc_mult: int = arg(unit="elements", doc="Residual streams.")
    sinkhorn_iters: int = arg(unit="elements", doc="Sinkhorn iterations.")
    dtype: DType = arg(doc="Activation dtype.")


for _impl in GLM_IMPLS[KIND]:
    register(neuron_spec(KIND, _impl, MhcPreArgs, "mHC pre (Neuron), impl " + _impl + "."))

__all__ = ["KIND", "MhcPreArgs"]
