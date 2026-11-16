"""LM head (Neuron).

Final stream mean + norm, the vocab-parallel lm_head GEMM over the rank's vocab shard and the local greedy argmax.

Wire string ``'lm_head'`` (Rust ``KernelSpec::KIND``); args and their order follow
``doc/pr/trn2-glm53-flash/L1_CONTRACT.md``. Every backend is ``neuron_<impl>``; rows are imported from the glm5-trn2
bench (bench_mhc_head.py kind lm_head) by ``tools/neuron-bench-import`` and are never profiled live here.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import arg
from profiling.db.registry import register
from profiling.kernels.neuron_common import GLM_IMPLS, neuron_spec

KIND = "lm_head"


@dataclass(frozen=True)
class LmHeadArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Rows sampled (decode rows, or one per prefill chunk).")
    hidden_size: int = arg(unit="elements", doc="Model hidden width.")
    vocab_size: int = arg(unit="elements", doc="Vocabulary rows on this rank (vocab / attn_tp).")
    hc_mult: int = arg(unit="elements", doc="Residual streams.")
    dtype: DType = arg(doc="Activation dtype.")


for _impl in GLM_IMPLS[KIND]:
    register(neuron_spec(KIND, _impl, LmHeadArgs, "LM head (Neuron), impl " + _impl + "."))

__all__ = ["KIND", "LmHeadArgs"]
