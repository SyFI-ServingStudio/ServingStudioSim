"""DSA decode (Neuron).

The whole DSA block of one rank over a decode batch at one context bucket.

Wire string ``'dsa_decode'`` (Rust ``KernelSpec::KIND``); args and their order follow
``doc/pr/trn2-glm53-flash/L1_CONTRACT.md``. Every backend is ``neuron_<impl>``; rows are imported from the glm5-trn2
bench (bench_dsa.py kind dsa_decode) by ``tools/neuron-bench-import`` and are never profiled live here.
"""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import arg
from profiling.db.registry import register
from profiling.kernels.neuron_common import GLM_IMPLS, neuron_spec

KIND = "dsa_decode"


@dataclass(frozen=True)
class DsaDecodeArgs(KernelArgs):
    batch_size: int = arg(unit="tokens", doc="Decode rows (requests, bucket-padded).")
    context_len: int = arg(unit="tokens", doc="Context bucket the region reads ([:C]).")
    hidden_size: int = arg(unit="elements", doc="Model hidden width.")
    num_heads: int = arg(unit="elements", doc="MLA heads on this rank (64 / attn_tp).")
    kv_lora_rank: int = arg(unit="elements", doc="Latent KV width.")
    head_dim: int = arg(unit="elements", doc="MLA query/key (NoPE) and value head dimension.")
    index_heads: int = arg(unit="elements", doc="Indexer heads (replicated).")
    index_head_dim: int = arg(unit="elements", doc="Indexer head dimension.")
    index_topk: int = arg(unit="elements", doc="Selected tokens per query row.")
    dtype: DType = arg(doc="Activation dtype.")
    weight_dtype: DType = arg(doc="Weight storage dtype.")


for _impl in GLM_IMPLS[KIND]:
    register(neuron_spec(KIND, _impl, DsaDecodeArgs, "DSA decode (Neuron), impl " + _impl + "."))

__all__ = ["KIND", "DsaDecodeArgs"]
