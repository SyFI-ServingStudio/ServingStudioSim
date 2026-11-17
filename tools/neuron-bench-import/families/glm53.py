"""GLM-5.3-Flash op rows from the glm5-trn2 benchmarks (``logs/bench/*.jsonl``).

Bench schema ``{kind, impl, phase, tokens, batch, context, tp, ep, dtype, device_ms, ...}``; one
compiled region per call, NRT device time (``runtime.device_bench``). Mapping to the sim kinds
(``doc/pr/trn2-glm53-flash/L1_CONTRACT.md``):

| bench kind | sim kind | shape args | config from |
|---|---|---|---|
| kda_prefill | kda_prefill | num_tokens = tokens | tp |
| kda_decode | kda_decode | batch_size = batch | tp |
| dsa_prefill | dsa_prefill | num_tokens = tokens, context_len = context | tp |
| dsa_decode | dsa_decode | batch_size = batch, context_len = context | tp |
| moe_partial | moe_partial | num_tokens = tokens (gathered rows) | ep |
| dense_ffn_partial | dense_ffn_partial | num_tokens | ep |
| router | moe_router (impl fp32) | num_tokens | - |
| mhc_pre, mhc_post | same | num_tokens | - |
| lm_head, embed | same | num_tokens | tp (vocab shard) |

Backend = ``neuron_<impl>``; a bench impl the registry does not know is skipped (counted).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import geometry as G
from analytic import estimate
from import_bench import ImportRow, device_ms, jsonl

SKIPPED: Counter = Counter()


def map_row(r: dict) -> tuple[str, str, dict] | None:
    from profiling.kernels.neuron_common import GLM_IMPLS

    kind, impl = r.get("kind"), r.get("impl")
    tp, ep = int(r.get("tp") or 1), int(r.get("ep") or 1)
    tokens, batch, ctx = r.get("tokens"), r.get("batch"), r.get("context")
    if kind == "router":
        kind, impl = "moe_router", "fp32"
    if kind not in GLM_IMPLS or impl not in GLM_IMPLS[kind]:
        SKIPPED[(kind, impl)] += 1
        return None
    if kind in ("kda_prefill", "kda_decode", "dsa_prefill", "dsa_decode", "lm_head", "embed"):
        if G.KDA_HEADS % tp:
            return None
    if kind == "kda_prefill":
        args = dict(num_tokens=int(tokens), **G.kda(tp))
    elif kind == "kda_decode":
        args = dict(batch_size=int(batch), **G.kda(tp))
    elif kind == "dsa_prefill":
        args = dict(num_tokens=int(tokens), context_len=int(ctx), **G.dsa(tp))
    elif kind == "dsa_decode":
        args = dict(batch_size=int(batch), context_len=int(ctx), **G.dsa(tp))
    elif kind == "moe_partial":
        args = dict(num_tokens=int(tokens), **G.moe(ep))
    elif kind == "dense_ffn_partial":
        args = dict(num_tokens=int(tokens), **G.dense(ep))
    elif kind == "moe_router":
        args = dict(num_tokens=int(tokens), **G.router())
    elif kind == "mhc_pre":
        args = dict(num_tokens=int(tokens), **G.mhc_pre())
    elif kind == "mhc_post":
        args = dict(num_tokens=int(tokens), **G.mhc_post())
    else:  # lm_head, embed
        args = dict(num_tokens=int(tokens), **G.head(tp))
    return kind, f"neuron_{impl}", args


def rows(bench_dir: Path):
    for path in sorted(Path(bench_dir).glob("*.jsonl")):
        for line, r in jsonl(path):
            ms = device_ms(r)
            if ms is None:
                continue
            mapped = map_row(r)
            if mapped is None:
                continue
            kind, backend, args = mapped
            flops, nbytes = estimate(kind, args)[1:]
            yield ImportRow(kind, backend, args, ms, f"imported:{path.name}:{line}", flops, nbytes)
