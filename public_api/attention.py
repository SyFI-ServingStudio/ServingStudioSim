"""``GET /attention``: one member's attention at a context length.

Today its ``kv_cache``: what one request's KV cache holds at ``tokens`` context.

The arch declares its cache part by part (the ``cost-trees`` ``kv_cache`` the
tree route serves): each part grows with every token of context (or every
``tokens_per_entry`` tokens, when compressed), keeps a sliding window of recent
tokens, or holds one state per request. Each part's entry comes in three sizes:
stored once (``one_copy``, the least the model needs), on one GPU of the KV
partition (``per_gpu``), and over the partition (``partition``). A request of
``tokens`` context holds ``entries`` of each part, so each size is entries
times the entry's.

These are the model's own bytes, as the Models page's KV slider shows them.
What an engine adds on top (vLLM's padded entries, block rounding, extra state
copies) and how many requests fit a memory budget are a simulation member's
(``/simulations/presets``), not this route's. A pipeline-parallel member's
layout spans its stages as one partition, so ``per_gpu`` spreads the layers
evenly over them; with unequal stages the stage with the most layers holds
more, as a sim member's ``kv_cache`` reports it for that stage.
"""

from __future__ import annotations

import math

SIZES = ("one_copy", "per_gpu", "partition")


class NoKvCache(LookupError):
    """The member keeps no KV cache, as an FFN side does."""


class BadTokens(ValueError):
    """A context length the member cannot hold."""


def _entry(part: dict) -> dict[str, float]:
    return {
        "one_copy": part["one_copy_bytes_per_entry"],
        "per_gpu": part["per_gpu_bytes_per_entry"],
        "partition": part["bytes_per_entry"],
    }


def _times(sizes: dict[str, float], count: float) -> dict[str, float]:
    return {name: sizes[name] * count for name in SIZES}


def _plus(a: dict[str, float] | None, b: dict[str, float]) -> dict[str, float]:
    return b if a is None else {name: a[name] + b[name] for name in SIZES}


def sizes(kv: dict, tokens: int) -> dict:
    """One request of ``tokens`` context on the layout ``kv``: per part, what
    it holds and its bytes in the three sizes; their total; and, by how parts
    grow, what each further token adds (``per_token``), the windowed parts when
    full (``window``) and the state each request keeps (``per_request``), each
    null when no part grows so. A part's ``share`` is of the per-GPU total."""
    per_token = window = per_request = None
    parts = []
    for part in kv["layout"]["parts"]:
        growth = part["growth"]
        entry = _entry(part)
        if growth["kind"] == "per_token":
            per_token = _plus(per_token, _times(entry, 1 / growth["tokens_per_entry"]))
            held = tokens
            entries = math.ceil(tokens / growth["tokens_per_entry"])
        elif growth["kind"] == "sliding_window":
            window = _plus(window, _times(entry, growth["window_tokens"]))
            held = entries = min(tokens, growth["window_tokens"])
        elif growth["kind"] == "per_request":
            per_request = _plus(per_request, entry)
            held, entries = None, 1
        else:
            raise RuntimeError(f"part {part['name']}: unknown growth {growth['kind']!r}")
        parts.append(
            {
                "name": part["name"],
                "growth": growth,
                "tokens_held": held,
                "entries": entries,
                "bytes_per_entry": entry,
                "bytes": _times(entry, entries),
                "placement": part.get("placement"),
                "prefix_reusable": part.get("prefix_reusable"),
            }
        )
    total = {name: sum(p["bytes"][name] for p in parts) for name in SIZES}
    for part in parts:
        part["share"] = part["bytes"]["per_gpu"] / total["per_gpu"] if total["per_gpu"] else 0.0
    return {
        "parts": parts,
        "total": total,
        "per_token": per_token,
        "window": window,
        "per_request": per_request,
    }


def document(index, preset_id: str, params: dict, tokens_text: str | None) -> dict:
    """The ``GET /attention`` answer for one member of ``index`` at the
    context ``tokens_text`` spells."""
    _, member = index.member(preset_id, params)
    if member.kv_cache is None:
        raise NoKvCache(f"{preset_id} {member.params} keeps no KV cache")
    try:
        tokens = int(tokens_text or "")
    except ValueError:
        raise BadTokens(f"tokens={tokens_text!r}: give a context length in tokens") from None
    longest = (member.predict or {}).get("max_model_len")
    if tokens < 1 or (longest is not None and tokens > longest):
        raise BadTokens(f"tokens must be from 1 to max_model_len {longest}, not {tokens}")
    return {
        "sim_commit": index.sim_commit,
        "preset": preset_id,
        "params": member.params,
        "gpu": member.gpu,
        "gpus_per_replica": member.gpus_per_replica,
        "max_model_len": longest,
        "tokens": tokens,
        "kv_cache": {
            "attention_shards": member.kv_cache["attention_shards"],
            **sizes(member.kv_cache, tokens),
        },
    }
