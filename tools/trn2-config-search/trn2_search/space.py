"""The configuration space of `qwen4_exp_neuron_bf16` on one trn2.48xlarge.

A *config* is a flat dict of knobs. Unified and MTP configs use the arch and
worker knob names directly (`attn_dp`, `max_batch_tokens`, ...); a P/D config
prefixes them with `prefill.` or `decode.` (plus `<pool>.replicas`). Knobs left
out keep the arch's own defaults (`simulator/src/arch/config.rs`
`NeuronFlashNextSpec`), so a config lists only what it changes.

`layout` is shorthand for the five layout knobs, `dp4-tp8-g8-ep32-mt1` =
attn_dp 4, attn_tp 8, gdn_tp 8, moe_ep 32, moe_tp 1. `replicas` defaults to as
many replicas as fit the host's 64 LNC2 cores.

The structural checks here only prune the enumeration; the simulator build
(`NeuronFlashNextParallel::validate`, the HBM check, the profile.db lookup) is
the authority, and `probe` runs it on every config.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re

HOST_CORES = 64
CORES_PER_CHIP = 4

# Model geometry (model/config/qwen3_8_flash_next_bf16.json).
QSA_HEADS = 24
GDN_KEY_HEADS, GDN_VALUE_HEADS = 16, 48
NUM_EXPERTS, MOE_INTERMEDIATE = 512, 640

FAMILIES = ("unified", "mtp", "pd")
POOLS = ("prefill", "decode")

# Knobs that live on the worker, not the arch.
WORKER_KNOBS = {
    "max_batch_tokens",
    "attn_gpu_memory_gb",
    "max_prefill_chunks_per_partition",
    "dp_placement",
}
# Knobs on the pool group.
GROUP_KNOBS = {"replicas"}
LAYOUT_KNOBS = ("attn_dp", "attn_tp", "gdn_tp", "moe_ep", "moe_tp")

_LAYOUT_RE = re.compile(r"^dp(\d+)-tp(\d+)-g(\d+)-ep(\d+)-mt(\d+)$")


def parse_layout(text: str) -> dict:
    m = _LAYOUT_RE.match(text)
    if not m:
        raise ValueError(f"layout {text!r} is not dp<D>-tp<T>-g<G>-ep<E>-mt<M>")
    return dict(zip(LAYOUT_KNOBS, map(int, m.groups())))


def layout_name(cfg: dict, prefix: str = "") -> str:
    v = [cfg.get(prefix + k) for k in LAYOUT_KNOBS]
    dp, tp, g, ep, mt = v
    g = g if g is not None else tp
    mt = mt if mt is not None else 1
    if ep is None and dp and tp:
        ep = dp * max(tp, g) // mt
    return f"dp{dp}-tp{tp}-g{g}-ep{ep}-mt{mt}"


def ranks(cfg: dict, prefix: str = "") -> int:
    tp = cfg.get(prefix + "attn_tp", 8)
    g = cfg.get(prefix + "gdn_tp", tp)
    return cfg.get(prefix + "attn_dp", 1) * max(tp, g)


def _pow2(x: int) -> bool:
    return x > 0 and x & (x - 1) == 0


def enumerate_layouts(rank_options=(32, 64)) -> list[str]:
    """Every layout the arch's structural rules accept for a replica of one of
    `rank_options` LNC2 cores."""
    out = []
    for r in rank_options:
        for attn_tp in (1, 2, 4, 8):
            for gdn_tp in (1, 2, 4, 8, 16):
                g = max(attn_tp, gdn_tp)
                if r % g or not _pow2(r // g):
                    continue
                for moe_tp in (1, 2, 4, 8, 16, 32, 64):
                    if r % moe_tp or MOE_INTERMEDIATE % moe_tp:
                        continue
                    ep = r // moe_tp
                    if NUM_EXPERTS % ep:
                        continue
                    out.append(f"dp{r // g}-tp{attn_tp}-g{gdn_tp}-ep{ep}-mt{moe_tp}")
    return out


def structural_errors(cfg: dict) -> list[str]:
    """Cheap mirror of the arch's structural checks (pruning only)."""
    family = cfg.get("family", "unified")
    errs: list[str] = []
    if family not in FAMILIES:
        return [f"unknown family {family!r}"]
    prefixes = [f"{p}." for p in POOLS] if family == "pd" else [""]
    used = 0
    for pre in prefixes:
        tp = cfg.get(pre + "attn_tp", 8)
        g = cfg.get(pre + "gdn_tp", tp)
        dp = cfg.get(pre + "attn_dp", 1)
        mt = cfg.get(pre + "moe_tp", 1)
        r = dp * max(tp, g)
        ep = cfg.get(pre + "moe_ep", r // max(mt, 1))
        where = pre.rstrip(".") or "replica"
        if not (_pow2(tp) and _pow2(g) and _pow2(dp)):
            errs.append(f"{where}: attn_tp, gdn_tp, attn_dp must be powers of two")
        if QSA_HEADS % tp:
            errs.append(f"{where}: attn_tp {tp} does not divide {QSA_HEADS} QSA heads")
        if GDN_KEY_HEADS % g or GDN_VALUE_HEADS % g:
            errs.append(f"{where}: gdn_tp {g} does not divide the GDN heads")
        if ep * mt != r:
            errs.append(f"{where}: moe_ep x moe_tp = {ep}x{mt} != {r} ranks")
        if NUM_EXPERTS % ep or MOE_INTERMEDIATE % mt:
            errs.append(f"{where}: moe_ep must divide 512 experts and moe_tp 640")
        if cfg.get(pre + "moe_dispatch", "all_gather") == "all_to_all" and mt != 1:
            errs.append(f"{where}: all_to_all dispatch needs moe_tp 1")
        reps = cfg.get(pre + "replicas", 1)
        used += reps * r
    if used > HOST_CORES:
        errs.append(f"{used} cores > the host's {HOST_CORES}")
    if family == "mtp":
        if int(cfg.get("draft_tokens", 3)) < 1:
            errs.append("mtp needs draft_tokens >= 1")
    elif "draft_tokens" in cfg:
        errs.append("draft_tokens only applies to the mtp family")
    return errs


def fill_replicas(cfg: dict) -> dict:
    """Default `replicas` to as many as fit the host (unified / mtp)."""
    if cfg.get("family", "unified") == "pd" or "replicas" in cfg:
        return cfg
    out = dict(cfg)
    out["replicas"] = max(1, HOST_CORES // ranks(cfg))
    return out


def expand_layout(cfg: dict) -> dict:
    """Replace `layout` / `<pool>.layout` shorthand by its five knobs."""
    out = {}
    for k, v in cfg.items():
        if k == "layout" or k.endswith(".layout"):
            pre = k[: -len("layout")]
            out.update({pre + kk: vv for kk, vv in parse_layout(v).items()})
        else:
            out[k] = v
    return out


def config_id(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def config_label(cfg: dict) -> str:
    """Path-safe, readable, unique: family, layout(s), hash."""
    fam = cfg.get("family", "unified")
    if fam == "pd":
        lay = "P" + layout_name(cfg, "prefill.") + "_D" + layout_name(cfg, "decode.")
    else:
        lay = layout_name(cfg)
    return f"{fam}_{lay}_{config_id(cfg)}"


def expand_stage(stage: dict) -> list[dict]:
    """The configs of one stage file (see `stages/README` in the tool README):

    - `family`, `base`: the starting point (`base` may hold `layout`).
    - `layouts`: a list, or `auto` (with `layout_ranks`), crossed with `grid`.
    - `grid`: knob -> values, crossed (cartesian).
    - `oat`: knob -> values, each varied alone from every grid point.
    - `seeds`: explicit configs (e.g. the top configs of an earlier stage),
      each also varied by `oat`.
    - `scenarios`: name -> knob overrides applied to every config (what-if).

    Returns configs carrying `family`, `scenario`, `label`, with duplicates and
    structurally invalid ones removed (the latter listed under `_pruned`).
    """
    family = stage.get("family", "unified")
    base = dict(stage.get("base", {}))
    base["family"] = family
    points: list[dict] = []
    layouts = stage.get("layouts")
    if layouts == "auto":
        layouts = enumerate_layouts(tuple(stage.get("layout_ranks", (32, 64))))
    grid = dict(stage.get("grid", {}))
    if layouts:
        grid = {"layout": list(layouts), **grid}
    keys = list(grid)
    # Without a grid the base is one point, unless seeds replace it.
    combos = itertools.product(*(grid[k] for k in keys)) if keys else [()]
    if not keys and stage.get("seeds"):
        combos = []
    for combo in combos:
        p = dict(base)
        p.update(dict(zip(keys, combo)))
        points.append(p)
    for seed in stage.get("seeds", []):
        p = dict(base)
        p.update(seed)
        p["family"] = family
        points.append(p)
    oat = stage.get("oat", {})
    varied = []
    for p in points:
        varied.append(p)
        for k, values in oat.items():
            for v in values:
                q = dict(p)
                q[k] = v
                varied.append(q)
    scenarios = stage.get("scenarios", {"base": {}})
    seen, out = set(), []
    for p in varied:
        core = fill_replicas(expand_layout(p))
        for sname, over in scenarios.items():
            cfg = dict(core)
            cfg.update(_pool_overrides(over) if family == "pd" else over)
            cfg["scenario"] = sname
            key = json.dumps(cfg, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            errs = structural_errors(cfg)
            if errs:
                continue
            cfg["label"] = config_label({k: v for k, v in cfg.items() if k != "label"})
            cfg["core_id"] = config_id(core)
            out.append(cfg)
    return out


def _pool_overrides(over: dict) -> dict:
    """A P/D scenario's unprefixed knobs apply to both pools."""
    out = {}
    for k, v in over.items():
        if k.split(".", 1)[0] in POOLS:
            out[k] = v
        else:
            out.update({f"{p}.{k}": v for p in POOLS})
    return out


def knobs(cfg: dict) -> dict:
    """The config without its bookkeeping keys."""
    return {k: v for k, v in cfg.items() if k not in BOOKKEEPING}


BOOKKEEPING = ("family", "scenario", "label", "core_id")
