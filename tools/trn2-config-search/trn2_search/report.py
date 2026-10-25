"""Tables over collected rows: Pareto frontiers, best configs, knob
sensitivity, what-if rank changes and the headroom ladder.

Every function takes the flat rows of `collect.collect` and returns plain
data; `markdown_*` helpers format them.
"""

from __future__ import annotations

import math
from collections import defaultdict

from .space import layout_name

NAN = float("nan")


def _num(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return NAN
    return v


def pareto(rows: list[dict], x: str, y: str) -> list[dict]:
    """Rows not dominated on (maximize `x`, minimize `y`), sorted by `x`."""
    pts = [r for r in rows if not math.isnan(_num(r.get(x))) and not math.isnan(_num(r.get(y)))]
    pts.sort(key=lambda r: (-_num(r[x]), _num(r[y])))
    out, best_y = [], math.inf
    for r in pts:
        if _num(r[y]) < best_y:
            out.append(r)
            best_y = _num(r[y])
    return sorted(out, key=lambda r: _num(r[x]))


def by_workload(rows: list[dict]) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        g[r.get("workload_base", r["workload"])].append(r)
    return dict(sorted(g.items()))


def best(rows: list[dict], metric: str, tiebreak: str = "total_tok_s") -> dict | None:
    pts = [r for r in rows if not math.isnan(_num(r.get(metric)))]
    if not pts:
        return None
    return max(pts, key=lambda r: (_num(r[metric]), _num(r.get(tiebreak))))


def knob_columns(rows: list[dict]) -> list[str]:
    """The `cfg.*` columns that take more than one value across `rows`."""
    cols = sorted({k for r in rows for k in r if k.startswith("cfg.")})
    skip = {"cfg.family", "cfg.core_id", "cfg.scenario", "cfg.accept_rate"}
    return [c for c in cols if c not in skip and len({repr(r.get(c)) for r in rows}) > 1]


def sensitivity(rows: list[dict], metric: str) -> list[dict]:
    """Per knob: the best `metric` reached at each of its values (over every
    other knob), and the spread `(max - min) / max` of those bests. A knob
    whose every value can reach the same best does not matter at this point
    of the space; a large spread means getting it wrong costs that fraction."""
    out = []
    for col in knob_columns(rows):
        per: dict[str, float] = {}
        for r in rows:
            v = _num(r.get(metric))
            if math.isnan(v):
                continue
            key = repr(r.get(col)) if col in r else "(default)"
            per[key] = max(per.get(key, -math.inf), v)
        if len(per) < 2:
            continue
        hi, lo = max(per.values()), min(per.values())
        out.append(
            {
                "knob": col[4:],
                "spread": (hi - lo) / hi if hi > 0 else NAN,
                "best_value": max(per, key=per.get),
                "per_value": per,
            }
        )
    out.sort(key=lambda d: -_num(d["spread"]))
    return out


def scenario_ranks(rows: list[dict], metric: str, base: str = "base") -> list[dict]:
    """Per scenario: the rank correlation (Kendall tau) of the core configs'
    `metric` against the `base` scenario, and the best core config in each."""
    per: dict[str, dict[str, float]] = defaultdict(dict)
    rep: dict[str, dict] = {}
    for r in rows:
        v = _num(r.get(metric))
        if math.isnan(v):
            continue
        key = f"{r['cfg.core_id']}@{r.get('cfg.accept_rate')}"
        scen = r.get("cfg.scenario", "base")
        per[scen][key] = v
        rep[(scen, key)] = r
    if base not in per:
        return []
    out = []
    for scen, vals in sorted(per.items()):
        common = sorted(set(vals) & set(per[base]))
        top = max(vals, key=vals.get)
        out.append(
            {
                "scenario": scen,
                "configs": len(common),
                "kendall_tau": kendall_tau(
                    [per[base][c] for c in common], [vals[c] for c in common]
                ),
                "best_core": top,
                "best_desc": describe(rep[(scen, top)]),
                "best_value": vals[top],
                "base_best_rank_here": _rank_of(vals, max(per[base], key=per[base].get)),
            }
        )
    return out


def _rank_of(vals: dict[str, float], key: str) -> int | None:
    if key not in vals:
        return None
    order = sorted(vals, key=lambda k: -vals[k])
    return order.index(key) + 1


def kendall_tau(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return NAN
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    tot = conc + disc
    return (conc - disc) / tot if tot else NAN


# What the R0 time is spent on, from the analyzer's cluster buckets.
LADDER_VIEW = (
    ("idle+imbalance", ("gap_idle", "gap_imbalance")),
    ("launch+batching (R2-R3)", ("gap_batching",)),
    ("communication (R3-R4)", ("gap_communication",)),
    ("kernel gap (R4-R4a)", ("gap_kernel_gap", "gap_power_gap")),
    ("hardware gap (R4b-R5)", ("gap_hardware_gap",)),
    ("excess over necessary", ("gap_excess_over_necessary", "gap_fusion")),
    ("necessary work", ("gap_hardware_necessary",)),
)


def ladder(row: dict) -> dict[str, float]:
    return {
        name: sum(_num(row.get(k)) or 0.0 for k in keys if row.get(k) is not None)
        for name, keys in LADDER_VIEW
    }


def describe(r: dict) -> str:
    """A compact config description from a row's `cfg.*` columns."""
    cfg = {k[4:]: v for k, v in r.items() if k.startswith("cfg.")}
    fam = cfg.get("family", "unified")
    skip = {
        "family",
        "scenario",
        "core_id",
        "accept_rate",
        "attn_dp",
        "attn_tp",
        "gdn_tp",
        "moe_ep",
        "moe_tp",
        "replicas",
    }
    if fam == "pd":
        lay = f"P[{layout_name(cfg, 'prefill.')}] D[{layout_name(cfg, 'decode.')}]"
        skip |= {
            f"{p}.{k}"
            for p in ("prefill", "decode")
            for k in ("attn_dp", "attn_tp", "gdn_tp", "moe_ep", "moe_tp", "replicas")
        }
    else:
        lay = f"{cfg.get('replicas', 1)}x {layout_name(cfg)}"
    rest = ", ".join(f"{k}={_short(v)}" for k, v in sorted(cfg.items()) if k not in skip)
    if cfg.get("accept_rate") is not None:
        lay += f" @accept {cfg['accept_rate']}"
    return f"{fam} {lay}" + (f" ({rest})" if rest else "")


def _short(v) -> str:
    if isinstance(v, list):
        return "[" + ",".join(map(str, v)) + "]"
    return str(v)


def fmt(x, digits: int = 1) -> str:
    v = _num(x)
    if math.isnan(v):
        return "-"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)
