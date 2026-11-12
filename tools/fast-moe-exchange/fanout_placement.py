"""Distinct EP ranks per token (D) under contiguous vs co-occurrence expert placement.

Reads a routing corpus (routes.u16, [token][layer][top_k]) and, per MoE layer,
places the experts on `ranks` ranks of equal size two ways:
  * contiguous: expert e on rank e // (E / ranks) (what the simulator and vLLM do);
  * co-occurrence: greedy grouping by pairwise co-occurrence, then pairwise-swap
    refinement of the intra-rank co-occurrence weight, fitted on the first half
    of the tokens and evaluated on the second half (held out).
Prints per-layer and mean D, and the busiest rank's share of (token, expert)
pairs (load balance cost of the placement). Usage:
  python fanout_placement.py <manifest.json> [ranks] [--out result.json]
"""
import json, sys
from pathlib import Path
import numpy as np

man = Path(sys.argv[1])
ranks = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 8
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
m = json.loads(man.read_text())
T, Lyr, E, K = m["num_tokens"], m["num_layers"], m["num_experts"], m["top_k"]
ids = np.fromfile(man.parent / m["data_file"], dtype=np.uint16).reshape(T, Lyr, K).astype(np.int64)
per = E // ranks
half = T // 2
rng = np.random.default_rng(0)


def fanout(sel, place):
    r = place[sel]                       # tokens x K rank ids
    r.sort(axis=1)
    return float((1 + (np.diff(r, axis=1) != 0).sum(1)).mean())


def busiest_share(sel, place):
    cnt = np.bincount(place[sel].ravel(), minlength=ranks)
    return float(cnt.max() / cnt.sum() * ranks)


def cooccur(sel):
    C = np.zeros((E, E))
    for a in range(K):
        for b in range(a + 1, K):
            np.add.at(C, (sel[:, a], sel[:, b]), 1)
    return C + C.T


def place_balanced(sel, cap):
    """Co-occurrence swaps from a popularity-snake start, keeping every rank's
    (token, expert) pair load <= cap x mean."""
    C = cooccur(sel)
    pop = np.bincount(sel.ravel(), minlength=E).astype(float)
    order = np.argsort(-pop)
    group = np.empty(E, int)
    for n, e in enumerate(order):
        r = n % (2 * ranks)
        group[e] = r if r < ranks else 2 * ranks - 1 - r
    limit = cap * pop.sum() / ranks
    load = np.bincount(group, weights=pop, minlength=ranks)
    for _ in range(12):
        W = np.stack([C[:, group == g].sum(1) for g in range(ranks)], 1)
        improved = 0
        for i in rng.permutation(E):
            gi = group[i]
            gj = group
            delta = (W[i, gj] - W[i, gi]) + (W[np.arange(E), gi] - W[np.arange(E), gj]) - 2 * C[i]
            new_i = load[gi] - pop[i] + pop          # load of gi after swap with j
            new_j = load[gj] - pop + pop[i]
            ok = (group != gi) & (new_i <= limit) & (new_j <= limit)
            delta[~ok] = -np.inf
            j = int(np.argmax(delta))
            if delta[j] > 1e-9:
                gjj = group[j]
                group[i], group[j] = gjj, gi
                load[gi] += pop[j] - pop[i]; load[gjj] += pop[i] - pop[j]
                W[:, gi] += C[:, j] - C[:, i]
                W[:, gjj] += C[:, i] - C[:, j]
                improved += 1
        if not improved:
            break
    return group


def place_cooccur(sel):
    C = cooccur(sel)
    pop = np.bincount(sel.ravel(), minlength=E)
    group = -np.ones(E, int)
    for g in range(ranks):
        free = np.flatnonzero(group < 0)
        seed = free[np.argmax(pop[free])]
        members = [seed]; group[seed] = g
        while len(members) < per:
            free = np.flatnonzero(group < 0)
            gain = C[np.ix_(free, members)].sum(1)
            pick = free[np.argmax(gain)]
            members.append(pick); group[pick] = g
    # pairwise swap refinement on intra-rank weight
    for _ in range(6):
        W = np.stack([C[:, group == g].sum(1) for g in range(ranks)], 1)  # E x ranks
        improved = 0
        for i in rng.permutation(E):
            gi = group[i]
            # best swap partner j in another group
            gj = group
            delta = (W[i, gj] - W[i, gi]) + (W[np.arange(E), gi] - W[np.arange(E), gj]) - 2 * C[i]
            delta[group == gi] = -np.inf
            j = int(np.argmax(delta))
            if delta[j] > 1e-9:
                gjj = group[j]
                group[i], group[j] = gjj, gi
                W[:, gi] += C[:, j] - C[:, i]
                W[:, gjj] += C[:, i] - C[:, j]
                improved += 1
        if not improved:
            break
    return group


contig = np.arange(E) // per
rows = []
for layer in range(Lyr):
    x = ids[:, layer]
    if (x == x[:1]).all():
        continue  # dense layer (no routing)
    train, test = x[:half], x[half:]
    g = place_cooccur(train)
    extra = {}
    for cap in (1.05, 1.26):
        b = place_balanced(train, cap)
        extra[f"D_balanced_cap{cap}"] = fanout(test, b)
        extra[f"busiest_share_balanced_cap{cap}"] = busiest_share(test, b)
    rows.append({"layer": layer, **extra,
                 "D_contiguous": fanout(test, contig),
                 "D_cooccur": fanout(test, g),
                 "D_cooccur_train": fanout(train, g),
                 "busiest_share_contiguous": busiest_share(test, contig),
                 "busiest_share_cooccur": busiest_share(test, g)})
    print(layer, {k: round(v, 3) for k, v in rows[-1].items() if k != "layer"}, flush=True)
p = 1.0
for i in range(K):
    p *= (E - per - i) / (E - i)
summary = {"ranks": ranks, "experts": E, "top_k": K, "tokens": T, "moe_layers": len(rows),
           "D_uniform": ranks * (1 - p),
           **{k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "layer"}}
print(json.dumps(summary, indent=1))
if out:
    Path(out).write_text(json.dumps({"summary": summary, "layers": rows}, indent=1))
