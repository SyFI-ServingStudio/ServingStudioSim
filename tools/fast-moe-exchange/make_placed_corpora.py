"""Write expert-renumbered routing corpora for three EP placements.

A placement is an offline renumbering of each layer's expert ids (router output
columns and expert weights permuted together); at run time the EP split is
still "contiguous by id", so a renumbered corpus run through the simulator's
standard `routing: corpus` path prices the placement's per-rank expert load.

Placements (per layer, `ranks` equal ranks of E / ranks experts):
  contig : the recorded ids (identity);
  balance: popularity snake only (load balance, no co-occurrence);
  cooccur: the snake, then pairwise swaps that raise intra-rank co-occurrence
           while every rank's (token, expert) pair load stays <= cap x mean.
Permutations are fitted on the FIRST half of the tokens (the corpus is the
capture's requests concatenated in order, so the halves are disjoint request
sets) and the corpora are written twice: `eval` = the second half only
(held out), `full` = all tokens (half of it is the fitting data: optimistic).

Usage: python make_placed_corpora.py <manifest.json> <out_dir> [ranks] [cap]
Writes <out_dir>/<placement>_<eval|full>/{manifest.json,routes.u16} and
<out_dir>/placement_stats.json (D and busiest-rank pair share per corpus).
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from alignment.profiler.token_corpus import fnv1a64  # noqa: E402

man = Path(sys.argv[1]); out = Path(sys.argv[2])
ranks = int(sys.argv[3]) if len(sys.argv) > 3 else 8
cap = float(sys.argv[4]) if len(sys.argv) > 4 else 1.05
m = json.loads(man.read_text())
T, L, E, K = m["num_tokens"], m["num_layers"], m["num_experts"], m["top_k"]
ids = np.fromfile(man.parent / m["data_file"], dtype=np.uint16).reshape(T, L, K).astype(np.int64)
per, half = E // ranks, T // 2
rng = np.random.default_rng(0)


def cooccur(sel):
    C = np.zeros((E, E))
    for a in range(K):
        for b in range(a + 1, K):
            np.add.at(C, (sel[:, a], sel[:, b]), 1)
    return C + C.T


def snake(pop):
    group = np.empty(E, int)
    for n, e in enumerate(np.argsort(-pop, kind="stable")):
        r = n % (2 * ranks)
        group[e] = r if r < ranks else 2 * ranks - 1 - r
    return group


def swaps(sel, group, pop):
    C = cooccur(sel)
    limit = cap * pop.sum() / ranks
    load = np.bincount(group, weights=pop, minlength=ranks)
    for _ in range(12):
        W = np.stack([C[:, group == g].sum(1) for g in range(ranks)], 1)
        improved = 0
        for i in rng.permutation(E):
            gi, gj = group[i], group
            delta = (W[i, gj] - W[i, gi]) + (W[np.arange(E), gi] - W[np.arange(E), gj]) - 2 * C[i]
            ok = (group != gi) & (load[gi] - pop[i] + pop <= limit) & (load[gj] - pop + pop[i] <= limit)
            delta[~ok] = -np.inf
            j = int(np.argmax(delta))
            if delta[j] > 1e-9:
                gjj = group[j]
                group[i], group[j] = gjj, gi
                load[gi] += pop[j] - pop[i]; load[gjj] += pop[i] - pop[j]
                W[:, gi] += C[:, j] - C[:, i]; W[:, gjj] += C[:, i] - C[:, j]
                improved += 1
        if not improved:
            break
    return group


def to_perm(group):
    """old id -> new id: rank r's experts become ids r*per .. r*per+per-1."""
    perm = np.empty(E, int)
    for r in range(ranks):
        members = np.flatnonzero(group == r)
        assert len(members) == per
        perm[members] = r * per + np.arange(per)
    return perm


perms = {"contig": [], "balance": [], "cooccur": []}
for layer in range(L):
    train = ids[:half, layer]
    pop = np.bincount(train.ravel(), minlength=E).astype(float)
    perms["contig"].append(np.arange(E))
    g = snake(pop)
    perms["balance"].append(to_perm(g))
    perms["cooccur"].append(to_perm(swaps(train, g.copy(), pop)))


def stats(x):
    r = x // per                                 # tokens x layers x K ranks
    r = np.sort(r, axis=2)
    D = float((1 + (np.diff(r, axis=2) != 0).sum(2)).mean())
    shares = [np.bincount(r[:, l].ravel(), minlength=ranks).max() / r[:, l].size * ranks for l in range(L)]
    return {"D": round(D, 4), "busiest_pair_share_x_mean": round(float(np.mean(shares)), 4)}


result = {"source_manifest": str(man), "fit": f"first half (tokens 0..{half})", "cap": cap, "ranks": ranks,
          "note": "busiest share = per layer max over ranks of (token, expert) pairs / mean, averaged over layers, whole corpus slice",
          "corpora": {}}
for name, ps in perms.items():
    P = np.stack(ps)                             # L x E
    new = P[np.arange(L)[None, :, None], ids]    # T x L x K
    np.save(out / f"perm_{name}.npy", P) if out.exists() else None
    for split, sl in (("eval", slice(half, T)), ("full", slice(0, T))):
        d = out / f"{name}_{split}"
        d.mkdir(parents=True, exist_ok=True)
        data = np.ascontiguousarray(new[sl]).astype("<u2")
        payload = data.tobytes()
        (d / "routes.u16").write_bytes(payload)
        (d / "manifest.json").write_text(json.dumps({
            "schema_version": 1, "data_file": "routes.u16", "num_tokens": int(data.shape[0]),
            "num_layers": L, "num_experts": E, "top_k": K, "checksum_fnv1a64": fnv1a64(payload)}, indent=2))
        result["corpora"][f"{name}_{split}"] = {"path": str((d / "manifest.json").resolve()), "tokens": int(data.shape[0]),
                                                **stats(new[sl])}
        print(name, split, result["corpora"][f"{name}_{split}"], flush=True)
    np.save(out / f"perm_{name}.npy", P)
(out / "placement_stats.json").write_text(json.dumps(result, indent=1))
