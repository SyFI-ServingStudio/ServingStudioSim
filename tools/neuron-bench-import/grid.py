"""Fill every Neuron grid cell the GLM-5.3-Flash arch can ask for that no measured row covers.

A derived cell is ``analytic(cell) x r``, where ``r`` is the measured/analytic ratio:

- ``interp``: interpolated (log-log, piecewise linear in the first shape axis) from the measured
  cells of the same series (kind, backend, config); outside the measured range the endpoint
  ratio is held. A 2D series (DSA) interpolates within the cell's context column, else uses the
  nearest measured column.
- ``kind-median``: the series has no measured cell; the median ratio of the kind's measured
  cells at other layouts (another tp / ep) or impls.
- ``analytic``: the kind has no measured cell at all (r = 1).

Collective cells missing from the trn2-flashnext rows are interpolated in log-log over the
message size of the same group (``comm-interp``).
"""

from __future__ import annotations

import bisect
import json
import math
import statistics
from collections import defaultdict

import geometry as G
from analytic import estimate
from import_bench import ImportRow


def _interp(points: list[tuple[float, float]], x: float) -> float:
    """Piecewise-linear interpolation of log(y) over log(x); endpoints held outside."""
    pts = sorted(points)
    xs = [p[0] for p in pts]
    if x <= xs[0]:
        return pts[0][1]
    if x >= xs[-1]:
        return pts[-1][1]
    i = bisect.bisect_left(xs, x)
    (x0, y0), (x1, y1) = pts[i - 1], pts[i]
    f = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
    return math.exp(math.log(y0) + f * (math.log(y1) - math.log(y0)))


def _comm_interp(points: list[tuple[float, float]], x: float) -> float:
    """Log-log over the message size; linear-in-bytes extrapolation above the top point."""
    pts = sorted(points)
    if len(pts) == 1 or x <= pts[0][0]:
        return pts[0][1] if x <= pts[0][0] else pts[0][1] * x / pts[0][0]
    if x >= pts[-1][0]:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        return y1 + (x - x1) * max(0.0, (y1 - y0) / (x1 - x0))
    return _interp(pts, x)


def fill_grid(measured: list[ImportRow], tps: list[int], world: int) -> list[ImportRow]:
    have = {r.key for r in measured}
    out: list[ImportRow] = []
    # Measured ratios per (kind, backend, config-json) -> {cell: ratio}
    ratios: dict[tuple[str, str, str], dict[tuple, float]] = defaultdict(dict)
    kind_ratios: dict[str, list[float]] = defaultdict(list)
    backends: dict[str, set[str]] = defaultdict(set)
    for r in measured:
        if r.comm or not r.measured:
            continue
        backends[r.kind].add(r.backend)
        shape_names = _shape_names(r.kind)
        cell = tuple(r.args[n] for n in shape_names)
        cfg = {k: v for k, v in r.args.items() if k not in shape_names}
        ratio = r.time_ms / estimate(r.kind, r.args)[0]
        ratios[(r.kind, r.backend, json.dumps(cfg, sort_keys=True))][cell] = ratio
        kind_ratios[r.kind].append(ratio)

    for kind, names, cfg, cells in G.series(tps, world):
        default = G.DEFAULT_IMPL[kind]
        defaults = {f"neuron_{i}" for i in ((default,) if isinstance(default, str) else default)}
        for backend in sorted(backends[kind] | defaults):
            series = ratios.get((kind, backend, json.dumps(cfg, sort_keys=True)), {})
            same_kind = [
                v
                for (k, b, _), cs in ratios.items()
                if k == kind and b == backend
                for v in cs.values()
            ] or kind_ratios.get(kind, [])
            for cell in cells:
                args = dict(zip(names, cell), **cfg)
                row_key = (kind, backend, json.dumps(args, sort_keys=True))
                if row_key in have:
                    continue
                base, flops, nbytes = estimate(kind, args)
                if series:
                    r, how = _series_ratio(series, cell), f"interp(n={len(series)})"
                elif same_kind:
                    r, how = statistics.median(same_kind), f"kind-median(n={len(same_kind)})"
                else:
                    r, how = 1.0, "analytic"
                out.append(
                    ImportRow(kind, backend, args, base * r, f"derived:{how}", flops, nbytes)
                )
                have.add(row_key)

    # Collectives: every (op, group) the arch builds, over COMM_BYTES.
    comm: dict[tuple[str, int, str], list[tuple[float, float]]] = defaultdict(list)
    for r in measured:
        if r.comm:
            comm[(r.kind, r.args["num_gpus"], r.args["fabric"])].append(
                (r.args["message_size_bytes"], r.time_ms)
            )
    for op in ("all_gather", "reduce_scatter"):
        for n, fabric in G.comm_groups(tps, world):
            pts = comm.get((op, n, fabric))
            for b in G.COMM_BYTES:
                args = dict(num_gpus=n, message_size_bytes=b, dtype="bf16", fabric=fabric)
                row_key = (op, "neuron_cc", json.dumps(args, sort_keys=True))
                if row_key in have:
                    continue
                if not pts:
                    raise SystemExit(f"no collective rows for {op} over {n} ranks ({fabric})")
                t = _comm_interp(pts, b)
                out.append(
                    ImportRow(
                        op,
                        "neuron_cc",
                        args,
                        t,
                        f"derived:comm-interp(n={len(pts)})",
                        bytes=b,
                        comm=True,
                    )
                )
                have.add(row_key)
    return out


def _shape_names(kind: str) -> tuple[str, ...]:
    for k, names, _, _ in G.series([1], 32):
        if k == kind:
            return names
    raise KeyError(kind)


def _series_ratio(series: dict[tuple, float], cell: tuple) -> float:
    if len(cell) == 1:
        return _interp([(c[0], v) for c, v in series.items()], cell[0])
    rows, ctx = cell
    cols: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for (r, c), v in series.items():
        cols[c].append((r, v))
    best = min(cols, key=lambda c: abs(math.log(c) - math.log(ctx)))
    return _interp(cols[best], rows)
