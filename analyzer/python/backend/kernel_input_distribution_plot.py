"""Render one scatter per cost-tree position from the Rust
`kernel_input_distribution_scatter.json` payload.

Each figure is a position's inputs projected by the Rust side: `raw_2d` (two
features), `pca` (≥3 features → top-2 PCs), or `feature_1d` (a single feature).
Every point is colored + shaped by the backend best-of-N selected there, so a
multi-backend position shows its selection boundary directly; a single-backend
position still renders, annotated as such. Point size encodes how many sampled
slots collapsed into that `(input, backend)` observation.

A `feature_1d` position is drawn as a genuine **1-D strip**: the single feature is
the x-axis and the vertical axis is hidden — points get only a cosmetic vertical
spread to de-overlap, never a second data dimension. `categorical` (no numeric
feature) is the same strip with the x-axis hidden too — only the backend mix
reads. The 2-D projections (`raw_2d` / `pca`) use both axes as real dimensions.

One PNG per position → `plots/kernel_input_dist/<position>.png`. Rendering is
resumable (skip-if-newer) and each position is an independent picklable job, so a
run with hundreds of positions parallelizes across the render pool and an
interrupted rerun keeps finished images.

Reads only the payload — no parquet. Reuses the shared style/formatting
(`common.style.backend_style`, `common.figure`).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from pathlib import Path

from common.figure import corner_box, fmt_count, new_axes
from common.layout import load_payload, plot_output_path, resolve_artifact
from common.style import backend_style, plt, save_plot

PAYLOAD = "kernel_input_distribution_scatter.json"
PLOT_SUBDIR = "kernel_input_dist"

# Point-area encoding of a `(input, backend)` observation's count: a gentle sqrt
# so a hot input stands out without swamping the axes.
_MIN_AREA = 12.0
_MAX_AREA = 160.0


def render(log_dir: Path) -> list[Callable[[], Path]]:
    payload_path = resolve_artifact(log_dir, PAYLOAD)
    if not payload_path.is_file():
        return []
    payload = load_payload(log_dir, PAYLOAD)
    positions = payload.get("positions") or []
    if not positions:
        reason = payload.get("meta", {}).get("reason", "no positions in payload")
        print(f"[kernel_input_distribution] nothing to render: {reason}")
        return []

    run_label = Path(payload.get("meta", {}).get("log_dir", str(log_dir))).name
    specs: list[tuple[dict, Path]] = []
    for position in positions:
        if not position.get("points"):
            continue
        out_path = plot_output_path(log_dir, f"{PLOT_SUBDIR}/{_safe_name(position['name'])}.png")
        specs.append((position, out_path))

    _remove_stale_outputs(log_dir, {out for _, out in specs})

    jobs: list[Callable[[], Path]] = []
    for position, out_path in specs:
        # Resumable: a newer payload or renderer invalidates the image; otherwise an
        # interrupted rerun keeps the completed scatters.
        if _output_is_current(out_path, payload_path, Path(__file__)):
            continue
        jobs.append(partial(_render_position, position, out_path, run_label=run_label))
    if not jobs and specs:
        print("[kernel_input_distribution] all position scatters already current")
    return jobs


def _render_position(position: dict, out_path: Path, *, run_label: str) -> Path:
    """Draw one position's backend-colored scatter/strip to `out_path`."""
    points = position["points"]
    labels = position.get("axis_labels", ["", ""])
    candidates = position.get("candidate_backends", [])
    projection = position.get("projection", "")
    one_d = projection in ("feature_1d", "categorical")

    # Group points by backend so each backend is one scatter call (one legend entry
    # with its fixed color+marker). Draw less-frequent backends last so a rare one
    # isn't buried under a dominant one.
    by_backend: dict[str, list[dict]] = defaultdict(list)
    for pt in points:
        by_backend[pt.get("backend_name", "?")].append(pt)
    order = sorted(by_backend, key=lambda k: -len(by_backend[k]))

    # A 1-D strip is short and wide; a 2-D projection is a square-ish scatter.
    fig, ax = new_axes(figsize=(9.5, 3.4) if one_d else (8.5, 6.0))
    for name in order:
        group = by_backend[name]
        color, marker = backend_style(name)
        sizes = [_area(p.get("count", 1)) for p in group]
        if one_d:
            # x = the single feature (feature_1d) or a cosmetic spread (categorical,
            # no feature); y is always a cosmetic spread that the hidden axis marks
            # as non-data.
            xs = [p["x"] for p in group] if projection == "feature_1d" else _spread(len(group), name, 0)
            ys = _spread(len(group), name, 1)
        else:
            xs = [p["x"] for p in group]
            ys = [p["y"] for p in group]
        ax.scatter(
            xs, ys, s=sizes, c=color, marker=marker, alpha=0.55,
            linewidths=0.3, edgecolors="white", label=name,
        )

    _annotate(ax, position, candidates)
    if one_d:
        # Only the feature axis (x) carries data; hide the cosmetic vertical spread
        # so it never reads as a second dimension.
        ax.set_ylim(-1.5, 1.5)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylabel("")
        if projection == "feature_1d":
            ax.set_xlabel(labels[0] or "feature")
            ax.grid(True, axis="x")
        else:
            ax.set_xticks([])
            ax.spines["bottom"].set_visible(False)
            ax.set_xlabel("no numeric features — backend mix only")
    else:
        ax.set_xlabel(labels[0] or "feature")
        ax.set_ylabel(labels[1] or "")
        ax.grid(True)
    ax.legend(loc="best", frameon=True, framealpha=0.85, fontsize=9, title="selected backend")
    title = f"Backend selection over inputs — {position['name']} [{position.get('kind', '')}]"
    ax.set_title(f"{run_label}\n{title}" if run_label else title)
    save_plot(fig, out_path)
    return out_path


def _spread(n: int, salt: str, axis: int) -> list[float]:
    """`n` deterministic cosmetic offsets in ~[-1, 1] for a 1-D strip's non-data
    axis. Keyed by backend name + axis so different backends don't stack on the
    same offsets and a re-render reproduces the same picture (a stable byte-fold,
    not Python's per-process salted `hash`, so it's reproducible across runs)."""
    base = axis * 2_654_435 + 1
    for ch in salt.encode("utf-8"):
        base = (base * 131 + ch) & 0xFFFF
    base = base or 1
    return [(((i * 2654435761 + base) % 2000) / 1000.0 - 1.0) for i in range(n)]


def _annotate(ax, position: dict, candidates: list[str]) -> None:
    """Corner box: candidate list, per-backend selection share, projection info."""
    n_points = len(position["points"])
    lines = [
        f"candidates: {', '.join(candidates) if candidates else '—'}",
        f"points: {fmt_count(n_points)}",
    ]
    selection = position.get("selection") or []
    if len(selection) <= 1:
        only = selection[0]["backend_name"] if selection else (candidates[0] if candidates else "?")
        lines.append(f"single backend: {only}")
    else:
        for s in selection:
            lines.append(f"  {s['backend_name']}: {s['ratio'] * 100:.0f}%  ({fmt_count(s['count'])})")
    proj = position.get("projection", "")
    ev = position.get("explained_variance")
    if proj == "pca" and ev:
        lines.append(f"PCA: PC1 {ev[0] * 100:.0f}% · PC2 {ev[1] * 100:.0f}% var")
    else:
        lines.append(f"projection: {proj}")
    corner_box(ax, lines, loc="upper left")


def _area(count: int) -> float:
    """Point marker area from an observation count (sqrt-scaled, clamped)."""
    return max(_MIN_AREA, min(_MAX_AREA, _MIN_AREA * math.sqrt(max(1, count))))


def _safe_name(name: str) -> str:
    """A position name → a filesystem-safe stem (path separators and spaces out;
    dots kept — they read as the dotted leaf path)."""
    safe = name.replace("/", "__").replace("\\", "__").replace(" ", "_")
    return safe or "position"


def _output_is_current(output_path: Path, *source_paths: Path) -> bool:
    """Whether an existing image is at least as new as every render input."""
    if not output_path.is_file():
        return False
    output_mtime_ns = output_path.stat().st_mtime_ns
    return all(output_mtime_ns >= source_path.stat().st_mtime_ns for source_path in source_paths)


def _remove_stale_outputs(log_dir: Path, desired: set[Path]) -> None:
    """Delete position PNGs no longer in the payload (a renamed/removed position),
    so a re-analyzed run's plot dir doesn't accumulate orphans."""
    plots_dir = log_dir / "plots" / PLOT_SUBDIR
    if not plots_dir.is_dir():
        return
    for path in plots_dir.glob("*.png"):
        if path not in desired:
            path.unlink()
