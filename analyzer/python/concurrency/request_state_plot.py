"""Render request-state occupancy at cluster, pool, and worker levels.

The Rust ``request-state`` subject owns stage decoding and aggregation.  This
module only draws its bounded JSON series; it deliberately never reads parquet
or reconstructs request state from transitions.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

from common.figure import add_legend, corner_box, finalize, fmt_value, new_axes
from common.style import ACCENT, CURVE, MARKER


def render(log_dir: Path) -> list[Callable[[], Path]]:
    """Return one cluster job plus one pool and one worker job per payload entry."""
    from common.layout import load_payload, plot_output_path

    payload = load_payload(log_dir, "request_state_series.json")
    meta = payload.get("meta", {})
    cluster_series = payload.get("cluster_series") or []
    if (
        payload.get("available") is False
        or meta.get("available") is False
        or not payload.get("t_start_ms")
        or not payload.get("t_end_ms")
        or not any(series.get("values") for series in cluster_series)
    ):
        reason = meta.get("reason", "no series in request_state_series.json")
        print(f"[request_state_plot] nothing to render: {reason}")
        return []

    run_label = Path(meta.get("log_dir", str(log_dir))).name
    jobs: list[Callable[[], Path]] = [
        partial(
            _render_cluster,
            payload,
            plot_output_path(log_dir, "request_state_cluster.png"),
            run_label=run_label,
        )
    ]
    for pool in payload.get("pools") or []:
        pool_tag = str(pool.get("pool_tag", pool.get("pool", "pool")))
        jobs.append(
            partial(
                _render_pool,
                payload,
                pool,
                plot_output_path(
                    log_dir,
                    f"request_state/request_state_pool_{pool_tag}.png",
                ),
                run_label=run_label,
            )
        )
        for worker in pool.get("workers") or []:
            worker_id = worker.get("worker_id", "unknown")
            jobs.append(
                partial(
                    _render_worker,
                    payload,
                    pool,
                    worker,
                    plot_output_path(
                        log_dir,
                        f"request_state/request_state_worker_{pool_tag}_{worker_id}.png",
                    ),
                    run_label=run_label,
                )
            )
    return jobs


def _edges_s(payload: dict) -> list[float]:
    """Convert contiguous millisecond bins to the n+1 stair edges in seconds."""
    starts = payload["t_start_ms"]
    ends = payload["t_end_ms"]
    return [value / 1000.0 for value in (*starts, ends[-1])]


def _request_label(value: float) -> str:
    return fmt_value(value, "requests")


def _render_cluster(payload: dict, out_path: Path, *, run_label: str) -> Path:
    """Draw every category as one conserved stacked request population."""
    fig, ax = new_axes(figsize=(9.5, 4.8))
    edges = _edges_s(payload)
    palette = (CURVE, ACCENT, MARKER)
    hatches = ("", "//", "\\\\", "..")
    peak_total = 0.0
    mean_total = 0.0
    totals = [0.0] * (len(edges) - 1)
    category_values: list[list[float]] = []
    labels: list[str] = []
    colors: list[str] = []
    for index, series in enumerate(payload.get("cluster_series") or []):
        values = series.get("values") or []
        if not values:
            continue
        labels.append(str(series.get("category", "other")))
        colors.append(palette[index % len(palette)])
        # Repeat the final bin value at the right edge so step="post" spans the
        # complete simulation range rather than ending at the last bin start.
        category_values.append([*values, values[-1]])
        totals = [total + value for total, value in zip(totals, values)]
    if category_values:
        layers = ax.stackplot(
            edges,
            *category_values,
            labels=labels,
            colors=colors,
            step="post",
            alpha=0.78,
        )
        for index, layer in enumerate(layers):
            layer.set_hatch(hatches[index % len(hatches)])
    if totals:
        peak_total = max(totals)
        mean_total = sum(totals) / len(totals)
    ax.set_xlim(left=edges[0], right=edges[-1])
    ax.set_ylim(bottom=0.0, top=max(1.0, peak_total * 1.08))
    corner_box(
        ax,
        [
            f"mean in system = {_request_label(mean_total)}",
            f"peak in system = {_request_label(peak_total)}",
            f"categories = {len(labels)}",
        ],
        loc="upper right",
    )
    add_legend(ax, loc="upper left")
    finalize(
        fig,
        ax,
        out_path,
        title="Request state · cluster",
        xlabel="sim time (s)",
        ylabel="requests",
        run_label=run_label,
    )
    return out_path


def _render_pool(
    payload: dict,
    pool: dict,
    out_path: Path,
    *,
    run_label: str,
) -> Path:
    """Draw worker pending shadows plus pool total and per-worker average."""
    fig, ax = new_axes(figsize=(9.5, 4.8))
    edges = _edges_s(payload)
    workers = pool.get("workers") or []
    for worker in workers:
        pending = worker.get("pending") or []
        if not pending:
            continue
        ax.stairs(
            pending,
            edges,
            label=f"worker {worker.get('worker_id', 'unknown')}",
            color=MARKER,
            linewidth=0.9,
            alpha=0.35,
            baseline=None,
        )

    total = pool.get("total_pending") or []
    average = pool.get("average_pending") or []
    if total:
        ax.stairs(
            total,
            edges,
            label="pool total",
            color=CURVE,
            linewidth=3.0,
            baseline=None,
        )
    if average:
        ax.stairs(
            average,
            edges,
            label="worker average",
            color=ACCENT,
            linestyle="--",
            linewidth=2.2,
            baseline=None,
        )

    peak = max(total, default=0.0)
    mean = sum(total) / len(total) if total else 0.0
    ax.set_xlim(left=edges[0], right=edges[-1])
    ax.set_ylim(bottom=0.0, top=max(1.0, peak * 1.08))
    corner_box(
        ax,
        [
            f"workers = {len(workers)}",
            f"mean total = {_request_label(mean)}",
            f"peak total = {_request_label(peak)}",
        ],
        loc="upper right",
    )
    add_legend(ax, loc="upper left")
    pool_name = pool.get("pool_tag", pool.get("pool", "pool"))
    finalize(
        fig,
        ax,
        out_path,
        title=f"Pending requests · pool {pool_name}",
        xlabel="sim time (s)",
        ylabel="pending requests",
        run_label=run_label,
    )
    return out_path


def _render_worker(
    payload: dict,
    pool: dict,
    worker: dict,
    out_path: Path,
    *,
    run_label: str,
) -> Path:
    """Draw one worker's pending queue over simulated time."""
    fig, ax = new_axes(figsize=(9.0, 4.5))
    edges = _edges_s(payload)
    pending = worker.get("pending") or []
    ax.stairs(pending, edges, color=CURVE, linewidth=2.2, label="pending", baseline=None)
    ax.fill_between(edges[:-1], pending, step="post", color=CURVE, alpha=0.14)

    peak = max(pending, default=0.0)
    mean = sum(pending) / len(pending) if pending else 0.0
    ax.set_xlim(left=edges[0], right=edges[-1])
    ax.set_ylim(bottom=0.0, top=max(1.0, peak * 1.08))
    corner_box(
        ax,
        [f"mean = {_request_label(mean)}", f"peak = {_request_label(peak)}"],
        loc="upper right",
    )
    add_legend(ax, loc="upper left")
    pool_name = pool.get("pool_tag", pool.get("pool", "pool"))
    worker_id = worker.get("worker_id", "unknown")
    finalize(
        fig,
        ax,
        out_path,
        title=f"Pending requests · {pool_name} worker {worker_id}",
        xlabel="sim time (s)",
        ylabel="pending requests",
        run_label=run_label,
    )
    return out_path
