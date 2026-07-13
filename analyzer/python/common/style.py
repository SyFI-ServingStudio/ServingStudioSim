"""Shared matplotlib style for analyzer plots (ported from ref plot_bridge).

Import for side effect: applies rcParams once on import. Keep all plot modules
consuming this so PNGs across metrics look uniform.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: no display, write PNGs directly
import matplotlib.pyplot as plt  # noqa: E402

# Palette — first entry is the primary curve color; extras for multi-series.
CURVE = "#4C78A8"
ACCENT = "#F58518"
GRID = "#98A2B3"
MARKER = "#667085"

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.labelweight": "semibold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "#FCFCFD",
        "savefig.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.linestyle": "--",
        "grid.alpha": 0.18,
    }
)


def save_plot(
    fig,
    path,
    dpi: int = 300,
    *,
    tight: bool = True,
    pil_kwargs: dict[str, object] | None = None,
) -> None:
    """Save and close one figure.

    Most subject-level figures benefit from a tight bounding-box pass. Large
    batches of pre-sized diagnostic figures may opt out: ``bbox_inches='tight'``
    triggers another full layout/draw and dominates their render time.
    """
    save_kwargs: dict[str, object] = {"dpi": dpi}
    if tight:
        save_kwargs["bbox_inches"] = "tight"
    if pil_kwargs is not None:
        # Forward only explicit image-format policy from a renderer. Subject
        # plots otherwise retain Matplotlib/Pillow defaults.
        save_kwargs["pil_kwargs"] = pil_kwargs
    fig.savefig(path, **save_kwargs)
    plt.close(fig)
