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

# Fixed per-backend (color, matplotlib marker) so the kernel-input-distribution
# scatters read consistently run-to-run and across positions: the same backend is
# always the same color+shape. A backend not in this table (a new kernel backend)
# gets a deterministic fallback keyed by its name, so old payloads never break.
_BACKEND_STYLE = {
    "fa2": ("#4C78A8", "o"),          # flashinfer attn v2 — blue circle
    "fa3": ("#F58518", "^"),          # flashinfer attn v3 — orange triangle
    "torch": ("#54A24B", "s"),        # torch reference — green square
    "torch_linear": ("#B279A2", "D"),  # torch linear — purple diamond
    "trt": ("#E45756", "v"),          # trtllm-gen — red down-triangle
    "deepgemm": ("#72B7B2", "P"),     # deep_gemm — teal plus
}
_FALLBACK_COLORS = ["#9D755D", "#BAB0AC", "#EECA3B", "#FF9DA6", "#79706E", "#D37295"]
_FALLBACK_MARKERS = ["X", "*", "p", "h", "<", ">"]


def backend_style(name: str) -> tuple[str, str]:
    """`(color, marker)` for a backend name. Known backends use the fixed palette
    above; an unknown name gets a deterministic fallback (hashed to a stable
    color+marker) so the plot never guesses and a new backend still renders."""
    if name in _BACKEND_STYLE:
        return _BACKEND_STYLE[name]
    # Stable hash → same fallback for the same name every run (Python's built-in
    # hash is salted per process, so fold the bytes ourselves).
    h = 0
    for ch in name.encode("utf-8"):
        h = (h * 131 + ch) & 0xFFFFFFFF
    return _FALLBACK_COLORS[h % len(_FALLBACK_COLORS)], _FALLBACK_MARKERS[h % len(_FALLBACK_MARKERS)]

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
