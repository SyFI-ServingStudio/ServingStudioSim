"""Pure-Torch reference for ``sliding_window_kv_append``.

Independent of TensorFold: a stream's new context is its old context followed
by its new rows, cut to the last ``window`` rows.
"""

from __future__ import annotations

from typing import Any


def sliding_window_kv_append_reference(torch: Any, old: Any, new: Any, window: int) -> Any:
    """``old`` ``(heads, n, d)`` or None, ``new`` ``(heads, add, d)`` -> ``(heads, min(window, n + add), d)``."""

    full = new if old is None else torch.cat((old, new), dim=1)
    keep = min(window, full.shape[1])
    return full[:, full.shape[1] - keep :].contiguous()
