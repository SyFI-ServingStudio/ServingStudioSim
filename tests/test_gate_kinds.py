"""mega_gate / engram_gate runner pieces that can break without a GPU."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from profiling.runners.engram import engram_gate_tile_kernels as engram  # noqa: E402
from profiling.runners.moe import mega_gate_deepgemm as gate  # noqa: E402


def test_slot_map_matches_deepinfer_initial_layout() -> None:
    """384 + 32: experts 0..31 get a second slot at 384..415, in order."""
    p2l, l2p, count = gate.slot_map(torch, 384, 416, "cpu")
    assert p2l[:384].tolist() == list(range(384))
    assert p2l[384:].tolist() == list(range(32))
    assert count[:32].tolist() == [2] * 32 and count[32:].tolist() == [1] * 352
    # DeepInfer pads every row to num_redundant_experts + 1 columns.
    assert l2p.shape == (384, 33)
    assert l2p[5, :3].tolist() == [5, 389, -1] and l2p[100, :2].tolist() == [100, -1]


def test_gate_rejects_non_bf16() -> None:
    with pytest.raises(ValueError, match="bf16"):
        gate._validate(16, 5120, 384, 416, 6, "sqrtsoftplus", "fp8_e4m3")


def test_engram_reference_is_residual_plus_gated_value() -> None:
    """Zero value row: the streams come back unchanged; the gate stays in (0, 1)."""
    torch.manual_seed(0)
    x = torch.randn(3, 4, 64)
    kv = torch.randn(3, 5, 64)
    kv[:, -1] = 0
    w = torch.ones(4, 64)
    torch.testing.assert_close(engram.reference(torch, x, kv, w, 1e-6, 1e-6), x)
    kv[:, -1] = 1
    delta = engram.reference(torch, x, kv, w, 1e-6, 1e-6) - x
    assert bool(((delta > 0) & (delta < 1)).all())
