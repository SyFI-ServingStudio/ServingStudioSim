"""mega_moe runner pieces that can break without a GPU."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from profiling.runners.moe import mega_moe_deepgemm as runner  # noqa: E402
from profiling.runners.moe import mega_moe_reference as reference  # noqa: E402


def _deepgemm_shared_sf_layout(sf, block_m: int, num_max_sf_tokens: int):
    """Transcription of DeepGEMM tests/test_mega_moe.py `_to_shared_mega_moe_sf_layout`."""
    num_tokens, packed_sf_k = sf.shape
    aligned_block_m = (block_m + 127) // 128 * 128
    num_m_blocks = (num_tokens + block_m - 1) // block_m
    result = torch.empty_strided(
        (num_max_sf_tokens, packed_sf_k), (1, num_max_sf_tokens), dtype=sf.dtype
    )
    result.zero_()
    for block_idx in range(num_m_blocks):
        for m_idx in range(min(block_m, num_tokens - block_idx * block_m)):
            transposed = (m_idx // 128) * 128 + (m_idx % 32) * 4 + (m_idx % 128) // 32
            result[block_idx * aligned_block_m + transposed].copy_(sf[block_idx * block_m + m_idx])
    return result


@pytest.mark.parametrize(("num_tokens", "block_m"), [(140, 16), (300, 192), (700, 240)])
def test_shared_sf_layout_matches_deepgemm(num_tokens: int, block_m: int) -> None:
    """A wrong shared-expert scale layout feeds the fused shared expert garbage
    scales; the kernel still runs, so only the layout itself catches it."""
    rows = 2048  # >= ceil(140/16) blocks x 128 rows
    sf = torch.randint(0, 2**30, (num_tokens, 40), dtype=torch.int32)
    dst = torch.empty_strided((rows, 40), (1, rows), dtype=torch.int32).fill_(-1)
    runner._write_shared_sf(torch, dst, sf, block_m)
    assert torch.equal(dst, _deepgemm_shared_sf_layout(sf, block_m, rows))


def test_validate_rejects_non_mxfp4_weights() -> None:
    """The runner times only the fp8xfp4 kernel; another dtype must not be stored under its key."""
    spec = dict(
        ep_size=4,
        num_tokens=16,
        hidden_size=5120,
        intermediate_size=2304,
        num_experts=416,
        num_shared_experts=1,
        top_k=6,
        activation_dtype="mxfp8_e4m3",
        weight_dtype="fp8_e4m3",
        per_slot_rows=[16 * 4 * 6 // 416] * 416,
    )
    with pytest.raises(Exception, match="mxfp4_e2m1"):
        runner._validate(**spec)


@pytest.mark.parametrize("ep_size", [1, 4])
def test_slot_topk_realizes_the_histogram_with_distinct_slots(ep_size: int) -> None:
    """The profiled routing must be exactly the histogram in the key, and a token
    may not pick one slot twice (the kernel would double-count it)."""
    import numpy as np

    num_tokens, top_k, slots = 96, 6, 32
    rng = np.random.default_rng(1)
    group = ep_size * num_tokens
    weights = np.r_[[8.0] * 4, [1.0] * (slots - 4)]
    rows = rng.multinomial(group * top_k, weights / weights.sum())
    # Clip a too-hot slot to the group's token count, moving its excess round-robin.
    while rows.max() > group:
        hot = rows.argmax()
        rows[(hot + 1) % slots] += rows[hot] - group
        rows[hot] = group
    ids = runner.slot_topk(rows.tolist(), ep_size, num_tokens, top_k)
    assert ids.shape == (ep_size, num_tokens, top_k)
    assert np.array_equal(np.bincount(ids.ravel(), minlength=slots), rows)
    flat = ids.reshape(-1, top_k)
    assert all(len(set(t)) == top_k for t in flat.tolist())
    spec = dict(
        ep_size=ep_size,
        num_tokens=num_tokens,
        hidden_size=5120,
        intermediate_size=2304,
        num_experts=slots,
        num_shared_experts=1,
        top_k=top_k,
        activation_dtype="mxfp8_e4m3",
        weight_dtype="mxfp4_e2m1",
        per_slot_rows=rows.tolist(),
    )
    runner._validate(**spec)
    with pytest.raises(ValueError, match="sum"):
        runner._validate(**{**spec, "per_slot_rows": [0] * slots})


def test_reference_routes_each_choice_to_its_slot() -> None:
    """Two ranks' partials over all tokens must sum to the dense top-k MoE."""
    torch.manual_seed(0)
    tokens, hidden, inter, experts, top_k = 8, 64, 32, 4, 2
    x = torch.randn(tokens, hidden)
    idx = torch.stack([torch.randperm(experts)[:top_k] for _ in range(tokens)])
    w = torch.rand(tokens, top_k)
    gate_up = torch.randn(experts, 2 * inter, hidden) / 8
    down = torch.randn(experts, hidden, inter) / 8
    partials = sum(
        reference.expert_partial(
            torch, x, idx, w, gate_up[r * 2 : r * 2 + 2], down[r * 2 : r * 2 + 2], r * 2, 10.0
        )
        for r in range(2)
    )
    dense = reference.expert_partial(torch, x, idx, w, gate_up, down, 0, 10.0)
    torch.testing.assert_close(partials, dense)
    # Every token reaches exactly its top_k experts: zeroing their weights zeroes the output.
    zeroed = reference.expert_partial(torch, x, idx, torch.zeros_like(w), gate_up, down, 0, 10.0)
    assert torch.count_nonzero(zeroed) == 0
