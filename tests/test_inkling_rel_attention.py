import math

import pytest
import torch

from profiling.runners.attention import inkling_rel_attention as runner
from profiling.runners.attention.inkling_rel_attention_reference import (
    gather_request_kv,
    rel_attention_reference,
)

_BASE = dict(
    num_qo_heads=8,
    num_kv_heads=2,
    head_dim=128,
    page_size=32,
    kv_layout="lbnhc",
    q_dtype="bf16",
    kv_dtype="bf16",
    max_model_len=8192,
)


def _shape(**overrides):
    args = dict(_BASE, is_local=False, rel_extent=1024, query_context_pairs=((1, 9),))
    args.update(overrides)
    return runner._validate_args(**args)


def _brute_force(q, positions, keys, values, rel, scale, is_local):
    """Loop form of the score definition, one (row, head) at a time."""
    rows, heads, dim = q.shape
    group = heads // keys.shape[1]
    extent = rel.shape[-1]
    out = torch.zeros(rows, heads, dim)
    for t in range(rows):
        pos = int(positions[t])
        for h in range(heads):
            logits, vals = [], []
            for j in range(pos + 1):
                d = pos - j
                if is_local and d >= extent:
                    continue
                bias = float(rel[t, h, d]) if d < extent else 0.0
                logits.append(float(q[t, h] @ keys[j, h // group]) * scale + bias)
                vals.append(values[j, h // group])
            weights = torch.softmax(torch.tensor(logits), 0)
            out[t, h] = (weights[:, None] * torch.stack(vals)).sum(0)
    return out


@pytest.mark.parametrize("is_local", [False, True])
def test_reference_matches_loop_definition_of_bias_window_and_gqa(is_local):
    # Defect caught: bias indexed by the wrong distance sign/offset, bias kept
    # (or keys kept) past rel_extent, or a query head reading the wrong KV head.
    torch.manual_seed(0)
    q = torch.randn(3, 4, 8)
    keys, values = torch.randn(11, 2, 8), torch.randn(11, 2, 8)
    rel = torch.randn(3, 4, 4)
    positions = torch.tensor([6, 9, 10])
    got = rel_attention_reference(
        q, positions, keys, values, rel, softmax_scale=0.125, is_local=is_local
    )
    expected = _brute_force(q, positions, keys, values, rel, 0.125, is_local)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_reference_single_visible_key_returns_its_value():
    # Defect caught: the causal mask letting a query see a future key.
    keys, values = torch.randn(5, 1, 4), torch.randn(5, 1, 4)
    out = rel_attention_reference(
        torch.randn(1, 2, 4),
        torch.tensor([0]),
        keys,
        values,
        torch.randn(1, 2, 3),
        softmax_scale=1.0,
        is_local=False,
    )
    torch.testing.assert_close(out[0], values[0].expand(2, 4))


def test_gather_reads_tokens_through_shuffled_lbnhc_pages():
    # Defect caught: page/offset arithmetic, or reading V where K lives in the
    # packed LBNHC page (K and V side by side in the last dimension).
    page, hkv, dim = 4, 2, 3
    raw = torch.arange(5 * page * hkv * 2 * dim, dtype=torch.float32).view(5, page, hkv, 2 * dim)
    key_cache, value_cache = raw.split(dim, dim=-1)
    block_row = torch.tensor([3, 1, 4, 0], dtype=torch.int32)
    keys, values = gather_request_kv(key_cache, value_cache, block_row, 10)
    assert keys.shape == (10, hkv, dim)
    for token in range(10):
        physical = int(block_row[token // page])
        torch.testing.assert_close(keys[token], raw[physical, token % page, :, :dim])
        torch.testing.assert_close(values[token], raw[physical, token % page, :, dim:])


def test_window_and_split_bound_follow_attention_py():
    # Defect caught: passing the global (-1, -1) window to a local layer, or the
    # local window as the split heuristic's max_kv_len on a global layer.
    local = _shape(is_local=True, rel_extent=512)
    assert local.window_size == (511, 0)
    assert local.max_kv_len == 512
    global_ = _shape()
    assert global_.window_size == (-1, -1)
    assert global_.max_kv_len == 8192


@pytest.mark.parametrize(
    "overrides",
    [
        dict(query_context_pairs=((2, 1),)),
        dict(query_context_pairs=((1, 9000),)),
        dict(query_context_pairs=()),
        dict(num_kv_heads=3),
    ],
)
def test_invalid_shapes_are_rejected(overrides):
    # Defect caught: profiling an impossible batch or ragged GQA silently.
    with pytest.raises((ValueError, TypeError, runner.ProfilerNotImplemented)):
        _shape(**overrides)


def test_unsupported_dtype_is_not_profiled():
    with pytest.raises(runner.ProfilerNotImplemented):
        _shape(kv_dtype="fp8_e4m3")


def test_visible_keys_and_bytes_count_window_and_causal_prefix():
    # Defect caught: TFLOPS/GB/s that ignore the local window or the prefix.
    global_ = _shape(query_context_pairs=((3, 10), (1, 4)))
    assert runner.visible_keys(global_) == (8 + 9 + 10) + 4
    local = _shape(is_local=True, rel_extent=4, query_context_pairs=((3, 10), (1, 4)))
    assert runner.visible_keys(local) == 4 * 3 + 4
    # Local request 0 reads keys 4..9 (6 tokens): positions 7..9 see 4 keys each.
    per_token = 8 * (2 * 128 + 4) * 2
    kv_bytes = (6 + 4) * 2 * 2 * 128 * 2
    assert math.isclose(runner.logical_bytes(local), 4 * per_token + kv_bytes)
