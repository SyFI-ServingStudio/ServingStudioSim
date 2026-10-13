import math

import pytest
import torch

from profiling.runners.attention import fa4_varlen_attention as runner
from profiling.runners.attention.fa4_varlen_attention_reference import (
    gather_request_kv,
    varlen_attention_reference,
)

_BASE = dict(
    num_qo_heads=8,
    num_kv_heads=2,
    head_dim_qk=192,
    head_dim_v=128,
    window=0,
    sinks=False,
    page_size=16,
    kv_layout="lbnhc",
    q_dtype="bf16",
    kv_dtype="bf16",
    max_model_len=8192,
    cuda_graph=False,
)


def _shape(**overrides):
    args = dict(_BASE, query_context_pairs=((1, 9),))
    args.update(overrides)
    return runner._validate_args(**args)


def _brute_force(q, positions, keys, values, scale, window, sinks):
    """Loop form of the definition, one (row, head) at a time."""
    rows, heads, _ = q.shape
    group = heads // keys.shape[1]
    out = torch.zeros(rows, heads, values.shape[-1])
    for t in range(rows):
        pos = int(positions[t])
        for h in range(heads):
            logits, vals = [], []
            for j in range(pos + 1):
                if window and pos - j >= window:
                    continue
                logits.append(float(q[t, h] @ keys[j, h // group]) * scale)
                vals.append(values[j, h // group])
            if sinks is not None:
                logits.append(float(sinks[h]))
                vals.append(torch.zeros_like(values[0, 0]))
            weights = torch.softmax(torch.tensor(logits), 0)
            out[t, h] = (weights[:, None] * torch.stack(vals)).sum(0)
    return out


@pytest.mark.parametrize("window", [0, 4])
@pytest.mark.parametrize("with_sinks", [False, True])
def test_reference_matches_loop_definition(window, with_sinks):
    # Defect caught: the sink left out of (or given a value in) the softmax, a
    # window off by one, a query head reading the wrong KV head, or the output
    # taking the QK head dim instead of V's.
    torch.manual_seed(0)
    q = torch.randn(3, 4, 6)
    keys, values = torch.randn(11, 2, 6), torch.randn(11, 2, 5)
    sinks = torch.randn(4) if with_sinks else None
    positions = torch.tensor([6, 9, 10])
    got = varlen_attention_reference(
        q, positions, keys, values, softmax_scale=0.3, window=window, sinks=sinks
    )
    expected = _brute_force(q, positions, keys, values, 0.3, window, sinks)
    assert got.shape == (3, 4, 5)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_large_sink_drains_the_output():
    # Defect caught: a sink applied with the wrong sign or scaled with the scores.
    keys, values = torch.randn(5, 1, 4), torch.randn(5, 1, 3)
    out = varlen_attention_reference(
        torch.randn(1, 2, 4),
        torch.tensor([4]),
        keys,
        values,
        softmax_scale=1.0,
        window=0,
        sinks=torch.full((2,), 60.0),
    )
    assert out.abs().max() < 1e-6


def test_gather_reads_k_then_v_channels_of_shuffled_lbnhc_pages():
    # Defect caught: page/offset arithmetic, or the K/V split of the packed
    # last dimension taken at the wrong channel (K is head_dim_qk wide, V after it).
    page, hkv, dqk, dv = 4, 2, 3, 2
    raw = torch.arange(5 * page * hkv * (dqk + dv), dtype=torch.float32)
    raw = raw.view(5, page, hkv, dqk + dv)
    key_cache, value_cache = raw.split(dqk, dim=-1)
    block_row = torch.tensor([3, 1, 4, 0], dtype=torch.int32)
    keys, values = gather_request_kv(key_cache, value_cache, block_row, 10)
    assert keys.shape == (10, hkv, dqk) and values.shape == (10, hkv, dv)
    for token in range(10):
        physical = int(block_row[token // page])
        torch.testing.assert_close(keys[token], raw[physical, token % page, :, :dqk])
        torch.testing.assert_close(values[token], raw[physical, token % page, :, dqk:])


def test_graph_step_uses_capture_time_shape():
    # Defect caught: a FULL-graph decode step called with the real batch and
    # longest context, which picks a different split count than the replayed
    # graph (fa4_varlen_attention.rs `tiling`).
    graph = _shape(cuda_graph=True, query_context_pairs=((1, 1000),) * 9, padded_requests=16)
    assert graph.replays_graph
    assert graph.launched_requests == 16
    assert (graph.max_seqlen_q, graph.max_seqlen_k) == (1, 8192)


@pytest.mark.parametrize(
    "overrides",
    [
        # Eager runs: no graph, a prefill token.
        dict(cuda_graph=False, query_context_pairs=((1, 1000),) * 9),
        dict(cuda_graph=True, query_context_pairs=((1, 1000),) * 8 + ((2, 900),)),
    ],
)
def test_eager_step_uses_real_shape_and_ignores_padding(overrides):
    shape = _shape(padded_requests=16, **overrides)
    assert not shape.replays_graph
    assert shape.launched_requests == len(shape.pairs)
    assert shape.max_seqlen_k == max(c for _, c in shape.pairs)
    assert shape.max_seqlen_q == max(q for q, _ in shape.pairs)


def test_window_size_and_first_visible_key_follow_vllm():
    # Defect caught: vLLM's (w - 1, 0) window encoding passed as (w, 0), or
    # the pages before a window not pointed at the null page.
    swa = _shape(window=128)
    assert swa.window_size == [127, 0]
    assert swa.first_visible_key(1, 1000) == 1000 - 128
    assert swa.first_visible_key(64, 100) == 0
    assert _shape().window_size == [-1, -1]
    assert _shape().first_visible_key(1, 1000) == 0


def test_padding_rows_are_empty_and_point_at_the_null_page():
    # Defect caught: padding requests that read real pages or carry a query.
    shape = _shape(
        window=8,
        cuda_graph=True,
        query_context_pairs=((1, 40), (1, 3)),
        padded_requests=4,
    )
    work = runner._build_workload(torch, shape, torch.device("cpu"))
    assert work.cu_seqlens_q.tolist() == [0, 1, 2, 2, 2]
    assert work.seqused_k.tolist() == [40, 3, 0, 0]
    assert (work.block_table[2:] == runner._NULL_PAGE).all()
    # Request 0 sees keys 32..39: page 2 only; pages 0 and 1 stay null.
    row = work.block_table[0].tolist()
    assert row[:2] == [runner._NULL_PAGE] * 2 and row[2] != runner._NULL_PAGE
    assert work.raw_cache.shape == (3, 16, 2, 320)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(query_context_pairs=((2, 1),)),
        dict(query_context_pairs=((1, 9000),)),
        dict(query_context_pairs=()),
        dict(num_kv_heads=3),
        dict(window=-1),
        dict(padded_requests=-1),
        dict(cuda_graph=1),
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
    # Defect caught: TFLOPS/GB/s that ignore the window, the prefix or the
    # different K and V widths.
    global_ = _shape(query_context_pairs=((3, 10), (1, 4)))
    assert runner.visible_keys(global_) == (8 + 9 + 10) + 4
    local = _shape(window=4, query_context_pairs=((3, 10), (1, 4)))
    assert runner.visible_keys(local) == 4 * 3 + 4
    # Local request 0 reads keys 4..9 (6 tokens).
    per_token = 8 * (192 + 128) * 2
    kv_bytes = (6 + 4) * 2 * (192 + 128) * 2
    assert math.isclose(runner.logical_bytes(local), 4 * per_token + kv_bytes)
