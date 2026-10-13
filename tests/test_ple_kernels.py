import pytest
import torch

from profiling.runners.embedding import embedding_row_gather as gather
from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.ple import ple_gate, ple_ngram_ids, ple_short_conv

_ROW_ARGS = dict(
    num_tokens=64,
    num_heads=16,
    head_dim=160,
    table_rows=320_001_536,
    residency="host_uva",
    weight_dtype="fp8_e4m3",
)


def test_id_pool_stays_inside_each_heads_range():
    # Defect caught: ids for head h outside its own range, which would gather a
    # different (or masked-out) part of the table than production touches.
    rows, heads = 1000, 16
    pool = gather.build_id_pool(torch, torch.device("cpu"), 8, heads, rows)
    assert pool.dtype == torch.int64
    for head, (lo, hi) in enumerate(gather.head_ranges(rows, heads)):
        column = pool[..., head]
        assert int(column.min()) >= lo and int(column.max()) < hi


def test_each_backend_runs_only_its_production_residency():
    # Defect caught: timing the pinned UVA kernel on an HBM table (or F.embedding
    # on a host table), a pairing production never runs.
    gather.validate_args("vllm_triton", **_ROW_ARGS)
    gather.validate_args("torch", **dict(_ROW_ARGS, residency="device"))
    with pytest.raises(ProfilerNotImplemented, match="host_uva"):
        gather.validate_args("vllm_triton", **dict(_ROW_ARGS, residency="device"))
    with pytest.raises(ProfilerNotImplemented, match="device"):
        gather.validate_args("torch", **_ROW_ARGS)


def test_gate_reference_matches_the_signed_sqrt_sigmoid_formula():
    # Defect caught: a gate without the signed square root, or norms that are
    # not per residual stream.
    hidden_size, hc = 4, 2
    key = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0]], dtype=torch.float32)
    hidden = torch.ones(1, hc * hidden_size)
    value = torch.full((1, hidden_size), 2.0)
    zeros = torch.zeros(hc * hidden_size)
    gated, normed = ple_gate.reference_gate(
        torch, key, value, hidden, zeros, zeros, zeros, hidden_size
    )
    # Stream 0: normalized dot = 4, d = 4 / 2 = 2 -> g = sigmoid(sqrt 2).
    # Stream 1: dot = 0 -> g = sigmoid(0) = 0.5.
    g0 = torch.sigmoid(torch.tensor(2.0).sqrt())
    assert torch.allclose(gated[0, :4], 2 * g0, atol=1e-5)
    assert torch.allclose(gated[0, 4:], torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(normed, torch.ones_like(normed), atol=1e-4)


def test_conv_reference_with_zero_state_is_a_causal_dilated_conv():
    # Defect caught: taps placed at the wrong dilation or in reverse order.
    channels, taps, dilation, q = 3, 4, 3, 11
    torch.manual_seed(0)
    x = torch.randn(q, channels, dtype=torch.float64)
    weight = torch.randn(channels, taps, dtype=torch.float64)
    state = torch.zeros(1, channels, (taps - 1) * dilation, dtype=torch.float64)
    zeros = torch.zeros_like(x)
    got = ple_short_conv.reference_conv(
        torch, x, zeros, zeros, state, weight, query_len=q, dilation=dilation, mode="prefill"
    )
    padded = torch.nn.functional.pad(x.T[None], ((taps - 1) * dilation, 0))
    conv = torch.nn.functional.conv1d(padded, weight[:, None, :], dilation=dilation, groups=channels)
    expected = torch.nn.functional.silu(conv[0].T)
    assert torch.allclose(got, expected, atol=1e-9)


def test_conv_args_reject_multi_token_decode():
    # Defect caught: a decode row with query_len > 1, which the decode kernel
    # would silently treat as separate requests.
    with pytest.raises(ValueError, match="decode"):
        ple_short_conv.validate_args(4, 2, 10240, 4, 3, "decode", "bf16")


def test_ngram_requests_split_tokens_nearly_evenly():
    assert ple_ngram_ids.query_start_loc(10, 3) == [0, 3, 6, 10]
    with pytest.raises(ValueError, match="num_reqs"):
        ple_ngram_ids.validate_args(4, 5, 2, 8)
