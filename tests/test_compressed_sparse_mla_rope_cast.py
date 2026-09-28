import math

import pytest
import torch

from profiling.runners.attention import compressed_sparse_mla_rope_cast as runner
from profiling.runners.attention.compressed_sparse_mla_rope_cast_reference import (
    decode_records,
    compressed_sparse_mla_rope_cast_reference,
    encode_records,
    output_from_fused_layout,
    q_to_fused_layout,
    write_records,
)

_BASE = dict(
    window_size=128,
    index_topk=512,
    num_heads=64,
    head_dim=512,
    rope_dim=64,
    max_model_len=65536,
    max_num_batched_tokens=8192,
    prefill_chunk_size=4,
    q_dtype="bf16",
    swa_cache_format="mxfp8",
    output_dtype="fp8_e4m3",
)


def _shape(**overrides):
    args = dict(
        _BASE,
        mode="decode",
        query_context_pairs=((1, 9),),
        compress_ratio=2,
        compressed_cache_format="nvfp4",
    )
    args.update(overrides)
    return runner._validate_args(**args)


def _cos_sin(max_pos):
    return runner._cos_sin(torch, max_pos, torch.device("cpu"))


def _single_row_cache(row, fmt):
    cache = torch.zeros((2, 32, {"mxfp8": 528, "nvfp4": 288}[fmt]), dtype=torch.uint8)
    write_records(cache, torch.tensor([37]), row[None], fmt)
    return cache


def test_reference_single_key_with_sink_is_scaled_inverse_roped_key():
    # Defect caught: wrong sink handling, or a missing/forward output rotation.
    torch.manual_seed(0)
    key = torch.randn(512)
    cache = _single_row_cache(key, "mxfp8")
    stored = decode_records(cache, torch.tensor([37]), "mxfp8")[0]
    q = torch.randn(1, 2, 512)
    positions = torch.tensor([5])
    cos_sin = _cos_sin(8)
    sink = torch.tensor([0.25, float("-inf")])
    out = compressed_sparse_mla_rope_cast_reference(
        q, positions, cos_sin, sink, 512**-0.5, cache, torch.tensor([[37, -1]])
    )
    cos, sin = cos_sin[5, :32], cos_sin[5, 32:]
    inverse = stored.clone()
    even, odd = stored[448::2], stored[449::2]
    inverse[448::2] = even * cos + odd * sin
    inverse[449::2] = odd * cos - even * sin
    rotated_q = q[0].clone()
    rotated_q[:, 448::2] = q[0, :, 448::2] * cos - q[0, :, 449::2] * sin
    rotated_q[:, 449::2] = q[0, :, 449::2] * cos + q[0, :, 448::2] * sin
    score = rotated_q[0] @ stored * 512**-0.5
    weight = 1.0 / (1.0 + math.exp(0.25 - score.item()))
    torch.testing.assert_close(out[0, 0], inverse * weight, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out[0, 1], inverse, rtol=1e-5, atol=1e-5)


def test_reference_attends_swa_and_compressed_keys_as_one_softmax():
    # Defect caught: normalizing the two key sources separately.
    torch.manual_seed(1)
    swa_rows, extra_rows = torch.randn(3, 512), torch.randn(2, 512)
    swa = torch.zeros((1, 32, 528), dtype=torch.uint8)
    extra = torch.zeros((1, 64, 288), dtype=torch.uint8)
    write_records(swa, torch.arange(3), swa_rows, "mxfp8")
    write_records(extra, torch.tensor([10, 20]), extra_rows, "nvfp4")
    q = torch.randn(1, 1, 512)
    kwargs = dict(
        positions=torch.tensor([0]),
        cos_sin_cache=_cos_sin(1),
        attn_sink=torch.tensor([float("-inf")]),
        softmax_scale=0.05,
    )
    out = compressed_sparse_mla_rope_cast_reference(
        q,
        swa_cache=swa,
        swa_slots=torch.tensor([[0, 1, 2, -1]]),
        extra_cache=extra,
        extra_slots=torch.tensor([[20, 10, -1]]),
        extra_format="nvfp4",
        **kwargs,
    )
    keys = torch.cat(
        (
            decode_records(swa, torch.arange(3), "mxfp8"),
            decode_records(extra, torch.tensor([20, 10]), "nvfp4"),
        )
    )
    weights = torch.softmax(q[0, 0] @ keys.T * 0.05, dim=-1)
    torch.testing.assert_close(out[0, 0], weights @ keys, rtol=1e-5, atol=1e-5)


def test_record_layouts_put_the_page_scale_region_after_all_data():
    # Defect caught: interleaving per-token scales with values inside a page.
    mx = torch.zeros((1, 32, 528), dtype=torch.uint8)
    flat = mx.view(-1)
    flat[3 * 512 : 4 * 512] = 0x38  # fp8 e4m3 1.0 for token 3
    flat[32 * 512 + 3 * 16 : 32 * 512 + 4 * 16] = 128  # ue8m0 2**1
    assert torch.all(decode_records(mx, torch.tensor([3]), "mxfp8") == 2.0)
    fp4 = torch.zeros((1, 64, 288), dtype=torch.uint8)
    flat = fp4.view(-1)
    flat[5 * 256] = 0x2A  # low nibble 0xA = -1.0 (even), high 0x2 = 1.0 (odd)
    flat[64 * 256 + 5 * 32] = 0x40  # e4m3 2.0 for dims 0..15
    row = decode_records(fp4, torch.tensor([5]), "nvfp4")[0]
    assert row[:2].tolist() == [-2.0, 2.0]


@pytest.mark.parametrize(("fmt", "tolerance"), [("mxfp8", 0.07), ("nvfp4", 0.2)])
def test_record_codec_round_trips_within_format_precision(fmt, tolerance):
    # Defect caught: wrong scale granularity or code table in the Torch codec.
    torch.manual_seed(2)
    rows = torch.randn(16, 512)
    data, scale = encode_records(rows, fmt)
    cache = torch.zeros((1, 16, data.shape[1] + scale.shape[1]), dtype=torch.uint8)
    write_records(cache, torch.arange(16), rows, fmt)
    decoded = decode_records(cache, torch.arange(16), fmt)
    assert ((decoded - rows).norm() / rows.norm()).item() < tolerance


def test_transport_layouts_match_the_fork_permutation_formulas():
    # Defect caught: a Q or O layout that disagrees with fused_layout.py.
    heads = 16
    q = torch.arange(heads * 512, dtype=torch.float32).view(1, heads, 512)
    fused = q_to_fused_layout(q).reshape(-1)
    h, d = 3, 37
    assert fused[(d // 16) * heads * 16 + h * 16 + d % 16] == q[0, h, d]
    data = torch.arange(8 * 512, dtype=torch.float32).view(1, 1, 4096)
    scale = torch.arange(128, dtype=torch.uint8).view(1, 1, 128).view(torch.int32)
    values, exponents = output_from_fused_layout(data, scale)
    c, j = d // 32, d % 32
    assert values[0, h, d] == data[0, 0, (c * 8 + h) * 32 + j]
    assert exponents[0, h, c] == c * 8 + h


def test_synthetic_batch_meets_the_indexer_and_window_count_contract():
    # Defect caught: top-k rows beyond the causal range, duplicates, or a
    # window that is not the last min(pos + 1, 128) tokens.
    shape = _shape(mode="prefill", query_context_pairs=((3, 700), (2, 2)), compress_ratio=2)
    work = runner._build_workload(torch, shape, torch.device("cpu"), runner._torch_writer)
    positions = work.positions
    for row, pos in enumerate(positions.tolist()):
        local = work.topk_local[row]
        valid = local[local >= 0]
        assert valid.numel() == min((pos + 1) // 2, 512) == work.extra_lens[row]
        assert valid.unique().numel() == valid.numel()
        assert bool((valid < (pos + 1) // 2).all())
        assert int(work.swa_lens[row]) == min(pos + 1, 128)
        assert int((work.swa_slots[row] >= 0).sum()) == int(work.swa_lens[row])


def test_invalid_shapes_fail_closed():
    # Defect caught: silently profiling a layer the kernel cannot serve.
    with pytest.raises(ValueError, match="compressed_cache_format='none'"):
        _shape(compress_ratio=0, compressed_cache_format="nvfp4")
    with pytest.raises(runner.ProfilerNotImplemented, match="compress_ratio"):
        _shape(compress_ratio=4)
    with pytest.raises(runner.ProfilerNotImplemented, match="padded head count"):
        _shape(num_heads=16)
    with pytest.raises(ValueError, match="query <= context"):
        _shape(query_context_pairs=((9, 8),))
