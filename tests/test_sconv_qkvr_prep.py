import pytest
import torch

from profiling.runners.attention import sconv_qkvr_prep_vllm_triton as runner
from profiling.runners.attention.sconv_qkvr_prep_reference import PrepShape, reference_conv
from profiling.runners.exceptions import ProfilerNotImplemented

_SHAPE = PrepShape(
    num_q_heads=8, num_kv_heads=2, head_dim=128, d_rel=16, rel_extent=512, sconv_kernel_size=4
)
_ARGS = dict(
    num_tokens=32,
    num_q_heads=8,
    num_kv_heads=2,
    head_dim=128,
    d_rel=16,
    rel_extent=512,
    apply_log_scaling=False,
    sconv_kernel_size=4,
    conv_block_size=4,
    page_size=32,
    dtype="bf16",
)


def _cpu_inputs(layout):
    return runner.build_inputs(
        torch,
        torch.device("cpu"),
        _SHAPE,
        layout,
        apply_log_scaling=True,
        conv_block_size=4,
        page_size=32,
    )


def test_reference_conv_takes_each_tap_from_its_true_source():
    # Defect caught: a reference that reads every tap from qkvr (or every tap
    # from the cache) would agree with a kernel that mixes up decode and
    # prefill taps. Request 0 continues a context of 2 tokens (taps at
    # positions 0-1 are cached, later ones in this forward); request 1 starts
    # at position 0 (taps before it are zero).
    x = _cpu_inputs(runner.Layout(((2, 3), (0, 2))))
    kv_w, q_w = _SHAPE.kv_width, _SHAPE.q_width
    k = x.qkvr[:, q_w : q_w + kv_w].float()
    w = torch.zeros(kv_w, 4, dtype=torch.bfloat16)
    w[:, 1] = 1  # one tap: position p - 2
    cache_k = torch.full((5, kv_w), 100.0)

    def cached_tap(pos, rows):
        return cache_k[rows] + pos[:, None].float(), torch.zeros(len(rows), kv_w)

    k_conv, _ = reference_conv(
        torch, _SHAPE, x.qkvr, w, w, x.positions, x.first_new_position, x.first_row, cached_tap
    )
    expected = k.clone()
    expected[0] += 100.0  # position 2 -> tap at cached position 0
    expected[1] += 101.0  # position 3 -> cached position 1
    expected[2] += k[0]  # position 4 -> position 2, row 0 of this forward
    # rows 3-4: request 1 at positions 0 and 1, tap before position 0 is zero.
    torch.testing.assert_close(k_conv.float(), expected.to(torch.bfloat16).float())


def test_timing_layout_reads_every_tap_from_the_conv_cache():
    # Defect caught: a decode layout whose contexts are too short, so some
    # taps silently come from qkvr or zero padding instead of the cache, or
    # two tokens sharing a cache slot.
    x = _cpu_inputs(runner.timing_layout(64))
    assert torch.all(x.positions - 3 >= 0)
    assert torch.equal(x.first_new_position, x.positions)
    assert len(set(x.conv_slot_mapping.tolist())) == 64
    assert len(set(x.attention_slot_mapping.tolist())) == 64
    # Every tap's conv block is a real (non-null) block of the token's request.
    for tap in range(3):
        pos = x.positions - 3 + tap
        blocks = x.conv_block_table[x.seq_idx.long(), pos // 4]
        assert torch.all(blocks > 0)


def test_pad_rows_carry_no_cache_slots():
    # Defect caught: CUDA-graph padding rows that write a cache slot.
    x = _cpu_inputs(runner.Layout(((None, 1), (5, 2)), num_pad=3))
    assert x.qkvr.shape[0] == 6
    assert x.conv_slot_mapping[-3:].tolist() == [-1, -1, -1]
    assert x.attention_slot_mapping[-3:].tolist() == [-1, -1, -1]


def test_args_the_production_call_cannot_take_are_rejected():
    # Defect caught: profiling a head_dim / d_rel / dtype fused_qkvr_prep asserts away.
    runner._validate_args(**_ARGS)
    with pytest.raises(ProfilerNotImplemented, match="head_dim"):
        runner._validate_args(**dict(_ARGS, head_dim=64))
    with pytest.raises(ProfilerNotImplemented, match="d_rel"):
        runner._validate_args(**dict(_ARGS, d_rel=32))
    with pytest.raises(ProfilerNotImplemented, match="bf16"):
        runner._validate_args(**dict(_ARGS, dtype="fp16"))
    with pytest.raises(ValueError, match="multiple"):
        runner._validate_args(**dict(_ARGS, num_kv_heads=3))


def test_logical_bytes_match_the_per_token_traffic():
    # Defect caught: GB/s computed from the wrong byte count. Inkling-Small at
    # TP4 moves 15,616 B per token on local layers (E = 512) and 23,808 B on
    # global layers (E = 1024), plus the rel_proj matrix once.
    assert runner.logical_bytes(_SHAPE, 1) == 15_616 + 2 * 16 * 512
    glob = PrepShape(8, 2, 128, 16, 1024, 4)
    assert runner.logical_bytes(glob, 10) == 10 * 23_808 + 2 * 16 * 1024
