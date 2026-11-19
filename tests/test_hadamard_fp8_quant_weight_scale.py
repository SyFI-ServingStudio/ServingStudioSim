"""CPU tests for the Hadamard FP8 quant with weight scaling: vendor bytes, geometry, fork pair."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from profiling.runners.elementwise.hadamard_fp8_quant_vllm_triton import reference
from profiling.runners.elementwise.hadamard_fp8_quant_weight_scale_triton import (
    _validate_args,
    fork_pair,
    logical_bytes,
)
from profiling.runners.exceptions import ProfilerNotImplemented

_VENDORED = (
    Path(__file__).resolve().parents[1]
    / "profiling/runners/elementwise/hadamard_wscale_vendor/triton_post_b.py"
)
# md5 of the measured kernel source (everything below the module docstring).
_ORIGIN_POST_B_MD5 = "3b112896e0b2eb284ee3047fb3992939"


def test_vendored_kernel_is_byte_identical_to_its_origin():
    text = _VENDORED.read_text()
    body = text[text.index('"""\n', 3) + 4 :]
    assert hashlib.md5(body.encode()).hexdigest() == _ORIGIN_POST_B_MD5


def test_other_head_geometry_is_refused_not_timed():
    # rows // 32 and the 0.015625 weight factor are compiled in; lead_cols is free.
    _validate_args(8, 32, 128, 16384, "bf16")
    _validate_args(8, 32, 128, 0, "bf16")
    with pytest.raises(ProfilerNotImplemented):
        _validate_args(8, 64, 128, 16384, "bf16")
    with pytest.raises(ProfilerNotImplemented):
        _validate_args(8, 32, 64, 16384, "bf16")
    with pytest.raises(ProfilerNotImplemented):
        _validate_args(8, 32, 128, 16384, "fp16")
    with pytest.raises(ValueError):
        _validate_args(8, 32, 128, -1, "bf16")
    # int32 input offsets: 65536 tokens of 20480 columns fit, 131072 wrap.
    _validate_args(65536, 32, 128, 16384, "bf16")
    with pytest.raises(ProfilerNotImplemented):
        _validate_args(131072, 32, 128, 16384, "bf16")


def _fake_fwht(q):
    out, scale = reference(torch, q)
    return out.to(torch.float8_e4m3fn), scale.float()


def test_fork_pair_reads_the_span_after_lead_cols_and_scales_by_its_head_scale():
    # Catches a fork-side reference that quantizes the wrong columns or uses
    # another weight factor than head_dim^-0.5 * num_heads^-0.5.
    g = torch.Generator().manual_seed(2)
    tokens, heads, dim, lead = 3, 32, 128, 40
    y = torch.randn(tokens, lead + heads * dim, generator=g).bfloat16()
    hw = torch.randn(tokens, heads, generator=g)
    q, s, w = fork_pair(torch, _fake_fwht, y, lead, hw, heads, dim)
    want_q, want_s = reference(torch, y[:, lead:].reshape(-1, dim))
    assert torch.equal(q, want_q.to(torch.float8_e4m3fn).view(torch.uint8))
    assert torch.equal(s, want_s.float().view(-1))
    assert torch.equal(w, hw.reshape(-1) * s * (1 / 64))


def test_logical_bytes_counts_heads_weights_scales():
    assert logical_bytes(1, 32, 128) == 32 * (128 * 3 + 12)
