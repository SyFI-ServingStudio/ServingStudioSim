"""CPU tests for the fp32 row-split LayerNorm kind: vendored bytes, geometry gate, acceptance."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.norm.fp32_row_split_layer_norm_reference import (
    bf16_ulp_distance,
    check_row_split,
    fp32_row_split_layer_norm_reference,
)
from profiling.runners.norm.fp32_row_split_layer_norm_triton import _validate_args

_VENDORED = (
    Path(__file__).resolve().parents[1] / "profiling/runners/norm/row_split_vendor/triton_post_a.py"
)
# md5 of the measured kernel source (everything below the module docstring).
_ORIGIN_POST_A_MD5 = "88d626c64ab27cc3128d4baad5639d21"


def test_vendored_kernel_is_byte_identical_to_its_origin():
    # Catches a reformat or edit of the vendored kernel: its rows would no
    # longer be the measured kernel's.
    text = _VENDORED.read_text()
    body = text[text.index('"""\n', 3) + 4 :]
    assert hashlib.md5(body.encode()).hexdigest() == _ORIGIN_POST_A_MD5


def test_other_row_geometry_is_refused_not_timed():
    # The kernel hard-codes its spans; another geometry would time the wrong work.
    _validate_args(64, 2048, 128, 128, 32)
    with pytest.raises(ProfilerNotImplemented):
        _validate_args(64, 1536, 128, 128, 32)


def _reference(y, w, b):
    return fp32_row_split_layer_norm_reference(torch, y, w, b, 1e-6, 6, 4, 3)


def _sample():
    g = torch.Generator().manual_seed(3)
    y = torch.randn(2, 15, generator=g) * 4
    w = 1 + 0.1 * torch.randn(4, generator=g)
    b = 0.1 * torch.randn(4, generator=g)
    return y, w, b


def test_reference_norms_the_bf16_rounded_span():
    # The fork LayerNorms a bf16 GEMM output: rounding must come first.
    y, w, b = _sample()
    lead, normed, trail = _reference(y, w, b)
    span = y[:, 6:10].bfloat16().float()
    mean = span.mean(-1, keepdim=True)
    var = (span - mean).square().mean(-1, keepdim=True)
    manual = ((span - mean) / torch.sqrt(var + 1e-6) * w + b).bfloat16()
    assert torch.equal(normed, manual)
    assert torch.equal(lead, y[:, :6].bfloat16())
    assert torch.equal(trail, y[:, 10:13].bfloat16())


def test_ulp_distance_crosses_zero_and_ignores_signed_zero():
    a = torch.tensor([0.0, -0.0, 1.0], dtype=torch.bfloat16)
    smallest = torch.tensor([0, 1], dtype=torch.int16).view(torch.bfloat16)
    assert bf16_ulp_distance(torch, a[:1], a[1:2]).item() == 0
    assert bf16_ulp_distance(torch, smallest[1:], -smallest[1:]).item() == 2
    one_up = torch.tensor([0x3F81], dtype=torch.int16).view(torch.bfloat16)
    assert bf16_ulp_distance(torch, one_up, a[2:]).item() == 1


def test_check_rejects_two_ulp_norm_or_touched_tail():
    y, w, b = _sample()
    expected = _reference(y, w, b)
    lead, normed, trail = (t.clone() for t in expected)
    check_row_split(torch, y, y.clone(), (lead, normed, trail), expected)
    bumped = normed.clone()
    bumped.view(torch.int16)[0, 0] += 1
    check_row_split(torch, y, y.clone(), (lead, bumped, trail), expected)
    bumped.view(torch.int16)[0, 0] += 1
    with pytest.raises(AssertionError, match="ulp"):
        check_row_split(torch, y, y.clone(), (lead, bumped, trail), expected)
    touched = y.clone()
    touched[1, -1] += 1
    with pytest.raises(AssertionError, match="fp32"):
        check_row_split(torch, y, touched, (lead, normed, trail), expected)
    lead_off = lead.clone()
    lead_off.view(torch.int16)[0, 0] += 1
    with pytest.raises(AssertionError, match="lead"):
        check_row_split(torch, y, y.clone(), (lead_off, normed, trail), expected)


def test_check_accepts_near_zero_cancellation_only():
    # x_hat * w + b cancelling to ~3e-7 is many bf16 ulps off after fp32
    # rounding (measured on B200 at 6912+ rows); a 2-ulp error at O(1) is not.
    y, w, b = _sample()
    lead, normed, trail = _reference(y, w, b)
    ref = normed.clone()
    ref[0, 0] = 2.75e-7
    near = ref.clone()
    near[0, 0] = 2.85e-7
    assert bf16_ulp_distance(torch, near, ref).max().item() > 1
    check_row_split(torch, y, y.clone(), (lead, near, trail), (lead, ref, trail))
