"""CPU tests for the static-FP8 batched GEMM: vendored bytes, geometry gate, reference."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.gemm.batched_gemm_static_fp8_quant_tcgen05 import (
    CONFIGS,
    _code_distance,
    _validate_args,
    reference,
)

_VENDOR = Path(__file__).resolve().parents[1] / "profiling/runners/gemm/absorb_fp8_vendor"
# md5s of the kernel sources as first written (fc2.cu before its ragged-M patch).
_ORIGIN_FC2_MD5 = "1779132c92e3aec06ea3ed67a482f4c3"
_ORIGIN_COMMON_MD5 = "557308223273ee89cb9aab346f032245"


def _strip_header(text: str, lines: int) -> str:
    return "".join(text.splitlines(keepends=True)[lines:])


def test_vendored_kernel_is_its_origin_plus_the_marked_ragged_m_patch():
    # Catches an edit beyond the three marked lines: rows would no longer be
    # the measured kernel's.
    common = _strip_header((_VENDOR / "common.cuh").read_text(), 2)
    assert hashlib.md5(common.encode()).hexdigest() == _ORIGIN_COMMON_MD5
    fc2 = _strip_header((_VENDOR / "fc2.cu").read_text(), 5)
    assert fc2.count("// PATCH ragged M") == 3
    unpatched = (
        fc2.replace("  int M;  // PATCH ragged M\n", "")
        .replace("if (tok < p.M) *reinterpret_cast", "*reinterpret_cast")
        .replace(" = u;  // PATCH ragged M\n", " = u;\n")
        .replace("p.mtiles = (M + 127) / 128; p.M = M;  // PATCH ragged M", "p.mtiles = M / 128;")
    )
    assert hashlib.md5(unpatched.encode()).hexdigest() == _ORIGIN_FC2_MD5


def test_other_geometry_or_dtype_is_refused_not_timed():
    # The kernel hard-codes 64 heads, k = 256 and n = 512.
    _validate_args(64, 7, 512, 256, "bf16")
    with pytest.raises(ProfilerNotImplemented):
        _validate_args(16, 128, 512, 256, "bf16")
    with pytest.raises(ProfilerNotImplemented):
        _validate_args(64, 128, 512, 192, "bf16")
    with pytest.raises(ProfilerNotImplemented):
        _validate_args(64, 128, 512, 256, "fp16")
    with pytest.raises(ValueError):
        _validate_args(64, 0, 512, 256, "bf16")


def test_every_config_is_one_the_kernel_instantiates():
    # run_c dispatches only these (n per CTA, stages) pairs.
    src = (_VENDOR / "fc2.cu").read_text()
    for nn, stages in CONFIGS:
        assert f"CR({nn}, {stages})" in src


def test_reference_rounds_to_bf16_then_scales_and_clamps():
    g = torch.Generator().manual_seed(1)
    q = torch.randn(5, 2 * 4, generator=g).bfloat16()
    w = (torch.randn(2, 4, 3, generator=g) * 3).bfloat16()
    got = reference(torch, q, w, 0.05)
    assert got.shape == (5, 2, 3) and got.dtype == torch.uint8
    prod = torch.einsum("mbk,bkn->mbn", q.view(5, 2, 4).float(), w.float())
    want = (prod.bfloat16().float() * (1 / 0.05)).clamp(-448, 448).to(torch.float8_e4m3fn)
    assert torch.equal(got, want.view(torch.uint8))
    # Large products saturate at the E4M3 maximum instead of overflowing.
    big = reference(torch, q, w * 100, 0.05).view(torch.float8_e4m3fn).float()
    assert big.abs().max().item() == 448.0


def test_code_distance_counts_steps_across_zero():
    codes = torch.tensor([1.0, -1.0, 0.0]).to(torch.float8_e4m3fn).view(torch.uint8)
    step = torch.tensor([1.125, -1.0, 0.0]).to(torch.float8_e4m3fn).view(torch.uint8)
    assert _code_distance(torch, codes, codes) == 0
    assert _code_distance(torch, codes, step) == 1
