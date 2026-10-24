"""TensorFold's activation quantizers: tensorfold_cuda on nvfp4_quant and static_fp8_quant."""

from __future__ import annotations

import pytest
import torch

from profiling.db.args import DType
from profiling.db.registry import find_kernel_profiler_spec
from profiling.runners.elementwise import tensorfold_quant as tq
from profiling.runners.exceptions import ProfilerNotImplemented


@pytest.mark.parametrize("kind", ["nvfp4_quant", "static_fp8_quant"])
def test_tensorfold_cuda_rows_run_in_tensorfold_env_on_sm12x_bf16(kind):
    # Catches the row landing on another env (no TensorFold) or advertising a
    # GPU family the shared SM12x extension is not built for.
    spec = find_kernel_profiler_spec(kind, "tensorfold_cuda")
    assert spec.subprocess_env == "tensorfold_env"
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA GB10")
    assert not spec.supports.allows(DType.FP16, gpu="NVIDIA GB10")
    assert spec.supports.sm_targets == frozenset({"sm_120f"})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale_format": "linear_e4m3"},
        {"scale_format": "swizzled_e4m3"},
        {"hidden_size": 5120 + 16},
        {"input_dtype": "fp16"},
        {"num_tokens": 65536},
    ],
)
def test_quant4_refuses_layouts_and_shapes_it_cannot_write(kwargs):
    # Catches a row recorded under another scale layout's identity, a K that
    # does not fill whole 64-input scale slices, or more rows than gridDim.y takes.
    spec = {
        "num_tokens": 4,
        "hidden_size": 5120,
        "group_size": 16,
        "input_dtype": "bf16",
        "scale_format": "kslice_e4m3",
        **kwargs,
    }
    with pytest.raises(ProfilerNotImplemented):
        tq.validate_nvfp4_args(**spec)


@pytest.mark.parametrize(
    "kwargs", [{"gated": True}, {"hidden_dim": 5128}, {"dtype": "fp16"}, {"num_tokens": 65536}]
)
def test_quant8_refuses_the_gated_form_and_ragged_rows(kwargs):
    # TensorFold applies the attention gate in its own launch; a gated row here
    # would be timed without the gate read it claims.
    spec = {"num_tokens": 4, "hidden_dim": 6144, "gated": False, "dtype": "bf16", **kwargs}
    with pytest.raises(ProfilerNotImplemented):
        tq.validate_fp8_args(**spec)


def test_kslice_unpack_inverts_the_kernel_scale_address():
    # act.cu quant4_kernel (tb = 0) stores block `blk` of `row` at byte
    # ((blk / 4) * mpad + row) * 4 + blk % 4; a wrong unpack makes the oracle
    # compare unrelated scales.
    m, k = 70, 256
    mpad = 128
    groups = k // 16
    want = torch.arange(m * groups, dtype=torch.int64).reshape(m, groups) % 251
    flat = torch.zeros((k // 64) * mpad * 4, dtype=torch.int64)
    for row in range(m):
        for blk in range(groups):
            flat[((blk // 4) * mpad + row) * 4 + blk % 4] = want[row, blk]
    got = tq.kslice_to_rows(flat.reshape(k // 64, mpad, 4), m)
    assert torch.equal(got, want)


def test_fp8_logical_order_inverts_the_fragment_order():
    # act.cu quant8_kernel: byte 4q + j of a 16-byte group holds input
    # 2q + j%2 + 8(j/2). The oracle must undo exactly that permutation.
    logical = torch.arange(32, dtype=torch.uint8).reshape(1, 32)
    stored = torch.empty_like(logical)
    for g in range(2):
        for q in range(4):
            for j in range(4):
                stored[0, 16 * g + 4 * q + j] = logical[0, 16 * g + 2 * q + j % 2 + 8 * (j // 2)]
    assert torch.equal(tq.fp8_logical_order(torch, stored), logical)


def test_nvfp4_reference_rounds_to_nearest_e2m1_with_ties_to_even():
    # One block whose amax is 6 under g = 1 makes the scale exactly 1.0, so the
    # codes are the e2m1 rounding of the inputs; midpoints go to the even code
    # (0.25 -> 0, 0.75 -> 1.0, 1.25 -> 1.0, 1.75 -> 2.0, 2.5 -> 2.0, 3.5 -> 4.0, 5.0 -> 4.0).
    vals = [6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.5, -3.0, 0.2, 0.3, 1.0, -6.0, 0.0, 4.0]
    x = torch.tensor([vals], dtype=torch.bfloat16)
    codes, sf = tq.nvfp4_reference(torch, x, act=1.0)
    assert sf.tolist() == [[0x38]]  # e4m3 1.0
    want = [7, 0, 2, 2, 4, 4, 6, 6, 9, 13, 0, 1, 2, 15, 0, 6]
    unpacked = torch.stack((codes & 0xF, codes >> 4), dim=-1).flatten().tolist()
    assert unpacked == want


def test_quant4_bytes_count_the_padded_scale_buffer_its_fill_writes():
    # The timed call zero-fills scales for ceil(M/64)*64 rows; counting only the
    # real rows would overstate GB/s at decode sizes.
    assert tq.nvfp4_logical_bytes(3, 64) == 3 * 64 * 2 + 3 * 32 + 64 * 4
    assert tq.nvfp4_logical_bytes(64, 64) == 64 * 64 * 2 + 64 * 32 + 64 * 4
    assert tq.fp8_logical_bytes(3, 64) == 3 * 3 * 64


def test_quant4_is_timed_as_its_whole_launch_sequence(monkeypatch):
    # checkpoint.quant4 always issues the scale buffer's zero fill before
    # quant4_kernel; filtering CUPTI by the kernel name would drop the fill.
    from types import SimpleNamespace

    seen = {}
    monkeypatch.setattr(tq, "check_quant4", lambda *a: {})
    monkeypatch.setattr(tq, "_rows", lambda torch_, m, k, seed: torch.ones(m, k))
    fake_checkpoint = SimpleNamespace(quant4=lambda x, act: None)
    monkeypatch.setattr(tq, "_load_checkpoint", lambda label: (torch, fake_checkpoint))

    def fake_cupti(fn, *, kernel_name):
        seen["kernel_name"] = kernel_name
        return 0.004

    monkeypatch.setattr(tq.Timer, "cupti", fake_cupti)
    monkeypatch.setattr(tq.Energy, "perf", lambda fn, *, per_iter_time_ms: 0.0)
    metrics = tq.profile_nvfp4_quant_tensorfold_cuda(16, 5120, 16, "bf16", "kslice_e4m3")
    assert seen["kernel_name"] is None
    assert metrics.time_ms == 0.004
