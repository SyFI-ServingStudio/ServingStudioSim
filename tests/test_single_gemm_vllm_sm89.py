"""CPU checks for the ``single_gemm`` ``vllm_cutlass_fp8`` and
``vllm_marlin_nvfp4`` backends (vLLM's FP8 and NVFP4 linears without FP4 tensor
cores, e.g. on an L40S).

GPU timing and the numerical check against the production calls are covered by
the public ``python -m profiling run`` smoke; these tests guard what can regress
without a GPU.
"""

from __future__ import annotations

import pytest

from profiling.db.args import DType
from profiling.db.registry import backend_supports, supported_backends
from profiling.runners.exceptions import KernelLaunchFailed
from profiling.runners.gemm.vllm_cutlass_fp8 import (
    profile_single_gemm_vllm_cutlass_fp8,
)
from profiling.runners.gemm.vllm_marlin_nvfp4 import (
    profile_single_gemm_vllm_marlin_nvfp4,
)


def test_ada_backends_follow_vllm_capability_rules():
    # cutlass_scaled_mm_supports_fp8: capability >= 89. Marlin W4A16 with a
    # bf16 activation: 8.0+. Neither may be offered on a GPU vLLM refuses, and
    # the SM10x-only FlashInfer rows must not be offered on an L40S.
    l40s = "NVIDIA L40S"
    assert backend_supports("single_gemm", "vllm_cutlass_fp8", DType.FP8_E4M3, gpu=l40s)
    assert backend_supports("single_gemm", "vllm_cutlass_fp8", DType.FP8_E4M3, gpu="NVIDIA H200")
    assert not backend_supports(
        "single_gemm", "vllm_cutlass_fp8", DType.FP8_E4M3, gpu="NVIDIA A100"
    )
    assert not backend_supports("single_gemm", "vllm_cutlass_fp8", DType.BF16, gpu=l40s)
    assert backend_supports("single_gemm", "vllm_marlin_nvfp4", DType.NVFP4_E2M1, gpu=l40s)
    assert not backend_supports("single_gemm", "vllm_marlin_nvfp4", DType.BF16, gpu=l40s)
    assert supported_backends("single_gemm", DType.NVFP4_E2M1, gpu=l40s) == ["vllm_marlin_nvfp4"]
    assert "flashinfer_fp8_tensor" not in supported_backends(
        "single_gemm", DType.FP8_E4M3, gpu=l40s
    )


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"m": 8, "n": 16384, "k": 5120, "dtype": "bf16"}, "dtype=fp8_e4m3"),
        ({"m": 8, "n": 16384, "k": 5120, "dtype": "nvfp4_e2m1"}, "dtype=fp8_e4m3"),
        ({"m": 0, "n": 16384, "k": 5120, "dtype": "fp8_e4m3"}, ">= 1"),
    ],
)
def test_cutlass_fp8_rejects_specs_before_gpu_work(spec, message):
    with pytest.raises(ValueError, match=message):
        profile_single_gemm_vllm_cutlass_fp8(**spec)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"m": 8, "n": 5120, "k": 17408, "dtype": "bf16"}, "dtype=nvfp4_e2m1"),
        ({"m": 8, "n": 5120, "k": 17408, "dtype": "fp8_e4m3"}, "dtype=nvfp4_e2m1"),
        ({"m": 8, "n": 5120, "k": 17400, "dtype": "nvfp4_e2m1"}, "divisible by 16"),
        ({"m": 0, "n": 5120, "k": 17408, "dtype": "nvfp4_e2m1"}, ">= 1"),
    ],
)
def test_marlin_nvfp4_rejects_specs_before_gpu_work(spec, message):
    with pytest.raises(ValueError, match=message):
        profile_single_gemm_vllm_marlin_nvfp4(**spec)


def test_cutlass_fp8_reference_applies_both_per_tensor_scales():
    # A production call that ignored scale_a or scale_b must fail the gate.
    torch = pytest.importorskip("torch")
    from profiling.runners.gemm.vllm_cutlass_fp8 import (
        _A_SCALE,
        _W_SCALE,
        check_against_reference,
    )

    gen = torch.Generator().manual_seed(0)
    a = torch.randn(32, 64, generator=gen).to(torch.float8_e4m3fn)
    w = torch.randn(48, 64, generator=gen).to(torch.float8_e4m3fn)
    unscaled = (a.float() @ w.float().t()).to(torch.bfloat16)
    check_against_reference(torch, a, w, unscaled * (_A_SCALE * _W_SCALE))
    with pytest.raises(KernelLaunchFailed):
        check_against_reference(torch, a, w, unscaled)


def test_nvfp4_oracle_decodes_the_checkpoint_nibble_order():
    # Element 2j is the LOW nibble of byte j (the order gptq_marlin_repack and
    # rand_marlin_weight_nvfp4_like assume). A weight made of exact E2M1 values
    # times power-of-two block scales must round-trip bit-exactly; a swapped
    # nibble order or a dropped scale breaks it.
    torch = pytest.importorskip("torch")
    from profiling.runners.gemm.vllm_marlin_nvfp4 import (
        _E2M1_VALUES,
        dequantize_nvfp4_rows,
        quantize_nvfp4_weight,
    )

    gen = torch.Generator().manual_seed(0)
    values = torch.tensor(_E2M1_VALUES)
    codes = torch.randint(0, 16, (8, 64), generator=gen)
    w = torch.where(codes >= 8, -values[codes % 8], values[codes % 8])
    w[:, 0::16] = 6.0  # every block reaches the FP4 max
    w = w * (2.0 ** torch.arange(-2, 2).repeat_interleave(16)).unsqueeze(0)
    w = w.to(torch.bfloat16)

    packed, scales, global_scale = quantize_nvfp4_weight(torch, w)
    assert packed.shape == (8, 32) and scales.shape == (8, 4)
    decoded = dequantize_nvfp4_rows(torch, packed, scales, global_scale)
    torch.testing.assert_close(decoded, w.float(), rtol=1e-6, atol=0)
    lo, hi = packed & 0xF, packed >> 4
    swapped = (hi | (lo << 4)).to(torch.uint8)
    assert not torch.allclose(
        dequantize_nvfp4_rows(torch, swapped, scales, global_scale), w.float()
    )


def test_nvfp4_oracle_rejects_an_unscaled_output():
    torch = pytest.importorskip("torch")
    from profiling.runners.gemm.vllm_marlin_nvfp4 import (
        check_against_reference,
        dequantize_nvfp4_rows,
        quantize_nvfp4_weight,
    )

    gen = torch.Generator().manual_seed(1)
    x = torch.randn(20, 64, generator=gen).to(torch.bfloat16)
    w = torch.randn(40, 64, generator=gen).to(torch.bfloat16)
    packed, scales, gs = quantize_nvfp4_weight(torch, w)
    good = (x.float() @ dequantize_nvfp4_rows(torch, packed, scales, gs).t()).to(torch.bfloat16)
    check_against_reference(torch, x, packed, scales, gs, good)
    with pytest.raises(KernelLaunchFailed):
        check_against_reference(torch, x, packed, scales, gs, good / gs.item())
