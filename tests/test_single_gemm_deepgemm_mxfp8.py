"""CPU checks for the ``single_gemm`` ``deepgemm_mxfp8`` backend.

GPU timing and the numerical check against DeepInfer's linear are covered by the
public ``python -m profiling run`` smoke in the DeepInfer image on a B200; these
tests guard the pieces that can regress without a GPU.
"""

from __future__ import annotations

import pytest

from profiling.db import DType, MetricFamily, find_kernel_profiler_spec
from profiling.kernels.single_gemm import SingleGemmArgs
from profiling.runners.gemm.deepgemm_mxfp8 import (
    _dequantize,
    profile_single_gemm_deepgemm_mxfp8,
)


def test_backend_runs_in_the_deepinfer_image_on_sm10x_mxfp8_only():
    # A wrong env would import a DeepGEMM other than the one DeepInfer vendors;
    # a wrong dtype gate would time per-128 FP8 or BF16 rows as this kernel.
    spec = find_kernel_profiler_spec("single_gemm", "deepgemm_mxfp8")
    assert spec.table_name == "single_gemm"
    assert spec.args_schema is SingleGemmArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.subprocess_env == "deepinfer_env"
    assert spec.supports.compute == frozenset({DType.MXFP8_E4M3})
    assert spec.supports.sm_targets == frozenset({"sm_100f"})
    assert spec.runner_ref.module_name == "profiling.runners.gemm.deepgemm_mxfp8"


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"m": 48, "n": 1792, "k": 5120, "dtype": "fp8_e4m3"}, "dtype=mxfp8_e4m3"),
        ({"m": 48, "n": 1792, "k": 5120, "dtype": "bf16"}, "dtype=mxfp8_e4m3"),
        ({"m": 0, "n": 1792, "k": 5120, "dtype": "mxfp8_e4m3"}, "m >= 1"),
        ({"m": 48, "n": 1792, "k": 5136, "dtype": "mxfp8_e4m3"}, "multiple of 32"),
    ],
)
def test_runner_rejects_specs_before_any_gpu_work(spec, message):
    with pytest.raises(ValueError, match=message):
        profile_single_gemm_deepgemm_mxfp8(**spec)


def test_reference_dequantization_spans_scale_blocks():
    # The oracle reads DeepGEMM's unpacked fp32 power-of-two scales: one per
    # row (activation) or per 32 rows (weight block), each over 32 columns.
    torch = pytest.importorskip("torch")
    data = torch.ones(64, 64, dtype=torch.float32).to(torch.float8_e4m3fn)
    scales = torch.tensor([[1.0, 2.0], [0.5, 8.0]])
    out = _dequantize(torch, data, scales, 32)
    assert torch.equal(out[:32, :32], torch.full((32, 32), 1.0))
    assert torch.equal(out[:32, 32:], torch.full((32, 32), 2.0))
    assert torch.equal(out[32:, :32], torch.full((32, 32), 0.5))
    assert torch.equal(out[32:, 32:], torch.full((32, 32), 8.0))
    per_row = _dequantize(torch, data[:2], scales, 1)
    assert torch.equal(per_row[1, 32:], torch.full((32,), 8.0))


def test_fp8_output_einsum_is_a_separate_deepinfer_backend_of_batched_gemm():
    # DeepInfer's wv einsum writes FP8 through DeepGEMM's cast epilogue, a
    # different kernel instance from vLLM's bf16-output einsum; routing it to
    # the vLLM row would price the wrong kernel in the wrong environment.
    from profiling.kernels.batched_gemm import BatchedGemmArgs

    spec = find_kernel_profiler_spec("batched_gemm", "deepgemm_mxfp8_einsum_grouped_o_proj_fp8_out")
    assert spec.args_schema is BatchedGemmArgs
    assert spec.subprocess_env == "deepinfer_env"
    assert spec.supports.compute == frozenset({DType.MXFP8_E4M3})
    vllm = find_kernel_profiler_spec("batched_gemm", "deepgemm_mxfp8_einsum_grouped_o_proj")
    assert vllm.subprocess_env == "vllm_env"


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"num_batches": 8, "m": 4, "n": 1024, "k": 4096, "dtype": "bf16"}, "mxfp8_e4m3"),
        ({"num_batches": 8, "m": 4, "n": 1000, "k": 4096, "dtype": "mxfp8_e4m3"}, "n to be"),
        ({"num_batches": 0, "m": 4, "n": 1024, "k": 4096, "dtype": "mxfp8_e4m3"}, "num_batches"),
    ],
)
def test_fp8_output_einsum_rejects_shapes_its_scale_packing_cannot_hold(spec, message):
    # DeepGEMM asserts n % 128 == 0 for the FP8 output pair (one packed scale
    # word may not cross a group); fail before any GPU work instead.
    from profiling.runners.gemm.deepgemm_mxfp8_einsum_fp8_out import (
        profile_batched_gemm_deepgemm_mxfp8_einsum_grouped_o_proj_fp8_out as run,
    )

    with pytest.raises(ValueError, match=message):
        run(**spec)


def test_packed_ue8m0_words_unpack_little_endian():
    # The FP8-output oracle reads the einsum's packed output scale; a byte-order
    # mistake would make a correct kernel fail (or a wrong one pass).
    torch = pytest.importorskip("torch")
    from profiling.runners.gemm.deepgemm_mxfp8_einsum_fp8_out import _unpack_ue8m0

    # DeepGEMM's pack_ue8m0_to_int views four consecutive exponent bytes as one int32.
    word = torch.tensor([[127, 128, 126, 130]], dtype=torch.uint8).view(torch.int32)
    assert _unpack_ue8m0(torch, word).tolist() == [[1.0, 2.0, 0.5, 8.0]]
