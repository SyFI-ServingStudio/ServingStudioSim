from __future__ import annotations

from dataclasses import fields

import pytest

from profiling.db.args import DType
from profiling.db.registry import find_kernel_profiler_spec
from profiling.kernels.gemm_fp32_output import GemmFp32OutputArgs
from profiling.kernels.moe_finalize_fuse_shared import MoeFinalizeFuseSharedArgs
from profiling.kernels.nvfp4_fused_moe import Nvfp4FusedMoeArgs
from profiling.kernels.nvfp4_quant import Nvfp4QuantArgs
from profiling.kernels.single_gemm import SingleGemmArgs
from profiling.runners.gemm import sglang as sglang_gemm
from profiling.runners.gemm.gemm_fp32_output_sglang_router import (
    _max_router_gemm_tokens,
)
from profiling.runners.moe import moe_finalize_fuse_shared, nvfp4_fused_moe


def test_sglang_registry_rows_reuse_existing_schemas_and_env() -> None:
    expected = {
        ("nvfp4_quant", "flashinfer_cutedsl"): Nvfp4QuantArgs,
        (
            "nvfp4_fused_moe",
            "flashinfer_trtllm_sm100_deferred_finalize",
        ): Nvfp4FusedMoeArgs,
        ("single_gemm", "sglang_bf16_auto"): SingleGemmArgs,
        ("single_gemm", "sglang_fused_a_auto"): SingleGemmArgs,
        ("gemm_fp32_output", "sglang_router_auto"): GemmFp32OutputArgs,
        ("moe_finalize_fuse_shared", "sglang_cuda"): MoeFinalizeFuseSharedArgs,
    }
    for (kind, backend), schema in expected.items():
        spec = find_kernel_profiler_spec(kind, backend)
        assert spec.table_name == kind
        assert spec.args_schema is schema
        assert spec.subprocess_env == "sglang_env"
        assert spec.supports.compute == frozenset({DType.BF16})
        assert spec.supports.gpus == frozenset({"NVIDIA B200"})


def test_moe_finalize_schema_keeps_physical_key_order() -> None:
    assert [field.name for field in fields(MoeFinalizeFuseSharedArgs)] == [
        "num_tokens",
        "top_k",
        "hidden_dim",
        "dtype",
        "fuse_shared_output",
    ]


def test_deferred_moe_wrapper_forwards_stack_and_finalize(monkeypatch) -> None:
    received = {}

    def fake_profile(**kwargs):
        received.update(kwargs)
        return object()

    monkeypatch.setattr(nvfp4_fused_moe, "_profile_nvfp4_fused_moe_sm100", fake_profile)
    marker = nvfp4_fused_moe.profile_nvfp4_fused_moe_deferred_finalize_sm100(num_tokens=8)
    assert marker is not None
    assert received == {"stack": "sglang", "do_finalize": False, "num_tokens": 8}


def test_finalize_launch_forwards_public_callable_arguments() -> None:
    calls = []

    def fake_fn(*args):
        calls.append(args)

    launch = moe_finalize_fuse_shared._Launch(
        fn=fake_fn,
        gemm2_out="rows",
        permuted_idx="perm",
        expert_weights="weights",
        shared_output="shared",
        top_k=8,
        enable_pdl=True,
    )
    launch.run()
    assert calls == [("rows", "perm", "weights", "shared", 8, True)]


class _FakeTensor:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def reshape(self, *shape):
        return _FakeTensor(shape, self.dtype)

    def view(self, dtype):
        return _FakeTensor(self.shape, dtype)


def test_sglang_quantization_forwards_per_token_scale_and_vllm_omits_it() -> None:
    torch = type("Torch", (), {"float8_e4m3fn": "float8_e4m3fn"})
    source = _FakeTensor((2, 32), "bf16")
    global_scale = _FakeTensor((1,), "float32")
    per_token_scale = _FakeTensor((2, 1), "float32")
    quant_calls = []

    def fake_quantize(*args, **kwargs):
        quant_calls.append((args, kwargs))
        return (
            _FakeTensor((2, 16), "uint8"),
            _FakeTensor((2, 2), "uint8"),
            per_token_scale,
        )

    hidden, hidden_scale, forwarded_scale = nvfp4_fused_moe._quantize_sglang_hidden(
        torch,
        fake_quantize,
        "linear",
        source,
        global_scale,
    )
    assert quant_calls == [
        (
            (source, global_scale),
            {
                "sfLayout": "linear",
                "per_token_activation": True,
                "backend": "cute-dsl",
            },
        )
    ]
    assert hidden.shape == (2, 16)
    assert hidden_scale.shape == (2, 2)
    assert hidden_scale.dtype == "float8_e4m3fn"
    assert forwarded_scale is per_token_scale
    assert forwarded_scale.shape == (2, 1)
    assert forwarded_scale.dtype == "float32"

    moe_calls = []
    nvfp4_fused_moe._call_trtllm_fp4_moe(
        lambda **kwargs: moe_calls.append(kwargs),
        stack="sglang",
        per_token_scale=forwarded_scale,
        kwargs={"hidden_states": hidden},
    )
    nvfp4_fused_moe._call_trtllm_fp4_moe(
        lambda **kwargs: moe_calls.append(kwargs),
        stack="vllm",
        per_token_scale=None,
        kwargs={"hidden_states": hidden},
    )
    assert moe_calls[0]["per_token_scale"] is per_token_scale
    assert "per_token_scale" not in moe_calls[1]


def test_sglang_router_uses_blackwell_production_threshold() -> None:
    assert _max_router_gemm_tokens(100) == 4
    assert _max_router_gemm_tokens(103) == 4
    assert _max_router_gemm_tokens(90) == 16


@pytest.mark.parametrize(
    "entry",
    [
        sglang_gemm.profile_single_gemm_sglang_bf16,
        sglang_gemm.profile_single_gemm_sglang_fused_a,
    ],
)
@pytest.mark.parametrize(
    "shape",
    [(0, 32, 32), (1, 0, 32), (1, 32, -1), (True, 32, 32)],
)
def test_sglang_gemm_rejects_invalid_shapes_before_framework_import(entry, shape) -> None:
    with pytest.raises(ValueError, match="must be a positive integer"):
        entry(*shape, dtype="bf16")
