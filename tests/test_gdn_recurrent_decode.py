"""Registration and CPU runner tests for ``gdn_recurrent_decode``."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import fields
from types import SimpleNamespace

import pytest

from profiling import perf_api
from profiling.db.args import DType
from profiling.db.batch import coerce_args
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import MetricFamily, find_kernel_profiler_spec, known_backends
from profiling.db.table import MissingEntry, Table
from profiling.kernels.gdn_recurrent_decode import (
    KIND,
    GdnRecurrentDecodeArgs,
)
from profiling.runners.exceptions import ProfilerNotImplemented

_SPEC = {
    "batch_size": 1,
    "num_qk_heads": 16,
    "num_value_heads": 32,
    "key_head_dim": 128,
    "value_head_dim": 128,
    "dtype": "bf16",
    "state_dtype": "fp32",
}


def test_args_field_order_and_coercion() -> None:
    assert [field.name for field in fields(GdnRecurrentDecodeArgs)] == [
        "batch_size",
        "num_qk_heads",
        "num_value_heads",
        "key_head_dim",
        "value_head_dim",
        "dtype",
        "state_dtype",
    ]
    args = coerce_args(
        GdnRecurrentDecodeArgs,
        _SPEC
        | {
            "batch_size": "1",
            "num_qk_heads": "16",
            "dtype": "bfloat16",
            "state_dtype": "float32",
        },
    )
    assert args == GdnRecurrentDecodeArgs(
        batch_size=1,
        num_qk_heads=16,
        num_value_heads=32,
        key_head_dim=128,
        value_head_dim=128,
        dtype=DType.BF16,
        state_dtype=DType.FP32,
    )
    with pytest.raises(Exception):
        args.batch_size = 2


def test_registration_table_kind_runner_and_support_contract() -> None:
    spec = find_kernel_profiler_spec(KIND, "torch")

    assert KIND == "gdn_recurrent_decode"
    assert known_backends(KIND) == ["torch", "vllm_triton"]
    assert spec.kernel_kind == spec.table_name == KIND
    assert spec.backend == "torch"
    assert spec.args_schema is GdnRecurrentDecodeArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.batch_outlier_policy == BatchOutlierPolicy()
    assert spec.subprocess_env is None
    assert spec.runner_ref.module_name == ("profiling.runners.attention.gdn_recurrent_decode_torch")
    assert spec.runner_ref.function_name == "profile_gdn_recurrent_decode"

    assert spec.supports.compute == frozenset({DType.BF16})
    assert spec.supports.kv is None
    assert spec.supports.gpus is None
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA H200")
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA B200")
    assert not spec.supports.allows(DType.FP16, gpu="NVIDIA H200")
    assert not spec.supports.allows(DType.FP32, gpu="NVIDIA H200")


def test_vllm_triton_registration_reuses_schema_table_and_is_h200_only() -> None:
    spec = find_kernel_profiler_spec(KIND, "vllm_triton")

    assert spec.kernel_kind == spec.table_name == KIND
    assert spec.backend == "vllm_triton"
    assert spec.args_schema is GdnRecurrentDecodeArgs
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.batch_outlier_policy == BatchOutlierPolicy()
    assert spec.subprocess_env == "vllm_env"
    assert spec.runner_ref.module_name == (
        "profiling.runners.attention.gdn_recurrent_decode_vllm_triton"
    )
    assert spec.runner_ref.function_name == "profile_gdn_recurrent_decode_vllm_triton"

    assert spec.supports.compute == frozenset({DType.BF16})
    assert spec.supports.kv is None
    assert spec.supports.gpus == frozenset({"NVIDIA H200"})
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA H200")
    assert not spec.supports.allows(DType.BF16, gpu="NVIDIA H100")
    assert not spec.supports.allows(DType.BF16, gpu="NVIDIA B200")
    assert not spec.supports.allows(DType.FP16, gpu="NVIDIA H200")
    assert not spec.supports.allows(DType.FP32, gpu="NVIDIA H200")


def test_registry_barrel_and_runner_ref_are_lazy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import profiling.kernels; "
                "print('torch' in sys.modules); "
                "print('profiling.runners.attention.gdn_recurrent_decode_torch' "
                "in sys.modules); "
                "print('profiling.runners.attention.gdn_recurrent_decode_vllm_triton' "
                "in sys.modules); "
                "print('profiling.runners.attention.gdn_recurrent_decode_reference' "
                "in sys.modules); print('vllm' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == ["False", "False", "False", "False", "False"]

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from profiling.db.registry import "
                "find_kernel_profiler_spec; runner = find_kernel_profiler_spec("
                "'gdn_recurrent_decode', 'torch').runner_ref.load(); "
                "print(runner.__module__); print(runner.__name__); "
                "print('torch' in sys.modules); "
                "print('profiling.runners.attention.gdn_recurrent_decode_reference' "
                "in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.attention.gdn_recurrent_decode_torch",
        "profile_gdn_recurrent_decode",
        "False",
        "False",
    ]

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from profiling.db.registry import "
                "find_kernel_profiler_spec; runner = find_kernel_profiler_spec("
                "'gdn_recurrent_decode', 'vllm_triton').runner_ref.load(); "
                "print(runner.__module__); print(runner.__name__); "
                "print('torch' in sys.modules); print('vllm' in sys.modules); "
                "print('profiling.runners.attention.gdn_recurrent_decode_reference' "
                "in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.splitlines() == [
        "profiling.runners.attention.gdn_recurrent_decode_vllm_triton",
        "profile_gdn_recurrent_decode_vllm_triton",
        "False",
        "False",
        "False",
    ]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("batch_size", 0),
        ("num_qk_heads", 0),
        ("num_value_heads", 0),
        ("key_head_dim", 0),
        ("value_head_dim", 0),
    ],
)
def test_runner_rejects_nonpositive_dimensions_before_cuda(
    name: str,
    value: int,
) -> None:
    from profiling.runners.attention.gdn_recurrent_decode_torch import (
        _validate_args,
    )

    spec = _SPEC | {name: value}
    with pytest.raises(ValueError, match="must be > 0"):
        _validate_args(**spec)


def test_runner_rejects_incompatible_heads_before_cuda() -> None:
    from profiling.runners.attention.gdn_recurrent_decode_torch import (
        _validate_args,
    )

    with pytest.raises(ValueError, match="must be divisible"):
        _validate_args(**(_SPEC | {"num_qk_heads": 3}))


@pytest.mark.parametrize(
    ("dtype", "state_dtype"),
    [
        ("fp16", "fp32"),
        ("fp32", "fp32"),
        ("bf16", "bf16"),
        ("bf16", "fp16"),
    ],
)
def test_runner_rejects_unsupported_dtypes_before_cuda(
    dtype: str,
    state_dtype: str,
) -> None:
    from profiling.runners.attention.gdn_recurrent_decode_torch import (
        _validate_args,
    )

    with pytest.raises(ValueError, match="requires dtype=bf16 and state_dtype=fp32"):
        _validate_args(**(_SPEC | {"dtype": dtype, "state_dtype": state_dtype}))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("batch_size", 0),
        ("num_qk_heads", 0),
        ("num_value_heads", 0),
        ("key_head_dim", 0),
        ("value_head_dim", 0),
    ],
)
def test_vllm_runner_rejects_nonpositive_dimensions_before_cuda(
    name: str,
    value: int,
) -> None:
    from profiling.runners.attention.gdn_recurrent_decode_vllm_triton import (
        _validate_args,
    )

    with pytest.raises(ValueError, match="must be > 0"):
        _validate_args(**(_SPEC | {name: value}))


def test_vllm_runner_rejects_heads_and_dtypes_before_cuda() -> None:
    from profiling.runners.attention.gdn_recurrent_decode_vllm_triton import (
        _validate_args,
    )

    with pytest.raises(ValueError, match="must be divisible"):
        _validate_args(**(_SPEC | {"num_qk_heads": 3}))
    with pytest.raises(ValueError, match="requires dtype=bf16 and state_dtype=fp32"):
        _validate_args(**(_SPEC | {"dtype": "fp16"}))
    with pytest.raises(ValueError, match="requires dtype=bf16 and state_dtype=fp32"):
        _validate_args(**(_SPEC | {"state_dtype": "bf16"}))


def test_vllm_runner_derived_layout_and_slot_mapping() -> None:
    from profiling.runners.attention.gdn_recurrent_decode_vllm_triton import (
        _operand_shapes,
        _valid_slot_indices,
        _validate_args,
    )

    args = _validate_args(**(_SPEC | {"batch_size": 3}))
    shapes = _operand_shapes(args)
    assert shapes.query == (3, 16, 128)
    assert shapes.value == (3, 32, 128)
    assert shapes.gates == (3, 32)
    assert shapes.parameters == (32,)
    assert shapes.mixed_qkv == (3, 8192)
    assert shapes.state_storage == (4, 32, 128, 128)
    assert shapes.output == (3, 1, 32, 128)
    assert _valid_slot_indices(3) == (1, 2, 3)


def test_vllm_correctness_guard_matches_reference_and_restores_cpu_state() -> None:
    import torch

    from profiling.runners.attention.gdn_recurrent_decode_reference import (
        gdn_recurrent_decode_reference,
    )
    from profiling.runners.attention.gdn_recurrent_decode_vllm_triton import (
        _build_operands,
        _check_correctness,
        _validate_args,
    )

    args = _validate_args(
        batch_size=2,
        num_qk_heads=2,
        num_value_heads=4,
        key_head_dim=4,
        value_head_dim=3,
        dtype="bf16",
        state_dtype="fp32",
    )
    operands = _build_operands(torch, args, device=torch.device("cpu"))
    state_before = operands.initial_state.clone()

    def fake_fused(**kwargs):
        assert kwargs["scale"] == args.key_head_dim**-0.5
        assert kwargs["use_qk_l2norm_in_kernel"] is True
        assert kwargs["mixed_qkv"].is_contiguous()
        assert tuple(kwargs["initial_state"].shape) == (3, 4, 3, 4)
        assert kwargs["ssm_state_indices"].tolist() == [1, 2]
        mixed = kwargs["mixed_qkv"]
        q_width = args.num_qk_heads * args.key_head_dim
        v_width = args.num_value_heads * args.value_head_dim
        query = mixed[:, :q_width].reshape(args.batch_size, args.num_qk_heads, args.key_head_dim)
        key = mixed[:, q_width : 2 * q_width].reshape_as(query)
        value = mixed[:, 2 * q_width : 2 * q_width + v_width].reshape(
            args.batch_size, args.num_value_heads, args.value_head_dim
        )
        slots = kwargs["ssm_state_indices"].long()
        semantic_state = kwargs["initial_state"].index_select(0, slots)
        semantic_state = semantic_state.transpose(-1, -2).contiguous()
        output, updated_state = gdn_recurrent_decode_reference(
            query,
            key,
            value,
            kwargs["a"],
            kwargs["b"],
            kwargs["A_log"],
            kwargs["dt_bias"],
            semantic_state,
        )
        kwargs["out"].copy_(output[:, None])
        kwargs["initial_state"].index_copy_(0, slots, updated_state.transpose(-1, -2))
        return kwargs["out"], kwargs["initial_state"]

    _check_correctness(
        torch,
        fake_fused,
        operands,
        args,
        synchronize=lambda: None,
    )
    assert torch.equal(operands.initial_state, state_before)
    assert torch.count_nonzero(operands.out) == 0


def test_runner_reports_missing_cuda_as_typed_unsupported() -> None:
    from profiling.runners.attention.gdn_recurrent_decode_torch import (
        _validate_cuda_device,
    )

    no_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(ProfilerNotImplemented, match="CUDA is required"):
        _validate_cuda_device(no_cuda)


def test_semantic_metrics_are_explicit_logical_counts() -> None:
    from profiling.runners.attention.gdn_recurrent_decode_torch import (
        _logical_bytes,
        _semantic_flops,
    )

    assert (
        _semantic_flops(
            batch_size=1,
            num_qk_heads=1,
            num_value_heads=2,
            key_head_dim=3,
            value_head_dim=4,
        )
        == 204
    )
    assert (
        _logical_bytes(
            batch_size=1,
            num_qk_heads=1,
            num_value_heads=2,
            key_head_dim=3,
            value_head_dim=4,
            dtype=DType.BF16,
            state_dtype=DType.FP32,
        )
        == 260
    )


def test_generated_facades_and_read_only_missing_query(tmp_path, monkeypatch) -> None:
    assert hasattr(perf_api, "get_gdn_recurrent_decode_times")
    assert hasattr(perf_api, "count_missing_gdn_recurrent_decode")
    monkeypatch.setattr(perf_api, "DB_PATH", tmp_path / "profile.db")

    assert (
        perf_api.count_missing_gdn_recurrent_decode(
            [_SPEC], backend="torch", gpu_name="NVIDIA H200"
        )
        == 1
    )
    result = perf_api.get_gdn_recurrent_decode_times(
        [_SPEC], backend="torch", gpu_name="NVIDIA H200"
    )[0]
    assert isinstance(result, MissingEntry)
    assert result.args == coerce_args(GdnRecurrentDecodeArgs, _SPEC)
    assert not perf_api.DB_PATH.exists()

    assert (
        perf_api.count_missing_gdn_recurrent_decode(
            [_SPEC], backend="vllm_triton", gpu_name="NVIDIA H200"
        )
        == 1
    )
    vllm_result = perf_api.get_gdn_recurrent_decode_times(
        [_SPEC], backend="vllm_triton", gpu_name="NVIDIA H200"
    )[0]
    assert isinstance(vllm_result, MissingEntry)
    assert vllm_result.args == coerce_args(GdnRecurrentDecodeArgs, _SPEC)
    assert not perf_api.DB_PATH.exists()

    table = Table(find_kernel_profiler_spec(KIND, "torch"), perf_api.DB_PATH)
    assert table.args_columns == [
        "batch_size",
        "num_qk_heads",
        "num_value_heads",
        "key_head_dim",
        "value_head_dim",
        "dtype",
        "state_dtype",
    ]
