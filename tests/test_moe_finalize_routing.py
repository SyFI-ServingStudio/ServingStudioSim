"""Focused tests for the ``moe_finalize_routing`` L1 contract."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import fields

import pytest

from profiling.db.args import DType
from profiling.db.batch import coerce_args
from profiling.db.registry import MetricFamily, find_kernel_profiler_spec
from profiling.kernels.moe_finalize_routing import KIND, MoeFinalizeRoutingArgs


def test_args_field_order_and_dtype_coercion():
    assert [field.name for field in fields(MoeFinalizeRoutingArgs)] == [
        "token_count",
        "hidden_size",
        "top_k",
        "num_experts_per_rank",
        "local_routed_token_count",
        "dtype",
    ]
    arguments = coerce_args(
        MoeFinalizeRoutingArgs,
        {
            "token_count": 64,
            "hidden_size": 4096,
            "top_k": 8,
            "num_experts_per_rank": 32,
            "local_routed_token_count": 128,
            "dtype": "bf16",
        },
    )
    assert arguments == MoeFinalizeRoutingArgs(64, 4096, 8, 32, 128, DType.BF16)


def test_registry_contract_and_backend_support():
    profiler_spec = find_kernel_profiler_spec(KIND, "flashinfer_trtllm")
    assert KIND == "moe_finalize_routing"
    assert profiler_spec.table_name == KIND
    assert profiler_spec.args_schema is MoeFinalizeRoutingArgs
    assert profiler_spec.metric_family is MetricFamily.COMPUTE
    assert profiler_spec.subprocess_env == "vllm_env"
    assert profiler_spec.runner_ref.module_name == (
        "profiling.runners.moe.flashinfer_trtllm_finalize"
    )
    assert profiler_spec.supports.allows(DType.BF16, gpu="NVIDIA H100")
    assert profiler_spec.supports.allows(DType.BF16, gpu="NVIDIA H200")
    assert not profiler_spec.supports.allows(DType.FP16, gpu="NVIDIA H200")
    assert not profiler_spec.supports.allows(DType.BF16, gpu="NVIDIA B200")


def test_import_is_lazy_for_runner_torch_and_flashinfer():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import profiling.kernels.moe_finalize_routing; "
                "print('profiling.runners.moe.flashinfer_trtllm_finalize' in sys.modules, "
                "'torch' in sys.modules, 'flashinfer' in sys.modules)"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "False False False"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"token_count": 0}, "token_count must be > 0"),
        ({"hidden_size": 10}, "divisible by 8"),
        ({"top_k": 0}, "top_k must be > 0"),
        ({"num_experts_per_rank": 0}, "num_experts_per_rank must be > 0"),
        ({"local_routed_token_count": 513}, "must be in"),
        ({"dtype": DType.FP16}, "requires dtype=bf16"),
    ],
)
def test_runner_validation(kwargs, message):
    from profiling.runners.moe import flashinfer_trtllm_finalize as runner

    arguments = {
        "token_count": 64,
        "hidden_size": 4096,
        "top_k": 8,
        "num_experts_per_rank": 32,
        "local_routed_token_count": 128,
        "dtype": DType.BF16,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        runner._validate_args(**arguments)


def test_balanced_routing_has_exact_local_count_and_valid_layout():
    from profiling.runners.moe import flashinfer_trtllm_finalize as runner

    selected, unpermute_map, scales = runner._routing_metadata(5, 3, 4, 7)
    assert len(selected) == len(unpermute_map) == len(scales) == 15
    assert sum(expert < 4 for expert in selected) == 7
    assert all(0 <= expert <= 4 for expert in selected)
    assert sorted(unpermute_map) == list(range(15))
    assert len(set(scales)) > 1


def test_generated_facade_symbols_exist():
    from profiling import perf_api

    assert hasattr(perf_api, "get_moe_finalize_routing_times")
    assert hasattr(perf_api, "count_missing_moe_finalize_routing")
