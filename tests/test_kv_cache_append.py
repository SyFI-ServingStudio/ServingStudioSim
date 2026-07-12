"""CPU-only contract tests for the ``kv_cache_append`` L1 kind."""

from __future__ import annotations

import subprocess
import sys

from profiling.db.args import DType, KvCacheAppendArgs
from profiling.db.registry import find_kernel_profiler_spec, known_backends
from profiling.kernels.kv_cache_append import KIND


def test_args_field_order_matches_runner_kwargs():
    assert list(KvCacheAppendArgs.__dataclass_fields__) == [
        "num_kv_heads",
        "head_dim",
        "block_size",
        "input_dtype",
        "kv_dtype",
        "cache_layout",
        "scale_granularity",
        "num_tokens",
    ]


def test_kind_and_backends_are_registered():
    assert KIND == "kv_cache_append"
    assert set(known_backends(KIND)) == {"torch", "vllm_cuda"}
    torch_spec = find_kernel_profiler_spec(KIND, "torch")
    vllm_spec = find_kernel_profiler_spec(KIND, "vllm_cuda")
    assert torch_spec.args_schema is KvCacheAppendArgs
    assert vllm_spec.args_schema is KvCacheAppendArgs
    assert torch_spec.subprocess_env is None
    assert vllm_spec.subprocess_env == "vllm_env"
    assert vllm_spec.supports.allows(DType.BF16, DType.FP8_E4M3)


def test_kernel_import_does_not_import_runner_or_vllm():
    command = [
        sys.executable,
        "-c",
        (
            "import sys; import profiling.kernels.kv_cache_append; "
            "print('profiling.runners.attention.kv_cache_append' in sys.modules, "
            "'vllm' in sys.modules)"
        ),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "False False"
