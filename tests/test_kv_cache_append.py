"""CPU-only contract tests for the ``kv_cache_append`` L1 kind."""

from __future__ import annotations

import subprocess
import sys

import pytest

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
        "head_dim_v",
    ]


def test_kind_and_backends_are_registered():
    assert KIND == "kv_cache_append"
    assert set(known_backends(KIND)) == {"torch", "vllm_cuda", "vllm_triton_diffkv"}
    torch_spec = find_kernel_profiler_spec(KIND, "torch")
    vllm_spec = find_kernel_profiler_spec(KIND, "vllm_cuda")
    diffkv_spec = find_kernel_profiler_spec(KIND, "vllm_triton_diffkv")
    assert torch_spec.args_schema is KvCacheAppendArgs
    assert vllm_spec.args_schema is KvCacheAppendArgs
    assert diffkv_spec.args_schema is KvCacheAppendArgs
    assert torch_spec.subprocess_env is None
    assert vllm_spec.subprocess_env == "vllm_env"
    assert diffkv_spec.subprocess_env == "vllm_env"
    assert vllm_spec.supports.allows(DType.BF16, DType.FP8_E4M3)
    assert not diffkv_spec.supports.allows(DType.BF16, DType.FP8_E4M3)


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


def _spec(**overrides):
    spec = {
        "num_kv_heads": 1,
        "head_dim": 192,
        "head_dim_v": 128,
        "block_size": 32,
        "input_dtype": "bf16",
        "kv_dtype": "bf16",
        "cache_layout": "NHD",
        "scale_granularity": "tensor",
        "num_tokens": 4,
    }
    return {**spec, **overrides}


def test_vllm_cuda_refuses_asymmetric_heads_before_touching_a_gpu():
    """reshape_and_cache_flash sizes V by K's head size; an asymmetric spec
    would otherwise time a write of the wrong shape."""
    from profiling.runners.attention.kv_cache_append import profile_kv_cache_append_vllm_cuda

    with pytest.raises(ValueError, match="vllm_triton_diffkv"):
        profile_kv_cache_append_vllm_cuda(**_spec())


def test_diffkv_refuses_per_head_scales():
    """The diffkv kernel loads one scalar k_scale / v_scale."""
    from profiling.runners.attention.kv_cache_append import (
        profile_kv_cache_append_vllm_triton_diffkv,
    )

    with pytest.raises(ValueError, match="tensor scale"):
        profile_kv_cache_append_vllm_triton_diffkv(**_spec(scale_granularity="head"))


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_packed_oracle_reads_k_then_v_from_each_slot(layout):
    """The oracle the diffkv runner checks before timing: a packed slot is
    K followed by V. A swapped half or a wrong page offset must fail it."""
    import torch

    from profiling.runners.attention.kv_cache_append import KernelLaunchFailed, check_packed_append

    blocks, page, heads, k_dim, v_dim = 4, 4, 2, 6, 3
    key = torch.randn(3, heads, k_dim)
    value = torch.randn(3, heads, v_dim)
    slots = torch.tensor([5, 0, 14])
    shape = (blocks, page, heads, k_dim + v_dim)
    if layout == "NHD":
        cache = torch.zeros(shape)
    else:
        cache = torch.zeros(blocks, heads, page, k_dim + v_dim).permute(0, 2, 1, 3)
    for token, slot in enumerate(slots.tolist()):
        cache[slot // page, slot % page] = torch.cat([key[token], value[token]], dim=-1)
    check_packed_append(
        torch, key=key, value=value, kv_cache=cache, slot_mapping=slots, block_size=page
    )

    cache[0, 0, :, :v_dim] = value[1]
    with pytest.raises(KernelLaunchFailed):
        check_packed_append(
            torch, key=key, value=value, kv_cache=cache, slot_mapping=slots, block_size=page
        )
