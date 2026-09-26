"""Cross-runner contracts shared by every vLLM GDN profiling backend."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from profiling.runners.exceptions import ProfilerNotImplemented

_VLLM_GDN_RUNNER_MODULES = (
    "profiling.runners.attention.gdn_chunk_local_cumsum_vllm_triton",
    "profiling.runners.attention.gdn_chunk_output_vllm_triton",
    "profiling.runners.attention.gdn_chunk_recompute_w_u_vllm_triton",
    "profiling.runners.attention.gdn_chunk_scaled_dot_kkt_vllm_triton",
    "profiling.runners.attention.gdn_chunk_solve_tril_vllm_triton",
    "profiling.runners.attention.gdn_chunk_state_update_vllm_triton",
    "profiling.runners.attention.gdn_gated_rms_norm_vllm_triton",
    "profiling.runners.attention.gdn_prefill_post_conv_vllm_triton",
    "profiling.runners.attention.gdn_recurrent_decode_vllm_triton",
)


def _fake_torch(gpu_name: str, *, cuda_available: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda_available,
            current_device=lambda: 0,
            get_device_name=lambda _device_index: gpu_name,
        )
    )


@pytest.mark.parametrize("runner_module_name", _VLLM_GDN_RUNNER_MODULES)
def test_vllm_gdn_runners_enforce_the_shared_h200_gate(runner_module_name: str) -> None:
    runner_module = importlib.import_module(runner_module_name)

    with pytest.raises(ProfilerNotImplemented, match="CUDA is required"):
        runner_module._require_h200(_fake_torch("NVIDIA H200", cuda_available=False))
    with pytest.raises(ProfilerNotImplemented, match="verified only on NVIDIA H200"):
        runner_module._require_h200(_fake_torch("NVIDIA B200"))
    runner_module._require_h200(_fake_torch("NVIDIA H200"))


@pytest.mark.parametrize(
    "runner_module_name",
    (
        "profiling.runners.attention.gdn_causal_conv_decode_vllm_triton",
        "profiling.runners.attention.gdn_causal_conv_prefill_vllm_triton",
    ),
)
def test_vllm_causal_conv_runners_admit_h200_and_b200_only(runner_module_name: str) -> None:
    runner_module = importlib.import_module(runner_module_name)

    with pytest.raises(ProfilerNotImplemented, match="CUDA is required"):
        runner_module._require_supported_gpu(_fake_torch("NVIDIA H200", cuda_available=False))
    with pytest.raises(ProfilerNotImplemented, match="verified only on"):
        runner_module._require_supported_gpu(_fake_torch("NVIDIA H100"))
    runner_module._require_supported_gpu(_fake_torch("NVIDIA H200"))
    runner_module._require_supported_gpu(_fake_torch("NVIDIA B200"))
