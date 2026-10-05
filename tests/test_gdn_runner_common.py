"""Cross-runner contracts shared by every vLLM GDN profiling backend."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from profiling.runners.exceptions import ProfilerNotImplemented

_VLLM_GDN_RUNNER_MODULES = (
    "profiling.runners.attention.gdn_gated_rms_norm_vllm_triton",
    "profiling.runners.attention.gdn_prefill_post_conv_vllm_triton",
    "profiling.runners.attention.gdn_recurrent_decode_vllm_triton",
    "profiling.runners.attention.gdn_causal_conv_decode_vllm_triton",
    "profiling.runners.attention.gdn_causal_conv_prefill_vllm_triton",
)


def _fake_torch(
    gpu_name: str,
    *,
    cuda_available: bool = True,
    capability: tuple[int, int] = (9, 0),
) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda_available,
            current_device=lambda: 0,
            get_device_name=lambda _device_index: gpu_name,
            get_device_capability=lambda _device_index=None: capability,
        )
    )


@pytest.mark.parametrize("runner_module_name", _VLLM_GDN_RUNNER_MODULES)
def test_vllm_gdn_triton_runners_require_cuda_but_no_gpu_name(runner_module_name: str) -> None:
    # The kernels are portable Triton: an unmeasured GPU is a data gap, not a
    # reason to refuse the shape.
    runner_module = importlib.import_module(runner_module_name)

    with pytest.raises(ProfilerNotImplemented, match="CUDA is required"):
        runner_module._require_cuda(_fake_torch("NVIDIA H200", cuda_available=False))
    for gpu_name in ("NVIDIA H200", "NVIDIA B200", "NVIDIA H100", "NVIDIA A100-SXM4-80GB"):
        runner_module._require_cuda(_fake_torch(gpu_name))
