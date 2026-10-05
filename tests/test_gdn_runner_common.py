"""Cross-runner contracts shared by every vLLM GDN profiling backend."""

from __future__ import annotations

import importlib

import pytest
from fixtures.fake_torch import fake_cuda_torch

from profiling.runners.exceptions import ProfilerNotImplemented

_VLLM_GDN_RUNNER_MODULES = (
    "profiling.runners.attention.gdn_gated_rms_norm_vllm_triton",
    "profiling.runners.attention.gdn_prefill_post_conv_vllm_triton",
    "profiling.runners.attention.gdn_recurrent_decode_vllm_triton",
    "profiling.runners.attention.gdn_causal_conv_decode_vllm_triton",
    "profiling.runners.attention.gdn_causal_conv_prefill_vllm_triton",
)


@pytest.mark.parametrize("runner_module_name", _VLLM_GDN_RUNNER_MODULES)
def test_vllm_gdn_triton_runners_require_cuda_but_no_gpu_name(runner_module_name: str) -> None:
    # The kernels are portable Triton: an unmeasured GPU is a data gap, not a
    # reason to refuse the shape.
    runner_module = importlib.import_module(runner_module_name)

    with pytest.raises(ProfilerNotImplemented, match="CUDA is required"):
        runner_module._require_cuda(fake_cuda_torch(name="NVIDIA H200", available=False))
    for gpu_name in ("NVIDIA H200", "NVIDIA B200", "NVIDIA H100", "NVIDIA A100-SXM4-80GB"):
        runner_module._require_cuda(fake_cuda_torch(name=gpu_name))
