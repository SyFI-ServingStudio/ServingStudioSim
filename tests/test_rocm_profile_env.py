"""CPU tests for the ROCm profiling environment (``profiling.exec.env``).

The ROCm env is registered parallel to the CUDA ``vllm_env`` so a ``(kernel,
backend)`` row can declare its MI300X dependency the same way. These assert the
registration and image-override behavior without needing docker or a GPU.
"""

from __future__ import annotations

import importlib

import profiling.exec.env as env_module
from profiling.exec.env import ContainerProfileEnv, resolve_profile_env


def test_rocm_env_registered_as_container_env():
    rocm_env = resolve_profile_env("vllm_rocm_env")
    assert isinstance(rocm_env, ContainerProfileEnv)
    assert rocm_env.name == "vllm_rocm_env"
    assert rocm_env.image  # non-empty image reference


def test_rocm_env_parallel_to_cuda_env():
    # Both the CUDA and ROCm envs are container envs; they must be distinct
    # images so a ROCm row never runs in the nvidia/cuda image.
    cuda_env = resolve_profile_env("vllm_env")
    rocm_env = resolve_profile_env("vllm_rocm_env")
    assert isinstance(cuda_env, ContainerProfileEnv)
    assert isinstance(rocm_env, ContainerProfileEnv)
    assert cuda_env.image != rocm_env.image


def test_rocm_image_respects_env_override(monkeypatch):
    monkeypatch.setenv("VIBESIM_VLLM_ROCM_PROFILE_IMAGE", "my-registry/vllm-rocm:test")
    reloaded = importlib.reload(env_module)
    try:
        assert reloaded.resolve_profile_env("vllm_rocm_env").image == "my-registry/vllm-rocm:test"
    finally:
        monkeypatch.delenv("VIBESIM_VLLM_ROCM_PROFILE_IMAGE", raising=False)
        importlib.reload(env_module)
