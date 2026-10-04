"""Device gate for the vLLM Triton FP8 fused-MoE runner."""

from types import SimpleNamespace

import pytest

from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.moe import vllm_fused_moe_triton as runner


def _fake_cuda(name: str, capability: tuple[int, int]) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_name=lambda _device: name,
            get_device_capability=lambda _device: capability,
        )
    )


@pytest.mark.parametrize(
    ("name", "capability"),
    [
        ("NVIDIA H200", (9, 0)),
        ("NVIDIA H100 80GB HBM3", (9, 0)),
        ("NVIDIA B200", (10, 0)),
        ("NVIDIA L40S", (8, 9)),
    ],
)
def test_any_fp8_capable_gpu_runs_the_triton_kernel(name: str, capability: tuple[int, int]) -> None:
    runner._require_fp8_device(_fake_cuda(name, capability))


def test_gpus_without_fp8_e4m3_are_rejected_by_capability() -> None:
    with pytest.raises(ProfilerNotImplemented, match="SM89"):
        runner._require_fp8_device(_fake_cuda("NVIDIA A100-SXM4-80GB", (8, 0)))
    no_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(ProfilerNotImplemented, match="CUDA is required"):
        runner._require_fp8_device(no_cuda)
