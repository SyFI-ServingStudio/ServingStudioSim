"""Profile DeepSeek V4's public fused inverse-RoPE FP8 transform."""

from dataclasses import dataclass
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "deepseek_v4_fused_inv_rope_fp8_quant:vllm_triton"
_GPU_NAME = "NVIDIA H200"
_MAX_TOKENS = 8192
_NUM_GROUPS = 8
_HEADS_PER_GROUP = 8
_HEAD_DIM = 512
_NOPE_DIM = 448
_ROPE_DIM = 64
_QUANT_GROUP_SIZE = 128


@dataclass(frozen=True)
class _Launch:
    callable: Any
    output: Any
    positions: Any
    cos_sin_cache: Any

    def run(self):
        return self.callable(
            self.output,
            self.positions,
            self.cos_sin_cache,
            n_groups=_NUM_GROUPS,
            heads_per_group=_HEADS_PER_GROUP,
            nope_dim=_NOPE_DIM,
            rope_dim=_ROPE_DIM,
            quant_group_size=_QUANT_GROUP_SIZE,
            tma_aligned_scales=False,
        )


def _validate_args(num_tokens: int) -> int:
    if type(num_tokens) is not int or not 1 <= num_tokens <= _MAX_TOKENS:
        raise ValueError(f"num_tokens must be an integer in [1, {_MAX_TOKENS}]")
    return num_tokens


def _require_h200(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"{_BACKEND} requires CUDA")
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name != _GPU_NAME:
        raise ProfilerNotImplemented(f"{_BACKEND} is verified only on {_GPU_NAME}, got {gpu_name}")


def _prepare(torch: Any, callable: Any, num_tokens: int) -> _Launch:
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(41)
    output = torch.randn(
        (num_tokens, _NUM_GROUPS * _HEADS_PER_GROUP, _HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    positions = torch.zeros(num_tokens, dtype=torch.int64, device=device)
    cos_sin_cache = torch.cat(
        (
            torch.ones((_ROPE_DIM // 2,), dtype=torch.float32, device=device),
            torch.zeros((_ROPE_DIM // 2,), dtype=torch.float32, device=device),
        )
    ).unsqueeze(0)
    return _Launch(callable, output, positions, cos_sin_cache)


def _check_output(torch: Any, launch: _Launch) -> None:
    actual_fp8, actual_scales = launch.run()
    torch.cuda.synchronize()
    grouped = launch.output.reshape(
        launch.output.shape[0], _NUM_GROUPS, _HEADS_PER_GROUP * _HEAD_DIM
    ).float()
    blocks = grouped.reshape(*grouped.shape[:-1], -1, _QUANT_GROUP_SIZE)
    expected_scales = torch.exp2(
        torch.ceil(torch.log2(torch.clamp(blocks.abs().amax(dim=-1) / 448.0, min=1e-10)))
    )
    torch.testing.assert_close(actual_scales, expected_scales, atol=0.0, rtol=0.0)
    expanded_scales = actual_scales.repeat_interleave(_QUANT_GROUP_SIZE, dim=-1)
    expected_fp8 = torch.clamp(grouped / expanded_scales, -448.0, 448.0).to(torch.float8_e4m3fn)
    if not torch.equal(actual_fp8.float(), expected_fp8.float()):
        raise AssertionError("fused inverse-RoPE FP8 output differs from Torch reference")


def _logical_bytes(num_tokens: int) -> int:
    input_bytes = 2 * num_tokens * _NUM_GROUPS * _HEADS_PER_GROUP * _HEAD_DIM
    position_bytes = 8 * num_tokens
    cos_sin_bytes = 4 * num_tokens * _ROPE_DIM
    output_bytes = num_tokens * _NUM_GROUPS * _HEADS_PER_GROUP * _HEAD_DIM
    scale_bytes = 4 * num_tokens * _NUM_GROUPS * (_HEADS_PER_GROUP * _HEAD_DIM // _QUANT_GROUP_SIZE)
    return input_bytes + position_bytes + cos_sin_bytes + output_bytes + scale_bytes


def profile_deepseek_v4_fused_inv_rope_fp8_quant_vllm_triton(
    num_tokens: int,
) -> ComputeMetrics:
    num_tokens = _validate_args(num_tokens)
    try:
        import torch
        from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
            fused_inv_rope_fp8_quant,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires pinned vLLM") from exc

    try:
        _require_h200(torch)
        launch = _prepare(torch, fused_inv_rope_fp8_quant, num_tokens)
        _check_output(torch, launch)
        time_ms = Timer.cupti(launch.run, warmup=3, kernel_name=None)
        energy_j = Energy.perf(launch.run, warmup=3, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed") from exc

    elapsed_seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(
            _logical_bytes(num_tokens) / elapsed_seconds / 1e9 if elapsed_seconds else 0.0
        ),
        energy_j=float(energy_j),
    )


__all__ = ["profile_deepseek_v4_fused_inv_rope_fp8_quant_vllm_triton"]
