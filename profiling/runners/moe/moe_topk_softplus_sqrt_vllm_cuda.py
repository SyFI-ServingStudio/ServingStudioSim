"""Profile vLLM's single learned/hash sqrt-softplus routing launch."""

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "moe_topk_softplus_sqrt:vllm_cuda"
_GPU_NAME = "NVIDIA H200"
_NUM_EXPERTS = 256
_TOP_K = 6
_VOCAB_SIZE = 129280
_ROUTED_SCALE = 1.5


@dataclass(frozen=True)
class _Shape:
    selection_mode: str
    num_tokens: int
    hash_vocab_size: int


@dataclass(frozen=True)
class _Launch:
    callable: Any
    weights: Any
    expert_ids: Any
    token_expert_indices: Any
    logits: Any
    correction_bias: Any
    input_tokens: Any
    hash_table: Any

    def run(self) -> None:
        self.callable(
            self.weights,
            self.expert_ids,
            self.token_expert_indices,
            self.logits,
            True,
            _ROUTED_SCALE,
            self.correction_bias,
            self.input_tokens,
            self.hash_table,
        )


def _validate_args(
    selection_mode: str,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hash_vocab_size: int,
    logits_dtype: DType | str,
) -> _Shape:
    if selection_mode not in ("learned", "hash"):
        raise ValueError("selection_mode must be 'learned' or 'hash'")
    if type(num_tokens) is not int or num_tokens <= 0:
        raise ValueError("num_tokens must be a positive integer")
    if num_experts != _NUM_EXPERTS or top_k != _TOP_K:
        raise ProfilerNotImplemented(
            f"{_BACKEND} requires num_experts={_NUM_EXPERTS}, top_k={_TOP_K}"
        )
    expected_vocab_size = _VOCAB_SIZE if selection_mode == "hash" else 0
    if hash_vocab_size != expected_vocab_size:
        raise ProfilerNotImplemented(
            f"{_BACKEND} requires hash_vocab_size={expected_vocab_size} for {selection_mode}"
        )
    if DType.from_value(logits_dtype) is not DType.FP32:
        raise ProfilerNotImplemented(f"{_BACKEND} requires logits_dtype=fp32")
    return _Shape(selection_mode, num_tokens, hash_vocab_size)


def _prepare(torch: Any, callable_: Any, shape: _Shape) -> _Launch:
    generator = torch.Generator().manual_seed(23)
    logits = torch.randn(
        (shape.num_tokens, _NUM_EXPERTS),
        dtype=torch.float32,
        generator=generator,
    ).cuda()
    weights = torch.empty((shape.num_tokens, _TOP_K), dtype=torch.float32, device="cuda")
    expert_ids = torch.empty((shape.num_tokens, _TOP_K), dtype=torch.int32, device="cuda")
    token_indices = torch.empty_like(expert_ids)
    if shape.selection_mode == "learned":
        correction_bias = torch.linspace(-0.125, 0.125, _NUM_EXPERTS).cuda()
        input_tokens = hash_table = None
    else:
        correction_bias = None
        input_tokens = torch.arange(shape.num_tokens, dtype=torch.int32)
        input_tokens.remainder_(shape.hash_vocab_size)
        rows = torch.arange(shape.hash_vocab_size, dtype=torch.int32)[:, None]
        slots = torch.arange(_TOP_K, dtype=torch.int32)[None, :]
        hash_table = (rows * 17 + slots * 29).remainder(_NUM_EXPERTS).contiguous().cuda()
        input_tokens = input_tokens.cuda()
    return _Launch(
        callable_,
        weights,
        expert_ids,
        token_indices,
        logits,
        correction_bias,
        input_tokens,
        hash_table,
    )


def _check_output(torch: Any, launch: _Launch) -> None:
    logits = launch.logits.cpu()
    scores = torch.sqrt(torch.nn.functional.softplus(logits))
    if launch.hash_table is None:
        selected_ids = torch.topk(scores + launch.correction_bias.cpu(), _TOP_K, dim=1).indices
    else:
        selected_ids = launch.hash_table.cpu()[launch.input_tokens.cpu().long()].long()
    launch.run()
    torch.cuda.synchronize()
    actual_ids = launch.expert_ids.cpu().long()
    # CUDA and Torch may order experts differently when corrected scores tie.
    # The routed expert set is the semantic result; weights follow the returned
    # IDs, so recompute them in the production order below.
    if not torch.equal(actual_ids.sort(dim=1).values, selected_ids.sort(dim=1).values):
        raise AssertionError("expert IDs differ from the independent routing reference")
    selected_weights = scores.gather(1, actual_ids)
    expected_weights = selected_weights / selected_weights.sum(dim=1, keepdim=True)
    torch.testing.assert_close(
        launch.weights.cpu(), expected_weights * _ROUTED_SCALE, atol=0.002, rtol=0.01
    )


def _logical_bytes(shape: _Shape) -> int:
    logits_and_outputs = 4 * (
        shape.num_tokens * _NUM_EXPERTS + 2 * shape.num_tokens * _TOP_K
    )
    # Learned routing writes source-row indices. Hash routing instead reads one
    # table entry per selected expert and leaves that output unused.
    mode_vector = _NUM_EXPERTS if shape.selection_mode == "learned" else shape.num_tokens
    mode_io = 4 * (mode_vector + shape.num_tokens * _TOP_K)
    return logits_and_outputs + mode_io


def profile_moe_topk_softplus_sqrt_vllm_cuda(
    selection_mode: str,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hash_vocab_size: int,
    logits_dtype: DType | str,
) -> ComputeMetrics:
    shape = _validate_args(
        selection_mode, num_tokens, num_experts, top_k, hash_vocab_size, logits_dtype
    )
    try:
        import torch
        from vllm import _custom_ops
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the pinned vLLM environment") from exc

    try:
        if not torch.cuda.is_available():
            raise ProfilerNotImplemented(f"{_BACKEND} requires CUDA")
        gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
        if gpu_name != _GPU_NAME:
            raise ProfilerNotImplemented(
                f"{_BACKEND} is verified only on {_GPU_NAME}, got {gpu_name}"
            )
        launch = _prepare(torch, _custom_ops.topk_hash_softplus_sqrt, shape)
        _check_output(torch, launch)
        time_ms = Timer.cupti(launch.run, warmup=5, kernel_name="topkGatingSoftplusSqrt")
        energy_j = Energy.perf(launch.run, warmup=5, per_iter_time_ms=time_ms)
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
            _logical_bytes(shape) / elapsed_seconds / 1e9 if elapsed_seconds else 0.0
        ),
        energy_j=float(energy_j),
    )


__all__ = ["profile_moe_topk_softplus_sqrt_vllm_cuda"]
