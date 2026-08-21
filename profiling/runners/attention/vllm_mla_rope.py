"""Exact vLLM MLA query-RoPE runner: the inductor fusion, not the rope slice.

The kernel this measures is the one the GLM-5.2 nsys capture labels
``attention.main_rope``:
``triton_poi_fused_add_copy_index_select_mul_slice_split_stack_sub_unsqueeze_view_*``.
Its generated source (recoverable from vLLM's ``torch_compile_cache``) iterates
``xnumel = num_heads * (qk_nope + rope) * num_tokens`` -- the FULL q -- loading
and storing every column and selecting ``tl.where(col >= qk_nope, roped, orig)``.

Reproducing that requires two things the obvious transcription gets wrong:

1. ``rotary_emb.forward_native``, not ``rotary_emb(...)``. RotaryEmbedding is a
   CustomOp; when enabled it dispatches to an opaque CUDA op that inductor
   cannot fuse. The traced kernel is a triton fusion, so the served run took
   the native path.
2. The block must RETURN a full-width q. vLLM writes ``q[..., nope:] = q_pe``
   and then feeds ``q`` to ``self.attn``, so functionalization materialises a
   new full q. Transcribing it as an in-place slice assignment whose result is
   unused makes inductor emit a rope-slice-sized kernel instead -- 4x less
   traffic, and it measured 136 us where the served kernel takes 600 us.

Assembling ``k`` is deliberately NOT included: the traced kernel takes three
inputs and one output and never writes k; that assembly is a separate kernel
outside this slot.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_SUPPORTED_GPUS = frozenset({"NVIDIA H100", "NVIDIA H200", "NVIDIA B200"})
_HOPPER_COMPUTE_CAPABILITY = (9, 0)
# Substring, not an exact name: inductor suffixes the fusion with a per-graph
# counter (`..._view_4`, `..._view_7`), so the digits are not stable.
_KERNEL_NAME = "triton_poi_fused"
# Only scales the cos/sin values, never the shape or the access pattern, so it
# stays out of the cache identity. GLM-5.2's value keeps the reference
# comparison against the served model exact.
_ROPE_THETA = 8_000_000
# Relative to the tensor's own magnitude, because the comparison is in BF16:
# one ulp is 2^-8 of the exponent's scale, so an absolute bound would flag
# ordinary rounding. The compiled fusion and the eager reference run the same
# math and differ only in accumulation order and the cat-vs-`tl.where` write.
_REFERENCE_RELATIVE_TOLERANCE = 4e-2


def _validate_args(
    num_tokens: int,
    num_heads: int,
    qk_nope_head_dim: int,
    rope_dim: int,
    max_position: int,
    is_neox_style: bool,
    input_dtype: DType | str,
) -> tuple[int, int, int, int, int, bool, DType]:
    for name, value in (
        ("num_tokens", num_tokens),
        ("num_heads", num_heads),
        ("qk_nope_head_dim", qk_nope_head_dim),
        ("rope_dim", rope_dim),
        ("max_position", max_position),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    rope_dim = int(rope_dim)
    if rope_dim % 2 != 0:
        raise ValueError(f"rope_dim must be even, got {rope_dim}")
    resolved_dtype = DType.from_value(input_dtype)
    if resolved_dtype is not DType.BF16:
        raise ProfilerNotImplemented(
            f"vllm_mla_rope supports bf16 activations only, got {resolved_dtype}"
        )
    return (
        int(num_tokens),
        int(num_heads),
        int(qk_nope_head_dim),
        rope_dim,
        int(max_position),
        bool(is_neox_style),
        resolved_dtype,
    )


def _validate_cuda_device(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("vllm_mla_rope requires a CUDA device")
    device_name = torch.cuda.get_device_name(0)
    if device_name not in _SUPPORTED_GPUS:
        raise ProfilerNotImplemented(
            f"vllm_mla_rope is validated on {sorted(_SUPPORTED_GPUS)}, got {device_name}"
        )
    if torch.cuda.get_device_capability(0) < _HOPPER_COMPUTE_CAPABILITY:
        raise ProfilerNotImplemented("vllm_mla_rope requires compute capability >= 9.0")


def _build_rotary_embedding(
    torch: Any,
    *,
    rope_dim: int,
    max_position: int,
    is_neox_style: bool,
    torch_dtype: Any,
) -> Any:
    try:
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.model_executor.layers.rotary_embedding import get_rope
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "the vLLM package is required for the vllm_mla_rope vllm_inductor backend"
        ) from exc

    # RotaryEmbedding is a CustomOp and reads the ambient compilation config
    # during construction; without a current config it raises on __init__.
    with set_current_vllm_config(VllmConfig()):
        rotary_emb = get_rope(
            rope_dim,
            max_position=max_position,
            is_neox_style=is_neox_style,
            rope_parameters={"rope_theta": _ROPE_THETA, "rope_type": "default"},
            dtype=torch_dtype,
        )
    return rotary_emb.to("cuda")


def _build_block(torch: Any, rotary_emb: Any, qk_nope_head_dim: int) -> Any:
    def block(positions: Any, query: Any, key_pe: Any) -> Any:
        query_nope = query[..., :qk_nope_head_dim]
        query_pe = query[..., qk_nope_head_dim:]
        query_pe, _ = rotary_emb.forward_native(positions, query_pe, key_pe)
        return torch.cat([query_nope, query_pe], dim=-1)

    return block


def _allocate_operands(
    torch: Any,
    *,
    num_tokens: int,
    num_heads: int,
    qk_head_dim: int,
    rope_dim: int,
    max_position: int,
    torch_dtype: Any,
) -> tuple[Any, Any, Any]:
    # Sequential positions, as a prefill chunk produces. Random positions would
    # scatter the cos/sin gather harder than the served workload ever does.
    span = min(num_tokens, max_position)
    positions = torch.arange(span, device="cuda", dtype=torch.int64)
    if num_tokens > span:
        positions = positions.repeat((num_tokens + span - 1) // span)[:num_tokens]
    query = torch.randn(num_tokens, num_heads, qk_head_dim, device="cuda", dtype=torch_dtype)
    key_pe = torch.randn(num_tokens, 1, rope_dim, device="cuda", dtype=torch_dtype)
    return positions, query, key_pe


def _validate_against_reference(
    torch: Any,
    compiled_block: Any,
    rotary_emb: Any,
    operands: tuple[Any, Any, Any],
    *,
    qk_nope_head_dim: int,
) -> None:
    """Compare the compiled fusion against the same math run eagerly."""
    positions, query, key_pe = operands
    actual = compiled_block(positions, query, key_pe)

    expected_pe, _ = rotary_emb.forward_native(
        positions, query[..., qk_nope_head_dim:].clone(), key_pe.clone()
    )
    expected = torch.cat(
        [query[..., :qk_nope_head_dim], expected_pe.reshape(query.shape[0], query.shape[1], -1)],
        dim=-1,
    )
    if actual.shape != expected.shape:
        raise KernelLaunchFailed(
            f"compiled MLA rope returned {tuple(actual.shape)}, expected {tuple(expected.shape)}"
        )
    difference = (actual.float() - expected.float()).abs().max().item()
    scale = max(expected.float().abs().max().item(), 1.0)
    if difference / scale > _REFERENCE_RELATIVE_TOLERANCE:
        raise KernelLaunchFailed(
            f"compiled MLA rope deviates from the eager reference by {difference:.4g} "
            f"({difference / scale:.3%} of the reference's peak magnitude {scale:.4g})"
        )


def _logical_bytes(
    *,
    num_tokens: int,
    num_heads: int,
    qk_head_dim: int,
    dtype_bytes: int,
) -> int:
    # The kernel reads every element of q and writes every element of the new q.
    # Re-reads of the rope columns and the cos/sin gather are excluded: they are
    # real traffic but land in cache, and counting them would overstate DRAM.
    row_bytes = num_heads * qk_head_dim * dtype_bytes
    return 2 * num_tokens * row_bytes


def profile_vllm_mla_rope_vllm_inductor(
    num_tokens: int,
    num_heads: int,
    qk_nope_head_dim: int,
    rope_dim: int,
    max_position: int,
    is_neox_style: bool,
    input_dtype: DType | str,
) -> ComputeMetrics:
    """Profile the inductor-fused MLA query RoPE exactly as vLLM emits it."""
    (
        num_tokens,
        num_heads,
        qk_nope_head_dim,
        rope_dim,
        max_position,
        is_neox_style,
        resolved_dtype,
    ) = _validate_args(
        num_tokens,
        num_heads,
        qk_nope_head_dim,
        rope_dim,
        max_position,
        is_neox_style,
        input_dtype,
    )

    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented("PyTorch is required for vllm_inductor") from exc

    _validate_cuda_device(torch)
    torch_dtype = torch.bfloat16
    qk_head_dim = qk_nope_head_dim + rope_dim

    rotary_emb = _build_rotary_embedding(
        torch,
        rope_dim=rope_dim,
        max_position=max_position,
        is_neox_style=is_neox_style,
        torch_dtype=torch_dtype,
    )
    compiled_block = torch.compile(
        _build_block(torch, rotary_emb, qk_nope_head_dim), dynamic=False
    )
    operands = _allocate_operands(
        torch,
        num_tokens=num_tokens,
        num_heads=num_heads,
        qk_head_dim=qk_head_dim,
        rope_dim=rope_dim,
        max_position=max_position,
        torch_dtype=torch_dtype,
    )

    # Compile and check semantics before timing so the first-call graph build is
    # not inside the measured window.
    _validate_against_reference(
        torch,
        compiled_block,
        rotary_emb,
        operands,
        qk_nope_head_dim=qk_nope_head_dim,
    )
    torch.cuda.synchronize()

    def kernel() -> None:
        try:
            compiled_block(*operands)
        except RuntimeError as exc:
            raise KernelLaunchFailed(f"compiled vLLM MLA rope failed: {exc}") from exc

    time_ms = Timer.cupti(kernel, kernel_name=_KERNEL_NAME)
    energy_j = Energy.perf(kernel, warmup=10, per_iter_time_ms=time_ms)
    logical_bytes = _logical_bytes(
        num_tokens=num_tokens,
        num_heads=num_heads,
        qk_head_dim=qk_head_dim,
        dtype_bytes=resolved_dtype.size_bytes(),
    )
    memory_bandwidth_gbps = (logical_bytes / (time_ms / 1000.0)) / 1e9
    return ComputeMetrics(
        time_ms=time_ms,
        tflops=0.0,
        memory_bandwidth_gbps=memory_bandwidth_gbps,
        energy_j=energy_j,
    )
