"""Kimi-K3 chunked-prefill profiling callables.

Each public function below follows the production SGLang boundary used by the
single-layer driver.  Imports stay inside the callables because this module is
loaded only by the ``sglang_k3_env`` profiling worker.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention import _common
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_K3_HEADS = 12
_K3_HEAD_DIM = 128
_K3_QK_DIM = 192
_K3_V_DIM = 128
_K3_KV_RANK = 512
_K3_ROPE_DIM = 64
_K3_CONV_CHANNELS = 3 * _K3_HEADS * _K3_HEAD_DIM
_K3_CONV_WIDTH = 4
_LOWER_BOUND = -5.0


@dataclass(frozen=True)
class _ChunkShape:
    num_tokens: int
    max_sequence_length: int
    num_sequences: int
    prefix_len: int


def _chunk_shape(
    num_tokens: int,
    max_sequence_length: int,
    num_sequences: int,
    prefix_len: int,
) -> _ChunkShape:
    shape = _ChunkShape(
        num_tokens=int(num_tokens),
        max_sequence_length=int(max_sequence_length),
        num_sequences=int(num_sequences),
        prefix_len=int(prefix_len),
    )
    if min(shape.num_tokens, shape.max_sequence_length, shape.num_sequences) <= 0:
        raise ValueError("prefill token and sequence dimensions must be positive")
    if shape.num_tokens != shape.max_sequence_length * shape.num_sequences:
        raise ValueError(
            "the K3 profile runner uses the canonical uniform chunk shape: "
            "num_tokens must equal max_sequence_length * num_sequences"
        )
    if shape.prefix_len < 0:
        raise ValueError("prefix_len must be non-negative")
    return shape


def _require_b200(torch: Any, backend: str) -> Any:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"{backend} requires CUDA")
    name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if name != "NVIDIA B200":
        raise ProfilerNotImplemented(f"{backend} is verified only on NVIDIA B200, got {name}")
    return torch.device("cuda", torch.cuda.current_device())


def _measure(
    kernel: Callable[[], object],
    *,
    flops: int,
    bytes_accessed: float,
) -> ComputeMetrics:
    try:
        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=flops / elapsed_s / 1e12 if elapsed_s else 0.0,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9 if elapsed_s else 0.0,
        energy_j=float(energy_j),
    )


def profile_k3_attn_res_prefill_sglang_k3(
    num_tokens: int,
    hidden_size: int,
    num_valid_blocks: int,
    num_launches: int,
    write_prefix: bool,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile the exact K3 attention-residual TMA call used by prefill.

    The layer invokes this fused callable twice around the attention result.
    Keeping both launches in one runner preserves the production boundary while
    avoiding a simulator leaf for each temporary tensor operation.
    """
    dtype = DType.from_value(dtype)
    if (hidden_size, num_valid_blocks, dtype) != (7_168, 1, DType.BF16):
        raise ProfilerNotImplemented(
            "K3 attn_res_prefill requires hidden_size=7168, num_valid_blocks=1, dtype=bf16"
        )
    if int(num_tokens) <= 0 or int(num_launches) <= 0:
        raise ValueError("num_tokens and num_launches must be positive")
    try:
        import torch
        from sglang.kernels.ops.kimi_k3.attn_res import attn_res_fused_tma
    except ImportError as exc:
        raise ProfilerNotImplemented("SGLang K3 attention-residual kernel is required") from exc
    device = _require_b200(torch, "k3_attn_res_prefill")
    num_tokens = int(num_tokens)
    num_launches = int(num_launches)
    # AttnResidual allocates the full K3 bank (ceil(93 / 12) = 8 rows) for
    # every layer.  ``num_valid_blocks`` controls the aggregation loop; it
    # does not reduce the physical bank stride.
    bank_rows = 8
    prefix_a = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device=device)
    prefix_b = torch.randn_like(prefix_a)
    prefix_sum = torch.empty_like(prefix_a)
    bank = torch.randn(
        (num_tokens, max(bank_rows, 1), hidden_size), dtype=torch.bfloat16, device=device
    )
    cw = torch.randn((hidden_size,), dtype=torch.bfloat16, device=device)
    ow = torch.randn_like(cw)
    out = torch.empty_like(prefix_sum)

    def kernel() -> None:
        for launch_idx in range(num_launches):
            # Aggregation 1 receives the current hidden state directly.  The
            # MLP-side aggregation is the only one with a pending residual
            # add on the single-layer path.
            if launch_idx:
                torch.add(prefix_a, prefix_b, out=prefix_sum)
            else:
                prefix_sum.copy_(prefix_a)
            attn_res_fused_tma(
                prefix_sum,
                bank,
                cw,
                ow,
                out,
                int(num_valid_blocks),
                1e-5,
                write_prefix=bool(write_prefix),
            )

    kernel()
    torch.cuda.synchronize(device)
    bytes_per_launch = (
        prefix_a.numel()
        + prefix_b.numel()
        + prefix_sum.numel()
        + bank.numel()
        + cw.numel()
        + ow.numel()
        + out.numel()
    ) * prefix_sum.element_size()
    flops_per_launch = num_tokens * hidden_size * (2 * int(num_valid_blocks) + 17)
    return _measure(
        kernel,
        flops=num_launches * flops_per_launch,
        bytes_accessed=num_launches * bytes_per_launch,
    )


def _uniform_metadata(torch: Any, shape: _ChunkShape, *, device: Any) -> tuple[Any, Any, Any, Any]:
    query_start_loc = (
        torch.arange(shape.num_sequences + 1, dtype=torch.int32, device=device)
        * shape.max_sequence_length
    )
    cache_indices = torch.arange(1, shape.num_sequences + 1, dtype=torch.int32, device=device)
    has_initial_state = torch.full(
        (shape.num_sequences,), shape.prefix_len > 0, dtype=torch.bool, device=device
    )
    seq_lens_cpu = [shape.max_sequence_length] * shape.num_sequences
    return query_start_loc, cache_indices, has_initial_state, seq_lens_cpu


def profile_causal_conv1d_prefill_sglang_triton(
    num_tokens: int,
    max_sequence_length: int,
    num_sequences: int,
    prefix_len: int,
    channels: int,
    kernel_size: int,
    dtype: DType | str,
    state_dtype: DType | str,
) -> ComputeMetrics:
    shape = _chunk_shape(num_tokens, max_sequence_length, num_sequences, prefix_len)
    dtype = DType.from_value(dtype)
    state_dtype = DType.from_value(state_dtype)
    if (channels, kernel_size, dtype, state_dtype) != (
        _K3_CONV_CHANNELS,
        _K3_CONV_WIDTH,
        DType.BF16,
        DType.BF16,
    ):
        raise ProfilerNotImplemented(
            "K3 causal_conv1d_prefill requires channels=4608, kernel_size=4, "
            "dtype=bf16, state_dtype=bf16"
        )

    try:
        import torch
        from sglang.srt.layers.attention.mamba.causal_conv1d import causal_conv1d_fn
    except ImportError as exc:
        raise ProfilerNotImplemented("SGLang K3 causal convolution is required") from exc
    device = _require_b200(torch, "causal_conv1d_prefill")
    query_start_loc, cache_indices, has_initial_state, seq_lens_cpu = _uniform_metadata(
        torch, shape, device=device
    )
    x = torch.randn((shape.num_tokens, channels), dtype=torch.bfloat16, device=device).transpose(
        0, 1
    )
    weight = torch.randn((channels, kernel_size), dtype=torch.float32, device=device)
    bias = torch.randn((channels,), dtype=torch.float32, device=device)
    conv_states = torch.randn(
        (shape.num_sequences + 1, channels, kernel_size - 1),
        dtype=torch.bfloat16,
        device=device,
    )

    def kernel() -> Any:
        return causal_conv1d_fn(
            x,
            weight,
            bias,
            conv_states=conv_states,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=seq_lens_cpu,
            activation="silu",
        )

    kernel()
    torch.cuda.synchronize(device)
    bytes_accessed = (
        2 * shape.num_tokens * channels * 2
        + channels * kernel_size * 4
        + shape.num_tokens * channels * 2
        + shape.num_sequences * channels * (kernel_size - 1) * 2
    )
    flops = shape.num_tokens * channels * (2 * kernel_size + 5)
    return _measure(kernel, flops=flops, bytes_accessed=bytes_accessed)


def profile_kda_chunk_prefill_sglang_triton(
    num_tokens: int,
    max_sequence_length: int,
    num_sequences: int,
    prefix_len: int,
    num_heads: int,
    head_dim: int,
    dtype: DType | str,
    state_dtype: DType | str,
    lower_bound: float,
) -> ComputeMetrics:
    shape = _chunk_shape(num_tokens, max_sequence_length, num_sequences, prefix_len)
    dtype = DType.from_value(dtype)
    state_dtype = DType.from_value(state_dtype)
    if (num_heads, head_dim, dtype, state_dtype) != (
        _K3_HEADS,
        _K3_HEAD_DIM,
        DType.BF16,
        DType.BF16,
    ):
        raise ProfilerNotImplemented(
            "K3 kda_chunk_prefill requires heads=12, head_dim=128, dtype=bf16, state_dtype=bf16"
        )
    if float(lower_bound) != _LOWER_BOUND:
        raise ProfilerNotImplemented(f"K3 kda_chunk_prefill requires lower_bound={_LOWER_BOUND}")

    try:
        import torch
        from sglang.kernels.ops.attention.fla.kda import chunk_kda
    except ImportError as exc:
        raise ProfilerNotImplemented("SGLang K3 chunk_kda is required") from exc
    device = _require_b200(torch, "kda_chunk_prefill")
    query_start_loc, cache_indices, _, _ = _uniform_metadata(torch, shape, device=device)
    activation = torch.bfloat16
    q = torch.randn((1, shape.num_tokens, num_heads, head_dim), dtype=activation, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.randn_like(q)
    beta = torch.sigmoid(
        torch.randn((1, shape.num_tokens, num_heads), dtype=activation, device=device)
    )
    a_log = torch.zeros((num_heads,), dtype=torch.float32, device=device)
    dt_bias = torch.zeros((num_heads * head_dim,), dtype=torch.float32, device=device)
    initial_state = torch.randn(
        (shape.num_sequences + 1, num_heads, head_dim, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )

    def kernel() -> Any:
        return chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            initial_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=query_start_loc,
            A_log=a_log,
            dt_bias=dt_bias,
            lower_bound=float(lower_bound),
            beta_is_raw=False,
        )

    kernel()
    torch.cuda.synchronize(device)
    state_elements = shape.num_sequences * num_heads * head_dim * head_dim
    bytes_accessed = (
        4 * shape.num_tokens * num_heads * head_dim * 2
        + shape.num_tokens * num_heads * 2
        + state_elements * 2 * 2
    )
    flops = shape.num_tokens * num_heads * head_dim * head_dim * 8
    return _measure(kernel, flops=flops, bytes_accessed=bytes_accessed)


def _build_mla_inputs(
    torch: Any,
    *,
    batch_size: int,
    q_len: int,
    kv_len: int,
    num_heads: int,
    device: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    if min(batch_size, q_len, kv_len) <= 0:
        raise ValueError("MLA batch and lengths must be positive")
    q = torch.randn(
        (batch_size * q_len, num_heads, _K3_QK_DIM), dtype=torch.bfloat16, device=device
    ).to(torch.float8_e4m3fn)
    k = torch.randn(
        (batch_size * kv_len, num_heads, _K3_QK_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).to(torch.float8_e4m3fn)
    v = torch.randn(
        (batch_size * kv_len, num_heads, _K3_V_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).to(torch.float8_e4m3fn)
    cum_q = torch.arange(batch_size + 1, dtype=torch.int32, device=device) * q_len
    cum_kv = torch.arange(batch_size + 1, dtype=torch.int32, device=device) * kv_len
    seq_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)
    out = torch.empty(
        (batch_size * q_len, num_heads, _K3_V_DIM), dtype=torch.bfloat16, device=device
    )
    return q, k, v, cum_q, cum_kv, seq_lens, out


def profile_mla_prefill_attention_sglang_trtllm(
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    q_dtype: DType | str,
    kv_dtype: DType | str,
    o_dtype: DType | str,
    causal: bool,
    batch_size: int,
    q_len: int,
    kv_len: int,
) -> ComputeMetrics:
    q_dtype = DType.from_value(q_dtype)
    kv_dtype = DType.from_value(kv_dtype)
    o_dtype = DType.from_value(o_dtype)
    if (num_heads, qk_head_dim, v_head_dim, q_dtype, kv_dtype, o_dtype) != (
        _K3_HEADS,
        _K3_QK_DIM,
        _K3_V_DIM,
        DType.FP8_E4M3,
        DType.FP8_E4M3,
        DType.BF16,
    ):
        raise ProfilerNotImplemented(
            "K3 mla_prefill_attention requires heads=12, qk=192, v=128, q/kv=fp8_e4m3, output=bf16"
        )
    try:
        import flashinfer
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented("FlashInfer is required for K3 MLA prefill") from exc
    device = _require_b200(torch, "mla_prefill_attention")
    q, k, v, cum_q, cum_kv, seq_lens, out = _build_mla_inputs(
        torch,
        batch_size=int(batch_size),
        q_len=int(q_len),
        kv_len=int(kv_len),
        num_heads=int(num_heads),
        device=device,
    )
    workspace = _common.make_workspace()

    def kernel() -> Any:
        return flashinfer.prefill.trtllm_ragged_attention_deepseek(
            query=q,
            key=k,
            value=v,
            workspace_buffer=workspace,
            batch_size=int(batch_size),
            window_left=-1,
            enable_pdl=False,
            max_q_len=int(q_len),
            bmm1_scale=_K3_QK_DIM**-0.5,
            bmm2_scale=1.0,
            cum_seq_lens_q=cum_q,
            cum_seq_lens_kv=cum_kv,
            seq_lens=seq_lens,
            max_kv_len=int(kv_len),
            is_causal=bool(causal),
            # The chunked-prefix handler requests LSE for both its prefix and
            # causal passes; the causal leaf is shared with the no-prefix path.
            return_lse=True,
            o_sf_scale=-1.0 if not causal else 1.0,
            out=out,
            skip_softmax_threshold_scale_factor=0,
        )

    kernel()
    torch.cuda.synchronize(device)
    work = int(batch_size) * int(q_len) * int(kv_len)
    if causal:
        work = int(batch_size) * int(q_len) * (2 * int(kv_len) - int(q_len)) // 2
    flops = 2 * work * int(num_heads) * (int(qk_head_dim) + int(v_head_dim))
    bytes_accessed = (
        q.numel() * q.element_size()
        + k.numel() * k.element_size()
        + v.numel() * v.element_size()
        + out.numel() * out.element_size()
    )
    return _measure(kernel, flops=flops, bytes_accessed=bytes_accessed)


def profile_mla_prefix_gather_sglang_triton(
    batch_size: int,
    num_tokens: int,
    kv_lora_rank: int,
    rope_dim: int,
    dtype: DType | str,
    cache_dtype: DType | str,
) -> ComputeMetrics:
    dtype = DType.from_value(dtype)
    cache_dtype = DType.from_value(cache_dtype)
    if (kv_lora_rank, rope_dim, dtype, cache_dtype) != (
        _K3_KV_RANK,
        _K3_ROPE_DIM,
        DType.BF16,
        DType.FP8_E4M3,
    ):
        raise ProfilerNotImplemented(
            "K3 mla_prefix_gather requires kv_lora_rank=512, rope_dim=64, "
            "dtype=bf16, cache_dtype=fp8_e4m3"
        )
    if min(int(batch_size), int(num_tokens)) <= 0:
        raise ValueError("MLA prefix gather dimensions must be positive")
    try:
        import torch
        from sglang.kernels.ops.kvcache.kv_indices import (
            create_chunked_prefix_cache_kv_indices,
        )
        from sglang.kernels.ops.kvcache.mla_buffer import get_mla_kv_buffer_triton
    except ImportError as exc:
        raise ProfilerNotImplemented("SGLang K3 MLA cache gather is required") from exc
    device = _require_b200(torch, "mla_prefix_gather")
    batch_size = int(batch_size)
    num_tokens = int(num_tokens)
    table_width = max(64, num_tokens + 1)
    req_to_token = torch.arange(table_width, dtype=torch.int32, device=device).repeat(batch_size, 1)
    req_pool_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
    base = num_tokens // batch_size
    remainder = num_tokens % batch_size
    chunk_seq_lens = torch.full((batch_size,), base, dtype=torch.int32, device=device)
    if remainder:
        chunk_seq_lens[:remainder] += 1
    chunk_start_idx = torch.zeros((batch_size,), dtype=torch.int32, device=device)
    chunk_cu_seq_lens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    chunk_cu_seq_lens[1:] = torch.cumsum(chunk_seq_lens, dim=0)
    locations = torch.empty((num_tokens,), dtype=torch.int32, device=device)
    kv_buffer = torch.randn(
        (table_width, kv_lora_rank + rope_dim), dtype=torch.bfloat16, device=device
    ).to(torch.float8_e4m3fn)
    cache_k_nope = torch.empty((num_tokens, kv_lora_rank), dtype=torch.bfloat16, device=device)
    cache_k_rope = torch.empty((num_tokens, rope_dim), dtype=torch.bfloat16, device=device)

    def kernel() -> Any:
        create_chunked_prefix_cache_kv_indices[(batch_size,)](
            req_to_token,
            req_pool_indices,
            chunk_start_idx,
            chunk_seq_lens,
            chunk_cu_seq_lens,
            locations,
            req_to_token.shape[1],
        )
        return get_mla_kv_buffer_triton(kv_buffer, locations, cache_k_nope, cache_k_rope)

    kernel()
    torch.cuda.synchronize(device)
    bytes_accessed = num_tokens * (
        (kv_lora_rank + rope_dim) * cache_dtype.size_bytes()
        + (kv_lora_rank + rope_dim) * dtype.size_bytes()
    )
    return _measure(kernel, flops=0, bytes_accessed=bytes_accessed)


def profile_mla_merge_state_sglang_triton(
    num_tokens: int,
    num_heads: int,
    value_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    dtype = DType.from_value(dtype)
    if (num_heads, value_dim, dtype) != (_K3_HEADS, _K3_V_DIM, DType.BF16):
        raise ProfilerNotImplemented(
            "K3 mla_merge_state requires heads=12, value_dim=128, dtype=bf16"
        )
    try:
        import torch
        from sglang.srt.layers.attention.merge_state import merge_state
    except ImportError as exc:
        raise ProfilerNotImplemented("SGLang merge_state is required") from exc
    device = _require_b200(torch, "mla_merge_state")
    num_tokens = int(num_tokens)
    prefix_output = torch.randn(
        (num_tokens, num_heads, value_dim), dtype=torch.bfloat16, device=device
    )
    suffix_output = torch.randn_like(prefix_output)
    prefix_lse = torch.randn((num_tokens, num_heads), dtype=torch.float32, device=device)
    suffix_lse = torch.randn_like(prefix_lse)

    def kernel() -> Any:
        return merge_state(prefix_output, prefix_lse, suffix_output, suffix_lse)

    kernel()
    torch.cuda.synchronize(device)
    bytes_accessed = (
        2 * (prefix_output.numel() * 2 + prefix_lse.numel() * 4)
        + prefix_output.numel() * 2
        + prefix_lse.numel() * 4
    )
    flops = num_tokens * num_heads * 16 * value_dim
    return _measure(kernel, flops=flops, bytes_accessed=bytes_accessed)


__all__ = [
    "profile_causal_conv1d_prefill_sglang_triton",
    "profile_k3_attn_res_prefill_sglang_k3",
    "profile_kda_chunk_prefill_sglang_triton",
    "profile_mla_merge_state_sglang_triton",
    "profile_mla_prefill_attention_sglang_trtllm",
    "profile_mla_prefix_gather_sglang_triton",
]
