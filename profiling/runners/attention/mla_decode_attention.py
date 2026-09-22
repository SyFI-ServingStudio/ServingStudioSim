"""Kimi-K3 absorbed MLA decode runners for SGLang's three backends."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.attention._common import WORKSPACE_BYTES, to_torch_dtype
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics


@dataclass(frozen=True)
class _Args:
    num_heads: int
    kv_lora_rank: int
    rope_dim: int
    q_dtype: DType
    kv_dtype: DType
    page_size: int
    batch_size: int
    kv_len: int


@dataclass(frozen=True)
class _Operands:
    query: Any
    cache: Any
    block_tables: Any
    seq_lens: Any
    workspace: Any


def _attention_scale(args: _Args) -> float:
    return (args.kv_lora_rank + args.rope_dim) ** -0.5


def _validate_args(
    num_heads: int,
    kv_lora_rank: int,
    rope_dim: int,
    q_dtype: DType | str,
    kv_dtype: DType | str,
    page_size: int,
    batch_size: int,
    kv_len: int,
    *,
    backend: str,
) -> _Args:
    args = _Args(
        int(num_heads),
        int(kv_lora_rank),
        int(rope_dim),
        DType.from_value(q_dtype),
        DType.from_value(kv_dtype),
        int(page_size),
        int(batch_size),
        int(kv_len),
    )
    if (
        min(
            args.num_heads,
            args.kv_lora_rank,
            args.rope_dim,
            args.page_size,
            args.batch_size,
            args.kv_len,
        )
        <= 0
    ):
        raise ValueError("MLA dimensions, page_size, batch_size, and kv_len must be positive")
    allowed = {DType.BF16, DType.FP8_E4M3}
    if args.q_dtype not in allowed or args.kv_dtype not in allowed:
        raise ValueError("SGLang MLA decode supports bf16 and fp8_e4m3")
    if backend == "sglang_trtllm_mla" and 64 < args.num_heads < 128:
        raise ValueError(
            "sglang_trtllm_mla rejects trtllm-gen's 64 < num_heads < 128 "
            f"range; got {args.num_heads}"
        )
    if backend == "sglang_cutedsl_mla" and args.page_size != 64:
        raise ValueError("SGLang cute-dsl MLA decode requires page_size=64")
    return args


def _build_operands(torch: Any, args: _Args) -> _Operands:
    device = torch.device("cuda")
    query_dtype = torch.float16 if args.q_dtype is DType.FP8_E4M3 else to_torch_dtype(args.q_dtype)
    cache_dtype = (
        torch.float16 if args.kv_dtype is DType.FP8_E4M3 else to_torch_dtype(args.kv_dtype)
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    pages_per_request = math.ceil(args.kv_len / args.page_size)
    total_pages = args.batch_size * pages_per_request
    latent_dim = args.kv_lora_rank + args.rope_dim
    query = torch.randn(
        (args.batch_size, 1, args.num_heads, latent_dim),
        dtype=query_dtype,
        device=device,
        generator=generator,
    )
    cache = torch.randn(
        (total_pages, 1, args.page_size, latent_dim),
        dtype=cache_dtype,
        device=device,
        generator=generator,
    )
    # SGLang quantizes the model-native query to FP8 whenever the MLA KV pool
    # is FP8, so the decode callable sees the same storage dtype as the cache.
    if args.q_dtype is DType.FP8_E4M3 or args.kv_dtype is DType.FP8_E4M3:
        query = query.to(torch.float8_e4m3fn)
    if args.kv_dtype is DType.FP8_E4M3:
        cache = cache.to(torch.float8_e4m3fn)
    block_tables = torch.arange(total_pages, dtype=torch.int32, device=device).view(
        args.batch_size, pages_per_request
    )
    seq_lens = torch.full((args.batch_size,), args.kv_len, dtype=torch.int32, device=device)
    return _Operands(
        query.contiguous(),
        cache.contiguous(),
        block_tables,
        seq_lens,
        torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device),
    )


def mla_decode_attention_reference(
    query: Any,
    cache: Any,
    block_tables: Any,
    seq_lens: Any,
    *,
    kv_lora_rank: int,
    rope_dim: int,
) -> Any:
    """CPU-testable dense semantic reference over the paged latent cache."""
    import torch

    if query.ndim == 4:
        query = query[:, 0]
    if query.ndim != 3 or cache.ndim != 4:
        raise ValueError("query must be [batch, heads, dim] and cache [pages, 1, page, dim]")
    batch, heads, dim = query.shape
    if dim != kv_lora_rank + rope_dim:
        raise ValueError("query latent dimension does not match MLA dimensions")
    outputs = []
    for row in range(batch):
        tokens = []
        length = int(seq_lens[row])
        for token in range(length):
            page = token // cache.shape[2]
            offset = token % cache.shape[2]
            page_id = int(block_tables[row, page])
            tokens.append(cache[page_id, 0, offset])
        keys_values = torch.stack(tokens, dim=0).float()
        scores = torch.matmul(query[row].float(), keys_values[:, : kv_lora_rank + rope_dim].T)
        scores = scores * ((kv_lora_rank + rope_dim) ** -0.5)
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.matmul(probs, keys_values[:, :kv_lora_rank]))
    return torch.stack(outputs, dim=0).to(query.dtype)


def _launch_trtllm(callable_: Any, operands: _Operands, args: _Args, *, cute: bool) -> Any:
    kwargs = {
        "query": operands.query,
        "kv_cache": operands.cache,
        "workspace_buffer": operands.workspace,
        "qk_nope_head_dim": args.kv_lora_rank,
        "kv_lora_rank": args.kv_lora_rank,
        "qk_rope_head_dim": args.rope_dim,
        "block_tables": operands.block_tables,
        "seq_lens": operands.seq_lens,
        "max_seq_len": args.kv_len,
        "bmm1_scale": _attention_scale(args),
        "bmm2_scale": 1.0,
    }
    if cute:
        kwargs["backend"] = "cute-dsl"
    else:
        # B200 is one of the architectures for which SGLang enables PDL.
        kwargs["enable_pdl"] = True
    return callable_(**kwargs)


def _run_flashinfer(args: _Args, *, cute: bool) -> ComputeMetrics:
    try:
        import torch
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
    except ImportError as exc:
        raise ProfilerNotImplemented("SGLang's FlashInfer MLA callable is required") from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for SGLang MLA profiling")
    operands = _build_operands(torch, args)

    def kernel() -> Any:
        return _launch_trtllm(trtllm_batch_decode_with_kv_cache_mla, operands, args, cute=cute)

    try:
        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
    seconds = time_ms / 1000.0
    flops = 2 * args.batch_size * args.num_heads * args.kv_len * (args.kv_lora_rank + args.rope_dim)
    bytes_accessed = (
        args.batch_size
        * args.num_heads
        * (args.kv_lora_rank + args.rope_dim)
        * args.q_dtype.size_bytes()
        + args.batch_size
        * args.kv_len
        * (args.kv_lora_rank + args.rope_dim)
        * args.kv_dtype.size_bytes()
        + args.batch_size * args.num_heads * args.kv_lora_rank * args.q_dtype.size_bytes()
    )
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=flops / seconds / 1e12 if seconds else 0.0,
        memory_bandwidth_gbps=bytes_accessed / seconds / 1e9 if seconds else 0.0,
        energy_j=float(energy_j),
    )


def _run_triton(args: _Args) -> ComputeMetrics:
    try:
        import torch
        from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd
    except ImportError as exc:
        raise ProfilerNotImplemented("SGLang Triton decode attention is required") from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for SGLang MLA profiling")
    operands = _build_operands(torch, args)
    pages_per_request = math.ceil(args.kv_len / args.page_size)
    token_count = args.batch_size * pages_per_request * args.page_size
    k_buffer = operands.cache.view(token_count, 1, args.kv_lora_rank + args.rope_dim)
    v_buffer = k_buffer[..., : args.kv_lora_rank]
    q = operands.query[:, 0].contiguous()
    kv_indptr = torch.arange(args.batch_size + 1, dtype=torch.int32, device="cuda") * args.kv_len
    indices = []
    for row in range(args.batch_size):
        base = row * pages_per_request * args.page_size
        indices.extend(base + token for token in range(args.kv_len))
    kv_indices = torch.tensor(indices, dtype=torch.int32, device="cuda")
    max_splits = max(1, math.ceil(args.kv_len / 1024))
    attn_logits = torch.empty(
        (args.batch_size, args.num_heads, max_splits, args.kv_lora_rank),
        dtype=torch.float32,
        device="cuda",
    )
    attn_lse = torch.empty(
        (args.batch_size, args.num_heads, max_splits), dtype=torch.float32, device="cuda"
    )
    num_kv_splits = torch.full((args.batch_size,), max_splits, dtype=torch.int32, device="cuda")
    output = torch.empty(
        (args.batch_size, args.num_heads, args.kv_lora_rank), dtype=torch.float32, device="cuda"
    )

    def kernel() -> Any:
        return decode_attention_fwd(
            q,
            k_buffer,
            v_buffer,
            output,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_splits,
            _attention_scale(args),
            1.0,
            1.0,
            has_mla=True,
            use_pdl=True,
            page_size=args.page_size,
        )

    try:
        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc
    seconds = time_ms / 1000.0
    flops = 2 * args.batch_size * args.num_heads * args.kv_len * (args.kv_lora_rank + args.rope_dim)
    bytes_accessed = k_buffer.numel() * k_buffer.element_size() + q.numel() * q.element_size()
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=flops / seconds / 1e12 if seconds else 0.0,
        memory_bandwidth_gbps=bytes_accessed / seconds / 1e9 if seconds else 0.0,
        energy_j=float(energy_j),
    )


def profile_mla_decode_attention_sglang_cutedsl(**kwargs: Any) -> ComputeMetrics:
    args = _validate_args(**kwargs, backend="sglang_cutedsl_mla")
    return _run_flashinfer(args, cute=True)


def profile_mla_decode_attention_sglang_trtllm(**kwargs: Any) -> ComputeMetrics:
    args = _validate_args(**kwargs, backend="sglang_trtllm_mla")
    return _run_flashinfer(args, cute=False)


def profile_mla_decode_attention_sglang_triton(**kwargs: Any) -> ComputeMetrics:
    args = _validate_args(**kwargs, backend="sglang_triton")
    return _run_triton(args)


__all__ = [
    "mla_decode_attention_reference",
    "profile_mla_decode_attention_sglang_cutedsl",
    "profile_mla_decode_attention_sglang_trtllm",
    "profile_mla_decode_attention_sglang_triton",
]
