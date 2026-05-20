"""FlashInfer decode attention runner — batched single-token decode (paged KV).

Decode processes one query token per request, attending to ``avg_len`` cached
tokens, across ``batch_size`` requests. It uses a different wrapper than
prefill/rect (``BatchDecodeWithPagedKVCacheWrapper``, paged), so it lives in its
own file with its own caller; only the op-agnostic mechanics (paged-KV
construction, dtype/backend helpers, do_bench + metrics) are shared via
``_common``.

Cache identity is ``(batch_size, total_tokens)`` — batch is a real axis, and
``total_tokens`` is the total kv across the batch (so a flat cap keeps every
grid corner feasible). The runner derives the per-request mean length
``avg_len = max(1, total_tokens // batch_size)`` (approximating a heterogeneous
batch by its mean kv length; error quantified separately), then sets
``q_len = 1`` per request, ``kv_len = avg_len``, non-causal.

Each ``(kind, backend)`` registers its own thin entry function because the worker
strips ``backend`` before calling the runner. The heavy libs are imported lazily;
this module is only loaded inside the profiling worker subprocess via
``RunnerRef``.
"""

from __future__ import annotations

from profiling.db.args import DType
from profiling.runners.attention import _common
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_PAGE_SIZE = 16


def _run_cudnn_decode(
    *,
    batch_size: int,
    seq_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: DType | str,
    kv_dtype: DType | str,
    o_dtype: DType | str,
) -> ComputeMetrics:
    """cuDNN SDPA decode path (torch): q is one token per request, non-causal."""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if any(_common.is_fp8(d) for d in (q_dtype, kv_dtype, o_dtype)):
        raise ProfilerNotImplemented("cudnn backend does not support fp8")

    torch_dtype = _common.to_torch_dtype(q_dtype)
    q = torch.randn(batch_size, num_qo_heads, 1, head_dim, dtype=torch_dtype, device="cuda")
    k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=torch_dtype, device="cuda")
    v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=torch_dtype, device="cuda")
    enable_gqa = num_qo_heads != num_kv_heads

    def benchmark_fn():
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=False, enable_gqa=enable_gqa
            )

    o_elem = torch.tensor([], dtype=_common.to_torch_dtype(o_dtype)).element_size()
    bytes_accessed = int(
        q.numel() * q.element_size()
        + k.numel() * k.element_size()
        + v.numel() * v.element_size()
        + batch_size * num_qo_heads * head_dim * o_elem
    )
    flops = _common.attention_flops(
        q_len=1, kv_len=seq_len, num_qo_heads=num_qo_heads,
        head_dim=head_dim, causal=False, batch_size=batch_size,
    )
    return _common.measure(benchmark_fn, flops=flops, bytes_accessed=bytes_accessed)


def _run_decode(
    *,
    backend: str,
    batch_size: int,
    total_tokens: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: DType | str,
    kv_dtype: DType | str,
    o_dtype: DType | str,
) -> ComputeMetrics:
    """Batched paged decode.

    Cache dims -> kernel params (the ONE place this mapping lives): the
    per-request (mean) kv length is ``avg_len = max(1, total_tokens //
    batch_size)``; each request emits ``q_len = 1`` token.
    """
    batch_size = int(batch_size)
    seq_len = max(1, int(total_tokens) // batch_size)

    if backend == "trt":
        # Same B200/sm100-only limitation as prefill: trtllm-gen FMHA errors with
        # "Unsupported architecture" on H200. Ref trt decode rows are all B200.
        raise ProfilerNotImplemented(
            "trt (trtllm-gen) decode is B200/sm100-only; unsupported on this GPU"
        )

    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented("torch is required for attention runners") from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for attention runners")

    if backend == "cudnn":
        return _run_cudnn_decode(
            batch_size=batch_size, seq_len=seq_len, num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads, head_dim=head_dim,
            q_dtype=q_dtype, kv_dtype=kv_dtype, o_dtype=o_dtype,
        )

    try:
        import flashinfer
    except ImportError as exc:
        raise ProfilerNotImplemented("flashinfer is required for attention runners") from exc

    try:
        inp = _common.build_paged_decode_inputs(
            batch_size=batch_size, seq_len=seq_len, num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads, head_dim=head_dim,
            q_dtype=q_dtype, kv_dtype=kv_dtype, o_dtype=o_dtype, page_size=_PAGE_SIZE,
        )
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            _common.make_workspace(),
            kv_layout="NHD",
            backend=_common.flashinfer_backend_name(backend),
            use_tensor_cores=True,
        )
        plan_kwargs = dict(
            indptr=inp.kv_indptr,
            indices=inp.kv_indices,
            last_page_len=inp.kv_last_page_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=_PAGE_SIZE,
            q_data_type=_common.to_torch_dtype(q_dtype),
        )
        if inp.scales is not None:  # fp8 KV
            plan_kwargs["kv_data_type"] = torch.float8_e4m3fn
        if _common.to_torch_dtype(q_dtype) != _common.to_torch_dtype(o_dtype):
            plan_kwargs["o_data_type"] = _common.to_torch_dtype(o_dtype)
        wrapper.plan(**plan_kwargs)

        paged_kv_cache = (inp.k_cache, inp.v_cache)
        if inp.scales is not None:
            s_k, s_v = inp.scales

            def benchmark_fn():
                return wrapper.run(inp.q, paged_kv_cache, k_scale=s_k, v_scale=s_v)
        else:

            def benchmark_fn():
                return wrapper.run(inp.q, paged_kv_cache)

        flops = _common.attention_flops(
            q_len=1, kv_len=seq_len, num_qo_heads=num_qo_heads,
            head_dim=head_dim, causal=False, batch_size=batch_size,
        )
        return _common.measure(benchmark_fn, flops=flops, bytes_accessed=inp.bytes_accessed)
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc


# --- flashinfer_attn_decode entry points, one per backend --------------------


def profile_flashinfer_attn_decode_fa2(**kwargs) -> ComputeMetrics:
    return _run_decode(backend="fa2", **kwargs)


def profile_flashinfer_attn_decode_fa3(**kwargs) -> ComputeMetrics:
    return _run_decode(backend="fa3", **kwargs)


def profile_flashinfer_attn_decode_trt(**kwargs) -> ComputeMetrics:
    return _run_decode(backend="trt", **kwargs)


def profile_flashinfer_attn_decode_cudnn(**kwargs) -> ComputeMetrics:
    return _run_decode(backend="cudnn", **kwargs)
