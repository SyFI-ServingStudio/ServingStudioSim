"""FlashInfer ragged attention runner — prefill (causal) + rect (non-causal).

These two kinds share one ragged-prefill caller (``_run_ragged_attention``) and
differ only by ``causal``, so they live together here rather than duplicating the
caller across two files. Each ``(kind, backend)`` registers its own thin entry
function because the worker strips ``backend`` before calling the runner — it
routes to the runner via the registry, so the runner is selected by backend, not
told it (see ``profiling/exec/local_worker.py``).

Prefill is causal and *merges* the ref's pure-prefill + chunked ops (pure prefill
= ``prefix_len == 0``); rect is the non-causal sibling. The cache-dim ->
kernel-param mapping (``q_len = append_len``, ``kv_len = prefix_len + append_len``)
lives in ``_run_ragged_attention`` — the ONE place it is documented.

Shared op-agnostic mechanics (tensor/indptr/fp8 construction, dtype/backend
helpers, do_bench + metrics) live in ``_common``. The heavy libs (``flashinfer`` /
``torch``) are imported lazily inside the functions; this module is only loaded
inside the profiling worker subprocess via ``RunnerRef``.
"""

from __future__ import annotations

from profiling.db.args import DType
from profiling.runners.attention import _common
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics


def _run_cudnn_attention(
    *,
    q_len: int,
    kv_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: DType | str,
    kv_dtype: DType | str,
    o_dtype: DType | str,
    causal: bool,
) -> ComputeMetrics:
    """cuDNN SDPA path (torch), used for the ``cudnn`` backend."""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if any(_common.is_fp8(d) for d in (q_dtype, kv_dtype, o_dtype)):
        raise ProfilerNotImplemented("cudnn backend does not support fp8")
    if causal and q_len != kv_len:
        raise ProfilerNotImplemented(
            "cudnn causal attention requires q_len == kv_len (prefix_len == 0)"
        )

    torch_dtype = _common.to_torch_dtype(q_dtype)
    q = torch.randn(1, num_qo_heads, q_len, head_dim, dtype=torch_dtype, device="cuda")
    k = torch.randn(1, num_kv_heads, kv_len, head_dim, dtype=torch_dtype, device="cuda")
    v = torch.randn(1, num_kv_heads, kv_len, head_dim, dtype=torch_dtype, device="cuda")
    enable_gqa = num_qo_heads != num_kv_heads

    def benchmark_fn():
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=causal, enable_gqa=enable_gqa
            )

    o_elem = torch.tensor([], dtype=_common.to_torch_dtype(o_dtype)).element_size()
    bytes_accessed = int(
        q.numel() * q.element_size()
        + k.numel() * k.element_size()
        + v.numel() * v.element_size()
        + num_qo_heads * q_len * head_dim * o_elem
    )
    flops = _common.attention_flops(
        q_len=q_len, kv_len=kv_len, num_qo_heads=num_qo_heads,
        head_dim=head_dim, causal=causal,
    )
    return _common.measure(benchmark_fn, flops=flops, bytes_accessed=bytes_accessed)


def _run_ragged_attention(
    *,
    backend: str,
    causal: bool,
    prefix_len: int,
    append_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: DType | str,
    kv_dtype: DType | str,
    o_dtype: DType | str,
) -> ComputeMetrics:
    """Shared ragged-prefill caller for prefill (causal) and rect (non-causal).

    Cache dims -> kernel params (the ONE place this mapping lives):
    ``q_len = append_len``, ``kv_len = prefix_len + append_len``.
    """
    if backend == "trt":
        # trtllm-gen has no variable-length/ragged prefill kernel, and on sm90
        # (H200) even the paged path errors with "Unsupported architecture" —
        # trtllm-gen FMHA is B200/sm100-only in this build, and ref only ever
        # profiled trt as paged on B200. Until the paged path lands (B200), trt
        # is cleanly unsupported here rather than a launch failure.
        raise ProfilerNotImplemented(
            "trt (trtllm-gen) prefill/rect is unavailable on the ragged path: it "
            "has no variable-length kernel and is B200/sm100-only; the paged path "
            "on B200 is not yet implemented"
        )

    q_len = int(append_len)
    kv_len = int(prefix_len) + int(append_len)

    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented("torch is required for attention runners") from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA is required for attention runners")

    if backend == "cudnn":
        return _run_cudnn_attention(
            q_len=q_len, kv_len=kv_len, num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads, head_dim=head_dim,
            q_dtype=q_dtype, kv_dtype=kv_dtype, o_dtype=o_dtype, causal=causal,
        )

    try:
        import flashinfer
    except ImportError as exc:
        raise ProfilerNotImplemented("flashinfer is required for attention runners") from exc

    try:
        inp = _common.build_ragged_inputs(
            q_len=q_len, kv_len=kv_len, num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads, head_dim=head_dim,
            q_dtype=q_dtype, kv_dtype=kv_dtype, o_dtype=o_dtype,
        )
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            _common.make_workspace(),
            kv_layout="NHD",
            backend=_common.flashinfer_backend_name(backend),
        )
        plan_kwargs = dict(
            qo_indptr=inp.qo_indptr,
            kv_indptr=inp.kv_indptr,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            causal=causal,
            q_data_type=_common.to_torch_dtype(q_dtype),
        )
        if inp.scales is not None:  # fp8: q == kv == fp8, o non-fp8
            plan_kwargs["q_data_type"] = torch.float8_e4m3fn
            plan_kwargs["kv_data_type"] = torch.float8_e4m3fn
            plan_kwargs["o_data_type"] = _common.to_torch_dtype(o_dtype)
        elif _common.to_torch_dtype(q_dtype) != _common.to_torch_dtype(o_dtype):
            plan_kwargs["o_data_type"] = _common.to_torch_dtype(o_dtype)
        wrapper.plan(**plan_kwargs)

        if inp.scales is not None:
            s_q, s_k, s_v = inp.scales

            def benchmark_fn():
                return wrapper.run(inp.q, inp.k, inp.v, s_q, s_k, s_v)
        else:

            def benchmark_fn():
                return wrapper.run(inp.q, inp.k, inp.v)

        flops = _common.attention_flops(
            q_len=q_len, kv_len=kv_len, num_qo_heads=num_qo_heads,
            head_dim=head_dim, causal=causal,
        )
        return _common.measure(benchmark_fn, flops=flops, bytes_accessed=inp.bytes_accessed)
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc


# --- flashinfer_attn_prefill (causal) entry points, one per backend ----------


def profile_flashinfer_attn_prefill_fa2(**kwargs) -> ComputeMetrics:
    return _run_ragged_attention(backend="fa2", causal=True, **kwargs)


def profile_flashinfer_attn_prefill_fa3(**kwargs) -> ComputeMetrics:
    return _run_ragged_attention(backend="fa3", causal=True, **kwargs)


def profile_flashinfer_attn_prefill_trt(**kwargs) -> ComputeMetrics:
    return _run_ragged_attention(backend="trt", causal=True, **kwargs)


def profile_flashinfer_attn_prefill_cudnn(**kwargs) -> ComputeMetrics:
    return _run_ragged_attention(backend="cudnn", causal=True, **kwargs)


# --- flashinfer_attn_rect (non-causal) entry points, one per backend ---------


def profile_flashinfer_attn_rect_fa2(**kwargs) -> ComputeMetrics:
    return _run_ragged_attention(backend="fa2", causal=False, **kwargs)


def profile_flashinfer_attn_rect_fa3(**kwargs) -> ComputeMetrics:
    return _run_ragged_attention(backend="fa3", causal=False, **kwargs)


def profile_flashinfer_attn_rect_trt(**kwargs) -> ComputeMetrics:
    return _run_ragged_attention(backend="trt", causal=False, **kwargs)


def profile_flashinfer_attn_rect_cudnn(**kwargs) -> ComputeMetrics:
    return _run_ragged_attention(backend="cudnn", causal=False, **kwargs)
