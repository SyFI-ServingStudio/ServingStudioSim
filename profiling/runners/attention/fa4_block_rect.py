"""Profile non-causal block attention on vLLM's FA4 over an fp8 paged cache.

This is the call the DFlash2 draft layer makes in the vLLM fork
(``vllm/model_executor/models/qwen3_dflash2.py`` -> ``FlashAttentionImpl`` ->
``fa_utils.flash_attn_varlen_func(fa_version=4)``): ``num_requests`` requests,
each a block of ``q_len`` fp8 query rows attending, without a causal mask, the
``kv_len`` fp8 keys and values of its own paged context. vLLM's FULL CUDA graph
passes ``num_splits = max_num_splits = 32`` (the attention config default), so
FA4 runs its split-KV path: a forward kernel over 32 KV splits per request plus
a combine kernel. ``num_splits = 1`` runs the forward kernel alone; its output
is bit-identical where FA4 does not split (lane J notes §3, job 22198: 16+
requests), and differs by bf16 rounding at 1-8 requests, where the 32-split call splits.

Cache layout is the fork's FlashAttention layout: per layer one raw
``[pages, kv_heads, page_size, 2 * head_dim]`` byte tensor viewed as fp8 and
transposed to ``[pages, page_size, kv_heads, 2 * head_dim]``, K then V in the last
dimension. Each request's pages are shuffled over a pool sized to the batch;
page 0 is the null page. Every request has the same ``kv_len`` (the draft's
context is capped by its 2048-token window, so a long-context batch is uniform).
The draft is unquantized, so the fork passes no q/k/v descales (scale 1.0).

The whole call is the L1 unit; CUPTI takes the busy union of its launches.
Before timing, sampled output rows are checked against a float32 PyTorch
reference over the dequantized cache, and the split and single-pass calls are
compared (a difference is printed; it fails past the reference tolerance).
"""

from __future__ import annotations

from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

KIND = "flashinfer_attn_rect"
PAGE_SIZE = 16
SPLITS = {"vllm_fa4_split32": 32, "vllm_fa4_nosplit": 1}
_CHECK_REQUESTS = 8
_ABS_TOL = 0.06
# FA4 rounds P to fp8 before the PV product on an fp8 call: ~3.5% row L2 error
# against the float32 reference at unit-scale inputs.
_REL_L2_TOL = 0.06
# Split and single-pass outputs are compared and must agree within the
# reference tolerance. They are bit-identical where FA4 does not split the KV
# range (16+ requests at the draft's 2048-key window, lane J's c128+ shapes);
# at 1-8 requests the 32-split call really splits and its combine rounds
# differently (max abs ~0.01 at unit-scale outputs; observed 2026-11-14).


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _validate(
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: Any,
    kv_dtype: Any,
    o_dtype: Any,
    q_len: int,
    kv_len: int,
    num_requests: int | None = None,
) -> int:
    requests = 1 if num_requests is None else int(num_requests)
    for name, value in (
        ("num_qo_heads", num_qo_heads),
        ("num_kv_heads", num_kv_heads),
        ("head_dim", head_dim),
        ("q_len", q_len),
        ("kv_len", kv_len),
        ("num_requests", requests),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if num_qo_heads % num_kv_heads:
        raise ValueError("num_qo_heads must be a multiple of num_kv_heads")
    if (str(q_dtype), str(kv_dtype), str(o_dtype)) != ("fp8_e4m3", "fp8_e4m3", "bf16"):
        raise ProfilerNotImplemented(
            f"{KIND} on vLLM FA4 measures fp8 queries and KV with a bf16 output"
        )
    if head_dim % 16 or head_dim > 256:
        raise ProfilerNotImplemented(f"{KIND} on vLLM FA4 needs head_dim a multiple of 16 <= 256")
    return requests


def _build(torch: Any, requests: int, q_len: int, kv_len: int, hq: int, hkv: int, d: int, dev: Any):
    fp8 = torch.float8_e4m3fn
    gen = torch.Generator(device=dev).manual_seed(0xD2A)
    pages_per_req = _cdiv(kv_len, PAGE_SIZE)
    num_pages = requests * pages_per_req + 1
    # Unit-scale values (the draft's normalized K and raw V), not random byte
    # codes: those decode to magnitudes up to 448 and a one-hot softmax.
    raw = (
        torch.randn(num_pages, hkv, PAGE_SIZE, 2 * d, device=dev, generator=gen)
        .to(fp8)
        .view(torch.uint8)
    )
    perm = (torch.randperm(num_pages - 1, device=dev, generator=gen) + 1).to(torch.int32)
    block_table = perm.view(requests, pages_per_req).contiguous()
    cache = raw.view(fp8).transpose(1, 2)
    key_cache, value_cache = cache.split(d, dim=-1)
    q = torch.randn(requests * q_len, hq, d, device=dev, generator=gen).to(fp8)
    cu_q = torch.arange(requests + 1, dtype=torch.int32, device=dev) * q_len
    seqused = torch.full((requests,), kv_len, dtype=torch.int32, device=dev)
    out = torch.empty(requests * q_len, hq, d, dtype=torch.bfloat16, device=dev)
    return q, key_cache, value_cache, block_table, cu_q, seqused, out


def _reference(
    torch: Any, q, key_cache, value_cache, block_table, req: int, q_len: int, kv_len: int
):
    pages = block_table[req, : _cdiv(kv_len, PAGE_SIZE)].long()
    k = key_cache[pages].reshape(-1, *key_cache.shape[2:])[:kv_len].float()
    v = value_cache[pages].reshape(-1, *value_cache.shape[2:])[:kv_len].float()
    qr = q[req * q_len : (req + 1) * q_len].float()  # [q, hq, d]
    group = qr.shape[1] // k.shape[1]
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", qr, k) * (qr.shape[-1] ** -0.5)
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)


def _profile(backend: str, **kwargs: Any) -> ComputeMetrics:
    requests = _validate(**kwargs)
    q_len, kv_len = kwargs["q_len"], kwargs["kv_len"]
    hq, hkv, d = kwargs["num_qo_heads"], kwargs["num_kv_heads"], kwargs["head_dim"]
    try:
        import torch
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ProfilerNotImplemented(f"{KIND}:{backend} requires vLLM's FA4 (vllm_env)") from exc
    try:
        dev = torch.device("cuda", torch.cuda.current_device())
        q, kc, vc, bt, cu_q, seqused, out = _build(torch, requests, q_len, kv_len, hq, hkv, d, dev)

        def call(splits: int, dst: Any) -> Any:
            return flash_attn_varlen_func(
                q=q,
                k=kc,
                v=vc,
                out=dst,
                cu_seqlens_q=cu_q,
                max_seqlen_q=q_len,
                seqused_k=seqused,
                max_seqlen_k=kv_len,
                softmax_scale=d**-0.5,
                causal=False,
                window_size=[-1, -1],
                block_table=bt,
                softcap=0.0,
                fa_version=4,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                num_splits=splits,
            )

        splits = SPLITS[backend]
        call(splits, out)
        other = torch.empty_like(out)
        call(33 - splits, other)  # the other split setting
        torch.cuda.synchronize(dev)
        if not torch.isfinite(out).all():
            raise KernelLaunchFailed(f"{KIND}:{backend} produced non-finite output")
        if not torch.equal(out, other):
            scale = float(out.float().abs().max().clamp(min=1.0))
            diff = float((out.float() - other.float()).abs().max()) / scale
            print(f"[{KIND}:{backend}] split/nosplit differ: max abs {diff} (scale-relative)")
            if diff > _ABS_TOL:
                raise KernelLaunchFailed(f"{KIND}:{backend}: num_splits 1 and 32 differ by {diff}")
        max_abs, max_rel = 0.0, 0.0
        for req in torch.linspace(0, requests - 1, min(requests, _CHECK_REQUESTS)).long().tolist():
            want = _reference(torch, q, kc, vc, bt, req, q_len, kv_len)
            got = out[req * q_len : (req + 1) * q_len].float()
            # Relative to the row scale: FA4 rounds P before the PV product.
            scale = float(want.abs().max().clamp(min=1.0))
            max_abs = max(max_abs, float((got - want).abs().max()) / scale)
            rel = (got - want).norm(dim=-1) / want.norm(dim=-1).clamp(min=1e-6)
            max_rel = max(max_rel, float(rel.max()))
        if max_abs > _ABS_TOL or max_rel > _REL_L2_TOL:
            raise KernelLaunchFailed(
                f"{KIND}:{backend} mismatches the reference: max_abs {max_abs} rel {max_rel}"
            )

        def run() -> Any:
            return call(splits, out)

        time_ms = Timer.cupti(run, kernel_name=None, interval_union=True)
        energy_j = Energy.perf(run, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{KIND}:{backend} ran out of CUDA memory") from exc
    except (KernelLaunchFailed, ProfilerNotImplemented, TypeError, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{KIND}:{backend} failed: {exc}") from exc
    seconds = time_ms / 1000.0
    flops = 4.0 * requests * q_len * kv_len * hq * d
    # Q (fp8) and O (bf16) once, and each request's own K and V (fp8) once.
    nbytes = requests * (q_len * hq * d * 3 + 2 * kv_len * hkv * d)
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(flops / seconds / 1e12),
        memory_bandwidth_gbps=float(nbytes / seconds / 1e9),
        energy_j=float(energy_j),
    )


def profile_flashinfer_attn_rect_vllm_fa4_split32(**kwargs: Any) -> ComputeMetrics:
    return _profile("vllm_fa4_split32", **kwargs)


def profile_flashinfer_attn_rect_vllm_fa4_nosplit(**kwargs: Any) -> ComputeMetrics:
    return _profile("vllm_fa4_nosplit", **kwargs)
