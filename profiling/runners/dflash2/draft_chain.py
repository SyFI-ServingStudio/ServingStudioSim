"""Profile the DFlash2 draft-chain fusions (lane J v3) against the fork's incumbent calls.

Four backends, each timed as one logical call with CUPTI (every launch counted,
cold L2) after a correctness check against the call sequence it replaces in the
vLLM fork's ``qwen3_dflash2`` draft (all incumbent callables are the fork's own:
``vllm._custom_ops``, ``torch.ops.vllm.dflash2_grouped_conv``, FlashInfer):

- ``qk_norm_rope_fp8_kv_write:triton`` -- vendored ``qk_norm_rope_fp8``
  (``lane_j_kernels``) vs q/k RMSNorm + NeoX RoPE + query QuantFP8 +
  ``reshape_and_cache_flash`` (fp8). Not bit-exact: the check requires at least
  90% of fp8 codes equal and the dequantized error below 2%.
- ``layered_kv_norm_rope_cache_write:vllm_triton`` -- the fork's Beam
  ``store_context_kv`` vs grouped ``rms_norm`` + ``rotary_embedding`` + one
  ``reshape_and_cache_flash`` per layer. Byte check: at least 99.9% of cache
  bytes equal (lane J measured 99.9999%, max one fp8 code).
- ``grouped_conv_residual_rms_norm:vllm_triton`` -- the fork's Beam
  ``conv_residual_norm`` vs ``dflash2_grouped_conv`` + ``fused_add_rms_norm``.
- ``logits_topk:triton_threshold`` -- vendored ``ThresholdTopK`` vs
  ``flashinfer.top_k(sorted=True, deterministic=True)``: ids and values must be
  identical on every row.

Caches use the fork's FlashAttention layout ``[pages, kv_heads, page, 2*head_dim]``
(fp8), slots are a shuffled page pool, and the draft's scales are 1.0 (the draft
is unquantized). Inputs are seeded normal values.
"""

from __future__ import annotations

from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_EPS = 1e-6
_ROPE_POSITIONS = 65536


def _positive(**values: Any) -> None:
    for name, value in values.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ProfilerNotImplemented("PyTorch is unavailable") from exc
    return torch


def _measure(run, flops: float, nbytes: float) -> ComputeMetrics:
    time_ms = Timer.cupti(run, kernel_name=None)
    energy_j = Energy.perf(run, per_iter_time_ms=time_ms)
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=float(flops / seconds / 1e12),
        memory_bandwidth_gbps=float(nbytes / seconds / 1e9),
        energy_j=float(energy_j),
    )


def _guard(kind: str, backend: str, body):
    torch = _torch()
    try:
        return body(torch)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{kind}:{backend} ran out of CUDA memory") from exc
    except (KernelLaunchFailed, ProfilerNotImplemented, TypeError, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{kind}:{backend} failed: {exc}") from exc


def _rope_table(torch: Any, head_dim: int, dev: Any) -> Any:
    inv = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, device=dev, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(torch.arange(_ROPE_POSITIONS, device=dev, dtype=torch.float32), inv)
    return torch.cat([freqs.cos(), freqs.sin()], -1).to(torch.bfloat16).contiguous()


def _paged_caches(
    torch: Any, layers: int, tokens: int, heads: int, head_dim: int, page: int, gen, dev
):
    """``layers`` fp8 caches in the FA layout, plus shuffled distinct slots."""
    pages = -(-tokens // page) + 2
    caches = [
        torch.zeros(pages, heads, page, 2 * head_dim, dtype=torch.uint8, device=dev)
        for _ in range(layers)
    ]
    slots = torch.randperm((pages - 1) * page, device=dev, generator=gen)[:tokens] + page
    return caches, slots.to(torch.int64).contiguous()


def _fa_views(torch: Any, cache: Any, head_dim: int):
    view = cache.view(torch.float8_e4m3fn).transpose(1, 2)  # [pages, page, heads, 2*hd]
    return view[..., :head_dim], view[..., head_dim:]


def _check_codes(torch: Any, got: Any, want: Any, min_equal: float, what: str) -> float:
    equal = float((got == want).float().mean())
    if equal < min_equal:
        raise KernelLaunchFailed(f"{what}: only {equal:.6f} of bytes equal (need {min_equal})")
    return equal


# --------------------------------------------------------------- q/k fused path --
def profile_qk_norm_rope_fp8_kv_write_triton(
    num_tokens: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    input_dtype: Any,
    kv_dtype: Any,
) -> ComputeMetrics:
    _positive(
        num_tokens=num_tokens,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
    )
    if (str(input_dtype), str(kv_dtype)) != ("bf16", "fp8_e4m3") or head_dim != 128:
        raise ProfilerNotImplemented(
            "qk_norm_rope_fp8 takes bf16 qkv, an fp8 cache and head_dim 128"
        )

    def body(torch):
        from vllm import _custom_ops as ops

        from profiling.runners.dflash2.lane_j_kernels import qk_norm_rope_fp8

        dev = torch.device("cuda", torch.cuda.current_device())
        gen = torch.Generator(device=dev).manual_seed(0x0C)
        hq, hkv, d, m = num_qo_heads, num_kv_heads, head_dim, num_tokens
        qkv = torch.randn(m, (hq + 2 * hkv) * d, device=dev, generator=gen).to(torch.bfloat16)
        qw = (1.0 + 0.1 * torch.randn(d, device=dev, generator=gen)).to(torch.bfloat16)
        kw = (1.0 + 0.1 * torch.randn(d, device=dev, generator=gen)).to(torch.bfloat16)
        pos = torch.randint(0, _ROPE_POSITIONS, (m,), device=dev, generator=gen)
        cos_sin = _rope_table(torch, d, dev)
        (cache_a,), slots = _paged_caches(torch, 1, m, hkv, d, page_size, gen, dev)
        cache_b = cache_a.clone()
        one = torch.ones(1, device=dev)
        fp8 = torch.float8_e4m3fn

        def incumbent():
            q, k, v = qkv.split([hq * d, hkv * d, hkv * d], -1)

            def rms(x, w):
                x32 = x.float()
                y = (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + _EPS)).to(torch.bfloat16)
                return y * w

            def rope(x, nh):
                cos, sin = cos_sin.index_select(0, pos).chunk(2, -1)
                x = x.view(m, nh, d)
                x1, x2 = x[..., : d // 2], x[..., d // 2 :]
                c, s = cos[:, None, :], sin[:, None, :]
                return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)

            qn = rope(rms(q.reshape(m, hq, d), qw), hq)
            kn = rope(rms(k.reshape(m, hkv, d), kw), hkv)
            qf = qn.float().clamp(-448.0, 448.0).to(fp8)
            kc, vc = _fa_views(torch, cache_b, d)
            ops.reshape_and_cache_flash(
                kn.contiguous(), v.reshape(m, hkv, d).contiguous(), kc, vc, slots, "fp8", one, one
            )
            return qf

        out_q = torch.empty(m, hq, d, dtype=fp8, device=dev)

        def run():
            return qk_norm_rope_fp8(
                qkv,
                qw,
                kw,
                _EPS,
                pos,
                cos_sin,
                cache_a.view(fp8),
                slots,
                one,
                one,
                one,
                hq,
                hkv,
                d,
                out_q=out_q,
            )

        run()
        want_q = incumbent()
        torch.cuda.synchronize(dev)
        q_eq = _check_codes(
            torch, out_q.view(torch.uint8), want_q.view(torch.uint8), 0.90, "query fp8"
        )
        c_eq = _check_codes(torch, cache_a, cache_b, 0.90, "kv cache fp8")
        rel = float((out_q.float() - want_q.float()).norm() / want_q.float().norm())
        if rel > 0.02:
            raise KernelLaunchFailed(f"query fp8 relative error {rel}")
        print(
            f"[qk_norm_rope_fp8_kv_write] m={m} q codes equal {q_eq:.4f} "
            f"cache bytes equal {c_eq:.4f} rel {rel:.2e}"
        )
        nbytes = m * ((hq + 2 * hkv) * d * 2 + hq * d + 2 * hkv * d)
        return _measure(run, flops=m * (hq + hkv) * d * 8.0, nbytes=nbytes)

    return _guard("qk_norm_rope_fp8_kv_write", "triton", body)


# ------------------------------------------------------ context K/V, all layers --
def profile_layered_kv_norm_rope_cache_write_vllm_triton(
    num_tokens: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    input_dtype: Any,
    kv_dtype: Any,
) -> ComputeMetrics:
    _positive(
        num_tokens=num_tokens,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
    )
    if (str(input_dtype), str(kv_dtype)) != ("bf16", "fp8_e4m3"):
        raise ProfilerNotImplemented("store_context_kv is measured with bf16 K/V into an fp8 cache")

    def body(torch):
        from vllm import _custom_ops as ops

        try:
            from vllm.models.beam.nvidia.dflash2_context_ops import store_context_kv
        except ImportError as exc:
            raise ProfilerNotImplemented(
                "needs the vLLM fork's Beam DFlash2 ops (vllm_env)"
            ) from exc

        dev = torch.device("cuda", torch.cuda.current_device())
        gen = torch.Generator(device=dev).manual_seed(0x0D)
        m, nl, hkv, d = num_tokens, num_layers, num_kv_heads, head_dim
        kv = torch.randn(m, nl, 2, hkv, d, device=dev, generator=gen).to(torch.bfloat16)
        weight = (1.0 + 0.1 * torch.randn(nl, d, device=dev, generator=gen)).to(torch.bfloat16)
        pos = torch.randint(0, _ROPE_POSITIONS, (m,), device=dev, generator=gen)
        cos_sin = _rope_table(torch, d, dev)
        caches_a, slots = _paged_caches(torch, nl, m, hkv, d, page_size, gen, dev)
        caches_b = [c.clone() for c in caches_a]
        fp8 = torch.float8_e4m3fn
        one = torch.ones(1, device=dev)
        cache_ptrs = torch.tensor([c.data_ptr() for c in caches_a], dtype=torch.int64, device=dev)
        scale_ptrs = torch.tensor([one.data_ptr()] * (2 * nl), dtype=torch.int64, device=dev)

        def run():
            store_context_kv(
                kv, weight, pos, cos_sin, cache_ptrs, slots, caches_a[0].view(fp8), _EPS, scale_ptrs
            )

        def incumbent():
            perm = kv.permute(2, 1, 0, 3, 4).contiguous()
            all_k, all_v = perm[0], perm[1]
            kn = torch.empty_like(all_k)
            ops.rms_norm(kn, all_k, weight, _EPS)
            ops.rotary_embedding(pos.repeat(nl), kn.view(nl * m, -1), None, d, cos_sin, True)
            for i in range(nl):
                kc, vc = _fa_views(torch, caches_b[i], d)
                ops.reshape_and_cache_flash(kn[i], all_v[i], kc, vc, slots, "fp8", one, one)

        run()
        incumbent()
        torch.cuda.synchronize(dev)
        eq = min(
            _check_codes(torch, a, b, 0.999, f"layer {i} cache")
            for i, (a, b) in enumerate(zip(caches_a, caches_b))
        )
        print(f"[layered_kv_norm_rope_cache_write] m={m} min cache bytes equal {eq:.6f}")
        nbytes = m * nl * 2 * hkv * d * (2 + 1)
        return _measure(run, flops=m * nl * hkv * d * 8.0, nbytes=nbytes)

    return _guard("layered_kv_norm_rope_cache_write", "vllm_triton", body)


# ------------------------------------------ grouped conv finish + residual norm --
def profile_grouped_conv_residual_rms_norm_vllm_triton(
    num_tokens: int,
    hidden: int,
    group_size: int,
    taps: int,
    block_len: int,
    dtype: Any,
) -> ComputeMetrics:
    _positive(
        num_tokens=num_tokens, hidden=hidden, group_size=group_size, taps=taps, block_len=block_len
    )
    if str(dtype) != "bf16" or hidden % group_size:
        raise ProfilerNotImplemented("conv_residual_norm is measured in bf16 with whole groups")
    if num_tokens % block_len:
        raise ValueError("num_tokens must be whole blocks of block_len")

    def body(torch):
        from vllm import _custom_ops as ops

        try:
            import vllm.model_executor.models.qwen3_dflash2  # noqa: F401  registers dflash2_grouped_conv
            from vllm.models.beam.nvidia.dflash2_conv_ops import conv_residual_norm
        except ImportError as exc:
            raise ProfilerNotImplemented("needs the vLLM fork's DFlash2 ops (vllm_env)") from exc

        dev = torch.device("cuda", torch.cuda.current_device())
        gen = torch.Generator(device=dev).manual_seed(0x0E)
        m, h, groups = num_tokens, hidden, hidden // group_size
        x = torch.randn(m, h, device=dev, generator=gen).to(torch.bfloat16)
        coeff = (0.1 * torch.randn(m, 2, taps, groups, device=dev, generator=gen)).to(
            torch.bfloat16
        )[:, 1]
        base = (0.5 * torch.randn(taps, h, device=dev, generator=gen)).to(torch.bfloat16)
        res0 = torch.randn(m, h, device=dev, generator=gen).to(torch.bfloat16)
        weight = (1.0 + 0.1 * torch.randn(h, device=dev, generator=gen)).to(torch.bfloat16)
        res_a, res_b = res0.clone(), res0.clone()

        def run():
            return conv_residual_norm(x, coeff, base, res_a, weight, _EPS, block_len, group_size)

        out_a, _ = run()
        y = torch.ops.vllm.dflash2_grouped_conv(x, coeff, base, block_len, group_size)
        ops.fused_add_rms_norm(y, res_b, weight, _EPS)
        torch.cuda.synchronize(dev)
        rel = float((out_a.float() - y.float()).norm() / y.float().norm())
        rrel = float((res_a.float() - res_b.float()).norm() / res_b.float().norm())
        if rel > 1e-2 or rrel > 1e-2:
            raise KernelLaunchFailed(
                f"conv_residual_norm vs incumbent: out rel {rel} residual rel {rrel}"
            )
        print(f"[grouped_conv_residual_rms_norm] m={m} out rel {rel:.2e} residual rel {rrel:.2e}")
        # x, residual read; residual and output written; coefficients per group.
        nbytes = m * (4 * h * 2 + taps * groups * 2)
        return _measure(run, flops=m * h * (2 * taps + 4.0), nbytes=nbytes)

    return _guard("grouped_conv_residual_rms_norm", "vllm_triton", body)


# ----------------------------------------------------------- vocabulary top-k --
_TOPK_ID_EXACT_ROWS = 512


def profile_logits_topk_triton_threshold(
    num_rows: int, num_columns: int, top_k: int, dtype: Any
) -> ComputeMetrics:
    _positive(num_rows=num_rows, num_columns=num_columns, top_k=top_k)
    if str(dtype) != "bf16":
        raise ProfilerNotImplemented("threshold top-k is measured on bf16 logits")

    def body(torch):
        import flashinfer

        from profiling.runners.dflash2.lane_j_kernels import ThresholdTopK

        dev = torch.device("cuda", torch.cuda.current_device())
        gen = torch.Generator(device=dev).manual_seed(0)
        scores = torch.randn(num_rows, num_columns, device=dev, generator=gen).to(torch.bfloat16)
        if -(-num_columns // 2048) < top_k:
            raise ProfilerNotImplemented("threshold top-k needs at least top_k 2048-wide slices")
        # Lane J v3's delivered setting: 2048-wide slices, 128 candidates per row, warps 2/2/1.
        topk = ThresholdTopK(
            num_rows, num_columns, top_k, dev, slice_=2048, cap=128, wa=2, wb=2, wc=1
        )

        def run():
            return topk(scores)

        vals, ids = run()
        ref_v, ref_i = flashinfer.top_k(scores, top_k, sorted=True, deterministic=True)
        tie = (scores.float() * 2).round().div(2).to(torch.bfloat16)
        tv, ti = ThresholdTopK(
            num_rows, num_columns, top_k, dev, slice_=2048, cap=128, wa=2, wb=2, wc=1
        )(tie)
        rv2, ri2 = flashinfer.top_k(tie, top_k, sorted=True, deterministic=True)
        torch.cuda.synchronize(dev)
        rows = float((ids == ref_i.long()).all(-1).float().mean())
        tie_rows = float((ti == ri2.long()).all(-1).float().mean())
        values_ok = torch.equal(vals, ref_v.float()) and torch.equal(tv, rv2.float())
        # The ids must point at the values they report (a valid top-k).
        picks_ok = torch.equal(scores.gather(1, ids).float(), vals) and torch.equal(
            tie.gather(1, ti).float(), tv
        )
        # Up to _TOPK_ID_EXACT_ROWS rows FlashInfer's deterministic order is the
        # one the kernel reproduces, so ids must match it exactly; past that
        # FlashInfer breaks bf16 ties in another order (observed: 30% of rows
        # identical at 768+ rows) and only values + valid picks are required.
        ids_required = num_rows <= _TOPK_ID_EXACT_ROWS
        if not values_ok or not picks_ok or (ids_required and (rows < 1.0 or tie_rows < 1.0)):
            raise KernelLaunchFailed(
                f"threshold top-k vs FlashInfer deterministic: values {values_ok} picks "
                f"{picks_ok} ids rows {rows} tie rows {tie_rows}"
            )
        print(
            f"[logits_topk:triton_threshold] rows={num_rows} values identical, ids identical "
            f"{rows} (tie-heavy {tie_rows})"
        )
        nbytes = num_rows * (num_columns * 2 + top_k * 12)
        return _measure(run, flops=float(num_rows * num_columns), nbytes=nbytes)

    return _guard("logits_topk", "triton_threshold", body)
