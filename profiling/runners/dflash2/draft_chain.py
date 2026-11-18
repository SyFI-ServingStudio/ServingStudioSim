"""Profile the DFlash2 draft-chain fusions against the fork's incumbent calls.

Each backend is timed as one logical call with CUPTI (every launch counted,
cold L2) after a correctness check against the call sequence it replaces in the
vLLM fork's ``qwen3_dflash2`` draft (all incumbent callables are the fork's own:
``vllm._custom_ops``, ``torch.ops.vllm.dflash2_grouped_conv``, FlashInfer):

- ``qk_norm_rope_fp8_kv_write:triton`` -- vendored ``qk_norm_rope_fp8``
  (``draft_chain_vendor``) vs q/k RMSNorm + NeoX RoPE + query QuantFP8 +
  ``reshape_and_cache_flash`` (fp8). Not bit-exact: the check requires at least
  90% of fp8 codes equal and the dequantized error below 2%.

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

        from profiling.runners.dflash2.draft_chain_vendor import qk_norm_rope_fp8

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
