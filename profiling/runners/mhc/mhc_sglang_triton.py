"""DeepSeek-V4 mHC kernels as SGLang v0.5.23 runs them (SM12x path): three kinds.

- ``mhc_mix_stats_sinkhorn:sglang_triton`` -- ``hc_mix_stats_sinkhorn``
  (``kernels/ops/layernorm/mhc.py:2465``): ``_hc_mix_stats_partial_kernel`` +
  ``_hc_mix_reduce_sinkhorn_kernel``.
- ``mhc_combine_rms_norm:sglang_triton`` -- ``hc_combine_norm``
  (``kernels/ops/layernorm/hc_combine_norm.py:55``): ``_hc_combine_norm`` /
  ``_hc_combine_norm_prefill``.
- ``mhc_post:sglang_auto`` -- ``srt/models/deepseek_v4_mhc.py`` ``post`` (line 525) with
  V4.1's config (``hc_pre_from_prev_sublayer``): ``_mhc_post_split_h_kernel`` up to 384 rows on
  Blackwell, else ``mhc_post_tilelang``.

Inputs are built with the shapes and dtypes the model passes (``make_hc_mixing_params``,
``deepseek_v4.py:636``): bf16 streams ``[T, hc_mult * hidden]``, fp32 ``hc_fn
[(2 + hc_mult) * hc_mult, hc_mult * hidden]``, fp32 ``hc_scale [3]`` and ``hc_base``, fp32
coefficients. Every call is checked against a reference before timing; every launch of the call is
counted (CUPTI). Runs in ``sglang_env``.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

# DeepSeek-V4.1-Flash config: rms_norm_eps (the mix RMS and the sublayer norms), hc_eps.
_RMS_EPS = 1e-20
_HC_EPS = 1e-6
_CHECK_ROWS = 256
_MAX_TOKENS = 65_536


def _validate(num_tokens: int, hidden_size: int, hc_mult: int, hidden_dtype: Any, backend: str) -> None:
    if DType.from_value(hidden_dtype) is not DType.BF16:
        raise ValueError(f"{backend} requires hidden_dtype=bf16, got {hidden_dtype!r}")
    if type(num_tokens) is not int or not 1 <= num_tokens <= _MAX_TOKENS:
        raise ValueError(f"{backend}: num_tokens must be an integer in [1, {_MAX_TOKENS}]")
    if type(hidden_size) is not int or hidden_size < 1 or type(hc_mult) is not int or hc_mult < 1:
        raise ValueError(f"{backend}: hidden_size and hc_mult must be positive integers")


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - sglang_env always has torch
        raise ProfilerNotImplemented("torch is required") from exc
    return torch


def _rel_err(torch: Any, got: Any, ref: Any) -> float:
    return ((got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30)).item()


def _metrics(time_ms: float, energy_j: float, logical_bytes: float) -> ComputeMetrics:
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(logical_bytes / elapsed_s / 1e9) if elapsed_s else 0.0,
        energy_j=float(energy_j),
    )


def _time(call: Any) -> tuple[float, float]:
    time_ms = Timer.cupti(call, warmup=3)
    energy_j = Energy.perf(call, per_iter_time_ms=time_ms)
    return time_ms, energy_j


def _guard(backend: str, fn: Any) -> Any:
    try:
        return fn()
    except (ProfilerNotImplemented, KernelLaunchFailed, ValueError):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError(f"{backend} ran out of memory") from exc
        raise KernelLaunchFailed(f"{backend} failed: {exc}") from exc


def profile_mhc_mix_stats_sinkhorn_sglang_triton(
    num_tokens: int, hidden_size: int, hc_mult: int, sinkhorn_iters: int, hidden_dtype: Any
) -> ComputeMetrics:
    backend = "mhc_mix_stats_sinkhorn:sglang_triton"
    _validate(num_tokens, hidden_size, hc_mult, hidden_dtype, backend)
    if type(sinkhorn_iters) is not int or sinkhorn_iters < 1:
        raise ValueError(f"{backend}: sinkhorn_iters must be a positive integer")
    torch = _torch()
    try:
        from sglang.kernels.ops.layernorm.mhc import (
            _hc_split_sinkhorn_triton,
            hc_mix_stats_sinkhorn,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{backend} requires the SGLang environment (sglang_env)") from exc

    def run() -> ComputeMetrics:
        g = torch.Generator(device="cuda").manual_seed(0)
        m, k, mix = num_tokens, hc_mult * hidden_size, (2 + hc_mult) * hc_mult
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
        fn = torch.randn(mix, k, device="cuda", generator=g) * (k**-0.5)
        scale = torch.rand(3, device="cuda", generator=g) + 0.5
        base = torch.randn(mix, device="cuda", generator=g) * 0.1

        def call() -> Any:
            return hc_mix_stats_sinkhorn(x, fn, scale, base, hc_mult, sinkhorn_iters, _RMS_EPS, _HC_EPS)

        pre, post, comb = call()
        torch.cuda.synchronize()
        rows = torch.linspace(0, m - 1, min(m, _CHECK_ROWS), device="cuda").round().long().unique()
        xs = x[rows].float()
        mixes = torch.nn.functional.linear(xs, fn) * torch.rsqrt(xs.square().mean(-1, keepdim=True) + _RMS_EPS)
        r_pre, r_post, r_comb = _hc_split_sinkhorn_triton(
            mixes.unsqueeze(1), scale, base, hc_mult, sinkhorn_iters, _HC_EPS
        )
        errs = (
            _rel_err(torch, pre[rows], r_pre.squeeze(1)),
            _rel_err(torch, post[rows], r_post.squeeze(1)),
            _rel_err(torch, comb[rows], r_comb.squeeze(1)),
        )
        # The fused path's split-K mix GEMM uses tl.dot (tf32) against an fp32 reference.
        if not max(errs) <= 2e-2:
            raise KernelLaunchFailed(f"{backend} pre/post/comb mismatch: relative errors {errs}")
        time_ms, energy_j = _time(call)
        return _metrics(time_ms, energy_j, 2 * m * k + 4 * mix * k)

    return _guard(backend, run)


def profile_mhc_combine_rms_norm_sglang_triton(
    num_tokens: int, hidden_size: int, hc_mult: int, hidden_dtype: Any
) -> ComputeMetrics:
    backend = "mhc_combine_rms_norm:sglang_triton"
    _validate(num_tokens, hidden_size, hc_mult, hidden_dtype, backend)
    if (hc_mult, hidden_size) != (4, 5120):
        # SGLang falls back to norm(mhc_combine(...)) for other geometries (deepseek_v4_mhc.py `_combine`).
        raise ProfilerNotImplemented(f"{backend}: SGLang's hc_combine_norm exists for hc_mult 4, hidden 5120 only")
    torch = _torch()
    try:
        from sglang.kernels.ops.layernorm.hc_combine_norm import hc_combine_norm
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{backend} requires the SGLang environment (sglang_env)") from exc

    def run() -> ComputeMetrics:
        g = torch.Generator(device="cuda").manual_seed(0)
        m = num_tokens
        x = torch.randn(m, hc_mult * hidden_size, device="cuda", dtype=torch.bfloat16, generator=g)
        pre = torch.rand(m, hc_mult, device="cuda", generator=g)
        weight = (torch.rand(hidden_size, device="cuda", generator=g) + 0.5).to(torch.bfloat16)

        def call() -> Any:
            return hc_combine_norm(x, pre, weight, _RMS_EPS)

        y = call()
        torch.cuda.synchronize()
        rows = torch.linspace(0, m - 1, min(m, _CHECK_ROWS), device="cuda").round().long().unique()
        streams = x[rows].float().view(-1, hc_mult, hidden_size)
        v = (streams * pre[rows].unsqueeze(-1)).sum(1).to(torch.bfloat16).float()
        ref = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + _RMS_EPS) * weight.float()
        err = _rel_err(torch, y[rows], ref)
        if not err <= 1e-2:
            raise KernelLaunchFailed(f"{backend} output mismatch: relative Frobenius error {err:.3e}")
        time_ms, energy_j = _time(call)
        return _metrics(time_ms, energy_j, 2 * m * (hc_mult + 1) * hidden_size)

    return _guard(backend, run)


def profile_mhc_post_sglang_auto(
    num_tokens: int, hidden_size: int, hc_mult: int, hidden_dtype: Any
) -> ComputeMetrics:
    backend = "mhc_post:sglang_auto"
    _validate(num_tokens, hidden_size, hc_mult, hidden_dtype, backend)
    torch = _torch()
    try:
        from sglang.srt.models.deepseek_v4_mhc import HcConfig, post
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{backend} requires the SGLang environment (sglang_env)") from exc

    def run() -> ComputeMetrics:
        g = torch.Generator(device="cuda").manual_seed(0)
        m, h = num_tokens, hidden_size
        # V4.1-Flash (configs/deepseek_v41.py): hc_pre_from_prev_sublayer = True; no prefill CP.
        cfg = HcConfig(
            mult=hc_mult,
            sinkhorn_iters=20,
            eps=_HC_EPS,
            rms_eps=_RMS_EPS,
            hidden=h,
            pre_from_prev=True,
            cp_prefill=False,
        )
        x = torch.randn(m, h, device="cuda", dtype=torch.bfloat16, generator=g)
        residual = torch.randn(m, hc_mult, h, device="cuda", dtype=torch.bfloat16, generator=g)
        post_mix = torch.rand(m, hc_mult, device="cuda", generator=g)
        comb = torch.rand(m, hc_mult, hc_mult, device="cuda", generator=g)

        def call() -> Any:
            return post(cfg, x, residual, post_mix, comb)

        out = call()
        torch.cuda.synchronize()
        rows = torch.linspace(0, m - 1, min(m, _CHECK_ROWS), device="cuda").round().long().unique()
        ref = post_mix[rows].unsqueeze(-1) * x[rows].float().unsqueeze(1) + (
            comb[rows].unsqueeze(-1) * residual[rows].float().unsqueeze(2)
        ).sum(dim=1)
        err = _rel_err(torch, out[rows], ref)
        if not err <= 1e-2:
            raise KernelLaunchFailed(f"{backend} output mismatch: relative Frobenius error {err:.3e}")
        time_ms, energy_j = _time(call)
        return _metrics(time_ms, energy_j, 2 * m * (2 * hc_mult + 1) * h)

    return _guard(backend, run)


__all__ = [
    "profile_mhc_combine_rms_norm_sglang_triton",
    "profile_mhc_mix_stats_sinkhorn_sglang_triton",
    "profile_mhc_post_sglang_auto",
]
