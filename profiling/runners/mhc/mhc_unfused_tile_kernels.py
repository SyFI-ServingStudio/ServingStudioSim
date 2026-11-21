"""DeepInfer's unfused mHC launches: ``tile_kernels`` backends of ``mhc_post``, ``mhc_shifted_pre``.

DeepInfer runs the mHC boundary unfused where ``deep_gemm.mega_mhc`` cannot
(``v4_modules/hc.py`` ``_hc_fused_post_pre``): at layer 0 (``expand_to_mhc``,
no previous post), at the Engram layers (the Engram hook sits between the post
and the pre) and at the model's end (``deepseek.py``: the last post, then the
final shifted mix). The calls are DeepInfer's ``v4_modules/mhc_ops.py``
wrappers:

- ``mhc_post``: ``HCLayerMixin.hc_post`` -> TileKernels ``mhc_post_fwd`` (one
  ``mhc_post_fwd_cuda_kernel``): ``post_mix * x + comb_res_mixᵀ @ residual``.
- ``mhc_shifted_pre``: ``_hc_pre`` with ``hc_shifted``, the mHC part before the
  layer norm: ``mhc_pre_apply_mix`` (collapse with the carried pre mix),
  ``mhc_pre_norm_fn`` (DeepGEMM ``tf32_hc_prenorm_gemm`` with 16 K-splits, then
  ``mhc_reduce_partials_and_rmsnorm``), ``mhc_pre_split_mixes`` and
  ``sinkhorn_normalize`` (20 iterations): five launches, all timed
  (``kernel_name=None``). The layer's RMSNorm (``_norm_kernel``) is a separate
  call and not part of this boundary.

Check: an fp32 Torch transcription of each (the math DeepGEMM's
``tests/test_mega_mhc.py`` reference uses for the same steps).
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BACKEND = "tile_kernels"
# DeepSeek-V4.1-Flash scalars (see mega_mhc_deepgemm); only the Sinkhorn
# iteration count changes the work.
HC_NORM_EPS = 1e-20
HC_PRE_EPS = 1e-6
HC_POST_SCALE = 2.0
SINKHORN_EPS = 1e-6
SINKHORN_ITERS = 20
CHECK_TOKENS = 2048
MAX_DIFF = 1e-3


def validate(
    num_tokens: int, hidden_size: int, hc_mult: int, hidden_dtype: DType | str
) -> tuple[int, int, int]:
    values = int(num_tokens), int(hidden_size), int(hc_mult)
    if min(values) < 1:
        raise ValueError(f"{BACKEND} mHC needs positive sizes, got {values}")
    if DType.from_value(hidden_dtype) is not DType.BF16:
        raise ValueError(f"DeepInfer's mHC streams are bf16, got {hidden_dtype!r}")
    return values


def calc_diff(x: Any, y: Any) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return float(1 - 2 * (x * y).sum() / denom) if float(denom) else 0.0


def sinkhorn_reference(value: Any, repeat: int, eps: float) -> Any:
    value = value.softmax(dim=-1) + eps
    value = value / (value.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        value = value / (value.sum(dim=-1, keepdim=True) + eps)
        value = value / (value.sum(dim=-2, keepdim=True) + eps)
    return value


def post_reference(x: Any, residual: Any, post: Any, comb: Any) -> Any:
    """[T, m, H]: post_mix * x + comb_res_mixᵀ @ residual, fp32."""
    import torch

    return x.float().unsqueeze(1) * post + torch.einsum("tij,tih->tjh", comb, residual.float())


def shifted_pre_reference(
    torch: Any, residual: Any, prev_mix: Any, fn: Any, scales: Any, bases: Any
) -> dict[str, Any]:
    t, m, h = residual.shape
    r = residual.float()
    layer_input = (r * prev_mix).sum(dim=1)
    mixes = r.flatten(1) @ fn.mT
    mixes = mixes * torch.rsqrt(r.square().sum((1, 2)) / (m * h) + HC_NORM_EPS).unsqueeze(1)
    full = torch.cat((scales[0].expand(m), scales[1].expand(m), scales[2].expand(m * m)))
    mixes = mixes * full + bases
    return {
        "layer_input": layer_input,
        "pre": mixes[:, :m].sigmoid().unsqueeze(2) + HC_PRE_EPS,
        "post": (mixes[:, m : 2 * m].sigmoid() * HC_POST_SCALE).unsqueeze(2),
        "comb": sinkhorn_reference(mixes[:, 2 * m :].view(t, m, m), SINKHORN_ITERS, SINKHORN_EPS),
    }


def _operands(torch: Any, t: int, h: int, m: int) -> dict[str, Any]:
    g = torch.Generator(device="cuda").manual_seed(41)

    def randn(*shape: int, dtype: Any = torch.float32) -> Any:
        return torch.randn(shape, device="cuda", dtype=dtype, generator=g)

    comb = torch.rand((t, m, m), device="cuda", generator=g)
    for _ in range(3):
        comb /= comb.sum(-1, keepdim=True)
        comb /= comb.sum(-2, keepdim=True)
    width = m * (m + 2)
    return dict(
        x=randn(t, h, dtype=torch.bfloat16),
        residual=randn(t, m, h, dtype=torch.bfloat16),
        prev_mix=randn(t, m, 1).sigmoid() + HC_PRE_EPS,
        post=randn(t, m, 1).sigmoid() * HC_POST_SCALE,
        comb=comb,
        fn=randn(width, m * h) * 1e-2,
        scales=randn(3) * 0.1,
        bases=randn(width) * 0.1,
    )


def _imports() -> tuple[Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"torch is required for {BACKEND}") from exc
    try:
        from deepinfer.model_executor.models.v4_modules import mhc_ops
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{BACKEND} mHC requires DeepInfer's mhc_ops and TileKernels (deepinfer_env)"
        ) from exc
    return torch, mhc_ops


def _measure(torch: Any, name: str, run_once: Any, bytes_accessed: int) -> ComputeMetrics:
    try:
        time_ms = Timer.cupti(run_once, warmup=3, kernel_name=None)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError(f"{BACKEND} {name} ran out of memory") from exc
        raise KernelLaunchFailed(f"{BACKEND} {name} failed: {exc}") from exc
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


def profile_mhc_post_tile_kernels(
    num_tokens: int, hidden_size: int, hc_mult: int, hidden_dtype: DType | str
) -> ComputeMetrics:
    t, h, m = validate(num_tokens, hidden_size, hc_mult, hidden_dtype)
    torch, mhc_ops = _imports()
    try:
        o = _operands(torch, t, h, m)
        args = (
            o["x"].unsqueeze(0),
            o["residual"].unsqueeze(0),
            o["post"].unsqueeze(0),
            o["comb"].unsqueeze(0),
        )
        got = mhc_ops.mhc_post(*args).squeeze(0)
        torch.cuda.synchronize()
        n = slice(0, min(t, CHECK_TOKENS))
        want = post_reference(o["x"][n], o["residual"][n], o["post"][n], o["comb"][n])
        diff = calc_diff(got[n].float(), want)
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError(f"{BACKEND} mhc_post ran out of memory") from exc
        raise KernelLaunchFailed(f"{BACKEND} mhc_post failed: {exc}") from exc
    if not diff < MAX_DIFF:
        raise KernelLaunchFailed(f"{BACKEND} mhc_post: diff {diff:.2e} vs fp32 reference")
    # x, residual and the mixes in, residual out.
    nbytes = t * (2 * h + 2 * 2 * m * h + 4 * (m + m * m))
    return _measure(torch, "mhc_post", lambda: mhc_ops.mhc_post(*args), nbytes)


def profile_mhc_shifted_pre_tile_kernels(
    num_tokens: int, hidden_size: int, hc_mult: int, hidden_dtype: DType | str
) -> ComputeMetrics:
    t, h, m = validate(num_tokens, hidden_size, hc_mult, hidden_dtype)
    torch, mhc_ops = _imports()
    try:
        o = _operands(torch, t, h, m)
        residual = o["residual"].unsqueeze(0)
        prev_mix = o["prev_mix"].unsqueeze(0)

        def run_once() -> tuple[Any, ...]:
            # hc.py _hc_pre, hc_shifted branch, up to the layer norm.
            layer_input = mhc_ops.mhc_pre_apply_mix(residual, prev_mix)
            mixes = mhc_ops.mhc_pre_norm_fn(residual, o["fn"], None, HC_NORM_EPS)
            pre, post, comb = mhc_ops.mhc_pre_split_mixes(
                mixes, o["scales"], o["bases"], m, HC_POST_SCALE, HC_PRE_EPS
            )
            comb = mhc_ops.sinkhorn_normalize(comb, repeat=SINKHORN_ITERS, eps=SINKHORN_EPS)
            return layer_input, pre, post, comb

        got = dict(zip(("layer_input", "pre", "post", "comb"), (v.squeeze(0) for v in run_once())))
        torch.cuda.synchronize()
        n = slice(0, min(t, CHECK_TOKENS))
        want = shifted_pre_reference(
            torch, o["residual"][n], o["prev_mix"][n], o["fn"], o["scales"], o["bases"]
        )
        diffs = {k: calc_diff(got[k][n].float(), want[k]) for k in want}
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError(f"{BACKEND} mhc_shifted_pre ran out of memory") from exc
        raise KernelLaunchFailed(f"{BACKEND} mhc_shifted_pre failed: {exc}") from exc
    bad = {k: v for k, v in diffs.items() if not v < MAX_DIFF}
    if bad:
        raise KernelLaunchFailed(f"{BACKEND} mhc_shifted_pre: diffs {bad} vs fp32 reference")
    # Streams read twice (apply-mix, GEMM), fn, the layer input and mixes written.
    nbytes = t * (2 * 2 * m * h + 2 * h + 4 * (2 * m + m * m)) + 4 * m * (m + 2) * m * h
    return _measure(torch, "mhc_shifted_pre", run_once, nbytes)


__all__ = [
    "post_reference",
    "profile_mhc_post_tile_kernels",
    "profile_mhc_shifted_pre_tile_kernels",
    "shifted_pre_reference",
    "sinkhorn_reference",
]
