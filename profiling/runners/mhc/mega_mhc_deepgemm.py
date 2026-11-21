"""DeepInfer's fused mHC boundary on SM100 -- the ``deepgemm_sm100`` backend of ``mega_mhc``.

The timed call is the one ``HCLayerMixin._mega_mhc`` issues
(``deepinfer/model_executor/models/v4_modules/hc.py``) at every sub-layer entry
that follows a sub-layer of the same trunk (every attention and FFN entry
except layer 0's attention, the Engram layers' attention and the model's
final mix): one ``deep_gemm.mega_mhc`` with the shifted pre mix
(``hc_shifted``), a BF16 output (``y_bf16``) and the FP8 E4M3 output with
UE8M0 scales per 32 channels packed four to an int32. Where the scales go
depends on the consumer, the only difference between the two entries:

- ``sf_layout="gemm"`` (attention entry): ``y_gemm_sf``, TMA-aligned
  column-major, read by the MXFP8 ``fp8_fp4_gemm_nt`` of the input
  projections.
- ``sf_layout="mega_moe"`` (FFN entry, ``fused_norm_cast_kwargs``): the FP8
  rows and ``y_routed_sf`` straight into the mega MoE symmetric buffer, plus
  ``y_shared_sf`` in the fused shared expert's layout with
  ``shared_sf_block_m = deep_gemm.get_block_m_for_mega_moe(...)``.

One ``sm100_mega_mhc_impl<hidden, splits, num_sms, shifted=1, store_bf16=1,
fp8=1, shared_sf_block_m>`` launch; DeepGEMM picks the split count from
``num_tokens`` on the host.

Check: against DeepInfer's own unfused shifted path on the same operands
(``v4_modules/mhc_ops.py``: TileKernels ``mhc_post`` + ``mhc_pre_apply_mix`` +
DeepGEMM ``tf32_hc_prenorm_gemm`` / ``mhc_reduce_partials_and_rmsnorm`` +
``mhc_pre_split_mixes`` + ``mhc_sinkhorn``, which ``_hc_pre`` runs at the
Engram entries) and an fp32 RMSNorm; the FP8 rows times their decoded scales
must reproduce the kernel's BF16 output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

BACKEND = "deepgemm_sm100"
SF_LAYOUTS = ("gemm", "mega_moe")
# DeepGEMM layout::mega_mhc: kNumRoutes, kNumMaxTokens, kDefaultNumSplits * BLOCK_K.
HC_MULT = 4
MAX_TOKENS = 1 << 20
HIDDEN_ALIGN = 16 * 64
# DeepSeek-V4.1-Flash scalars (hc_eps, hc_sinkhorn_iters, rms_norm_eps; the
# fork's hc_post_alpha). Only num_sinkhorn_iters changes the work.
HC_NORM_EPS = 1e-20
HC_PRE_EPS = 1e-6
HC_POST_SCALE = 2.0
SINKHORN_EPS = 1e-6
SINKHORN_ITERS = 20
RMSNORM_EPS = 1e-20
RMSNORM_SCALE = 1.0
# The mega MoE the FFN entry feeds (DeepInfer DP4/EP4, 384 + 32 physical
# slots, top-6, fp8 x fp4): only the shared-scale tile size depends on it.
MEGA_MOE_EP = 4
MEGA_MOE_EXPERTS = 416
MEGA_MOE_TOPK = 6
MEGA_MOE_MMA = "fp8xfp4"
CHECK_TOKENS = 2048
# calc_diff-style similarity bounds (DeepGEMM's test uses 5e-5 against its own
# fp32 reference; the unfused TileKernels path rounds at other points).
MAX_DIFF = 1e-3
MAX_FP8_DIFF = 2e-3


@dataclass(frozen=True)
class Call:
    """Operands and outputs of one production ``mega_mhc`` call."""

    kwargs: dict[str, Any]
    shared_sf_block_m: int

    def run(self, deep_gemm: Any) -> None:
        deep_gemm.mega_mhc(**self.kwargs)


def validate(
    num_tokens: int, hidden_size: int, hc_mult: int, hidden_dtype: DType | str, sf_layout: str
) -> tuple[int, int]:
    if type(num_tokens) is not int or num_tokens < 1:
        raise ValueError(f"num_tokens must be a positive integer, got {num_tokens!r}")
    if num_tokens > MAX_TOKENS:
        raise ValueError(f"mega_mhc takes at most {MAX_TOKENS} tokens, got {num_tokens}")
    if hc_mult != HC_MULT:
        raise ValueError(f"DeepGEMM mega_mhc is built for hc_mult={HC_MULT}, got {hc_mult}")
    if hidden_size < 1 or hidden_size % HIDDEN_ALIGN:
        raise ValueError(f"hidden_size must be a positive multiple of {HIDDEN_ALIGN}")
    if DType.from_value(hidden_dtype) is not DType.BF16:
        raise ValueError(f"mega_mhc streams are bf16, got {hidden_dtype!r}")
    if sf_layout not in SF_LAYOUTS:
        raise ValueError(f"sf_layout must be one of {SF_LAYOUTS}, got {sf_layout!r}")
    return num_tokens, hidden_size


def shared_sf_block_m(deep_gemm: Any, num_tokens: int, capacity: int) -> int:
    """The mega MoE tile height DeepInfer passes as ``extra_sf_block_m``."""
    return int(
        deep_gemm.get_block_m_for_mega_moe(
            MEGA_MOE_EP, MEGA_MOE_EXPERTS, capacity, num_tokens, MEGA_MOE_TOPK, MEGA_MOE_MMA
        )
    )


def build_call(
    torch: Any,
    deep_gemm: Any,
    num_tokens: int,
    hidden: int,
    sf_layout: str,
    block_m: int | None = None,
) -> Call:
    """Operands as the previous sub-layer's lazy post data leaves them.

    Outputs are allocated as ``_mega_mhc`` allocates them.
    """
    g = torch.Generator(device="cuda").manual_seed(41)
    dev = "cuda"
    t, h, m = num_tokens, hidden, HC_MULT
    width = m * (m + 2)

    def randn(*shape: int, dtype: Any = torch.float32) -> Any:
        return torch.randn(shape, device=dev, dtype=dtype, generator=g)

    comb = torch.rand((t, m, m), device=dev, generator=g)
    for _ in range(3):
        comb /= comb.sum(-1, keepdim=True)
        comb /= comb.sum(-2, keepdim=True)
    kwargs: dict[str, Any] = dict(
        x=randn(t, h, dtype=torch.bfloat16),
        residual=randn(t, m, h, dtype=torch.bfloat16),
        shifted_prev_mix=randn(t, m, 1).sigmoid() + HC_PRE_EPS,
        post_mix=randn(t, m, 1).sigmoid() * HC_POST_SCALE,
        comb_res_mix=comb,
        fn=randn(width, m * h) * 1e-2,
        mix_scales=randn(3) * 0.1,
        mix_bases=randn(width) * 0.1,
        hc_mult=m,
        hc_norm_eps=HC_NORM_EPS,
        hc_pre_eps=HC_PRE_EPS,
        hc_post_scale=HC_POST_SCALE,
        sinkhorn_eps=SINKHORN_EPS,
        num_sinkhorn_iters=SINKHORN_ITERS,
        rmsnorm_weight=(randn(h) * 0.1 + 1).bfloat16(),
        rmsnorm_eps=RMSNORM_EPS,
        rmsnorm_scale=RMSNORM_SCALE,
        new_residual=torch.empty((t, m, h), device=dev, dtype=torch.bfloat16),
        new_prev_mix=torch.empty((t, m, 1), device=dev),
        new_post_mix=torch.empty((t, m, 1), device=dev),
        new_comb_res_mix=torch.empty((t, m, m), device=dev),
        y_bf16=torch.empty((t, h), device=dev, dtype=torch.bfloat16),
        y_fp8=torch.empty((t, h), device=dev, dtype=torch.float8_e4m3fn),
    )
    words = h // 128
    if sf_layout == "gemm":
        # hc.py: torch.empty(hidden // 128, m_pad, int32).mT[:num_tokens]
        m_pad = (t + 3) // 4 * 4
        kwargs["y_gemm_sf"] = torch.empty((words, m_pad), device=dev, dtype=torch.int32).mT[:t]
        return Call(kwargs, 0)
    if block_m is None:
        block_m = shared_sf_block_m(deep_gemm, t, t)
    # Symmetric-buffer views: routed SF row-major [T, H/128]; shared SF
    # column-major over ceil(T / block_m) tiles of align(block_m, 128) rows.
    rows = (t + block_m - 1) // block_m * ((block_m + 127) // 128 * 128)
    kwargs["y_routed_sf"] = torch.empty((t, words), device=dev, dtype=torch.int32)
    kwargs["y_shared_sf"] = torch.empty_strided(
        (rows, words), (1, rows), device=dev, dtype=torch.int32
    )[:t]
    kwargs["shared_sf_block_m"] = block_m
    return Call(kwargs, block_m)


def calc_diff(x: Any, y: Any) -> float:
    """DeepGEMM's similarity measure: 1 - 2<x,y> / (|x|^2 + |y|^2)."""
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return float(1 - 2 * (x * y).sum() / denom) if float(denom) else 0.0


def decode_fp8(torch: Any, y_fp8: Any, sf_words: Any) -> Any:
    """fp32 rows from E4M3 data and UE8M0 exponents packed four per int32 (one per 32 channels)."""
    t, h = y_fp8.shape
    exps = sf_words.contiguous().view(torch.uint8).view(t, h // 32).float()
    scale = torch.exp2(exps - 127).repeat_interleave(32, dim=1)
    return y_fp8.float() * scale


def check(torch: Any, call: Call, sf_layout: str) -> None:
    """Compare the first CHECK_TOKENS rows with DeepInfer's unfused shifted path."""
    from deepinfer.model_executor.models.v4_modules.mhc_ops import (
        mhc_post,
        mhc_pre_apply_mix,
        mhc_pre_norm_fn,
        mhc_pre_split_mixes,
        sinkhorn_normalize,
    )

    k = call.kwargs
    n = min(k["x"].shape[0], CHECK_TOKENS)
    rows = slice(0, n)
    residual = mhc_post(
        k["x"][rows].unsqueeze(0),
        k["residual"][rows].unsqueeze(0),
        k["post_mix"][rows].unsqueeze(0),
        k["comb_res_mix"][rows].unsqueeze(0),
    )
    layer_input = mhc_pre_apply_mix(residual, k["shifted_prev_mix"][rows].unsqueeze(0))
    mixes = mhc_pre_norm_fn(residual, k["fn"], None, HC_NORM_EPS)
    pre, post, comb = mhc_pre_split_mixes(
        mixes, k["mix_scales"], k["mix_bases"], HC_MULT, HC_POST_SCALE, HC_PRE_EPS
    )
    comb = sinkhorn_normalize(comb, repeat=SINKHORN_ITERS, eps=SINKHORN_EPS)
    li = layer_input.squeeze(0).float()
    y = li * torch.rsqrt(li.square().mean(-1, keepdim=True) + RMSNORM_EPS)
    y = (y * k["rmsnorm_weight"].float() * RMSNORM_SCALE).bfloat16()
    expected = {
        "new_residual": residual.squeeze(0),
        "new_prev_mix": pre.squeeze(0),
        "new_post_mix": post.squeeze(0),
        "new_comb_res_mix": comb.squeeze(0),
        "y_bf16": y,
    }
    for name, want in expected.items():
        diff = calc_diff(k[name][rows].float(), want.float())
        if not diff < MAX_DIFF:
            raise KernelLaunchFailed(
                f"mega_mhc {name}: diff {diff:.2e} vs DeepInfer's unfused path"
            )
    sf = k["y_gemm_sf"] if sf_layout == "gemm" else k["y_routed_sf"]
    dequant = decode_fp8(torch, k["y_fp8"][rows], sf[rows])
    diff = calc_diff(dequant, k["y_bf16"][rows].float())
    if not diff < MAX_FP8_DIFF:
        raise KernelLaunchFailed(f"mega_mhc y_fp8 x {sf_layout} scales: diff {diff:.2e} vs y_bf16")
    if sf_layout == "mega_moe":
        storage = k["y_shared_sf"].as_strided(
            (k["y_shared_sf"].stride(1), k["y_shared_sf"].shape[1]), k["y_shared_sf"].stride()
        )
        shared = storage[shared_sf_rows(torch, n, call.shared_sf_block_m, storage.device)]
        if not torch.equal(shared, k["y_routed_sf"][rows]):
            raise KernelLaunchFailed("mega_mhc y_shared_sf differs from y_routed_sf")


def shared_sf_rows(torch: Any, num_tokens: int, block_m: int, device: Any) -> Any:
    """Row of each token in the mega MoE shared-expert scale layout (DeepGEMM test_mega_mhc)."""
    token = torch.arange(num_tokens, device=device)
    i = token % block_m
    return (
        token // block_m * ((block_m + 127) // 128 * 128)
        + i // 128 * 128
        + i % 32 * 4
        + i % 128 // 32
    )


def profile_mega_mhc_deepgemm(
    num_tokens: int,
    hidden_size: int,
    hc_mult: int,
    hidden_dtype: DType | str,
    sf_layout: str,
) -> ComputeMetrics:
    num_tokens, hidden = validate(num_tokens, hidden_size, hc_mult, hidden_dtype, sf_layout)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"torch is required for {BACKEND}") from exc
    try:
        import deep_gemm
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{BACKEND} mega_mhc requires DeepInfer's DeepGEMM (deepinfer_env)"
        ) from exc
    if not hasattr(deep_gemm, "mega_mhc"):
        raise ProfilerNotImplemented("this DeepGEMM has no mega_mhc")
    try:
        call = build_call(torch, deep_gemm, num_tokens, hidden, sf_layout)
        call.run(deep_gemm)  # JIT compile, outside timing.
        torch.cuda.synchronize()
        check(torch, call, sf_layout)

        def run_once() -> None:
            call.run(deep_gemm)

        time_ms = Timer.cupti(run_once, warmup=3, kernel_name=None)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError(f"{BACKEND} mega_mhc ran out of memory") from exc
        raise KernelLaunchFailed(f"{BACKEND} mega_mhc failed: {exc}") from exc

    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=2 * num_tokens * HC_MULT * hidden * HC_MULT * (HC_MULT + 2) / elapsed_s / 1e12,
        memory_bandwidth_gbps=hbm_bytes(num_tokens, hidden) / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


def hbm_bytes(num_tokens: int, hidden: int) -> int:
    """One pass over the call's tensors: x, residual in and out, BF16 and FP8 outputs, scales."""
    m = HC_MULT
    per_token = hidden * (2 + 2 * m * 2 + 2 + 1) + 2 * (hidden // 128) * 4 + 4 * (m + m + m + m * m)
    return num_tokens * per_token + 4 * m * (m + 2) * m * hidden


__all__ = ["profile_mega_mhc_deepgemm", "build_call", "check", "decode_fp8", "validate"]
