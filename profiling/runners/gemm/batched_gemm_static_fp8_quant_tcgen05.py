"""Profile the vendored tcgen05 query absorption with an FP8 epilogue (backend ``cuda_tcgen05``).

The kernel (``absorb_fp8_vendor/fc2.cu``, the MoE session's lane F cut C "fc2")
computes, for every token t, head h and latent column n,

    out[t, h, n] = fp8(clamp(bf16(sum_k q[t, h*256 + k] * W_UK[h, k, n]) * (1/scale), +-448))

in one launch: TMA loads of the token tile and the K-major weight copy, tcgen05
MMA into TMEM, and an epilogue that rounds to bf16, scales, clamps and stores
E4M3. That is what vLLM's MLA does in two launches: ``torch.bmm`` of the query
heads by ``W_UK_T`` into the bf16 latent query, then the static
``scaled_fp8_quant`` the FP8 sparse-attention kernel reads.

The kernel has three tile configurations (n per CTA, ring stages): (128, 2),
(128, 3), (256, 2). The best differs by m (lane F: c128 -> 128/3, c256 and c512
-> 256/2, c384 and c640 -> 128/2), so each spec checks and times all three and
reports the fastest, as an autotuned launch would. The configuration is an
implementation detail of one backend, not a row key.

Gate before timing, for every configuration: the FP8 bytes equal vLLM's unfused
pair (the fork's ``torch.bmm(..., out=ql_nope.transpose(0, 1))`` then
``vllm._custom_ops.scaled_fp8_quant``) on the same operands, and a Torch
float32 reference (independent of both) agrees within one FP8 code.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "batched_gemm_static_fp8_quant:cuda_tcgen05"
_SOURCE = Path(__file__).resolve().parent / "absorb_fp8_vendor" / "fc2.cu"
_MODULE = "batched_gemm_static_fp8_quant_fc2"
# (num_batches, k, n) compiled into fc2.cu: 64 heads, K = 256 (4 k-blocks), N = 512.
_COMPILED_GEOMETRY = (64, 256, 512)
# (n per CTA, ring stages) the kernel instantiates; rung 3 is the full kernel.
CONFIGS = ((128, 2), (128, 3), (256, 2))
_RUNG = 3
_FP8_MAX = 448.0
# A static q scale like the checkpoint's: the products stay well inside +-448.
_SCALE = 0.05
# Tokens checked against the float32 reference (the GEMM tiles are 128 tokens;
# a prefix plus the ragged tail covers every tile shape).
_CHECK_TOKENS = 1024
_ext: Any = None


def _validate_args(num_batches: int, m: int, n: int, k: int, dtype: DType | str) -> None:
    for name, value in (("num_batches", num_batches), ("m", m), ("n", n), ("k", k)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_BACKEND} requires dtype=bf16")
    if (num_batches, k, n) != _COMPILED_GEOMETRY:
        raise ProfilerNotImplemented(
            f"{_BACKEND} has (num_batches, k, n) = {_COMPILED_GEOMETRY} compiled in, "
            f"got {(num_batches, k, n)}"
        )


def _extension() -> Any:
    """Build the vendored kernel once per process (cached under HOME)."""
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load

        _ext = load(
            name=_MODULE,
            sources=[str(_SOURCE)],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_100a,code=sm_100a", "-std=c++20"],
            extra_ldflags=["-lcuda"],
            verbose=False,
        )
    return _ext


def fork_pair(torch: Any, q: Any, w_uk_t: Any, scale: Any) -> Any:
    """vLLM's unfused pair: bf16 bmm into the token-major latent query, then the static quant."""
    from vllm import _custom_ops as vops

    m, heads = q.shape[0], w_uk_t.shape[0]
    n = w_uk_t.shape[2]
    ql_nope = torch.empty((m, heads, n), dtype=torch.bfloat16, device=q.device)
    torch.bmm(q.view(m, heads, -1).transpose(0, 1), w_uk_t, out=ql_nope.transpose(0, 1))
    out, _ = vops.scaled_fp8_quant(ql_nope.view(m, heads * n), scale)
    return out.view(torch.uint8).view(m, heads, n)


def reference(torch: Any, q: Any, w_uk_t: Any, scale: float) -> Any:
    """Float32 product, bf16 materialization, x * (1/scale), clamp, E4M3."""
    m, heads = q.shape[0], w_uk_t.shape[0]
    y = torch.einsum("bmk,bkn->mbn", q.view(m, heads, -1).transpose(0, 1).float(), w_uk_t.float())
    y = y.to(torch.bfloat16).float() * (1.0 / scale)
    return y.clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def _code_distance(torch: Any, a: Any, b: Any) -> int:
    """Largest distance between E4M3 codes on the sign-magnitude number line."""

    def signed(u: Any) -> Any:
        u = u.int()
        return torch.where(u >= 128, -(u - 128), u)

    return int((signed(a) - signed(b)).abs().max().item())


def _check(torch: Any, ext: Any, ops: dict[str, Any], want: Any, nn: int, stages: int) -> None:
    out = ops["out"]
    out.fill_(0x7F)
    ext.run_c2(ops["q"], ops["wt"], out, ops["scale"], 0, _RUNG, stages, nn)
    torch.cuda.synchronize()
    if not torch.equal(out, want):
        same = (out == want).float().mean().item()
        raise AssertionError(
            f"{_BACKEND} (n {nn}, {stages} stages) FP8 output is not byte-identical to vLLM's "
            f"bmm + scaled_fp8_quant ({same:.6f} of bytes equal)"
        )


def profile_batched_gemm_static_fp8_quant_cuda_tcgen05(
    num_batches: int,
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    _validate_args(num_batches, m, n, k, dtype)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires Torch") from exc
    try:
        ext = _extension()
        device = torch.device("cuda", torch.cuda.current_device())
        g = torch.Generator(device=device).manual_seed(0)
        q = torch.randn((m, num_batches * k), device=device, generator=g).to(torch.bfloat16)
        w_uk_t = (torch.randn((num_batches, k, n), device=device, generator=g) * 0.05).to(
            torch.bfloat16
        )
        # The weight-load-time K-major copy the kernel reads: [h * n + j, k].
        wt = w_uk_t.transpose(1, 2).contiguous().view(num_batches * n, k)
        scale = torch.tensor([_SCALE], dtype=torch.float32, device=device)
        out = torch.empty((m, num_batches, n), dtype=torch.uint8, device=device)
        ops = {"q": q, "wt": wt, "scale": scale, "out": out}

        want = fork_pair(torch, q, w_uk_t, scale)
        rows = torch.cat(
            (
                torch.arange(min(m, _CHECK_TOKENS), device=device),
                torch.arange(max(m - 64, min(m, _CHECK_TOKENS)), m, device=device),
            )
        )
        ref = reference(torch, q[rows], w_uk_t, _SCALE)
        drift = _code_distance(torch, want[rows], ref)
        if drift > 1:
            raise AssertionError(f"vLLM's pair is {drift} FP8 codes from the float32 reference")
        del ref, rows
        calls = {}
        for nn, stages in CONFIGS:
            _check(torch, ext, ops, want, nn, stages)

            def call(nn: int = nn, stages: int = stages) -> None:
                ext.run_c2(q, wt, out, scale, 0, _RUNG, stages, nn)

            calls[(nn, stages)] = call
        del want
        times = {cfg: Timer.cupti(fn, warmup=3, kernel_name=None) for cfg, fn in calls.items()}
        best = min(times, key=times.__getitem__)
        time_ms = times[best]
        print(
            f"[{_BACKEND}] m={m} "
            + " ".join(f"n{c[0]}/st{c[1]}={t * 1e3:.2f}us" for c, t in times.items())
            + f" -> n{best[0]}/st{best[1]}",
            file=sys.stderr,
            flush=True,
        )
        energy_j = Energy.perf(calls[best], warmup=3, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {type(exc).__name__}: {exc}") from exc

    seconds = time_ms / 1000.0
    flops = 2.0 * num_batches * m * n * k
    moved = num_batches * (2 * m * k + 2 * k * n + m * n)
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=flops / seconds / 1e12,
        memory_bandwidth_gbps=moved / seconds / 1e9,
        energy_j=float(energy_j),
    )


__all__ = [
    "CONFIGS",
    "fork_pair",
    "profile_batched_gemm_static_fp8_quant_cuda_tcgen05",
    "reference",
]
