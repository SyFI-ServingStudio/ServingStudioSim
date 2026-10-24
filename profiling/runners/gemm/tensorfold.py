"""TensorFold checkpoint-math dense linears: the ``tensorfold_lane`` and ``tensorfold_ck``
backends of ``single_gemm``.

TensorFold serves NVFP4 and per-tensor FP8 projections "in the checkpoint's own
math" (``tensorfold.cuda.nvfp4.checkpoint``): the bf16 rows are quantized under
the projection's static input scale (``quant4`` / ``quant8``, a separate launch
and a separate kernel kind), then one GEMM multiplies the quantized rows by the
stored weight and writes bf16.

- ``tensorfold_lane`` times ``checkpoint._lane``, the GEMM half of
  ``checkpoint.matmul`` (decode / verify rows): ``lane_kernel`` from ``lane4.cu``.
  Its K split comes from the weight shape alone (``qmm.split_k``, at most 8),
  its block shape from ``lane_tile(m, n, sk, SMs)``; on sm_90+ the K slices meet
  through a thread-block cluster in the same launch, so there is no reduce launch.
- ``tensorfold_ck`` times ``checkpoint._gemm`` with the prompt path's ``tile=0``,
  the GEMM half of ``checkpoint.prompt`` and of ``matmul_group(prompt_rows=True)``
  (prefill rows): ``gemm_kernel`` with ``EPI=0`` from ``gemm_ck.cu``. The fused
  gate|up SwiGLU launch (``gemm_gu_ck``, ``EPI>0``) is another kind.

Both closures are exactly the production GEMM call on already-quantized rows;
CUPTI sums every launch in them (``kernel_name=None``) so a split that ever
needed its reduce kernel would still be priced whole.

Weights are built from random checkpoint bytes through the production loaders
``Fp4Linear.from_checkpoint`` / ``Fp8Linear.from_checkpoint`` with ``act`` set
(checkpoint math). The oracle decodes the GEMM's actual inputs -- the quantized
rows and the checkpoint bytes -- in Torch and multiplies in fp32, outside timing.

Runs in ``tensorfold_env``: the TensorFold checkout's own Torch stack, against
which its CUDA extensions JIT-compile on first use.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

# Both loaders tile K in steps of 64 inputs (``from_checkpoint`` refuses others).
K_STEP = 64
# The prompt GEMM's blocks are 128 columns wide; an NVFP4 weight pads only to 64
# (linear.py ``Fp4Linear.tiles``: "the prompt GEMM wants 128-column multiples").
CK_NVFP4_N_STEP = 128
NVFP4_GROUP = 16
_FP4_MAX = 6.0
_E4M3_MAX = 448.0
# bf16 output rounding (2^-8 relative) plus accumulation-order differences.
REL_FROBENIUS_TOL = 1e-2
# The oracle checks at most this many rows and columns (each output element is
# independent, and a full fp32 lm_head reference would not fit beside the weight).
_CHECK_ROWS = 256
_CHECK_COLS = 4096
# Non-power-of-two weight scales, so a dropped or swapped factor shows up.
_FP8_W_SCALE = 1.13e-2
_FP4_W_GLOBAL = 7.3e-4
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _validate(backend: str, m: int, n: int, k: int, dtype: DType | str) -> tuple[int, int, int, DType]:
    dtype = DType.from_value(dtype)
    if dtype not in (DType.FP8_E4M3, DType.NVFP4_E2M1):
        raise ValueError(f"{backend} requires dtype fp8_e4m3 or nvfp4_e2m1, got {dtype.value}")
    m, n, k = int(m), int(n), int(k)
    if m < 1 or n < 1:
        raise ValueError(f"{backend} requires m >= 1 and n >= 1, got m={m}, n={n}")
    if k < K_STEP or k % K_STEP:
        raise ValueError(f"{backend} requires k to be a positive multiple of {K_STEP}, got k={k}")
    if backend == "tensorfold_ck" and dtype is DType.NVFP4_E2M1 and n % CK_NVFP4_N_STEP:
        raise ValueError(
            f"tensorfold_ck nvfp4_e2m1 requires n to be a multiple of {CK_NVFP4_N_STEP}, got n={n}"
        )
    return m, n, k, dtype


def _load_runtime(backend: str) -> tuple[Any, Any, Any]:
    try:
        import torch
        from tensorfold.cuda.nvfp4 import checkpoint
        from tensorfold.cuda.nvfp4.linear import Fp4Linear, Fp8Linear
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{backend} requires the TensorFold environment (tensorfold_env)") from exc
    return torch, checkpoint, (Fp4Linear, Fp8Linear)


class _Problem:
    """One projection in checkpoint math: its checkpoint bytes, the loaded linear and quantized rows."""

    def __init__(self, torch: Any, checkpoint: Any, linears: tuple[Any, Any], m: int, n: int, k: int,
                 dtype: DType, prompt: bool) -> None:
        fp4_linear, fp8_linear = linears
        gen = torch.Generator(device="cuda").manual_seed(0)
        x = torch.randn(m, k, device="cuda", generator=gen, dtype=torch.float32).to(torch.bfloat16)
        amax = float(x.float().abs().amax())
        self.mode = checkpoint.A4 if dtype is DType.NVFP4_E2M1 else checkpoint.A8
        if self.mode == checkpoint.A4:
            # Static input scale: the rows' amax lands on e2m1 max times e4m3 max.
            act = amax / (_FP4_MAX * _E4M3_MAX)
            self.codes = torch.randint(0, 256, (n, k // 2), device="cuda", generator=gen, dtype=torch.uint8)
            sf = torch.rand(n, k // NVFP4_GROUP, device="cuda", generator=gen) * 1.5 + 0.5
            self.sf = sf.to(torch.float8_e4m3fn)
            self.w_scale = _FP4_W_GLOBAL
            self.lin = fp4_linear.from_checkpoint(self.codes, self.sf, self.w_scale, act=act)
        else:
            act = amax / _E4M3_MAX
            self.codes = torch.randn(n, k, device="cuda", generator=gen).mul_(64.0).to(torch.float8_e4m3fn)
            self.sf = None
            self.w_scale = _FP8_W_SCALE
            self.lin = fp8_linear.from_checkpoint(self.codes, self.w_scale, act=act)
        self.act = float(self.lin.act)
        # The quantization the production call runs before the GEMM (its own kind).
        self.rows = checkpoint._rowsq(self.mode, x, self.act, checkpoint._tb(0) if prompt else 0)
        self.out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")


def _decode_e2m1(torch: Any, packed: Any) -> Any:
    """uint8 [..., K/2], element 2j in byte j's low nibble -> fp32 [..., K]."""
    lut = torch.tensor(E2M1_VALUES + tuple(-v for v in E2M1_VALUES), device=packed.device)
    lo, hi = (packed & 0xF).long(), (packed >> 4).long()
    return torch.stack((lut[lo], lut[hi]), dim=-1).flatten(-2)


def _fp8_logical_order(torch: Any, stored: Any) -> Any:
    """``quant8``'s fragment order undone: byte 4q + j of a 16-byte group holds input 2q + j%2 + 8(j//2)."""
    perm = [2 * (p // 4) + (p % 4) % 2 + 8 * ((p % 4) // 2) for p in range(16)]
    rows, k = stored.shape
    logical = torch.empty_like(stored)
    idx = torch.tensor(perm, device=stored.device)
    logical.view(rows, k // 16, 16)[:, :, idx] = stored.view(rows, k // 16, 16)
    return logical


def check_against_reference(torch: Any, checkpoint: Any, problem: _Problem) -> float:
    """Relative Frobenius error of the GEMM output against Torch on its decoded inputs."""
    m, n = problem.out.shape
    rows = torch.unique(torch.linspace(0, m - 1, min(m, _CHECK_ROWS), device="cuda").long())
    cols = torch.unique(torch.linspace(0, n - 1, min(n, _CHECK_COLS), device="cuda").long())
    if problem.mode == checkpoint.A4:
        codes, scales = problem.rows
        x = _decode_e2m1(torch, codes[rows])
        # scales [K/64, mpad, 4]: byte b of step s is the scale of 16-input block 4s + b.
        sx = scales.view(torch.float8_e4m3fn).float()[:, rows, :].permute(1, 0, 2).reshape(len(rows), -1)
        x = x * sx.repeat_interleave(NVFP4_GROUP, dim=1) * problem.act
        w = _decode_e2m1(torch, problem.codes[cols])
        w = w * problem.sf[cols].float().repeat_interleave(NVFP4_GROUP, dim=1) * problem.w_scale
    else:
        x8 = _fp8_logical_order(torch, problem.rows[0][rows].contiguous())
        x = x8.view(torch.float8_e4m3fn).float() * problem.act
        w = problem.codes[cols].float() * problem.w_scale
    ref = x @ w.t()
    got = problem.out[rows][:, cols].float()
    err = ((got - ref).norm() / ref.norm().clamp_min(1e-30)).item()
    if not err <= REL_FROBENIUS_TOL:
        raise KernelLaunchFailed(
            f"TensorFold GEMM output mismatch: relative Frobenius error {err:.3e} > {REL_FROBENIUS_TOL:.0e}"
        )
    return err


def _profile(backend: str, m: int, n: int, k: int, dtype: DType | str) -> ComputeMetrics:
    m, n, k, dtype = _validate(backend, m, n, k, dtype)
    torch, checkpoint, linears = _load_runtime(backend)
    if not checkpoint.available() and dtype is DType.NVFP4_E2M1:
        raise ProfilerNotImplemented(f"{backend} nvfp4_e2m1 needs the SM12x block-scaled FP4 mma")
    prompt = backend == "tensorfold_ck"
    try:
        problem = _Problem(torch, checkpoint, linears, m, n, k, dtype, prompt)
        mode, rows, lin, out = problem.mode, problem.rows, problem.lin, problem.out

        if prompt:
            def run_once() -> None:
                # checkpoint.prompt / matmul_group(prompt_rows=True) after the row quantization.
                checkpoint._gemm(mode, rows, lin, out, False, 0)
        else:
            def run_once() -> None:
                # checkpoint.matmul / matmul_group after the row quantization.
                checkpoint._lane(mode, rows, lin, out, False)

        # The first call JIT-compiles the extension (minutes, once per cache).
        run_once()
        torch.cuda.synchronize()
        check_against_reference(torch, checkpoint, problem)
        del problem.codes, problem.sf

        time_ms = Timer.cupti(run_once, warmup=3)
        energy_j = Energy.perf(run_once, per_iter_time_ms=time_ms)
    except (ProfilerNotImplemented, KernelLaunchFailed):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError(f"{backend} ran out of memory") from exc
        raise KernelLaunchFailed(f"{backend} failed: {exc}") from exc

    bytes_accessed = logical_bytes(m, n, k, dtype)
    elapsed_s = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=2 * m * n * k / elapsed_s / 1e12,
        memory_bandwidth_gbps=bytes_accessed / elapsed_s / 1e9,
        energy_j=float(energy_j),
    )


def logical_bytes(m: int, n: int, k: int, dtype: DType) -> int:
    """One read of each quantized operand (with its e4m3 group scales for NVFP4) and the bf16 output write."""
    if dtype is DType.NVFP4_E2M1:
        return (m * k + n * k) // 2 + (m * k + n * k) // NVFP4_GROUP + 2 * m * n
    return m * k + n * k + 2 * m * n


def profile_single_gemm_tensorfold_lane(m: int, n: int, k: int, dtype: DType | str) -> ComputeMetrics:
    return _profile("tensorfold_lane", m, n, k, dtype)


def profile_single_gemm_tensorfold_ck(m: int, n: int, k: int, dtype: DType | str) -> ComputeMetrics:
    return _profile("tensorfold_ck", m, n, k, dtype)


__all__ = [
    "check_against_reference",
    "logical_bytes",
    "profile_single_gemm_tensorfold_ck",
    "profile_single_gemm_tensorfold_lane",
]
