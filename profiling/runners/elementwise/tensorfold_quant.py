"""TensorFold's checkpoint-math activation quantizers: the ``tensorfold_cuda``
backends of ``nvfp4_quant`` and ``static_fp8_quant``.

This runner only allocates tensors, times one public call, and returns metrics.
DB writes, subprocess selection and registry routing live in L1b.

TensorFold (perf-study 6469730a) serves NVFP4 and per-tensor FP8 projections in
the checkpoint's own math (``tensorfold.cuda.nvfp4.checkpoint``): before each
GEMM the bf16 rows are quantized under the projection's static input scale
``act``. Both quantizers live in ``cuda/nvfp4/act.cu``; one CUDA thread handles
16 inputs of one row, so a row's bytes never depend on another row.

- ``checkpoint.quant4(x, act)`` -> ``Rows4(codes, scales)``: one
  ``quant4_kernel`` launch. Per 16-input block, scale byte
  ``sf8 = e4m3(g * amax / 6)`` with ``g = 1 / act`` (the checkpoint's static
  global scale), codes ``e2m1(x * g / sf)`` to nearest, ties to the even code,
  saturating at 6. ``codes`` is ``uint8 [M, K/2]``, input ``2j`` in byte ``j``'s
  low nibble (the usual packed layout). ``scales`` is ``uint8 [K/64, mpad, 4]``
  with ``mpad = ceil(M/64)*64``: for each 64-input K slice, every row's four
  block scales together; this is ``scale_format="kslice_e4m3"``. The pad rows
  are zero because ``scales`` is allocated with ``torch.zeros`` inside
  ``quant4``: every call issues a ``FillFunctor`` launch over the whole scale
  tensor, then ``quant4_kernel``.
- ``checkpoint.quant8(x, act)`` -> ``uint8 [M, K]``: one ``quant8_kernel``
  launch, ``e4m3(x * (1 / act))`` saturating to +-448, with each 16-byte group
  written in the FP8 weights' fragment order (byte ``4q + j`` holds input
  ``2q + j%2 + 8*(j//2)``) for the ``lane`` / ``gemm_ck`` consumer. The kind
  does not fix the byte order of the output; the order is a permutation inside
  each 16-byte group, so traffic and the scale semantics are the kind's.

Both are called with ``tb=0`` in production: decode/verify (``matmul``,
``matmul_group``) and prefill (``prompt`` with its default ``tile=0`` and
``matmul_group(prompt_rows=True)`` with ``tile=0``) all quantize row-major. The
``tb=128`` tiled variant only feeds the warp-specialized prompt GEMM at
``tile > WS``, which the Qwen3.5-family prefill never selects. TensorFold's
attention output gate is its own ``_gate_mul`` launch before ``quant8``, so
the ``tensorfold_cuda`` backend has no gated form.

The timed closure is the public call, on an input built outside it. For
``quant4`` CUPTI counts every launch of the call (``kernel_name=None``): the
scale-buffer zero fill and ``quant4_kernel``, the fixed sequence production
issues each time. ``quant8`` allocates with ``torch.empty`` and is the single
``quant8_kernel`` launch, counted by name. An independent Torch oracle checks
codes and scales outside timing.
"""

from __future__ import annotations

from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

KSLICE_SCALE_FORMAT = "kslice_e4m3"
QUANT8_KERNEL_NAME = "quant8_kernel"
NVFP4_GROUP = 16
# quant4's scale tensor is [K/64, mpad, 4]: K must hold whole 64-input slices.
NVFP4_K_STEP = 64
# quant8 reads and writes 16 inputs a thread.
FP8_K_STEP = 16
# Both kernels launch one block row per token row (gridDim.y = M), so one call
# takes at most CUDA's 65535 rows; TensorFold's prefill chunks are <= 4096 rows.
MAX_ROWS = 65535
# quant4 pads its scale rows to whole 64-row blocks (checkpoint.quant4: ``tb or 64``).
SCALE_ROW_PAD = 64
_FP4_MAX = 6.0
_E4M3_MAX = 448.0
# Rows checked by the oracle (each row is independent; covers several 64-row
# scale blocks and a partial one).
_CHECK_ROWS = 131
# quant8: the kernel multiplies by fp32(1 / act), the reference divides by
# act; the two fp32 values can differ by an ulp and land on either side of an
# e4m3 rounding boundary. Allow such single-step flips on a tiny fraction.
FP8_MAX_FLIP_FRACTION = 1e-4
_LABEL4 = "nvfp4_quant:tensorfold_cuda"
_LABEL8 = "static_fp8_quant:tensorfold_cuda"
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def validate_nvfp4_args(
    num_tokens: int,
    hidden_size: int,
    group_size: int,
    input_dtype: DType | str,
    scale_format: str,
) -> tuple[int, int]:
    num_tokens, hidden_size = int(num_tokens), int(hidden_size)
    if num_tokens <= 0:
        raise ValueError(f"{_LABEL4} needs num_tokens > 0, got {num_tokens}")
    if num_tokens > MAX_ROWS:
        raise ProfilerNotImplemented(
            f"{_LABEL4} launches one block row a token: at most {MAX_ROWS} rows"
        )
    if int(group_size) != NVFP4_GROUP:
        raise ValueError(f"NVFP4 requires group_size={NVFP4_GROUP}, got {group_size}")
    if DType.from_value(input_dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_LABEL4} quantizes bf16 rows only, got {input_dtype}")
    if scale_format != KSLICE_SCALE_FORMAT:
        raise ProfilerNotImplemented(
            f"{_LABEL4} writes scale_format={KSLICE_SCALE_FORMAT!r} only, got {scale_format!r}"
        )
    if hidden_size < NVFP4_K_STEP or hidden_size % NVFP4_K_STEP:
        raise ProfilerNotImplemented(
            f"{_LABEL4} needs hidden_size a positive multiple of {NVFP4_K_STEP}, got {hidden_size}"
        )
    return num_tokens, hidden_size


def validate_fp8_args(
    num_tokens: int, hidden_dim: int, gated: bool, dtype: DType | str
) -> tuple[int, int]:
    num_tokens, hidden_dim = int(num_tokens), int(hidden_dim)
    if num_tokens <= 0:
        raise ValueError(f"{_LABEL8} needs num_tokens > 0, got {num_tokens}")
    if num_tokens > MAX_ROWS:
        raise ProfilerNotImplemented(
            f"{_LABEL8} launches one block row a token: at most {MAX_ROWS} rows"
        )
    if DType.from_value(dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_LABEL8} quantizes bf16 rows only, got {dtype}")
    if gated:
        # TensorFold multiplies by the attention gate in its own _gate_mul launch.
        raise ProfilerNotImplemented(f"{_LABEL8} has no gated form")
    if hidden_dim < FP8_K_STEP or hidden_dim % FP8_K_STEP:
        raise ProfilerNotImplemented(
            f"{_LABEL8} needs hidden_dim a positive multiple of {FP8_K_STEP}, got {hidden_dim}"
        )
    return num_tokens, hidden_dim


def nvfp4_logical_bytes(num_tokens: int, hidden_size: int) -> int:
    """bf16 read, packed FP4 written, and the whole E4M3 scale tensor.

    The timed call zero-fills the scale tensor padded to whole 64-row blocks,
    then the quantizer writes the real rows' scales, so the scale term counts
    the padded tensor once, as swizzled_e4m3 does.
    """
    scale_rows = -(-num_tokens // SCALE_ROW_PAD) * SCALE_ROW_PAD
    return (
        num_tokens * hidden_size * 2
        + num_tokens * hidden_size // 2
        + scale_rows * hidden_size // NVFP4_GROUP
    )


def fp8_logical_bytes(num_tokens: int, hidden_dim: int) -> int:
    """The kind's formula with gated=False: bf16 read, one e4m3 byte written."""
    return 3 * num_tokens * hidden_dim


def _load_checkpoint(label: str) -> tuple[Any, Any]:
    try:
        import torch
        from tensorfold.cuda.nvfp4 import checkpoint
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"{label} requires the TensorFold environment (tensorfold_env)"
        ) from exc
    return torch, checkpoint


def _rows(torch: Any, m: int, k: int, seed: int) -> Any:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", generator=gen, dtype=torch.float32)
    # A few large entries a row so blocks see a spread of amax values and some
    # blocks are far below the tensor's amax (small block scales).
    x[:, ::97] *= 8.0
    return x.to(torch.bfloat16)


# ---------------------------------------------------------------- oracles


def nvfp4_reference(torch: Any, x: Any, act: float) -> tuple[Any, Any]:
    """Static-global-scale NVFP4 in Torch: (codes uint8 [M, K/2], scale bytes uint8 [M, K/16]).

    Per 16-input block: sf = e4m3(g * (amax / 6)), g = fp32(1 / act); each input
    times g / sf, rounded to the nearest e2m1 magnitude (ties to even code).
    """
    m, k = x.shape
    g = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(act, dtype=torch.float32)
    g = g.to(x.device)
    blocks = x.float().reshape(m, k // NVFP4_GROUP, NVFP4_GROUP)
    amax = blocks.abs().amax(-1)
    sf8 = (g * (amax * torch.tensor(1.0 / 6.0, dtype=torch.float32, device=x.device))).clamp(
        max=_E4M3_MAX
    )
    sf8 = sf8.to(torch.float8_e4m3fn)
    sf = sf8.float()
    mul = torch.where(sf != 0, g / sf, torch.zeros_like(sf))
    v = blocks * mul.unsqueeze(-1)
    mag = v.abs()
    # Nearest e2m1 magnitude; at a midpoint the even code wins (0, 1.0, 2.0, 4.0).
    code = torch.zeros_like(mag, dtype=torch.int64)
    bounds = (
        (0.25, True),
        (0.75, False),
        (1.25, True),
        (1.75, False),
        (2.5, True),
        (3.5, False),
        (5.0, True),
    )
    for i, (edge, tie_down) in enumerate(bounds):
        above = mag > edge if tie_down else mag >= edge
        code = torch.where(above, torch.full_like(code, i + 1), code)
    code = torch.where((v < 0) & (code != 0), code | 8, code).to(torch.uint8).reshape(m, k)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    return packed, sf8.view(torch.uint8)


def kslice_to_rows(scales: Any, m: int) -> Any:
    """quant4's [K/64, mpad, 4] scale bytes -> row-major [M, K/16]."""
    slices, mpad, four = scales.shape
    return scales.permute(1, 0, 2).reshape(mpad, slices * four)[:m]


def fp8_logical_order(torch: Any, stored: Any) -> Any:
    """Undo quant8's fragment order.

    Byte 4q + j of a 16-byte group holds input 2q + j%2 + 8(j//2).
    """
    perm = [2 * (p // 4) + (p % 4) % 2 + 8 * ((p % 4) // 2) for p in range(16)]
    rows, k = stored.shape
    logical = torch.empty_like(stored)
    idx = torch.tensor(perm, device=stored.device)
    logical.view(rows, k // 16, 16)[:, :, idx] = stored.view(rows, k // 16, 16)
    return logical


def check_quant4(torch: Any, checkpoint: Any, k: int) -> dict[str, float]:
    x = _rows(torch, _CHECK_ROWS, k, seed=1)
    act = float(x.float().abs().amax()) / (_FP4_MAX * _E4M3_MAX)
    codes, scales = checkpoint.quant4(x, act)
    torch.cuda.synchronize()
    mpad = -(-_CHECK_ROWS // SCALE_ROW_PAD) * SCALE_ROW_PAD
    if tuple(codes.shape) != (_CHECK_ROWS, k // 2) or tuple(scales.shape) != (k // 64, mpad, 4):
        raise AssertionError(
            f"{_LABEL4} returned codes {tuple(codes.shape)} scales {tuple(scales.shape)}"
        )
    want_codes, want_sf = nvfp4_reference(torch, x, act)
    got_sf = kslice_to_rows(scales, _CHECK_ROWS)
    sf_bad = int((got_sf != want_sf).sum())
    code_bad = int((codes != want_codes).sum())
    pad_bad = int(scales[:, _CHECK_ROWS:, :].count_nonzero())
    if sf_bad or code_bad or pad_bad:
        raise AssertionError(
            f"{_LABEL4} differs from the Torch oracle: {sf_bad} scale bytes, "
            f"{code_bad} code bytes, "
            f"{pad_bad} nonzero pad scales"
        )
    return {"scale_mismatch": 0.0, "code_mismatch": 0.0}


def check_quant8(torch: Any, checkpoint: Any, k: int) -> dict[str, float]:
    from profiling.runners.elementwise.static_fp8_quant_reference import static_fp8_quant_reference

    x = _rows(torch, _CHECK_ROWS, k, seed=2)
    # A scale below amax / 448 so the largest entries saturate.
    act = 0.8 * float(x.float().abs().amax()) / _E4M3_MAX
    out = checkpoint.quant8(x, act)
    torch.cuda.synchronize()
    if tuple(out.shape) != (_CHECK_ROWS, k) or out.dtype != torch.uint8:
        raise AssertionError(f"{_LABEL8} returned {tuple(out.shape)} {out.dtype}")
    got = fp8_logical_order(torch, out)
    scale = torch.tensor(act, dtype=torch.float32, device="cuda")
    want = static_fp8_quant_reference(torch, x, None, scale).view(torch.uint8)
    diff = got != want
    flips = int(diff.sum())
    if flips:
        # Only a one-step e4m3 rounding flip is allowed: adjacent byte, same sign.
        step = (got.to(torch.int16) - want.to(torch.int16)).abs()[diff]
        if int(step.max()) != 1 or flips > FP8_MAX_FLIP_FRACTION * got.numel():
            raise AssertionError(
                f"{_LABEL8} differs from the Torch oracle on {flips} bytes "
                f"(max step {int(step.max())})"
            )
    return {"flip_fraction": flips / got.numel()}


# ---------------------------------------------------------------- runners


def _metrics(time_ms: float, energy_j: float, logical_bytes: int) -> ComputeMetrics:
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(logical_bytes / seconds / 1e9) if time_ms > 0 else 0.0,
        energy_j=float(energy_j),
    )


def profile_nvfp4_quant_tensorfold_cuda(
    num_tokens: int,
    hidden_size: int,
    group_size: int,
    input_dtype: DType | str,
    scale_format: str,
) -> ComputeMetrics:
    m, k = validate_nvfp4_args(num_tokens, hidden_size, group_size, input_dtype, scale_format)
    torch, checkpoint = _load_checkpoint(_LABEL4)
    try:
        # Also absorbs the extension's JIT build.
        check_quant4(torch, checkpoint, k)
        x = _rows(torch, m, k, seed=0)
        act = float(x.float().abs().amax()) / (_FP4_MAX * _E4M3_MAX)

        def kernel() -> Any:
            return checkpoint.quant4(x, act)

        # Compound call: the scale buffer's zero fill, then quant4_kernel.
        time_ms = Timer.cupti(kernel, kernel_name=None)
        energy_j = Energy.perf(kernel, per_iter_time_ms=time_ms)
        return _metrics(time_ms, energy_j, nvfp4_logical_bytes(m, k))
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_LABEL4} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError, AssertionError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_LABEL4} failed: {exc}") from exc


def profile_static_fp8_quant_tensorfold_cuda(
    num_tokens: int,
    hidden_dim: int,
    gated: bool,
    dtype: DType | str,
) -> ComputeMetrics:
    m, k = validate_fp8_args(num_tokens, hidden_dim, gated, dtype)
    torch, checkpoint = _load_checkpoint(_LABEL8)
    try:
        check_quant8(torch, checkpoint, k)
        x = _rows(torch, m, k, seed=0)
        act = float(x.float().abs().amax()) / _E4M3_MAX

        def kernel() -> Any:
            return checkpoint.quant8(x, act)

        time_ms = Timer.cupti(kernel, kernel_name=QUANT8_KERNEL_NAME)
        energy_j = Energy.perf(kernel, per_iter_time_ms=time_ms)
        return _metrics(time_ms, energy_j, fp8_logical_bytes(m, k))
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_LABEL8} ran out of CUDA memory") from exc
    except (ProfilerNotImplemented, ValueError, AssertionError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_LABEL8} failed: {exc}") from exc
