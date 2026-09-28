"""Profile DeepSeek V4.1's Engram row lookup (``_engram_lookup_kernel``).

``vllm_triton`` times one call of the fork's
``ParallelEngramEmbedding.lookup`` (``common/engram.py``), bound to a minimal
stand-in for the module so the fork's own grid rule and launch run unchanged:
``grid = min(ceil(rows / 16), SMs // 2 if background else SMs)``. Production
passes ``background=True`` exactly when the table is CPU-offloaded
(``nvidia/engram.py`` ``_start_prefetch``), so ``residency`` selects both the
table placement and the grid:

- ``host_uva``: the FP8 table and its UE8M0 scale table are allocated like
  production (``torch.empty(..., pin_memory=True)``) and read through the
  fork's UVA device view (``get_accelerator_view_from_cpu_tensor``).
- ``device``: both tables in HBM (``cpu_offload=False``), full-SM grid.

The table is ``table_rows`` long, so the random gather sees the production
TLB footprint (23.6 GiB per rank per layer on V4.1-Flash TP4). It is filled
once per worker process and reused across specs with the same table key
(allocating and pinning tens of GiB costs seconds); a new key releases the
old table first and the pinned cache is emptied at exit.

Hash ids are uniform over each local head's bucket range (production heads own
disjoint, equal-sized prime ranges, and every id a rank sees at its own head
columns falls inside its slice). Each timed launch reads a different id set
from a pool of at least ~1M distinct rows, so repeats do not hit rows an
earlier launch left in L2 or the TLBs.

Correctness (outside timing): the gathered BF16 rows must equal, bit for bit,
a Torch ``index_select`` from the table followed by FP8 -> FP32, times
``2 ** (scale - 127)`` per 32-value block, rounded to BF16 (both steps are
exact in FP32, so the only rounding is the shared final one). Ids outside the
rank's slice must produce zero rows.
"""

from __future__ import annotations

import atexit
from types import SimpleNamespace
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

KIND = "engram_lookup"
_BACKEND = f"{KIND}:vllm_triton"
_GPU_NAME = "NVIDIA B200"
_KERNEL_NAME = "_engram_lookup_kernel"
_HEAD_DIM = 256
_QUANT_BLOCK = 32
_RESIDENCIES = ("host_uva", "device")
_MAX_TOKENS = 65_536
# (max_ngram - 1) * n_heads on V4.1-Flash; a rank never owns more columns.
_MAX_LOCAL_HEADS = 24
# 200M rows x 264 B = 49 GiB: above the TP1 slice of one V4.1-Flash layer
# (384M rows would be TP1; its 94 GiB table is outside what a profiling job pins).
_MAX_TABLE_ROWS = 200_000_000
# Distinct rows the timing pool spans (~1 GB of rows at 264 B), well past L2.
_POOL_ROWS = 4 << 20
_MAX_POOL = 4096
_FILL_CHUNK_ROWS = 1 << 22
# Scale exponents 2**-10 .. 2**7 keep 448 * scale finite and normal in BF16.
_SCALE_LO, _SCALE_HI = 117, 135

_TABLE: dict[str, Any] = {}


def _validate_args(
    num_tokens: int,
    local_heads: int,
    head_dim: int,
    quant_block_size: int,
    table_rows: int,
    residency: str,
    weight_dtype: object,
) -> None:
    if type(num_tokens) is not int or not 1 <= num_tokens <= _MAX_TOKENS:
        raise ValueError(f"num_tokens must be an integer in [1, {_MAX_TOKENS}]")
    if type(local_heads) is not int or not 1 <= local_heads <= _MAX_LOCAL_HEADS:
        raise ValueError(f"local_heads must be an integer in [1, {_MAX_LOCAL_HEADS}]")
    if type(table_rows) is not int or not local_heads <= table_rows <= _MAX_TABLE_ROWS:
        raise ValueError(f"table_rows must be an integer in [local_heads, {_MAX_TABLE_ROWS}]")
    if (head_dim, quant_block_size) != (_HEAD_DIM, _QUANT_BLOCK):
        raise ProfilerNotImplemented(
            f"{_BACKEND} supports head_dim={_HEAD_DIM}, quant_block_size={_QUANT_BLOCK}"
        )
    if residency not in _RESIDENCIES:
        raise ValueError(f"residency must be one of {_RESIDENCIES}")
    if str(weight_dtype) != "fp8_e4m3":
        raise ProfilerNotImplemented(f"{_BACKEND} supports fp8_e4m3 tables")


def row_bytes(head_dim: int, quant_block_size: int) -> int:
    """Table bytes one gathered row reads: FP8 data plus its UE8M0 scales."""
    return head_dim + head_dim // quant_block_size


def logical_bytes(num_tokens: int, local_heads: int, head_dim: int, quant_block_size: int) -> int:
    """Hash ids read + table rows read + BF16 rows written."""
    rows = num_tokens * local_heads
    return rows * (4 + row_bytes(head_dim, quant_block_size) + 2 * head_dim)


def head_ranges(table_rows: int, local_heads: int) -> list[tuple[int, int]]:
    """Equal, disjoint [lo, hi) bucket ranges, one per local head."""
    bounds = [table_rows * h // local_heads for h in range(local_heads + 1)]
    return list(zip(bounds[:-1], bounds[1:]))


def reference_rows(torch: Any, weight_u8: Any, scales_u8: Any, ids: Any, quant_block: int) -> Any:
    """Torch lookup: gather, FP8 -> FP32, times 2**(e - 127) per block, BF16.

    ``ids`` is [tokens, heads] int64 on the table's device; ids outside
    ``[0, rows)`` yield zero rows, as the kernel's ownership mask does.
    """
    rows = weight_u8.shape[0]
    flat = ids.reshape(-1)
    owned = (flat >= 0) & (flat < rows)
    local = torch.where(owned, flat, torch.zeros_like(flat))
    values = weight_u8.index_select(0, local).view(torch.float8_e4m3fn).float()
    exps = scales_u8.index_select(0, local).to(torch.int32)
    scale = (exps << 23).view(torch.float32)
    out = values.view(values.shape[0], -1, quant_block) * scale[:, :, None]
    out = out.reshape(values.shape) * owned[:, None]
    return out.to(torch.bfloat16).view(*ids.shape, weight_u8.shape[1])


def _release_table() -> None:
    if not _TABLE:
        return
    import torch

    _TABLE.clear()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch._C._host_emptyCache()


atexit.register(_release_table)


def _fill_table(torch: Any, device: Any, weight: Any, scales: Any) -> None:
    """Random FP8 codes (NaN codes zeroed) and bounded UE8M0 exponents."""
    generator = torch.Generator(device=device).manual_seed(0xE9A3)
    rows, dim = weight.shape
    for start in range(0, rows, _FILL_CHUNK_ROWS):
        end = min(rows, start + _FILL_CHUNK_ROWS)
        codes = torch.randint(
            0, 256, (end - start, dim), dtype=torch.uint8, device=device, generator=generator
        )
        codes.masked_fill_((codes & 0x7F) == 0x7F, 0)
        weight[start:end].copy_(codes)
        exps = torch.randint(
            _SCALE_LO,
            _SCALE_HI + 1,
            (end - start, scales.shape[1]),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )
        scales[start:end].copy_(exps)
    torch.cuda.synchronize(device)


def _get_table(torch: Any, device: Any, table_rows: int, residency: str) -> dict[str, Any]:
    key = f"{residency}:{table_rows}"
    if _TABLE.get("key") == key:
        return _TABLE
    _release_table()
    host = residency == "host_uva"
    kwargs = dict(device="cpu", pin_memory=True) if host else dict(device=device)
    weight = torch.empty(table_rows, _HEAD_DIM, dtype=torch.uint8, **kwargs)
    scales = torch.empty(table_rows, _HEAD_DIM // _QUANT_BLOCK, dtype=torch.uint8, **kwargs)
    _fill_table(torch, device, weight, scales)
    if host:
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        weight_view = get_accelerator_view_from_cpu_tensor(weight.view(torch.float8_e4m3fn))
        scale_view = get_accelerator_view_from_cpu_tensor(scales)
    else:
        weight_view, scale_view = weight.view(torch.float8_e4m3fn), scales
    _TABLE.update(key=key, weight=weight, scales=scales, views=(weight_view, scale_view))
    return _TABLE


def build_id_pool(
    torch: Any, device: Any, num_tokens: int, local_heads: int, table_rows: int
) -> Any:
    """[pool, tokens, heads] int32 ids, uniform within each head's range."""
    pool = max(4, min(_MAX_POOL, -(-_POOL_ROWS // (num_tokens * local_heads))))
    ranges = head_ranges(table_rows, local_heads)
    lo = torch.tensor([r[0] for r in ranges], dtype=torch.int64, device=device)
    size = torch.tensor([r[1] - r[0] for r in ranges], dtype=torch.int64, device=device)
    generator = torch.Generator(device=device).manual_seed(0x5EED)
    raw = torch.randint(
        0, 1 << 62, (pool, num_tokens, local_heads), device=device, generator=generator
    )
    return (raw % size + lo).to(torch.int32)


def _lookup_module(torch: Any, device: Any, views: tuple[Any, Any], local_heads: int, rows: int):
    """The attributes ``ParallelEngramEmbedding.lookup`` reads, for rank 0's slice."""
    return SimpleNamespace(
        part_n_hash_cols=local_heads,
        n_hash_cols=local_heads,
        head_start=0,
        vocab_start_idx=0,
        vocab_end_idx=rows,
        dim=_HEAD_DIM,
        block_size=_QUANT_BLOCK,
        _num_sms=torch.cuda.get_device_properties(device).multi_processor_count,
        _storage=lambda: views,
    )


def check_lookup(torch: Any, lookup: Any, table: dict[str, Any], ids: Any, out: Any) -> dict:
    """Bit-exact comparison against the Torch reference, plus unowned ids -> 0."""
    lookup(ids, out)
    torch.cuda.synchronize()
    weight, scales = table["weight"], table["scales"]
    ref = reference_rows(torch, weight, scales, ids.to(weight.device).long(), _QUANT_BLOCK)
    got = out.to(weight.device)
    mismatches = int((got.view(torch.int16) != ref.view(torch.int16)).sum().item())
    probe = ids[:1].clone()
    probe[0, 0] = -1
    probe[0, -1] = weight.shape[0]
    probe_out = torch.full_like(out[:1], float("nan"))
    lookup(probe, probe_out)
    torch.cuda.synchronize()
    unowned_zero = bool(
        (probe_out[0, 0] == 0).all().item() and (probe_out[0, -1] == 0).all().item()
    )
    report = {"mismatched_values": mismatches, "unowned_rows_zero": unowned_zero}
    if mismatches or not unowned_zero:
        raise KernelLaunchFailed(f"{_BACKEND} rows differ from the Torch lookup: {report}")
    return report


def profile_engram_lookup_vllm_triton(
    num_tokens: int,
    local_heads: int,
    head_dim: int,
    quant_block_size: int,
    table_rows: int,
    residency: str,
    weight_dtype: object,
) -> ComputeMetrics:
    """Time one Engram lookup launch (fork Triton kernel) on a B200."""
    _validate_args(
        num_tokens, local_heads, head_dim, quant_block_size, table_rows, residency, weight_dtype
    )
    try:
        import torch
        import vllm._C_stable_libtorch  # noqa: F401  (get_cuda_view_from_cpu_tensor)
        from vllm.models.deepseek_v41.common.engram import ParallelEngramEmbedding
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_BACKEND} requires the upstream-rebased vLLM fork (vllm_upstream_fork_env)") from exc
    try:
        if not torch.cuda.is_available():
            raise ProfilerNotImplemented(f"{_BACKEND} requires CUDA")
        device = torch.device("cuda", torch.cuda.current_device())
        name = str(torch.cuda.get_device_name(device))
        if name != _GPU_NAME:
            raise ProfilerNotImplemented(f"{_BACKEND} requires {_GPU_NAME}, got {name}")
        table = _get_table(torch, device, table_rows, residency)
        module = _lookup_module(torch, device, table["views"], local_heads, table_rows)
        background = residency == "host_uva"

        def lookup(ids: Any, out: Any) -> None:
            ParallelEngramEmbedding.lookup(module, ids, out, background=background)

        pool = build_id_pool(torch, device, num_tokens, local_heads, table_rows)
        out = torch.empty((num_tokens, local_heads, _HEAD_DIM), dtype=torch.bfloat16, device=device)
        check_lookup(torch, lookup, table, pool[0], out)
        cursor = [0]

        def launch() -> None:
            index = cursor[0]
            cursor[0] = index + 1 if index + 1 < pool.shape[0] else 0
            lookup(pool[index], out)

        time_ms = Timer.cupti(launch, kernel_name=_KERNEL_NAME)
        energy_j = Energy.perf(launch, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        _release_table()
        raise OOMError(f"{_BACKEND} ran out of memory") from exc
    except (KernelLaunchFailed, ProfilerNotImplemented):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(
            logical_bytes(num_tokens, local_heads, head_dim, quant_block_size)
            / (time_ms / 1000.0)
            / 1e9
        ),
        energy_j=float(energy_j),
    )


__all__ = [
    "build_id_pool",
    "head_ranges",
    "logical_bytes",
    "profile_engram_lookup_vllm_triton",
    "reference_rows",
    "row_bytes",
]
