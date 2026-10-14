"""Profile the raw embedding row gather (``embedding_row_gather``).

``vllm_triton`` times the fork's ``Qwen4ExpPLEPinnedHostEmbedding._lookup``
(``models/qwen4_exp/nvidia/ngram_embedding.py``), bound to a minimal stand-in
for the module so its own launch (``_lookup_ple_embedding_from_pinned_kernel``,
one program per id, ``BLOCK_D = next_power_of_2(head_dim)``) runs unchanged.
The table is allocated like production (``torch.empty(..., pin_memory=True)``)
and read through ``get_accelerator_view_from_cpu_tensor``. Only
``residency=host_uva`` exists for this backend: production never runs this
kernel on an HBM table.

``torch`` times ``F.embedding`` on an HBM table, the ``cpu_offload=False``
path (``Qwen4ExpPLEDeviceEmbedding`` -> ``VocabParallelEmbedding`` at TP1 ->
``F.embedding``); ``residency=device`` only.

The table is ``table_rows`` long, so the random gather sees the production
footprint (51.2 GB on the 320M-row PLE table). It is filled once per worker
process and reused across specs with the same key; a new key releases the old
table first, and the pinned cache is emptied at exit.

Ids are int64 and uniform over each head's equal share of the table
(production heads own disjoint prime-sized ranges of nearly equal size). Each
timed call reads the next of 4 to 4096 id sets, as many as bring the pool to
about 4M rows (fewer for calls under 1024 ids), so consecutive calls do not
reuse rows an earlier call left in L2 or the TLBs.

Correctness (outside timing): the gathered rows must equal, byte for byte, a
Torch ``index_select`` of the table; for ``vllm_triton`` ids outside the table
must produce zero rows (the kernel's ownership mask).
"""

from __future__ import annotations

import atexit
from types import SimpleNamespace
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

KIND = "embedding_row_gather"
_TRITON_KERNEL = "_lookup_ple_embedding_from_pinned_kernel"
_BACKEND_RESIDENCY = {"vllm_triton": "host_uva", "torch": "device"}
# Table element dtype -> (bytes per element, torch dtype attribute).
_DTYPES = {"fp8_e4m3": (1, "float8_e4m3fn"), "bf16": (2, "bfloat16")}
_BACKEND_DTYPES = {"vllm_triton": ("fp8_e4m3",), "torch": ("fp8_e4m3", "bf16")}
_RESIDENCIES = ("host_uva", "device")
# 400M rows x 160 B = 64 GB: above the 51.2 GB PLE table, inside what one
# profiling job can pin or keep in HBM.
_MAX_TABLE_BYTES = 64 << 30
_POOL_ROWS = 4 << 20
_MAX_POOL = 4096
_FILL_CHUNK_ROWS = 1 << 24

_TABLE: dict[str, Any] = {}


def validate_args(
    backend: str,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    table_rows: int,
    residency: str,
    weight_dtype: object,
) -> None:
    for name, value in (
        ("num_tokens", num_tokens),
        ("num_heads", num_heads),
        ("head_dim", head_dim),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if type(table_rows) is not int or table_rows < num_heads:
        raise ValueError("table_rows must be an integer >= num_heads")
    if str(weight_dtype) not in _BACKEND_DTYPES[backend]:
        raise ProfilerNotImplemented(
            f"{KIND}:{backend} supports {', '.join(_BACKEND_DTYPES[backend])} tables"
        )
    if table_rows * head_dim * _DTYPES[str(weight_dtype)][0] > _MAX_TABLE_BYTES:
        raise ProfilerNotImplemented(f"tables above {_MAX_TABLE_BYTES >> 30} GiB are not pinned")
    if residency not in _RESIDENCIES:
        raise ValueError(f"residency must be one of {_RESIDENCIES}")
    if residency != _BACKEND_RESIDENCY[backend]:
        raise ProfilerNotImplemented(
            f"{KIND}:{backend} runs only with residency={_BACKEND_RESIDENCY[backend]}"
        )


def logical_bytes(
    num_tokens: int, num_heads: int, head_dim: int, weight_dtype: object = "fp8_e4m3"
) -> int:
    """Int64 ids read + table rows read + rows written (table dtype)."""
    return num_tokens * num_heads * (8 + 2 * head_dim * _DTYPES[str(weight_dtype)][0])


def head_ranges(table_rows: int, num_heads: int) -> list[tuple[int, int]]:
    """Equal, disjoint [lo, hi) id ranges, one per head."""
    bounds = [table_rows * h // num_heads for h in range(num_heads + 1)]
    return list(zip(bounds[:-1], bounds[1:]))


def build_id_pool(torch: Any, device: Any, num_tokens: int, num_heads: int, table_rows: int):
    """[pool, tokens, heads] int64 ids, uniform within each head's range."""
    pool = max(4, min(_MAX_POOL, -(-_POOL_ROWS // (num_tokens * num_heads))))
    ranges = head_ranges(table_rows, num_heads)
    lo = torch.tensor([r[0] for r in ranges], dtype=torch.int64, device=device)
    size = torch.tensor([r[1] - r[0] for r in ranges], dtype=torch.int64, device=device)
    generator = torch.Generator(device=device).manual_seed(0x5EED)
    raw = torch.randint(
        0, 1 << 62, (pool, num_tokens, num_heads), device=device, generator=generator
    )
    return raw % size + lo


def release_table() -> None:
    if not _TABLE:
        return
    import torch

    _TABLE.clear()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch._C._host_emptyCache()


atexit.register(release_table)


def _get_table(
    torch: Any, device: Any, table_rows: int, head_dim: int, residency: str, weight_dtype: str
) -> dict:
    key = f"{residency}:{table_rows}:{head_dim}:{weight_dtype}"
    elem_bytes, dtype_name = _DTYPES[weight_dtype]
    row_bytes = head_dim * elem_bytes
    if _TABLE.get("key") == key:
        return _TABLE
    release_table()
    host = residency == "host_uva"
    kwargs = dict(device="cpu", pin_memory=True) if host else dict(device=device)
    table = torch.empty(table_rows, row_bytes, dtype=torch.uint8, **kwargs)
    generator = torch.Generator(device=device).manual_seed(0xE9A3)
    for start in range(0, table_rows, _FILL_CHUNK_ROWS):
        end = min(table_rows, start + _FILL_CHUNK_ROWS)
        codes = torch.randint(
            0, 256, (end - start, row_bytes), dtype=torch.uint8, device=device, generator=generator
        )
        table[start:end].copy_(codes)
    torch.cuda.synchronize(device)
    _TABLE.update(key=key, table=table.view(getattr(torch, dtype_name)))
    if host:
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        _TABLE["view"] = get_accelerator_view_from_cpu_tensor(_TABLE["table"])
    return _TABLE


def _pinned_lookup(torch: Any, table: dict, head_dim: int):
    """``Qwen4ExpPLEPinnedHostEmbedding._lookup`` bound to a TP1 stand-in."""
    from vllm.models.qwen4_exp.nvidia.ngram_embedding import Qwen4ExpPLEPinnedHostEmbedding
    from vllm.triton_utils import triton

    rows = table["table"].shape[0]
    module = SimpleNamespace(
        embedding_dim=head_dim,
        weight=table["table"],
        _uva_weight=table["view"],
        _block_d=triton.next_power_of_2(head_dim),
        shard_indices=SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=rows),
    )

    def lookup(ids: Any, out: Any) -> None:
        Qwen4ExpPLEPinnedHostEmbedding._lookup(module, ids, output=out)

    return lookup


def check_rows(torch: Any, table: dict, ids: Any, got: Any) -> None:
    ref = (
        table["table"].view(torch.uint8).index_select(0, ids.reshape(-1).to(table["table"].device))
    )
    got = got.contiguous().view(torch.uint8).reshape(-1, ref.shape[1]).to(ref.device)
    mismatches = int((got != ref).sum().item())
    if mismatches:
        raise KernelLaunchFailed(f"{KIND} rows differ from index_select: {mismatches} bytes")


def _profile(backend: str, **args: Any) -> ComputeMetrics:
    validate_args(backend, **args)
    num_tokens, num_heads, head_dim = args["num_tokens"], args["num_heads"], args["head_dim"]
    table_rows, residency = args["table_rows"], args["residency"]
    weight_dtype = str(args["weight_dtype"])
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{KIND}:{backend} requires torch") from exc
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        table = _get_table(torch, device, table_rows, head_dim, residency, weight_dtype)
        pool = build_id_pool(torch, device, num_tokens, num_heads, table_rows)
        out = None  # F.embedding allocates its own output
        if backend == "vllm_triton":
            out = torch.empty(
                (num_tokens, num_heads, head_dim), dtype=table["table"].dtype, device=device
            )
            try:
                import vllm._C_stable_libtorch  # noqa: F401  (get_cuda_view_from_cpu_tensor)

                lookup = _pinned_lookup(torch, table, head_dim)
            except ImportError as exc:
                raise ProfilerNotImplemented(f"{KIND}:{backend} requires the vLLM fork") from exc
            lookup(pool[0], out)
            torch.cuda.synchronize()
            check_rows(torch, table, pool[0], out)
            probe = pool[0][:1].clone()
            probe[0, 0], probe[0, -1] = -1, table_rows
            probe_out = torch.full_like(out[:1].view(torch.uint8), 0xFF).view(out.dtype)
            lookup(probe, probe_out)
            torch.cuda.synchronize()
            raw = probe_out.view(torch.uint8)
            if int(raw[0, 0].sum()) or int(raw[0, -1].sum()):
                raise KernelLaunchFailed(f"{KIND}:{backend} out-of-table ids must give zero rows")
            kernel_name = _TRITON_KERNEL
        else:
            weight = table["table"]
            torch.nn.functional.embedding(pool[0], weight)  # warm the dispatch
            check_rows(torch, table, pool[0], torch.nn.functional.embedding(pool[0], weight))

            def lookup(ids: Any, _out: None) -> None:
                torch.nn.functional.embedding(ids, weight)

            kernel_name = None
        cursor = [0]

        def launch() -> None:
            index = cursor[0]
            cursor[0] = index + 1 if index + 1 < pool.shape[0] else 0
            lookup(pool[index], out)

        time_ms = Timer.cupti(launch, kernel_name=kernel_name)
        energy_j = Energy.perf(launch, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        release_table()
        raise OOMError(f"{KIND}:{backend} ran out of memory") from exc
    except (KernelLaunchFailed, ProfilerNotImplemented):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{KIND}:{backend} failed: {exc}") from exc
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(
            logical_bytes(num_tokens, num_heads, head_dim, weight_dtype) / (time_ms / 1000.0) / 1e9
        ),
        energy_j=float(energy_j),
    )


def profile_embedding_row_gather_vllm_triton(**args: Any) -> ComputeMetrics:
    """Time one pinned-host UVA row gather (fork Triton kernel)."""
    return _profile("vllm_triton", **args)


def profile_embedding_row_gather_torch(**args: Any) -> ComputeMetrics:
    """Time one HBM ``F.embedding`` row gather."""
    return _profile("torch", **args)


__all__ = [
    "build_id_pool",
    "head_ranges",
    "logical_bytes",
    "profile_embedding_row_gather_torch",
    "profile_embedding_row_gather_vllm_triton",
    "validate_args",
]
