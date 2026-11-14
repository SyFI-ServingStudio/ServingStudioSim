"""DeepSeek-V4.1 Engram as SGLang v0.5.23 runs it: the shared host-table lookup and the fused gate.

``engram_lookup:sglang_triton`` -- ``srt/layers/engram.py`` ``EngramEmbedding.forward`` (line 1098),
shared host layout (``SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1``,
``SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=shared``, the default layout): every TP rank maps the whole
layer table and gathers all ``(max_ngram - 1) * n_heads`` hash columns of its tokens itself (no TP
all-reduce). The call is ``kernels/ops/embeddings/engram_gather.py`` ``engram_gather`` (line 52):
with ``sort_ids`` -- ``_sorted_host_reads`` (engram.py:656), true on GPUs whose pageable-memory access
does not use the host page tables, i.e. every x86 PCIe GPU -- one ``torch.sort`` of the ids, then the
Triton ``_engram_gather_kernel`` (one program per row, fp8 row * 2**(e8m0 - 127) -> bf16). The row
times both.

The table is allocated the way ``_HostTable`` (engram.py:684) does for the shared layout: a memfd,
``mmap(MAP_SHARED)``, ``madvise`` before the first touch, ``cudaHostRegister``, then filled. Weight
bytes ``[rows, 256]`` are followed by the e8m0 scale bytes ``[rows, 8]`` in the same buffer. Whether
the kernel hands out huge pages for that shmem mapping depends on
``/sys/kernel/mm/transparent_hugepage/shmem_enabled``; SGLang logs ~10x slower lookups without them,
and the 2026-11-05 4x RTX PRO 6000 run had ``shmem_enabled=never`` (0% huge). So ``residency``
names the page backing:

- ``host_shared_4k``: shmem on 4 KiB pages (``MADV_NOHUGEPAGE``);
- ``host_shared_huge``: shmem on huge pages (``MADV_HUGEPAGE``; needs shmem_enabled advise/always,
  else the runner refuses);
- ``device``: the non-host-table layout's device memory (no sort), for comparison.

The time depends on PCIe and on host memory/IOMMU behavior of the profiling machine, not only on the
GPU: rows are keyed by GPU name, so profile on the host topology you intend to model.

``engram_gate:sglang_triton`` -- ``srt/layers/engram.py`` ``engram_gate`` (line 1253) ->
``kernels/ops/embeddings/engram_gate.py`` ``fused_engram_gate`` (line 49): one Triton
``_engram_gate_kernel`` program per (token, hc copy).

Runs in ``sglang_env``.
"""

from __future__ import annotations

import atexit
import mmap
import os
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.embedding.engram_lookup import (
    build_id_pool,
    logical_bytes,
    reference_rows,
    row_bytes,
)
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_LOOKUP = "engram_lookup:sglang_triton"
_GATE = "engram_gate:sglang_triton"
_HEAD_DIM = 256
_QUANT_BLOCK = 32
_RESIDENCIES = ("host_shared_4k", "host_shared_huge", "device")
_MAX_TOKENS = 65_536
# Shared layout: a rank gathers every hash column, (max_ngram - 1) * n_heads = 24 on V4.1-Flash.
_MAX_HEADS = 24
# A whole V4.1-Flash layer: 384,016,682 rows x 264 B = 101 GB of host memory.
_MAX_TABLE_ROWS = 400_000_000
_CHECK_TOKENS = 256
_FILL_CHUNK_ROWS = 1 << 22
_SCALE_LO, _SCALE_HI = 117, 135
_MADV_HUGEPAGE = getattr(mmap, "MADV_HUGEPAGE", 14)
_MADV_NOHUGEPAGE = getattr(mmap, "MADV_NOHUGEPAGE", 15)
_TABLE: dict[str, Any] = {}


def _validate_lookup(
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
    if type(local_heads) is not int or not 1 <= local_heads <= _MAX_HEADS:
        raise ValueError(f"local_heads must be an integer in [1, {_MAX_HEADS}]")
    if type(table_rows) is not int or not local_heads <= table_rows <= _MAX_TABLE_ROWS:
        raise ValueError(f"table_rows must be an integer in [local_heads, {_MAX_TABLE_ROWS}]")
    if (head_dim, quant_block_size) != (_HEAD_DIM, _QUANT_BLOCK):
        raise ProfilerNotImplemented(f"{_LOOKUP} supports head_dim={_HEAD_DIM}, quant_block_size={_QUANT_BLOCK}")
    if residency not in _RESIDENCIES:
        raise ValueError(f"{_LOOKUP}: residency must be one of {_RESIDENCIES}")
    if str(weight_dtype) != "fp8_e4m3":
        raise ProfilerNotImplemented(f"{_LOOKUP} supports fp8_e4m3 tables")


def _release_table() -> None:
    if not _TABLE:
        return
    import torch

    import gc

    torch.cuda.synchronize()
    host = _TABLE.get("host")
    if host is not None:
        torch.cuda.cudart().cudaHostUnregister(host["bytes"].data_ptr())
    # Drop every tensor view of the mapping before closing it (mmap refuses to close while
    # exported buffers exist).
    _TABLE.clear()
    if host is not None:
        host.pop("bytes")
        gc.collect()
        try:
            host["mm"].close()
        except BufferError:
            pass  # a caller still holds a view; the mapping goes with the process
        os.close(host["fd"])
    torch.cuda.empty_cache()


atexit.register(_release_table)


def _fill(torch: Any, device: Any, weight: Any, scales: Any) -> None:
    """Random FP8 codes (NaN codes zeroed) and bounded UE8M0 exponents, written chunk by chunk."""
    generator = torch.Generator(device=device).manual_seed(0xE9A3)
    rows, dim = weight.shape
    for start in range(0, rows, _FILL_CHUNK_ROWS):
        end = min(rows, start + _FILL_CHUNK_ROWS)
        codes = torch.randint(0, 256, (end - start, dim), dtype=torch.uint8, device=device, generator=generator)
        codes.masked_fill_((codes & 0x7F) == 0x7F, 0)
        weight[start:end].copy_(codes)
        exps = torch.randint(
            _SCALE_LO, _SCALE_HI + 1, (end - start, scales.shape[1]), dtype=torch.uint8, device=device, generator=generator
        )
        scales[start:end].copy_(exps)
    torch.cuda.synchronize(device)


def _get_table(torch: Any, device: Any, table_rows: int, residency: str) -> dict[str, Any]:
    key = f"{residency}:{table_rows}"
    if _TABLE.get("key") == key:
        return _TABLE
    _release_table()
    w_bytes = table_rows * _HEAD_DIM
    s_bytes = table_rows * (_HEAD_DIM // _QUANT_BLOCK)
    if residency == "device":
        raw = torch.empty(w_bytes + s_bytes, dtype=torch.uint8, device=device)
        host = None
    else:
        from sglang.srt.layers.engram import _huge_pages_backing

        nbytes = w_bytes + s_bytes
        fd = os.memfd_create("servingstudio_engram", 0)
        os.ftruncate(fd, nbytes)
        mm = mmap.mmap(fd, nbytes, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        # Advisory before the first touch, as _HostTable does; 4k explicitly opts out.
        mm.madvise(_MADV_HUGEPAGE if residency == "host_shared_huge" else _MADV_NOHUGEPAGE)
        raw = torch.frombuffer(mm, dtype=torch.uint8)
        err = torch.cuda.cudart().cudaHostRegister(raw.data_ptr(), nbytes, 0)
        if int(err) != 0:
            raw = None
            mm.close()
            os.close(fd)
            raise KernelLaunchFailed(f"{_LOOKUP}: cudaHostRegister({nbytes} bytes) failed: {err}")
        host = {"fd": fd, "mm": mm, "bytes": raw}
    weight = raw[:w_bytes].view(table_rows, _HEAD_DIM)
    scales = raw[w_bytes:].view(table_rows, _HEAD_DIM // _QUANT_BLOCK)
    _TABLE.update(key=key, host=host, weight=weight, scales=scales)
    _fill(torch, device, weight, scales)
    if host is not None:
        mapped_kb, huge_kb = _huge_pages_backing(raw.data_ptr())
        del raw, weight, scales  # the table holds its own views; let a release close the mapping
        _TABLE["huge_fraction"] = huge_kb / mapped_kb if mapped_kb else 0.0
        if residency == "host_shared_huge" and _TABLE["huge_fraction"] < 0.98:
            frac = _TABLE["huge_fraction"]
            _release_table()
            raise ProfilerNotImplemented(
                f"{_LOOKUP}: host_shared_huge needs shmem huge pages, got {frac:.0%} "
                "(set /sys/kernel/mm/transparent_hugepage/shmem_enabled to advise)"
            )
        if residency == "host_shared_4k" and huge_kb:
            _release_table()
            raise KernelLaunchFailed(f"{_LOOKUP}: host_shared_4k table came back partly on huge pages")
    return _TABLE


def profile_engram_lookup_sglang_triton(
    num_tokens: int,
    local_heads: int,
    head_dim: int,
    quant_block_size: int,
    table_rows: int,
    residency: str,
    weight_dtype: object,
) -> ComputeMetrics:
    """Time one SGLang ``engram_gather`` call (sort + gather) on a shared host table."""
    _validate_lookup(num_tokens, local_heads, head_dim, quant_block_size, table_rows, residency, weight_dtype)
    try:
        import torch
        from sglang.kernels.ops.embeddings.engram_gather import engram_gather
        from sglang.srt.layers.engram import _sorted_host_reads
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_LOOKUP} requires the SGLang environment (sglang_env)") from exc
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        table = _get_table(torch, device, table_rows, residency)
        weight, scales = table["weight"], table["scales"]
        w_ptr, s_ptr = weight.data_ptr(), scales.data_ptr()
        sort_ids = residency != "device" and _sorted_host_reads(device.index)
        # SGLang's hash ids are int64 (EngramHasher.new_hash_ids).
        pool = build_id_pool(torch, device, num_tokens, local_heads, table_rows).long()
        out = torch.empty((num_tokens * local_heads, _HEAD_DIM), dtype=torch.bfloat16, device=device)

        def lookup(ids: Any) -> None:
            engram_gather(w_ptr, s_ptr, ids.reshape(-1), out, _HEAD_DIM, _QUANT_BLOCK, sort_ids=sort_ids)

        # Bit-exact check on the first tokens of one pool entry (CPU index_select for a host table).
        lookup(pool[0])
        torch.cuda.synchronize()
        n_check = min(num_tokens, _CHECK_TOKENS)
        ids = pool[0][:n_check]
        ref = reference_rows(torch, weight, scales, ids.to(weight.device), _QUANT_BLOCK)
        got = out[: n_check * local_heads].view(n_check, local_heads, _HEAD_DIM).to(weight.device)
        mismatches = int((got.view(torch.int16) != ref.view(torch.int16)).sum().item())
        if mismatches:
            raise KernelLaunchFailed(f"{_LOOKUP} rows differ from the Torch lookup: {mismatches} values")
        cursor = [0]

        def launch() -> None:
            index = cursor[0]
            cursor[0] = index + 1 if index + 1 < pool.shape[0] else 0
            lookup(pool[index])

        # The sort's launches are part of SGLang's call: count every launch.
        time_ms = Timer.cupti(launch)
        energy_j = Energy.perf(launch, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        _release_table()
        raise OOMError(f"{_LOOKUP} ran out of memory") from exc
    except (KernelLaunchFailed, ProfilerNotImplemented, ValueError):
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_LOOKUP} failed: {exc}") from exc
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(
            logical_bytes(num_tokens, local_heads, head_dim, quant_block_size) / (time_ms / 1000.0) / 1e9
        ),
        energy_j=float(energy_j),
    )


def profile_engram_gate_sglang_triton(
    num_tokens: int, hidden_size: int, hc_mult: int, hidden_dtype: object
) -> ComputeMetrics:
    """Time one SGLang ``fused_engram_gate`` launch."""
    if str(hidden_dtype) != "bf16":
        raise ValueError(f"{_GATE} requires hidden_dtype=bf16, got {hidden_dtype!r}")
    if type(num_tokens) is not int or not 1 <= num_tokens <= _MAX_TOKENS:
        raise ValueError(f"{_GATE}: num_tokens must be an integer in [1, {_MAX_TOKENS}]")
    if type(hidden_size) is not int or hidden_size < 1 or type(hc_mult) is not int or hc_mult < 1:
        raise ValueError(f"{_GATE}: hidden_size and hc_mult must be positive integers")
    try:
        import torch
        from sglang.kernels.ops.embeddings.engram_gate import fused_engram_gate
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{_GATE} requires the SGLang environment (sglang_env)") from exc
    # V4.1-Flash: Engram(...).eps = rms_norm_eps, clamp_value 1e-6 (engram.py Engram.__init__).
    eps, clamp = 1e-20, 1e-6
    try:
        g = torch.Generator(device="cuda").manual_seed(0)
        t, hc, d = num_tokens, hc_mult, hidden_size
        x = torch.randn(t, hc, d, device="cuda", dtype=torch.bfloat16, generator=g)
        kv = torch.randn(t, (hc + 1) * d, device="cuda", dtype=torch.bfloat16, generator=g)
        q_w = (torch.rand(hc, d, device="cuda", generator=g) + 0.5).to(torch.bfloat16)
        k_w = (torch.rand(hc, d, device="cuda", generator=g) + 0.5).to(torch.bfloat16)

        def call() -> Any:
            return fused_engram_gate(x, kv, q_w, k_w, eps, clamp)

        out = call()
        torch.cuda.synchronize()
        n = min(t, _CHECK_TOKENS)
        key, value = kv[:n].split([hc * d, d], dim=-1)
        key = key.float().unflatten(-1, (hc, d))
        h = x[:n].float()
        weight = q_w.float() * k_w.float()
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (h * weight * key).sum(-1) * rstd * d**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp).sqrt(), dot))
        ref = h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)
        err = ((out[:n].float() - ref).norm() / ref.norm().clamp_min(1e-30)).item()
        if not err <= 1e-2:
            raise KernelLaunchFailed(f"{_GATE} output mismatch: relative Frobenius error {err:.3e}")
        time_ms = Timer.cupti(call, warmup=3)
        energy_j = Energy.perf(call, per_iter_time_ms=time_ms)
    except (KernelLaunchFailed, ProfilerNotImplemented, ValueError):
        raise
    except Exception as exc:
        if exc.__class__.__name__ == "OutOfMemoryError":
            raise OOMError(f"{_GATE} ran out of memory") from exc
        raise KernelLaunchFailed(f"{_GATE} failed: {exc}") from exc
    # x read + kv read + out write, bf16.
    logical = 2 * (t * hc * d + t * (hc + 1) * d + t * hc * d)
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=0.0,
        memory_bandwidth_gbps=float(logical / (time_ms / 1000.0) / 1e9),
        energy_j=float(energy_j),
    )


__all__ = ["profile_engram_gate_sglang_triton", "profile_engram_lookup_sglang_triton", "row_bytes"]
