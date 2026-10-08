"""Shared machinery for the MI300X rope-free BF16 sparse-MLA attention backend.

Both the decode (``dsa_sparse_mla_attention``) and prefill
(``dsa_sparse_mla_prefill``) kinds of GLM-5.3-Flash's DSA sparse attention run,
on ROCm, through the *same* vLLM Triton entry point
``vllm.v1.attention.ops.rocm_aiter_mla_sparse.rocm_sparse_attn_prefill`` -- the
ragged kernel ``_sparse_attn_prefill_ragged_kernel``. This module holds the parts
they share: loading the entry point, building rope-free BF16 operands and the
ragged index structure, the Torch reference for correctness, the GPU-identity
guard, and the Triton-path (anti-fallback) assertion.

Why one callable for both phases. The separate decode entry point
``rocm_sparse_attn_decode`` hard-asserts a 448+64 NoPE/RoPE split
(``_validate_dsv4_sparse_dims``) and a uint8 ``fp8_ds_mla`` cache, so it cannot
serve GLM's rope-free 512-wide BF16 latent. ``rocm_sparse_attn_prefill`` validates
only ``head_dim == nope_head_dim + rope_head_dim`` (``_validate_sparse_dims``), so
``head_dim=512, nope_head_dim=512, rope_head_dim=0`` is accepted; a decode step is
just a ragged batch whose every request contributes a single query row. The
backend-level gate ``_use_rocm_sparse_triton`` selects exactly this path for a
non-FP8 cache with ``head_size == kv_lora_rank`` (rope-free). Signatures confirmed
by in-container ``inspect.signature`` on the pinned image (vLLM v0.3.1.dev190,
torch 2.12, ROCm 7.2).

The AITER "opus" sparse-prefill fast path inside the entry point is gfx950-only
(``_can_use_aiter_sparse_prefill_opus`` requires ``on_gfx950``); on MI300X
(gfx942) it is skipped and the Triton ragged kernel runs. We still assert the
Triton kernel is present in the capture -- and no AITER opus kernel is -- before
reporting any number, so a silent fall-through can never be mislabeled.

Timing is kernel-only via rocprofv3, the ROCm counterpart of the B200 flashinfer
backend's CUPTI path. The call fans out to several dispatches (ragged attention
plus any candidate-mask / reduce launches); as with ``kda_chunk_prefill`` these
are summed per launch with the autotune-robust trailing fold
``dispatches_per_launch=D``. ``D`` is pinned from the first real MI300X capture
(``_DISPATCHES_PER_LAUNCH``); until it is set the profiler raises
``ProfilerNotImplemented`` rather than guessing a time.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from profiling.runners.attention.kda_recurrent_decode_torch_rocm import (
    _device_arch,
    _device_name,
    _is_mi300_series,
)
from profiling.runners.exceptions import ProfilerNotImplemented

# The rope-free MLA geometry GLM-5.3-Flash uses on ROCm: a 512-wide BF16 latent
# carrying the whole head (no separate RoPE part), value width 512, one MQA KV
# head. These are the coordinate the MI300X arch branch composes; they must line
# up with the kind's reused args struct.
#
# selected_k is the width of the sparse page table the attention call indexes
# into. It is NOT the raw index top-k: the model's index_topk is 2048, but vLLM
# allocates a kpool buffer round_up(index_topk + index_kpool - 1, 128) wide, so
# with index_kpool=4 the page table is round_up(2051, 128) = 2176. The arch
# (simulator/src/arch/glm53_flash_vllm_fp8_kda_dsa_moe.rs `SELECTED_K`) composes
# the 2176 page-table width, so that is what the sweep grid enumerates. We accept
# both the raw 2048 (index_topk, used by other DSA layouts) and the 2176 kpool
# page-table width, matching the B200 runner's `_TRTLLM_FP8_NOPE_SELECTED_K`. The
# padded width only sizes the host-side -1-padded index buffer; the Triton ragged
# kernel consumes the packed valid indices, so a wider table adds padding, not a
# tile constraint.
HEAD_DIM = 512
NOPE_HEAD_DIM = 512
ROPE_HEAD_DIM = 0
VALUE_DIM = 512
NUM_KV_HEADS = 1
# Raw index top-k, kept for provenance; `ALLOWED_SELECTED_K` is what validators use.
SELECTED_K = 2048
ALLOWED_SELECTED_K = frozenset({2048, 2176})
SOFTMAX_SCALE = 0.0625
# A rope-free latent-only cache row; distinct from the B200 fp8 layouts and from
# the H200 BF16 latent+rope layout, so a MI300X row never collides with them.
CACHE_LAYOUT = "token_major_mqa_bf16_latent"

_MODULE = "vllm.v1.attention.ops.rocm_aiter_mla_sparse"
_CALLABLE = "rocm_sparse_attn_prefill"
_RAGGED_BUILDER = "build_ragged_indices_from_dense"
# The @triton.jit ragged-prefill kernel that MUST appear in the capture: proof
# the rope-free Triton path ran. The AITER opus kernel that MUST be absent (it is
# gfx950-only, so its presence would mean a wrong device or a mislabeled run).
TRITON_KERNEL_SUBSTR = "sparse_attn_prefill_ragged"
AITER_OPUS_SUBSTR = "pa_sparse_prefill_opus"

GPU_SKU_TOKEN = "MI300X"
WARMUP = 5
REP = 20


@dataclass(frozen=True)
class RaggedBatch:
    """One sparse-MLA call's query rows and their selected cache positions.

    ``dense_indices`` is ``[num_queries, SELECTED_K]`` int32; row ``i`` holds that
    query's selected cache-token ids in its first ``lengths[i]`` slots and ``-1``
    after. ``lengths`` is ``[num_queries]`` int32. ``num_cache_tokens`` is the KV
    latent pool size. ``valid_counts`` mirrors ``lengths`` as a Python tuple for
    byte accounting.
    """

    dense_indices: Any
    lengths: Any
    num_cache_tokens: int
    num_queries: int
    valid_counts: tuple[int, ...]


def load_callables(torch: Any) -> tuple[Any, Any]:
    """Return ``(rocm_sparse_attn_prefill, build_ragged_indices_from_dense)``."""
    if getattr(torch.version, "hip", None) is None:
        raise ProfilerNotImplemented(f"{_CALLABLE} requires a ROCm (HIP) torch build")
    try:
        module = importlib.import_module(_MODULE)
    except Exception as exc:  # pragma: no cover - requires the vLLM-ROCm image
        raise ProfilerNotImplemented(
            f"rocm_triton_mla_sparse requires the vllm_rocm_env sparse-MLA ops ({_MODULE})"
        ) from exc
    prefill = getattr(module, _CALLABLE, None)
    ragged = getattr(module, _RAGGED_BUILDER, None)
    if not callable(prefill) or not callable(ragged):
        raise ProfilerNotImplemented(
            f"rocm_triton_mla_sparse requires {_MODULE}.{_CALLABLE} and .{_RAGGED_BUILDER}"
        )
    return prefill, ragged


def require_mi300(torch: Any, *, backend: str) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"a ROCm device is required for {backend}")
    if getattr(torch.version, "hip", None) is None:
        raise ProfilerNotImplemented(f"{backend} requires a ROCm (HIP) torch build")
    if not _is_mi300_series(torch):
        raise ProfilerNotImplemented(
            f"{backend} is verified only on {GPU_SKU_TOKEN} (CDNA3 gfx942), got "
            f"name={_device_name(torch)!r} arch={_device_arch(torch)!r}"
        )


def _fill_operand_values(torch: Any, tensor: Any) -> None:
    """Deterministic bounded BF16 fill (fixed seed), built on host, no kernels."""
    gen = torch.Generator(device="cpu").manual_seed(42)
    flat = torch.randn(tensor.numel(), generator=gen, dtype=torch.float32)
    tensor.copy_(flat.reshape(tensor.shape).to(torch.bfloat16))


def build_operands(
    torch: Any,
    batch: RaggedBatch,
    *,
    num_heads: int,
    device: Any,
) -> dict[str, Any]:
    """Build BF16 q/kv, the precomputed ragged indices, and the output buffer.

    All tensors are built on the host and moved with ``.to(device)`` so operand
    setup launches no GPU kernels. ``build_ragged_indices_from_dense`` DOES launch
    a packing kernel, so it is called here (once, outside the timed call) and its
    result is passed in -- keeping the per-call dispatch count exactly the
    attention call's own.
    """
    _prefill, build_ragged = load_callables(torch)

    q = torch.empty((batch.num_queries, num_heads, HEAD_DIM), dtype=torch.bfloat16)
    _fill_operand_values(torch, q)
    q = q.to(device)
    # kv is the [skv, 1, 512] BF16 latent cache the call views as kv.view(-1,1,d).
    kv = torch.empty((batch.num_cache_tokens, NUM_KV_HEADS, HEAD_DIM), dtype=torch.bfloat16)
    _fill_operand_values(torch, kv)
    kv = kv.to(device)

    dense_indices = batch.dense_indices.to(device)
    lengths = batch.lengths.to(device)
    ragged_indices, ragged_indptr = build_ragged(
        dense_indices, lengths, batch.num_cache_tokens
    )
    output = torch.empty((batch.num_queries, num_heads, VALUE_DIM), dtype=torch.bfloat16, device=device)
    return {
        "q": q,
        "kv": kv,
        "dense_indices": dense_indices,
        "lengths": lengths,
        "ragged_indices": ragged_indices,
        "ragged_indptr": ragged_indptr,
        "output": output,
    }


def make_kernel(torch: Any, operands: dict[str, Any]) -> Any:
    """Return a zero-arg closure that replays one rocm_sparse_attn_prefill call."""
    prefill, _ragged = load_callables(torch)
    q = operands["q"]
    kv = operands["kv"]
    ragged_indices = operands["ragged_indices"]
    ragged_indptr = operands["ragged_indptr"]
    output = operands["output"]

    def kernel() -> Any:
        prefill(
            q=q,
            kv=kv.view(-1, 1, q.shape[-1]),
            indices=None,
            topk_length=None,
            scale=SOFTMAX_SCALE,
            head_dim=HEAD_DIM,
            nope_head_dim=NOPE_HEAD_DIM,
            rope_head_dim=ROPE_HEAD_DIM,
            attn_sink=None,
            output=output,
            ragged_indices=ragged_indices,
            ragged_indptr=ragged_indptr,
        )
        return output

    return kernel


def reference_output(
    torch: Any,
    q: Any,
    kv: Any,
    batch: RaggedBatch,
    *,
    softmax_scale: float,
) -> Any:
    """Rope-free BF16 selected-MLA attention in FP32, as a ``[Q, H, 512]`` BF16.

    For query ``i`` the valid cache set is ``dense_indices[i, :lengths[i]]``;
    scores are ``q_i . kv_selected`` over all 512 dims, softmax-scaled, and the
    output is ``probs . kv_selected`` over the same 512 dims (K and V are the one
    latent). Rows with no valid index return zeros. This mirrors the production
    ragged kernel's math (K == V == the latent) at nope=512/rope=0.
    """
    num_queries, num_heads, _ = q.shape
    out = torch.zeros((num_queries, num_heads, VALUE_DIM), dtype=torch.bfloat16, device=q.device)
    kv_rows = kv[:, 0, :].float()
    dense = batch.dense_indices
    for i in range(num_queries):
        count = int(batch.valid_counts[i])
        if count == 0:
            continue
        idx = dense[i, :count].to(torch.int64).to(q.device)
        valid = (idx >= 0) & (idx < kv.shape[0])
        if not bool(valid.any()):
            continue
        idx = idx[valid]
        selected = kv_rows.index_select(0, idx)  # [count, 512]
        scores = (q[i].float() @ selected.t()) * softmax_scale  # [H, count]
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        out[i] = (probs @ selected).to(torch.bfloat16)
    return out


def check_correctness(
    torch: Any,
    operands: dict[str, Any],
    batch: RaggedBatch,
    *,
    num_heads: int,
    backend: str,
    rel_l2_tol: float = 0.05,
) -> dict[str, float]:
    """Run the call once and compare with the rope-free Torch reference."""
    kernel = make_kernel(torch, operands)
    actual = kernel().float()
    torch.cuda.synchronize()
    expected = reference_output(
        torch, operands["q"], operands["kv"], batch, softmax_scale=SOFTMAX_SCALE
    ).float()
    denom = expected.norm().clamp(min=1e-30)
    rel_l2 = float((actual - expected).norm() / denom)
    finite = bool(torch.isfinite(actual).all())
    if not finite or rel_l2 > rel_l2_tol:
        from profiling.runners.exceptions import KernelLaunchFailed

        raise KernelLaunchFailed(
            f"{backend} correctness failed: rel_l2={rel_l2:.4f} finite={finite}"
        )
    return {"rel_l2": rel_l2, "finite": float(finite)}


def assert_triton_path(*, kind: str, backend: str, spec: dict[str, Any]) -> list[str]:
    """Prove the Triton ragged kernel ran (and no AITER opus kernel did).

    A standalone rocprofv3 capture of the same launch driver, read for kernel
    names only. The mandatory silent-fallback guard: run once before timing.
    Returns the captured kernel-name set (for logging the exact HIP strings).
    """
    from profiling.profilers.rocprof_kernel_profiler import _captured_names, _find_rocprofv3

    rocprofv3 = _find_rocprofv3()
    with tempfile.TemporaryDirectory(prefix="vibesim-sparse-mla-") as tmp:
        rocpd_dir = Path(tmp) / "rocpd"
        driver = [
            "python3", "-m", "profiling.profilers.rocprof_run",
            "--kind", kind, "--backend", backend,
            "--spec", json.dumps(spec), "--warmup", str(WARMUP), "--rep", str(REP),
        ]
        cmd = [
            rocprofv3, "--kernel-trace", "--output-format", "rocpd",
            "-d", str(rocpd_dir), "-o", "run", "--", *driver,
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            from profiling.runners.exceptions import KernelLaunchFailed

            tail = (completed.stderr or completed.stdout or "")[-1500:]
            raise KernelLaunchFailed(f"{backend} path-assertion capture failed: {tail}")
        db_files = sorted(rocpd_dir.rglob("*.db"))
        if not db_files:
            from profiling.runners.exceptions import KernelLaunchFailed

            raise KernelLaunchFailed(f"{backend} path-assertion produced no rocpd database")
        names = [n for n in _captured_names(str(db_files[0])) if n]
    has_triton = any(TRITON_KERNEL_SUBSTR in n for n in names)
    has_opus = any(AITER_OPUS_SUBSTR in n for n in names)
    if not has_triton or has_opus:
        from profiling.runners.exceptions import KernelLaunchFailed

        raise KernelLaunchFailed(
            f"{backend} is NOT on the rope-free Triton sparse path: "
            f"triton_ragged={has_triton} aiter_opus={has_opus}. "
            "Refusing to mislabel the run."
        )
    return names


def logical_flops(*, num_queries: int, num_heads: int, selected_k: int) -> int:
    return 2 * num_queries * num_heads * selected_k * (HEAD_DIM + VALUE_DIM)


def logical_bytes(
    *,
    num_queries: int,
    num_heads: int,
    selected_k: int,
    valid_counts: tuple[int, ...],
) -> int:
    q_read = 2 * num_queries * num_heads * HEAD_DIM
    index_read = 4 * num_queries * selected_k
    cache_read = 2 * sum(valid_counts) * HEAD_DIM
    output_write = 2 * num_queries * num_heads * VALUE_DIM
    max_lse_write = 8 * num_queries * num_heads
    return q_read + index_read + cache_read + output_write + max_lse_write
