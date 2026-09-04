"""Torch composite profiler for GLM-5.2 selected sparse MLA attention.

The composite reproduces the semantic boundary of FlashMLA's sparse attention
callable. Query concatenation, index remapping, cache conversion, and any other
setup launches remain outside this profiler kind.
"""

from __future__ import annotations

import importlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_NUM_HEADS = 64
_NUM_KV_HEADS = 1
_SELECTED_K = 2048
_LATENT_DIM = 512
_ROPE_DIM = 64
_SCORE_DIM = _LATENT_DIM + _ROPE_DIM
_VALUE_DIM = 512
_SOFTMAX_SCALE = 0.0625
_CACHE_LAYOUT = "token_major_mqa_bf16_latent_rope"
_QUERY_CHUNK_SIZE = 8
_CACHE_TEMPLATE_ROWS = 64
_FLASHMLA_BACKEND = "dsa_sparse_mla_attention:vllm_flashmla_bf16"
_FLASHMLA_KERNEL_NAME = "sparse_attn_fwd_kernel"
_FLASHMLA_ATOL = 8e-4
_FLASHMLA_RTOL = 3.01 / 128
_FLASHMLA_COSINE_LIMIT = 7e-6
_TRTLLM_FP8_BACKEND = "dsa_sparse_mla_attention:flashinfer_trtllm_fp8"
_TRTLLM_FP8_CACHE_LAYOUT = "hnd_paged_mqa_fp8_latent_rope"
_TRTLLM_KERNEL_NAME = "fmhaSm100"
# Per-rank q-head counts of GLM's 64 heads at validated TP degrees (TP8, TP4);
# operands, output check and metrics are parametric in num_heads.
_TRTLLM_SUPPORTED_NUM_HEADS = frozenset({8, 16})
_TRTLLM_WORKSPACE_BYTES = 128 * 1024 * 1024
_TRTLLM_PAGE_SIZE = 64

_UINT = r"(?:0|[1-9][0-9]*)"
_UNIFORM_RE = re.compile(rf"u:({_UINT})x({_UINT})\Z")
_RAMP_RE = re.compile(rf"r:({_UINT})\.\.({_UINT})\Z")
_CLIPPED_RE = re.compile(rf"c:({_UINT})\.\.({_UINT})@({_UINT})\Z")
_GROUP_RE = re.compile(rf"g:\((?P<group>{_UINT}(?:,{_UINT})*)\)x(?P<groups>{_UINT})\Z")

_INDEX_DISTRIBUTIONS = frozenset(
    {
        "recent_contiguous",
        "unique_scattered_pages",
        "clustered_pages",
        "uniform_stride",
    }
)


@dataclass(frozen=True)
class _ValidatedArgs:
    num_queries: int
    num_cache_tokens: int
    valid_counts: tuple[int, ...]
    index_distribution: str


@dataclass(frozen=True)
class _Operands:
    q: Any
    cache: Any
    selected_indices: Any


@dataclass(frozen=True)
class _TrtllmFp8Operands:
    query: Any
    cache: Any
    block_tables: Any
    seq_lens: Any
    workspace: Any


def _encode_valid_counts(counts: Sequence[int], *, selected_k: int, num_cache_tokens: int) -> str:
    """Return the unique canonical encoding for a flattened count vector."""
    values = tuple(counts)
    if not values:
        raise ValueError("valid_counts must contain at least one row")
    limit = min(selected_k, num_cache_tokens)
    if any(type(value) is not int for value in values):
        raise TypeError("valid_counts values must be integers")
    if any(value < 0 or value > limit for value in values):
        raise ValueError(f"valid_counts values must be in 0..{limit}")

    if all(value == values[0] for value in values):
        return f"u:{values[0]}x{len(values)}"

    if all(values[index] == values[0] + index for index in range(len(values))):
        return f"r:{values[0]}..{values[-1]}"

    unclipped_last = values[0] + len(values) - 1
    clipped = tuple(min(values[0] + index, selected_k) for index in range(len(values)))
    if values[0] < selected_k < unclipped_last <= num_cache_tokens and values == clipped:
        return f"c:{values[0]}..{unclipped_last}@{selected_k}"

    period = len(values)
    for candidate in range(1, len(values) + 1):
        if len(values) % candidate:
            continue
        if all(values[index] == values[index % candidate] for index in range(len(values))):
            period = candidate
            break
    group = ",".join(str(value) for value in values[:period])
    return f"g:({group})x{len(values) // period}"


def _decode_valid_counts(
    encoded: str,
    *,
    num_queries: int,
    selected_k: int,
    num_cache_tokens: int,
) -> tuple[int, ...]:
    """Parse and validate the canonical flattened count-vector syntax."""
    if not isinstance(encoded, str):
        raise TypeError("valid_counts must be a string")
    if not encoded.isascii():
        raise ValueError("valid_counts must use ASCII compact syntax")

    values: tuple[int, ...]
    if match := _UNIFORM_RE.fullmatch(encoded):
        count, rows = (int(value) for value in match.groups())
        if rows <= 0:
            raise ValueError("valid_counts uniform row count must be positive")
        if rows != num_queries:
            raise ValueError(f"valid_counts expands to {rows} rows, expected {num_queries}")
        values = (count,) * rows
    elif match := _RAMP_RE.fullmatch(encoded):
        first, last = (int(value) for value in match.groups())
        if first >= last:
            raise ValueError("valid_counts ramp must be a strict +1 ramp")
        rows = last - first + 1
        if rows != num_queries:
            raise ValueError(f"valid_counts expands to {rows} rows, expected {num_queries}")
        values = tuple(range(first, last + 1))
    elif match := _CLIPPED_RE.fullmatch(encoded):
        first, last, cap = (int(value) for value in match.groups())
        if cap != selected_k:
            raise ValueError("valid_counts clipped-ramp cap must equal selected_k")
        if not first < selected_k < last <= num_cache_tokens:
            raise ValueError(
                "valid_counts clipped ramp requires first < selected_k < last <= num_cache_tokens"
            )
        rows = last - first + 1
        if rows != num_queries:
            raise ValueError(f"valid_counts expands to {rows} rows, expected {num_queries}")
        values = tuple(min(value, selected_k) for value in range(first, last + 1))
    elif match := _GROUP_RE.fullmatch(encoded):
        group = tuple(int(value) for value in match.group("group").split(","))
        groups = int(match.group("groups"))
        if groups <= 0:
            raise ValueError("valid_counts tuple repetition must be positive")
        rows = len(group) * groups
        if rows != num_queries:
            raise ValueError(f"valid_counts expands to {rows} rows, expected {num_queries}")
        values = group * groups
    else:
        raise ValueError("valid_counts must be canonical u:, r:, c:, or g: compact syntax")

    if len(values) != num_queries:
        raise ValueError(f"valid_counts expands to {len(values)} rows, expected {num_queries}")
    limit = min(selected_k, num_cache_tokens)
    if any(value > limit for value in values):
        raise ValueError(f"valid_counts values must be in 0..{limit}")

    canonical = _encode_valid_counts(
        values, selected_k=selected_k, num_cache_tokens=num_cache_tokens
    )
    if canonical != encoded:
        raise ValueError(f"valid_counts is noncanonical; canonical encoding is {canonical!r}")
    return values


def _validate_args(
    *,
    num_queries: int,
    num_cache_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    selected_k: int,
    latent_dim: int,
    rope_dim: int,
    value_dim: int,
    softmax_scale: float,
    q_dtype: DType | str,
    cache_dtype: DType | str,
    index_dtype: str,
    output_dtype: DType | str,
    valid_counts: str,
    index_distribution: str,
    cache_layout: str,
    expected_num_heads: int | None = _NUM_HEADS,
    expected_q_dtype: DType = DType.BF16,
    expected_cache_dtype: DType = DType.BF16,
    expected_output_dtype: DType = DType.BF16,
    expected_cache_layout: str = _CACHE_LAYOUT,
) -> _ValidatedArgs:
    integers = {
        "num_queries": num_queries,
        "num_cache_tokens": num_cache_tokens,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "selected_k": selected_k,
        "latent_dim": latent_dim,
        "rope_dim": rope_dim,
        "value_dim": value_dim,
    }
    for name, value in integers.items():
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer")

    # Positivity is the only shape law here. The former 4,096 / 131,072 ceilings
    # recorded where the accuracy test had been run, not what the kernel accepts,
    # and every context extension had to chase them. Operand size is bounded by
    # the sweep grid's `infeasible_mask` instead.
    if num_queries < 1:
        raise ProfilerNotImplemented(f"num_queries must be >= 1, got {num_queries}")
    if num_cache_tokens < 1:
        raise ProfilerNotImplemented(f"num_cache_tokens must be >= 1, got {num_cache_tokens}")
    expected = {
        "num_kv_heads": (num_kv_heads, _NUM_KV_HEADS),
        "selected_k": (selected_k, _SELECTED_K),
        "latent_dim": (latent_dim, _LATENT_DIM),
        "rope_dim": (rope_dim, _ROPE_DIM),
        "value_dim": (value_dim, _VALUE_DIM),
    }
    for name, (actual, required) in expected.items():
        if actual != required:
            raise ProfilerNotImplemented(f"{name} must be {required}, got {actual}")
    if expected_num_heads is None:
        if num_heads < 1:
            raise ProfilerNotImplemented(f"num_heads must be >= 1, got {num_heads}")
    elif num_heads != expected_num_heads:
        raise ProfilerNotImplemented(f"num_heads must be {expected_num_heads}, got {num_heads}")
    if type(softmax_scale) not in {int, float} or isinstance(softmax_scale, bool):
        raise TypeError("softmax_scale must be a real number")
    if float(softmax_scale) != _SOFTMAX_SCALE:
        raise ProfilerNotImplemented(
            f"softmax_scale must be exactly {_SOFTMAX_SCALE}, got {softmax_scale}"
        )
    if DType.from_value(q_dtype) is not expected_q_dtype:
        raise ProfilerNotImplemented(f"q_dtype must be {expected_q_dtype.value}")
    if DType.from_value(cache_dtype) is not expected_cache_dtype:
        raise ProfilerNotImplemented(f"cache_dtype must be {expected_cache_dtype.value}")
    if index_dtype != "int32":
        raise ProfilerNotImplemented("index_dtype must be int32")
    if DType.from_value(output_dtype) is not expected_output_dtype:
        raise ProfilerNotImplemented(f"output_dtype must be {expected_output_dtype.value}")
    if cache_layout != expected_cache_layout:
        raise ProfilerNotImplemented(f"cache_layout must be {expected_cache_layout!r}")
    if index_distribution not in _INDEX_DISTRIBUTIONS:
        modes = ", ".join(sorted(_INDEX_DISTRIBUTIONS))
        raise ProfilerNotImplemented(f"index_distribution must be one of: {modes}")

    counts = _decode_valid_counts(
        valid_counts,
        num_queries=num_queries,
        selected_k=selected_k,
        num_cache_tokens=num_cache_tokens,
    )
    return _ValidatedArgs(
        num_queries=num_queries,
        num_cache_tokens=num_cache_tokens,
        valid_counts=counts,
        index_distribution=index_distribution,
    )


def _require_h200(torch: Any, *, backend: str = "dsa_sparse_mla_attention:torch") -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"CUDA is required for {backend}")
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    if gpu_name != "NVIDIA H200":
        raise ProfilerNotImplemented(f"{backend} requires NVIDIA H200, got {gpu_name!r}")


def _require_b200(torch: Any, *, backend: str = _TRTLLM_FP8_BACKEND) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"CUDA is required for {backend}")
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    if gpu_name != "NVIDIA B200":
        raise ProfilerNotImplemented(f"{backend} requires NVIDIA B200, got {gpu_name!r}")


def _load_flashmla_sparse_fwd() -> Any:
    """Load the packaged FlashMLA callable only inside a vllm_env worker."""
    try:
        module = importlib.import_module("vllm.v1.attention.ops.flashmla")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ProfilerNotImplemented(
            f"{_FLASHMLA_BACKEND} requires vllm_env with packaged FlashMLA"
        ) from exc
    callable_ = getattr(module, "flash_mla_sparse_fwd", None)
    if not callable(callable_):
        raise ProfilerNotImplemented(
            f"{_FLASHMLA_BACKEND} requires vllm.v1.attention.ops.flashmla.flash_mla_sparse_fwd"
        )
    return callable_


def _coprime_stride(size: int, preferred: int) -> int:
    if size <= 1:
        return 1
    stride = preferred % size or 1
    while math.gcd(stride, size) != 1:
        stride += 1
    return stride


def _row_indices(
    torch: Any,
    *,
    row: int,
    count: int,
    num_cache_tokens: int,
    distribution: str,
    device: Any,
) -> Any:
    if count == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)
    positions = torch.arange(count, dtype=torch.int64, device=device)
    if distribution == "recent_contiguous":
        return positions + (num_cache_tokens - count)
    if distribution == "uniform_stride":
        return torch.div(positions * num_cache_tokens, count, rounding_mode="floor")
    if distribution == "unique_scattered_pages":
        stride = _coprime_stride(num_cache_tokens, 4099 + 2 * row)
        return (positions * stride + 17 * row) % num_cache_tokens

    page_size = 64
    num_pages = (num_cache_tokens + page_size - 1) // page_size
    page_stride = _coprime_stride(num_pages, 17 + 2 * row)
    values: list[int] = []
    for page_offset in range(num_pages):
        page = (row * 11 + page_offset * page_stride) % num_pages
        base = page * page_size
        for offset in range(page_size):
            index = base + (offset + row * 7) % page_size
            if index < num_cache_tokens:
                values.append(index)
                if len(values) == count:
                    return torch.tensor(values, dtype=torch.int64, device=device)
    raise AssertionError("clustered index construction did not produce enough indices")


def _build_operands(torch: Any, validated: _ValidatedArgs, *, device: Any) -> _Operands:
    q = torch.empty(
        (validated.num_queries, _NUM_HEADS, _SCORE_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    cache = torch.empty(
        (validated.num_cache_tokens, _NUM_KV_HEADS, _SCORE_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    selected_indices = torch.full(
        (validated.num_queries, _NUM_KV_HEADS, _SELECTED_K),
        -1,
        dtype=torch.int32,
        device=device,
    )

    feature = torch.linspace(-0.875, 0.875, _SCORE_DIM, device=device)
    heads = (torch.arange(_NUM_HEADS, dtype=torch.float32, device=device) - 31.5) / 256.0
    rows = (torch.arange(_QUERY_CHUNK_SIZE, dtype=torch.float32, device=device) - 3.5) / 128.0
    q_template = (rows[:, None, None] + heads[None, :, None] + feature).to(torch.bfloat16)
    for start in range(0, validated.num_queries, _QUERY_CHUNK_SIZE):
        stop = min(start + _QUERY_CHUNK_SIZE, validated.num_queries)
        q[start:stop].copy_(q_template[: stop - start])

    cache_rows = (
        torch.arange(_CACHE_TEMPLATE_ROWS, dtype=torch.float32, device=device) - 31.5
    ) / 192.0
    cache_template = (cache_rows[:, None] + feature[None, :]).to(torch.bfloat16)
    for start in range(0, validated.num_cache_tokens, _CACHE_TEMPLATE_ROWS):
        stop = min(start + _CACHE_TEMPLATE_ROWS, validated.num_cache_tokens)
        phase = ((start // _CACHE_TEMPLATE_ROWS) % 17 - 8) / 128.0
        cache[start:stop, 0].copy_(
            (cache_template[: stop - start].float() + phase).to(torch.bfloat16)
        )

    for row, count in enumerate(validated.valid_counts):
        indices = _row_indices(
            torch,
            row=row,
            count=count,
            num_cache_tokens=validated.num_cache_tokens,
            distribution=validated.index_distribution,
            device=device,
        )
        selected_indices[row, 0, :count].copy_(indices.to(torch.int32))
    return _Operands(q=q, cache=cache, selected_indices=selected_indices)


def _torch_composite(
    torch: Any,
    q: Any,
    cache: Any,
    selected_indices: Any,
    *,
    softmax_scale: float,
    query_chunk_size: int = _QUERY_CHUNK_SIZE,
) -> Any:
    outputs: list[Any] = []
    cache_2d = cache[:, 0, :]
    num_cache_tokens = cache.shape[0]
    for start in range(0, q.shape[0], query_chunk_size):
        stop = min(start + query_chunk_size, q.shape[0])
        indices = selected_indices[start:stop, 0, :].to(torch.int64)
        valid = (indices >= 0) & (indices < num_cache_tokens)
        safe = torch.where(valid, indices, torch.zeros_like(indices))
        gathered = cache_2d.index_select(0, safe.reshape(-1)).reshape(
            stop - start, _SELECTED_K, _SCORE_DIM
        )
        gathered = torch.where(valid[..., None], gathered, torch.zeros_like(gathered))

        scores = torch.einsum("qhd,qkd->qhk", q[start:stop].float(), gathered.float())
        scores.mul_(softmax_scale)
        scores.masked_fill_(~valid[:, None, :], float("-inf"))
        all_invalid = ~valid.any(dim=-1)
        if all_invalid.any():
            scores[all_invalid] = 0.0
        probabilities = torch.softmax(scores, dim=-1)
        probabilities = torch.where(
            valid[:, None, :], probabilities, torch.zeros_like(probabilities)
        )
        output = torch.einsum("qhk,qkv->qhv", probabilities, gathered[..., :_VALUE_DIM].float())
        outputs.append(output.to(torch.bfloat16))
    return torch.cat(outputs, dim=0)


def _check_correctness(
    torch: Any,
    operands: _Operands,
    *,
    softmax_scale: float,
) -> None:
    from profiling.runners.attention.dsa_sparse_mla_attention_reference import (
        dsa_sparse_mla_attention_reference,
    )

    q_before = operands.q.clone()
    cache_before = operands.cache.clone()
    indices_before = operands.selected_indices.clone()
    expected = dsa_sparse_mla_attention_reference(
        operands.q,
        operands.cache,
        operands.selected_indices,
        softmax_scale=softmax_scale,
    )
    actual = _torch_composite(
        torch,
        operands.q,
        operands.cache,
        operands.selected_indices,
        softmax_scale=softmax_scale,
    )

    if actual.shape != (operands.q.shape[0], _NUM_HEADS, _VALUE_DIM):
        raise AssertionError(f"unexpected output shape {tuple(actual.shape)}")
    if actual.dtype is not torch.bfloat16 or not torch.isfinite(actual).all():
        raise AssertionError("Torch composite output must be finite BF16")
    torch.testing.assert_close(actual.float(), expected.float(), atol=8e-4, rtol=3.01 / 128)
    numerator = 2.0 * torch.sum(actual.float() * expected.float()).item()
    denominator = (
        torch.sum(actual.float().square()).item() + torch.sum(expected.float().square()).item()
    )
    cosine_difference = 0.0 if denominator == 0.0 else 1.0 - numerator / denominator
    if abs(cosine_difference) > 7e-6:
        raise AssertionError(f"cosine difference {cosine_difference} exceeds 7e-6")

    if not torch.equal(operands.q, q_before):
        raise AssertionError("Torch composite mutated q")
    if not torch.equal(operands.cache, cache_before):
        raise AssertionError("Torch composite mutated cache")
    if not torch.equal(operands.selected_indices, indices_before):
        raise AssertionError("Torch composite mutated selected_indices")
    for source in (operands.q, operands.cache, operands.selected_indices):
        if actual.untyped_storage().data_ptr() == source.untyped_storage().data_ptr():
            raise AssertionError("Torch composite output aliases an input")


def _validate_flashmla_layouts(operands: _Operands) -> None:
    """Reject layouts that violate the SM90 TMA/cp.async address contract."""

    for name, tensor in (("q", operands.q), ("cache", operands.cache)):
        if tensor.data_ptr() % 16:
            raise KernelLaunchFailed(
                f"{_FLASHMLA_BACKEND} requires {name} base address divisible by 16"
            )
        if tensor.stride(-1) != 1:
            raise KernelLaunchFailed(f"{_FLASHMLA_BACKEND} requires {name} final stride to equal 1")
        outer_byte_strides = tuple(
            stride * tensor.element_size() for stride in tensor.stride()[:-1]
        )
        if any(stride % 16 for stride in outer_byte_strides):
            raise KernelLaunchFailed(
                f"{_FLASHMLA_BACKEND} requires every {name} outer byte stride "
                f"to be divisible by 16, got {outer_byte_strides}"
            )

    indices = operands.selected_indices
    if indices.stride(-1) != 1:
        raise KernelLaunchFailed(
            f"{_FLASHMLA_BACKEND} requires selected_indices final stride to equal 1"
        )
    if indices.data_ptr() % indices.element_size():
        raise KernelLaunchFailed(
            f"{_FLASHMLA_BACKEND} requires ordinary int32 alignment for selected_indices"
        )


def _check_flashmla_correctness(
    torch: Any,
    flash_mla_sparse_fwd: Any,
    operands: _Operands,
    *,
    softmax_scale: float,
) -> None:
    """Validate the packaged one-launch BF16 callable before formal timing."""
    from profiling.runners.attention.dsa_sparse_mla_attention_reference import (
        dsa_sparse_mla_attention_reference,
    )

    snapshots = (
        operands.q.clone(),
        operands.cache.clone(),
        operands.selected_indices.clone(),
    )
    input_pointers = {
        operands.q.untyped_storage().data_ptr(),
        operands.cache.untyped_storage().data_ptr(),
        operands.selected_indices.untyped_storage().data_ptr(),
    }

    def invoke() -> tuple[Any, Any, Any]:
        returned = flash_mla_sparse_fwd(
            operands.q,
            operands.cache,
            operands.selected_indices,
            softmax_scale,
            _VALUE_DIM,
            None,
            None,
        )
        torch.cuda.synchronize()
        if not isinstance(returned, (tuple, list)) or len(returned) != 3:
            raise AssertionError("FlashMLA sparse forward must return output, max_logits, lse")
        return returned[0], returned[1], returned[2]

    output, max_logits, lse = invoke()
    expected = dsa_sparse_mla_attention_reference(
        operands.q,
        operands.cache,
        operands.selected_indices,
        softmax_scale=softmax_scale,
    )
    expected_output_shape = (operands.q.shape[0], _NUM_HEADS, _VALUE_DIM)
    expected_diagnostic_shape = (operands.q.shape[0], _NUM_HEADS)
    if tuple(output.shape) != expected_output_shape or output.dtype is not torch.bfloat16:
        raise AssertionError(
            f"FlashMLA output must be BF16 {expected_output_shape}, "
            f"got {output.dtype} {tuple(output.shape)}"
        )
    for name, tensor in (("max_logits", max_logits), ("lse", lse)):
        if tuple(tensor.shape) != expected_diagnostic_shape or tensor.dtype is not torch.float32:
            raise AssertionError(
                f"FlashMLA {name} must be FP32 {expected_diagnostic_shape}, "
                f"got {tensor.dtype} {tuple(tensor.shape)}"
            )
    if not torch.isfinite(output).all():
        raise AssertionError("FlashMLA output must be finite for supported finite operands")
    torch.testing.assert_close(
        output.float(),
        expected.float(),
        atol=_FLASHMLA_ATOL,
        rtol=_FLASHMLA_RTOL,
    )
    numerator = 2.0 * torch.sum(output.float() * expected.float()).item()
    denominator = (
        torch.sum(output.float().square()).item() + torch.sum(expected.float().square()).item()
    )
    cosine_difference = 0.0 if denominator == 0.0 else 1.0 - numerator / denominator
    if abs(cosine_difference) > _FLASHMLA_COSINE_LIMIT:
        raise AssertionError(
            f"FlashMLA cosine difference {cosine_difference} exceeds {_FLASHMLA_COSINE_LIMIT}"
        )

    valid = (operands.selected_indices >= 0) & (operands.selected_indices < operands.cache.shape[0])
    all_invalid = ~valid.any(dim=-1).squeeze(1)
    if all_invalid.any():
        if torch.count_nonzero(output[all_invalid]).item() != 0:
            raise AssertionError("FlashMLA all-invalid rows must return exact zero output")
        if not torch.isneginf(max_logits[all_invalid]).all():
            raise AssertionError("FlashMLA all-invalid rows must return max_logits=-inf")
        if not torch.isposinf(lse[all_invalid]).all():
            raise AssertionError("FlashMLA all-invalid rows must return lse=+inf")
    nonempty = ~all_invalid
    if nonempty.any() and not (
        torch.isfinite(max_logits[nonempty]).all() and torch.isfinite(lse[nonempty]).all()
    ):
        raise AssertionError("FlashMLA diagnostics must be finite for nonempty finite rows")

    repeated = invoke()
    if not all(
        torch.equal(first, second)
        for first, second in zip((output, max_logits, lse), repeated, strict=True)
    ):
        raise AssertionError("FlashMLA repeated calls must be deterministic")
    for name, tensor, snapshot in zip(
        ("q", "cache", "selected_indices"),
        (operands.q, operands.cache, operands.selected_indices),
        snapshots,
        strict=True,
    ):
        if not torch.equal(tensor, snapshot):
            raise AssertionError(f"FlashMLA mutated {name}")
    outputs = (output, max_logits, lse, *repeated)
    output_pointers = [tensor.untyped_storage().data_ptr() for tensor in outputs]
    if input_pointers.intersection(output_pointers):
        raise AssertionError("FlashMLA output aliases an input")


def _logical_flops(*, num_queries: int, num_heads: int, selected_k: int) -> int:
    return 2 * num_queries * num_heads * selected_k * (_SCORE_DIM + _VALUE_DIM)


def _logical_bytes(
    *,
    num_queries: int,
    num_heads: int,
    selected_k: int,
    valid_counts: Sequence[int],
    q_bytes: int = 2,
    cache_bytes: int = 2,
    output_bytes: int = 2,
) -> int:
    q_read = q_bytes * num_queries * num_heads * _SCORE_DIM
    index_read = 4 * num_queries * selected_k
    cache_read = cache_bytes * sum(valid_counts) * _SCORE_DIM
    output_write = output_bytes * num_queries * num_heads * _VALUE_DIM
    max_lse_write = 8 * num_queries * num_heads
    return q_read + index_read + cache_read + output_write + max_lse_write


def profile_dsa_sparse_mla_attention_torch(
    *,
    num_queries: int,
    num_cache_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    selected_k: int,
    latent_dim: int,
    rope_dim: int,
    value_dim: int,
    softmax_scale: float,
    q_dtype: DType | str,
    cache_dtype: DType | str,
    index_dtype: str,
    output_dtype: DType | str,
    valid_counts: str,
    index_distribution: str,
    cache_layout: str,
) -> ComputeMetrics:
    """Profile the complete Torch semantic composite on an NVIDIA H200."""
    validated = _validate_args(
        num_queries=num_queries,
        num_cache_tokens=num_cache_tokens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        selected_k=selected_k,
        latent_dim=latent_dim,
        rope_dim=rope_dim,
        value_dim=value_dim,
        softmax_scale=softmax_scale,
        q_dtype=q_dtype,
        cache_dtype=cache_dtype,
        index_dtype=index_dtype,
        output_dtype=output_dtype,
        valid_counts=valid_counts,
        index_distribution=index_distribution,
        cache_layout=cache_layout,
    )

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ProfilerNotImplemented("PyTorch is unavailable") from exc

    try:
        _require_h200(torch)
        device = torch.device("cuda", torch.cuda.current_device())
        operands = _build_operands(torch, validated, device=device)
        _check_correctness(torch, operands, softmax_scale=float(softmax_scale))

        def kernel() -> Any:
            return _torch_composite(
                torch,
                operands.q,
                operands.cache,
                operands.selected_indices,
                softmax_scale=float(softmax_scale),
            )

        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed("dsa_sparse_mla_attention:torch composite failed") from exc

    flops = _logical_flops(num_queries=num_queries, num_heads=num_heads, selected_k=selected_k)
    logical_bytes = _logical_bytes(
        num_queries=num_queries,
        num_heads=num_heads,
        selected_k=selected_k,
        valid_counts=validated.valid_counts,
    )
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=time_ms,
        energy_j=energy_j,
        tflops=flops / seconds / 1e12,
        memory_bandwidth_gbps=logical_bytes / seconds / 1e9,
    )


def _build_trtllm_fp8_operands(
    torch: Any,
    validated: _ValidatedArgs,
    *,
    num_heads: int,
    device: Any,
) -> _TrtllmFp8Operands:
    fp8 = torch.float8_e4m3fn
    query = torch.empty(
        (validated.num_queries, 1, num_heads, _SCORE_DIM),
        dtype=fp8,
        device=device,
    )
    query.copy_(
        torch.linspace(-0.75, 0.75, _SCORE_DIM, device=device)
        .view(1, 1, 1, _SCORE_DIM)
        .expand_as(query)
    )

    num_pages = math.ceil(validated.num_cache_tokens / _TRTLLM_PAGE_SIZE)
    cache = torch.empty(
        (num_pages, 1, _TRTLLM_PAGE_SIZE, _SCORE_DIM),
        dtype=fp8,
        device=device,
    )
    cache.copy_(torch.linspace(-0.5, 0.5, _SCORE_DIM, device=device).view(1, 1, 1, _SCORE_DIM))

    block_tables = torch.zeros(
        (validated.num_queries, 1, _SELECTED_K),
        dtype=torch.int32,
        device=device,
    )
    for row, count in enumerate(validated.valid_counts):
        indices = _row_indices(
            torch,
            row=row,
            count=count,
            num_cache_tokens=validated.num_cache_tokens,
            distribution=validated.index_distribution,
            device=device,
        )
        block_tables[row, 0, :count].copy_(indices.to(torch.int32))

    return _TrtllmFp8Operands(
        query=query,
        cache=cache,
        block_tables=block_tables,
        seq_lens=torch.tensor(validated.valid_counts, dtype=torch.int32, device=device),
        workspace=torch.zeros(_TRTLLM_WORKSPACE_BYTES, dtype=torch.uint8, device=device),
    )


def _launch_trtllm_fp8(
    callable_: Any,
    operands: _TrtllmFp8Operands,
    *,
    softmax_scale: float,
) -> Any:
    return callable_(
        query=operands.query,
        kv_cache=operands.cache,
        workspace_buffer=operands.workspace,
        qk_nope_head_dim=_LATENT_DIM,
        kv_lora_rank=_LATENT_DIM,
        qk_rope_head_dim=_ROPE_DIM,
        block_tables=operands.block_tables,
        seq_lens=operands.seq_lens,
        max_seq_len=_SELECTED_K,
        bmm1_scale=softmax_scale,
        bmm2_scale=1.0,
        sparse_mla_top_k=_SELECTED_K,
    )


def profile_dsa_sparse_mla_attention_flashinfer_trtllm_fp8(
    *,
    num_queries: int,
    num_cache_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    selected_k: int,
    latent_dim: int,
    rope_dim: int,
    value_dim: int,
    softmax_scale: float,
    q_dtype: DType | str,
    cache_dtype: DType | str,
    index_dtype: str,
    output_dtype: DType | str,
    valid_counts: str,
    index_distribution: str,
    cache_layout: str,
) -> ComputeMetrics:
    """Profile vLLM's one-launch B200 FlashInfer sparse-MLA callable."""
    if num_heads not in _TRTLLM_SUPPORTED_NUM_HEADS:
        raise ProfilerNotImplemented(
            f"{_TRTLLM_FP8_BACKEND} supports num_heads in "
            f"{sorted(_TRTLLM_SUPPORTED_NUM_HEADS)}, got {num_heads}"
        )
    validated = _validate_args(
        num_queries=num_queries,
        num_cache_tokens=num_cache_tokens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        selected_k=selected_k,
        latent_dim=latent_dim,
        rope_dim=rope_dim,
        value_dim=value_dim,
        softmax_scale=softmax_scale,
        q_dtype=q_dtype,
        cache_dtype=cache_dtype,
        index_dtype=index_dtype,
        output_dtype=output_dtype,
        valid_counts=valid_counts,
        index_distribution=index_distribution,
        cache_layout=cache_layout,
        expected_num_heads=num_heads,
        expected_q_dtype=DType.FP8_E4M3,
        expected_cache_dtype=DType.FP8_E4M3,
        expected_cache_layout=_TRTLLM_FP8_CACHE_LAYOUT,
    )

    try:
        import torch
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
    except (ImportError, ModuleNotFoundError) as exc:
        raise ProfilerNotImplemented(
            f"{_TRTLLM_FP8_BACKEND} requires the repository vllm_env"
        ) from exc

    try:
        _require_b200(torch)
        device = torch.device("cuda", torch.cuda.current_device())
        operands = _build_trtllm_fp8_operands(
            torch,
            validated,
            num_heads=num_heads,
            device=device,
        )

        def kernel() -> Any:
            return _launch_trtllm_fp8(
                trtllm_batch_decode_with_kv_cache_mla,
                operands,
                softmax_scale=float(softmax_scale),
            )

        output = kernel()
        torch.cuda.synchronize(device)
        expected_shape = (num_queries, 1, num_heads, value_dim)
        if output.dtype is not torch.bfloat16 or tuple(output.shape) != expected_shape:
            raise KernelLaunchFailed(
                f"{_TRTLLM_FP8_BACKEND} returned {output.dtype} {tuple(output.shape)}, "
                f"expected BF16 {expected_shape}"
            )
        if not torch.isfinite(output).all():
            raise KernelLaunchFailed(f"{_TRTLLM_FP8_BACKEND} output must be finite")

        time_ms = Timer.cupti(kernel, kernel_name=_TRTLLM_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except (KernelLaunchFailed, ProfilerNotImplemented):
        raise
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_TRTLLM_FP8_BACKEND} ran out of GPU memory") from exc
    except Exception as exc:
        raise KernelLaunchFailed(f"{_TRTLLM_FP8_BACKEND} native callable failed") from exc

    flops = _logical_flops(
        num_queries=num_queries,
        num_heads=num_heads,
        selected_k=selected_k,
    )
    logical_bytes = _logical_bytes(
        num_queries=num_queries,
        num_heads=num_heads,
        selected_k=selected_k,
        valid_counts=validated.valid_counts,
        q_bytes=1,
        cache_bytes=1,
    )
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        energy_j=float(energy_j),
        tflops=flops / seconds / 1e12,
        memory_bandwidth_gbps=logical_bytes / seconds / 1e9,
    )


def profile_dsa_sparse_mla_attention_vllm_flashmla_bf16(
    *,
    num_queries: int,
    num_cache_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    selected_k: int,
    latent_dim: int,
    rope_dim: int,
    value_dim: int,
    softmax_scale: float,
    q_dtype: DType | str,
    cache_dtype: DType | str,
    index_dtype: str,
    output_dtype: DType | str,
    valid_counts: str,
    index_distribution: str,
    cache_layout: str,
) -> ComputeMetrics:
    """Profile the packaged one-launch FlashMLA BF16 sparse forward callable."""
    validated = _validate_args(
        num_queries=num_queries,
        num_cache_tokens=num_cache_tokens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        selected_k=selected_k,
        latent_dim=latent_dim,
        rope_dim=rope_dim,
        value_dim=value_dim,
        softmax_scale=softmax_scale,
        q_dtype=q_dtype,
        cache_dtype=cache_dtype,
        index_dtype=index_dtype,
        output_dtype=output_dtype,
        valid_counts=valid_counts,
        index_distribution=index_distribution,
        cache_layout=cache_layout,
    )

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ProfilerNotImplemented(f"{_FLASHMLA_BACKEND} requires PyTorch in vllm_env") from exc

    try:
        _require_h200(torch, backend=_FLASHMLA_BACKEND)
        flash_mla_sparse_fwd = _load_flashmla_sparse_fwd()
        device = torch.device("cuda", torch.cuda.current_device())
        operands = _build_operands(torch, validated, device=device)
        # q is consumed through SM90 TMA and cache through 16-byte cp.async;
        # the profiler constructs aligned storage and never realigns in timing.
        _validate_flashmla_layouts(operands)
        _check_flashmla_correctness(
            torch,
            flash_mla_sparse_fwd,
            operands,
            softmax_scale=float(softmax_scale),
        )

        latest_native_tuple: tuple[Any, Any, Any] | list[Any] | None = None

        def kernel() -> Any:
            nonlocal latest_native_tuple
            latest_native_tuple = flash_mla_sparse_fwd(
                operands.q,
                operands.cache,
                operands.selected_indices,
                float(softmax_scale),
                _VALUE_DIM,
                None,
                None,
            )
            return latest_native_tuple

        time_ms = Timer.cupti(kernel, kernel_name=_FLASHMLA_KERNEL_NAME)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)
    except ProfilerNotImplemented:
        raise
    except KernelLaunchFailed:
        raise
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_FLASHMLA_BACKEND} ran out of GPU memory") from exc
    except Exception as exc:
        raise KernelLaunchFailed(f"{_FLASHMLA_BACKEND} native callable failed") from exc

    flops = _logical_flops(num_queries=num_queries, num_heads=num_heads, selected_k=selected_k)
    logical_bytes = _logical_bytes(
        num_queries=num_queries,
        num_heads=num_heads,
        selected_k=selected_k,
        valid_counts=validated.valid_counts,
    )
    seconds = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=time_ms,
        energy_j=energy_j,
        tflops=flops / seconds / 1e12,
        memory_bandwidth_gbps=logical_bytes / seconds / 1e9,
    )
