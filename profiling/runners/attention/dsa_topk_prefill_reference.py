"""Torch semantics for DSA prefill top-k index selection."""

from __future__ import annotations

import torch


def dsa_topk_prefill_reference(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    out: torch.Tensor,
    *,
    top_k: int,
) -> torch.Tensor:
    """Select per-row span-local indices into caller-owned output storage.

    For spans no longer than ``top_k``, the natural local indices are returned
    and unused slots are ``-1``. Longer spans use value-descending order with
    local-index ascending as the deterministic tie-break. Production CUDA does
    not contractually order selected indices or resolve equal-value ties, so a
    later backend compares selected sets and values for those longer spans.

    Source boundary: vLLM v0.23.0 commit
    0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665,
    ``top_k_per_row_prefill`` in ``csrc/libtorch_stable/sampler.cu`` and the
    sparse indexer. GLM-5.2 config revision
    b4734de4facf877f85769a911abafc5283eab3d9 sets ``index_topk=2048``.
    This function defines mathematical and valid-output semantics only; later
    profiling backends construct padded logits and time the production launch.
    """
    _validate(logits, row_starts, row_ends, out, top_k=top_k)

    out.fill_(-1)
    for row in range(logits.shape[0]):
        start = int(row_starts[row].item())
        end = int(row_ends[row].item())
        span_length = end - start
        if span_length == 0:
            continue

        if span_length <= top_k:
            selected = torch.arange(
                span_length,
                dtype=torch.int32,
                device=logits.device,
            )
        else:
            # Stable sorting preserves the original ascending local-index order
            # among equal values, providing the reference's explicit tie-break.
            selected = torch.argsort(
                logits[row, start:end],
                descending=True,
                stable=True,
            )[:top_k].to(torch.int32)
        out[row, : selected.numel()].copy_(selected)

    return out


def _validate(
    logits: object,
    row_starts: object,
    row_ends: object,
    out: object,
    *,
    top_k: object,
) -> None:
    tensors = {
        "logits": logits,
        "row_starts": row_starts,
        "row_ends": row_ends,
        "out": out,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    assert isinstance(logits, torch.Tensor)
    assert isinstance(row_starts, torch.Tensor)
    assert isinstance(row_ends, torch.Tensor)
    assert isinstance(out, torch.Tensor)

    if type(top_k) is not int:
        raise TypeError("top_k must be a Python int")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    _validate_ranks_and_shapes(logits, row_starts, row_ends, out, top_k)
    _validate_dtypes_and_devices(logits, row_starts, row_ends, out)
    _validate_layouts(logits, row_starts, row_ends, out)

    for name, tensor in (
        ("logits", logits),
        ("row_starts", row_starts),
        ("row_ends", row_ends),
    ):
        if torch._C._overlaps(out, tensor):
            raise ValueError(f"out must not alias {name}")

    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("logits must contain only finite values")

    num_keys = logits.shape[1]
    if bool(torch.any(row_starts < 0).item()):
        raise ValueError("row_starts must be nonnegative")
    if bool(torch.any(row_starts > row_ends).item()):
        raise ValueError("each row start must be less than or equal to its end")
    if bool(torch.any(row_ends > num_keys).item()):
        raise ValueError(f"row_ends must not exceed the logits width {num_keys}")


def _validate_ranks_and_shapes(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
) -> None:
    for name, tensor, expected_rank in (
        ("logits", logits, 2),
        ("row_starts", row_starts, 1),
        ("row_ends", row_ends, 1),
        ("out", out, 2),
    ):
        if tensor.ndim != expected_rank:
            raise ValueError(f"{name} must be rank {expected_rank}, got rank {tensor.ndim}")
        if any(dimension <= 0 for dimension in tensor.shape):
            raise ValueError(f"{name} dimensions must be positive, got {tuple(tensor.shape)}")

    num_rows = logits.shape[0]
    if row_starts.shape != (num_rows,):
        raise ValueError(f"row_starts shape must be ({num_rows},), got {tuple(row_starts.shape)}")
    if row_ends.shape != (num_rows,):
        raise ValueError(f"row_ends shape must be ({num_rows},), got {tuple(row_ends.shape)}")
    expected_out_shape = (num_rows, top_k)
    if out.shape != expected_out_shape:
        raise ValueError(f"out shape must be {expected_out_shape}, got {tuple(out.shape)}")


def _validate_dtypes_and_devices(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    out: torch.Tensor,
) -> None:
    if logits.dtype is not torch.float32:
        raise TypeError(f"logits dtype must be torch.float32, got {logits.dtype}")
    for name, tensor in (
        ("row_starts", row_starts),
        ("row_ends", row_ends),
        ("out", out),
    ):
        if tensor.dtype is not torch.int32:
            raise TypeError(f"{name} dtype must be torch.int32, got {tensor.dtype}")

    if any(tensor.device.type == "meta" for tensor in (logits, row_starts, row_ends, out)):
        raise ValueError("meta tensors are not supported")
    for name, tensor in (
        ("row_starts", row_starts),
        ("row_ends", row_ends),
        ("out", out),
    ):
        if tensor.device != logits.device:
            raise ValueError(
                f"{name} must be on the same device as logits, "
                f"got {tensor.device} and {logits.device}"
            )


def _validate_layouts(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    out: torch.Tensor,
) -> None:
    for name, tensor in (
        ("logits", logits),
        ("row_starts", row_starts),
        ("row_ends", row_ends),
        ("out", out),
    ):
        if tensor.layout is not torch.strided:
            raise ValueError(f"{name} must have torch.strided layout")
        # Status 1 is definite overlap. Status 2 means Torch cannot prove the
        # layout either way and must remain accepted for padded logits views.
        if int(torch._debug_has_internal_overlap(tensor)) == 1:
            raise ValueError(f"{name} must not have internal overlap")

    if logits.stride(-1) != 1:
        raise ValueError("logits innermost stride must be 1")
    for name, tensor in (("row_starts", row_starts), ("row_ends", row_ends)):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
