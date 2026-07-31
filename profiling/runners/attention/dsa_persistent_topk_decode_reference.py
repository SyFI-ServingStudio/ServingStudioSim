"""Torch semantics for pinned persistent DSA decode top-k selection."""

from __future__ import annotations

import torch

_SUPPORTED_TOP_K = frozenset({512, 1024, 2048})


def dsa_persistent_topk_decode_reference(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor,
    *,
    top_k: int,
    max_seq_len: int,
) -> torch.Tensor:
    """Select valid-prefix, row-local top-k indices into caller-owned storage.

    Rows no longer than ``top_k`` produce their natural local indices followed
    by ``-1``. Longer rows use value-descending order with ascending local index
    as the deterministic tie-break. Pinned CUDA does not contractually specify
    output order or equal-value tie selection, so its eventual correctness gate
    must compare selected sets and values for long rows.

    This models the vLLM v0.23.0 boundary at commit
    0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665: schema
    ``persistent_topk(Tensor logits, Tensor lengths, Tensor! output, Tensor
    workspace, int k, int max_seq_len) -> ()`` in
    ``csrc/libtorch_stable/topk.cu`` and ``persistent_topk.cuh``. GLM-5.2 uses
    K=2048. The reference defines mathematical and valid-output semantics only;
    scratch-workspace reset, persistent/FilteredTopK dispatch, launch sequencing,
    and timing belong to a later production backend once that API is runnable.

    ``max_seq_len`` is validated dispatch metadata and otherwise does not alter
    the result. Invalid logits tails are intentionally never inspected because
    production ``clean_logits=false`` leaves them undefined.
    """
    _validate(
        logits,
        lengths,
        out,
        top_k=top_k,
        max_seq_len=max_seq_len,
    )

    flat_lengths = lengths.view(-1)
    out.fill_(-1)
    for row in range(logits.shape[0]):
        length = int(flat_lengths[row].item())
        if length == 0:
            continue

        if length <= top_k:
            selected = torch.arange(
                length,
                dtype=torch.int32,
                device=logits.device,
            )
        else:
            # Stable sorting preserves ascending local-index order among equal
            # values and touches only the contractual valid prefix.
            selected = torch.argsort(
                logits[row, :length],
                descending=True,
                stable=True,
            )[:top_k].to(torch.int32)
        out[row, : selected.numel()].copy_(selected)

    return out


def _validate(
    logits: object,
    lengths: object,
    out: object,
    *,
    top_k: object,
    max_seq_len: object,
) -> None:
    for name, tensor in (("logits", logits), ("lengths", lengths), ("out", out)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    assert isinstance(logits, torch.Tensor)
    assert isinstance(lengths, torch.Tensor)
    assert isinstance(out, torch.Tensor)

    if type(top_k) is not int:
        raise TypeError("top_k must be a Python int")
    if top_k not in _SUPPORTED_TOP_K:
        raise ValueError(f"top_k must be one of {sorted(_SUPPORTED_TOP_K)}, got {top_k}")
    if type(max_seq_len) is not int:
        raise TypeError("max_seq_len must be a Python int")
    if max_seq_len < 0:
        raise ValueError("max_seq_len must be nonnegative")

    _validate_ranks_and_shapes(logits, lengths, out, top_k)
    _validate_dtypes_and_devices(logits, lengths, out)
    _validate_layouts(logits, lengths, out)

    if max_seq_len > logits.shape[1]:
        raise ValueError(
            f"max_seq_len must not exceed the logits width {logits.shape[1]}, got {max_seq_len}"
        )

    for name, tensor in (("logits", logits), ("lengths", lengths)):
        if torch._C._overlaps(out, tensor):
            raise ValueError(f"out must not alias {name}")

    flat_lengths = lengths.view(-1)
    if bool(torch.any(flat_lengths < 0).item()):
        raise ValueError("lengths must be nonnegative")
    if bool(torch.any(flat_lengths > max_seq_len).item()):
        raise ValueError(f"lengths must not exceed max_seq_len {max_seq_len}")

    # Check only prefixes the production result contracts. Invalid tails can
    # contain arbitrary NaN/Inf and must remain unread.
    for row in range(logits.shape[0]):
        length = int(flat_lengths[row].item())
        if length and not bool(torch.isfinite(logits[row, :length]).all().item()):
            raise ValueError(f"logits valid prefix for row {row} must be finite")


def _validate_ranks_and_shapes(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
) -> None:
    if logits.ndim != 2:
        raise ValueError(f"logits must be rank 2, got rank {logits.ndim}")
    if any(dimension <= 0 for dimension in logits.shape):
        raise ValueError(f"logits dimensions must be positive, got {tuple(logits.shape)}")
    if lengths.ndim not in (1, 2):
        raise ValueError(f"lengths must be rank 1 or 2, got rank {lengths.ndim}")
    if any(dimension <= 0 for dimension in lengths.shape):
        raise ValueError(f"lengths dimensions must be positive, got {tuple(lengths.shape)}")
    if out.ndim != 2:
        raise ValueError(f"out must be rank 2, got rank {out.ndim}")
    if any(dimension <= 0 for dimension in out.shape):
        raise ValueError(f"out dimensions must be positive, got {tuple(out.shape)}")

    num_rows = logits.shape[0]
    if lengths.numel() != num_rows:
        raise ValueError(f"lengths must contain exactly {num_rows} values, got {lengths.numel()}")
    expected_out_shape = (num_rows, top_k)
    if out.shape != expected_out_shape:
        raise ValueError(f"out shape must be {expected_out_shape}, got {tuple(out.shape)}")


def _validate_dtypes_and_devices(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor,
) -> None:
    if logits.dtype is not torch.float32:
        raise TypeError(f"logits dtype must be torch.float32, got {logits.dtype}")
    if lengths.dtype is not torch.int32:
        raise TypeError(f"lengths dtype must be torch.int32, got {lengths.dtype}")
    if out.dtype is not torch.int32:
        raise TypeError(f"out dtype must be torch.int32, got {out.dtype}")

    if any(tensor.device.type == "meta" for tensor in (logits, lengths, out)):
        raise ValueError("meta tensors are not supported")
    for name, tensor in (("lengths", lengths), ("out", out)):
        if tensor.device != logits.device:
            raise ValueError(
                f"{name} must be on the same device as logits, "
                f"got {tensor.device} and {logits.device}"
            )


def _validate_layouts(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor,
) -> None:
    for name, tensor in (("logits", logits), ("lengths", lengths), ("out", out)):
        if tensor.layout is not torch.strided:
            raise ValueError(f"{name} must have torch.strided layout")
        # Status 1 is definite overlap. Status 2 is indeterminate and remains
        # accepted for otherwise valid padded logits views.
        if int(torch._debug_has_internal_overlap(tensor)) == 1:
            raise ValueError(f"{name} must not have internal overlap")

    if logits.stride(-1) != 1:
        raise ValueError("logits innermost stride must be 1")
    if not lengths.is_contiguous():
        raise ValueError("lengths must be contiguous")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
