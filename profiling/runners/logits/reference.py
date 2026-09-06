"""Independent value oracles for vocabulary gathering and greedy sampling."""

from collections.abc import Sequence
from typing import Any


def _validate_logits(torch: Any, logits: Any) -> None:
    if logits.ndim != 2 or any(size <= 0 for size in logits.shape):
        raise ValueError("logits must have non-empty shape [num_rows, vocab_size]")
    if logits.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("logits must use bfloat16 or float32")
    if logits.stride(1) != 1 or logits.stride(0) < logits.shape[1]:
        raise ValueError("logits require unit column stride and row_stride >= vocab_size")


def vocab_parallel_all_gather_reference(torch: Any, shards: Sequence[Any]) -> Any:
    """Concatenate equal contiguous rank shards in rank order along vocabulary.

    All shards share shape, dtype and device. NaN and infinity are preserved.
    """
    if not shards:
        raise ValueError("at least one rank shard is required")
    first = shards[0]
    for shard in shards:
        _validate_logits(torch, shard)
        if not shard.is_contiguous():
            raise ValueError("rank shards must be contiguous")
        if shard.shape != first.shape or shard.dtype != first.dtype or shard.device != first.device:
            raise ValueError("rank shards must share shape, dtype and device")
    return torch.cat(tuple(shards), dim=-1).contiguous()


def logits_copy_reference(torch: Any, logits: Any, output_dtype: Any) -> Any:
    """Copy logits into independent contiguous storage with the requested dtype.

    BF16 and FP32 input/output are supported; NaN and infinity are preserved.
    """
    _validate_logits(torch, logits)
    if output_dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("logits copy output must use bfloat16 or float32")
    result = torch.empty(logits.shape, dtype=output_dtype, device=logits.device)
    result.copy_(logits)
    return result


def logits_argmax_reference(torch: Any, logits: Any) -> Any:
    """Return the first maximal column as int64; allow infinity, reject NaN.

    This oracle deliberately avoids argmax, the production reduction being
    checked. NaN ordering is outside the synthetic profiling contract.
    """
    _validate_logits(torch, logits)
    if torch.isnan(logits).any().item():
        raise ValueError("argmax reference does not accept NaN logits")
    maxima = logits.amax(dim=-1, keepdim=True)
    columns = torch.arange(logits.shape[1], dtype=torch.int64, device=logits.device)
    candidates = torch.where(logits == maxima, columns, logits.shape[1])
    return candidates.amin(dim=-1)


__all__ = [
    "vocab_parallel_all_gather_reference",
    "logits_copy_reference",
    "logits_argmax_reference",
]
