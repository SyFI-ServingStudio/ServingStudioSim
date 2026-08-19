"""Torch semantics for Qwen GDN chunk-local triangular inversion."""

from __future__ import annotations

import torch

__all__ = ["gdn_chunk_solve_tril_reference"]

_CHUNK_SIZE = 64


def gdn_chunk_solve_tril_reference(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Compute Qwen's BF16 chunk-local ``(I + A)^{-1}`` result.

    ``A`` is token-major FP32 ``[T, H, 64]`` positive-sign KKT storage and
    ``cu_seqlens`` is an int32 prefix array for positive ragged sequences.
    Each sequence is split independently into consecutive chunks of at most
    64 tokens. Within one valid ``p x p`` strict-lower block ``L``, the FP32
    recurrence is

    ``M[i, j] = -L[i, j] - sum(L[i, k] * M[k, j], k=j+1..i-1)``

    below the diagonal, with an identity diagonal and zero upper triangle.
    The completed FP32 block is cast once to BF16. Sequence and chunk
    boundaries reset the recurrence; columns outside a partial chunk remain
    zero.

    Chunk size 64, FP32 input arithmetic, and BF16 output are frozen Qwen
    semantics rather than options. Strided CPU inputs are accepted. The
    returned ``[T, H, 64]`` tensor is newly allocated and contiguous, and no
    input is mutated.
    """
    boundaries = _validate(A, cu_seqlens)
    num_tokens, num_heads, _ = A.shape
    output = torch.zeros(
        (num_tokens, num_heads, _CHUNK_SIZE),
        dtype=torch.bfloat16,
        device="cpu",
    )

    for sequence_start, sequence_end in zip(boundaries, boundaries[1:]):
        for chunk_start in range(sequence_start, sequence_end, _CHUNK_SIZE):
            chunk_end = min(chunk_start + _CHUNK_SIZE, sequence_end)
            chunk_length = chunk_end - chunk_start
            block = A[chunk_start:chunk_end, :, :chunk_length].permute(1, 0, 2)

            # Store only the strict-lower part while applying the same row
            # recurrence independently per head. Keeping this local avoids a
            # global padded solve that could obscure reset and FP32 ordering.
            strict_inverse = torch.zeros(
                (num_heads, chunk_length, chunk_length),
                dtype=torch.float32,
                device="cpu",
            )
            for row in range(1, chunk_length):
                negative_input_row = -block[:, row, :row]
                solved_row = negative_input_row
                if row > 1:
                    solved_row = solved_row + torch.bmm(
                        negative_input_row.unsqueeze(1),
                        strict_inverse[:, :row, :row],
                    ).squeeze(1)
                strict_inverse[:, row, :row].copy_(solved_row)

            inverse = strict_inverse + torch.eye(
                chunk_length,
                dtype=torch.float32,
                device="cpu",
            ).unsqueeze(0)
            output[chunk_start:chunk_end, :, :chunk_length].copy_(
                inverse.permute(1, 0, 2).to(torch.bfloat16)
            )

    return output


def _validate(A: object, cu_seqlens: object) -> list[int]:
    tensors = {"A": A, "cu_seqlens": cu_seqlens}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    assert isinstance(A, torch.Tensor)
    assert isinstance(cu_seqlens, torch.Tensor)

    for name, tensor in tensors.items():
        if tensor.layout is not torch.strided:
            raise ValueError(f"{name} must have torch.strided layout")

    if A.device != cu_seqlens.device:
        raise ValueError(
            f"A and cu_seqlens must be on the same device, got {A.device} and {cu_seqlens.device}"
        )
    if A.device.type == "meta":
        raise ValueError("meta tensors are not supported")
    if A.device.type != "cpu":
        raise ValueError(f"semantic reference requires CPU tensors, got {A.device.type}")

    if A.ndim != 3:
        raise ValueError(f"A must be rank 3, got rank {A.ndim}")
    if cu_seqlens.ndim != 1:
        raise ValueError(f"cu_seqlens must be rank 1, got rank {cu_seqlens.ndim}")
    if A.dtype is not torch.float32:
        raise TypeError(f"A dtype must be torch.float32, got {A.dtype}")
    if cu_seqlens.dtype is not torch.int32:
        raise TypeError(f"cu_seqlens dtype must be torch.int32, got {cu_seqlens.dtype}")

    num_tokens, num_heads, storage_width = A.shape
    if num_tokens <= 0:
        raise ValueError(f"A token dimension must be positive, got {num_tokens}")
    if num_heads <= 0:
        raise ValueError(f"A head dimension must be positive, got {num_heads}")
    if storage_width != _CHUNK_SIZE:
        raise ValueError(
            f"A last dimension must equal chunk size {_CHUNK_SIZE}, got {storage_width}"
        )
    if cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must contain at least one sequence")
    if not torch.isfinite(A).all().item():
        raise ValueError("A must contain only finite values")

    boundaries = [int(boundary) for boundary in cu_seqlens.tolist()]
    if any(boundary < 0 or boundary > num_tokens for boundary in boundaries):
        raise ValueError(f"cu_seqlens boundaries must be in [0, {num_tokens}]")
    if boundaries[0] != 0:
        raise ValueError(f"cu_seqlens must start at zero, got {boundaries[0]}")
    if boundaries[-1] != num_tokens:
        raise ValueError(f"cu_seqlens must end at token count {num_tokens}, got {boundaries[-1]}")
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("cu_seqlens must be strictly increasing with no empty sequences")

    # Production consumes only strict-lower local matrices. Validate all
    # diagonal, upper, and partial-chunk unused storage before doing any math.
    for sequence_start, sequence_end in zip(boundaries, boundaries[1:]):
        for chunk_start in range(sequence_start, sequence_end, _CHUNK_SIZE):
            chunk_end = min(chunk_start + _CHUNK_SIZE, sequence_end)
            chunk_length = chunk_end - chunk_start
            local = A[chunk_start:chunk_end]
            for row in range(chunk_length):
                if torch.count_nonzero(local[row, :, row:]).item() != 0:
                    raise ValueError(
                        "A must be exactly strict-lower within every local chunk; "
                        "diagonal, upper triangle, and unused columns must be zero"
                    )
    return boundaries
