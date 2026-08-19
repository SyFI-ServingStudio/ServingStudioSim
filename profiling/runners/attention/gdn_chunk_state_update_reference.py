"""Torch semantics for Qwen GDN's fused chunk state update."""

from __future__ import annotations

import torch

__all__ = ["gdn_chunk_state_update_reference"]

_CHUNK_SIZE = 64


def gdn_chunk_state_update_reference(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g_cumsum: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run Qwen's ragged BF16 chunk-state recurrence on CPU.

    The token-major semantic inputs are BF16 ``k [T, Hg, K]``, BF16
    ``w [T, H, K]``, BF16 ``u [T, H, V]``, FP32 chunk-local inclusive
    ``g_cumsum [T, H]``, FP32 initial state ``[N, H, V, K]`` oriented as
    ``[V, K]``, and int32 ragged sequence boundaries.

    For each sequence-local chunk and output head, the incoming FP32 state is
    first rounded to a BF16 snapshot. The value residual uses that rounded
    snapshot in a BF16-input/FP32-accumulating dot. The residual is stored once
    as BF16 ``v_new``, while a *separate* BF16 rounding of the FP32 residual
    after chunk-end gate decay feeds the state update. The recurrent state
    remains FP32 across chunks and resets to the corresponding initial state at
    each sequence boundary.

    Chunk size 64, ordinary exponential decay, all three output dtypes, and the
    enabled initial/final-state and new-value paths are frozen Qwen semantics,
    not options. Strided CPU inputs are accepted. Outputs are fresh contiguous
    tensors and no input is mutated.
    """
    boundaries = _validate(k, w, u, g_cumsum, initial_state, cu_seqlens)
    num_tokens, num_key_heads, key_head_dim = k.shape
    num_heads, value_head_dim = u.shape[1:]
    heads_per_key = num_heads // num_key_heads
    num_chunks = sum(
        (sequence_end - sequence_start + _CHUNK_SIZE - 1) // _CHUNK_SIZE
        for sequence_start, sequence_end in zip(boundaries, boundaries[1:])
    )

    h = torch.empty(
        (num_chunks, num_heads, value_head_dim, key_head_dim),
        dtype=torch.bfloat16,
        device="cpu",
    )
    v_new = torch.empty(
        (num_tokens, num_heads, value_head_dim),
        dtype=torch.bfloat16,
        device="cpu",
    )
    final_state = torch.empty_like(initial_state, memory_format=torch.contiguous_format)

    global_chunk = 0
    for sequence, (sequence_start, sequence_end) in enumerate(zip(boundaries, boundaries[1:])):
        # Clone makes the recurrent state independent of input storage and
        # preserves FP32 state across chunks within this sequence.
        state = initial_state[sequence].contiguous().clone()
        for chunk_start in range(sequence_start, sequence_end, _CHUNK_SIZE):
            chunk_end = min(chunk_start + _CHUNK_SIZE, sequence_end)
            chunk_gate = g_cumsum[chunk_start:chunk_end]

            for head in range(num_heads):
                key_head = head // heads_per_key
                snapshot = state[head].to(torch.bfloat16)
                h[global_chunk, head].copy_(snapshot)

                # Production rounds the incoming state to BF16 before this
                # dot, but accumulates the products in FP32.
                correction = torch.matmul(
                    w[chunk_start:chunk_end, head].float(),
                    snapshot.float().transpose(0, 1),
                )
                residual = u[chunk_start:chunk_end, head].float() - correction
                v_new[chunk_start:chunk_end, head].copy_(residual.to(torch.bfloat16))

                end_gate = chunk_gate[-1, head]
                decay = torch.exp(end_gate - chunk_gate[:, head])

                # Do not reuse v_new: production decays the unrounded FP32
                # residual and rounds that factor independently to BF16.
                state_factor = (residual * decay.unsqueeze(-1)).to(torch.bfloat16)
                update = torch.matmul(
                    state_factor.float().transpose(0, 1),
                    k[chunk_start:chunk_end, key_head].float(),
                )
                state[head].copy_(state[head] * torch.exp(end_gate) + update)

            global_chunk += 1
        final_state[sequence].copy_(state)

    return h.contiguous(), v_new.contiguous(), final_state.contiguous()


def _validate(
    k: object,
    w: object,
    u: object,
    g_cumsum: object,
    initial_state: object,
    cu_seqlens: object,
) -> list[int]:
    tensors = {
        "k": k,
        "w": w,
        "u": u,
        "g_cumsum": g_cumsum,
        "initial_state": initial_state,
        "cu_seqlens": cu_seqlens,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    assert all(isinstance(tensor, torch.Tensor) for tensor in tensors.values())
    k = tensors["k"]
    w = tensors["w"]
    u = tensors["u"]
    g_cumsum = tensors["g_cumsum"]
    initial_state = tensors["initial_state"]
    cu_seqlens = tensors["cu_seqlens"]

    for name, tensor in tensors.items():
        if tensor.layout is not torch.strided:
            raise ValueError(f"{name} must have torch.strided layout")

    devices = {tensor.device for tensor in tensors.values()}
    if len(devices) != 1:
        rendered = ", ".join(f"{name}={tensor.device}" for name, tensor in tensors.items())
        raise ValueError(f"all inputs must be on the same device, got {rendered}")
    device = next(iter(devices))
    if device.type == "meta":
        raise ValueError("meta tensors are not supported")
    if device.type != "cpu":
        raise ValueError(f"semantic reference requires CPU tensors, got {device.type}")

    expected_ranks = {
        "k": 3,
        "w": 3,
        "u": 3,
        "g_cumsum": 2,
        "initial_state": 4,
        "cu_seqlens": 1,
    }
    for name, tensor in tensors.items():
        expected_rank = expected_ranks[name]
        if tensor.ndim != expected_rank:
            raise ValueError(f"{name} must be rank {expected_rank}, got rank {tensor.ndim}")

    for name, tensor in (("k", k), ("w", w), ("u", u)):
        if tensor.dtype is not torch.bfloat16:
            raise TypeError(f"{name} dtype must be torch.bfloat16, got {tensor.dtype}")
    for name, tensor in (("g_cumsum", g_cumsum), ("initial_state", initial_state)):
        if tensor.dtype is not torch.float32:
            raise TypeError(f"{name} dtype must be torch.float32, got {tensor.dtype}")
    if cu_seqlens.dtype is not torch.int32:
        raise TypeError(f"cu_seqlens dtype must be torch.int32, got {cu_seqlens.dtype}")

    num_tokens, num_key_heads, key_head_dim = k.shape
    if num_tokens <= 0:
        raise ValueError(f"k token dimension must be positive, got {num_tokens}")
    if num_key_heads <= 0:
        raise ValueError(f"k head dimension must be positive, got {num_key_heads}")
    if key_head_dim <= 0:
        raise ValueError(f"k feature dimension must be positive, got {key_head_dim}")

    if w.shape[0] != num_tokens:
        raise ValueError(f"w token dimension must equal {num_tokens}, got {w.shape[0]}")
    num_heads = w.shape[1]
    if num_heads <= 0:
        raise ValueError(f"w head dimension must be positive, got {num_heads}")
    if w.shape[2] != key_head_dim:
        raise ValueError(
            f"w feature dimension must equal key dimension {key_head_dim}, got {w.shape[2]}"
        )
    if num_heads % num_key_heads != 0:
        raise ValueError(
            "output head count must be divisible by key head count, got "
            f"H={num_heads}, Hg={num_key_heads}"
        )

    if u.shape[0] != num_tokens or u.shape[1] != num_heads:
        raise ValueError(
            f"u leading dimensions must be {(num_tokens, num_heads)}, got {tuple(u.shape[:2])}"
        )
    value_head_dim = u.shape[2]
    if value_head_dim <= 0:
        raise ValueError(f"u feature dimension must be positive, got {value_head_dim}")
    expected_gate_shape = (num_tokens, num_heads)
    if g_cumsum.shape != expected_gate_shape:
        raise ValueError(
            f"g_cumsum must have shape {expected_gate_shape}, got {tuple(g_cumsum.shape)}"
        )

    if cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must contain at least one sequence")
    boundaries = [int(boundary) for boundary in cu_seqlens.tolist()]
    if any(boundary < 0 or boundary > num_tokens for boundary in boundaries):
        raise ValueError(f"cu_seqlens boundaries must be in [0, {num_tokens}]")
    if boundaries[0] != 0:
        raise ValueError(f"cu_seqlens must start at zero, got {boundaries[0]}")
    if boundaries[-1] != num_tokens:
        raise ValueError(f"cu_seqlens must end at token count {num_tokens}, got {boundaries[-1]}")
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("cu_seqlens must be strictly increasing with no empty sequences")

    num_sequences = len(boundaries) - 1
    expected_state_shape = (
        num_sequences,
        num_heads,
        value_head_dim,
        key_head_dim,
    )
    if initial_state.shape != expected_state_shape:
        raise ValueError(
            f"initial_state must have shape {expected_state_shape}, "
            f"got {tuple(initial_state.shape)}"
        )

    for name, tensor in (
        ("k", k),
        ("w", w),
        ("u", u),
        ("g_cumsum", g_cumsum),
        ("initial_state", initial_state),
    ):
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} must contain only finite values")

    return boundaries
