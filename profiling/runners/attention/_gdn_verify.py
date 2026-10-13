"""Operands of a Gated DeltaNet recurrence on a speculative (MTP verify) step.

Shared by the two verify recurrences vLLM runs (``qwen_gdn_linear_attn.py``):
``fused_sigmoid_gating_delta_rule_update`` with ``num_accepted_tokens`` (a step
that also carries a prefill) and ``fused_gdn_decode_post_conv_mtp`` (a pure
verify step). Each request carries ``q`` request-major tokens; the FP32 state
pool holds one checkpoint slot per token (``state_indices`` ``[B, q]``, slot
zero reserved) in vLLM's ``[Hv, Dv, Dk]`` slot layout. Activations come from
one backing ``[T, 2·Hq·Dk + 2·Hv·Dv]`` qkvz projection and one ``[T, 2·Hv]``
ba projection, as in production, so the fused kernel sees strided rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiling.runners.attention.gdn_causal_conv_verify_vllm_triton import (
    accepted_counts,
    checkpoint_indices,
)


@dataclass(frozen=True)
class VerifyRecurrenceOperands:
    batch_size: int
    per_request: int
    num_qk_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    qkvz: Any
    ba: Any
    A_log: Any
    dt_bias: Any
    norm_weight: Any
    state: Any
    state_indices: Any
    num_accepted: Any
    cu_seqlens: Any

    @property
    def num_tokens(self) -> int:
        return self.batch_size * self.per_request

    @property
    def _key_width(self) -> int:
        return self.num_qk_heads * self.key_head_dim

    @property
    def _value_width(self) -> int:
        return self.num_value_heads * self.value_head_dim

    @property
    def mixed_qkv(self) -> Any:
        """Strided ``[T, 2·Hq·Dk + Hv·Dv]`` view, as split from qkvz."""
        return self.qkvz[:, : 2 * self._key_width + self._value_width]

    @property
    def output_gate(self) -> Any:
        """Strided ``[T, Hv, Dv]`` view of qkvz's z part."""
        z = self.qkvz[:, 2 * self._key_width + self._value_width :]
        return z.view(self.num_tokens, self.num_value_heads, self.value_head_dim)

    @property
    def b(self) -> Any:
        return self.ba[:, : self.num_value_heads]

    @property
    def a(self) -> Any:
        return self.ba[:, self.num_value_heads :]

    def semantic_qkv(self) -> tuple[Any, Any, Any]:
        """Contiguous ``[T, H, D]`` query, key and value."""
        k, v = self._key_width, self._value_width
        tokens = self.num_tokens
        query = self.qkvz[:, :k].reshape(tokens, self.num_qk_heads, self.key_head_dim)
        key = self.qkvz[:, k : 2 * k].reshape(tokens, self.num_qk_heads, self.key_head_dim)
        value = self.qkvz[:, 2 * k : 2 * k + v].reshape(
            tokens, self.num_value_heads, self.value_head_dim
        )
        return query.contiguous(), key.contiguous(), value.contiguous()


def build_verify_recurrence_operands(
    torch: Any,
    *,
    batch_size: int,
    per_request: int,
    num_qk_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    dtype: Any,
    state_dtype: Any,
    device: Any,
) -> VerifyRecurrenceOperands:
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    tokens = batch_size * per_request
    width = 2 * num_qk_heads * key_head_dim + 2 * num_value_heads * value_head_dim
    qkvz = 0.25 * torch.randn((tokens, width), dtype=dtype, device=device, generator=generator)
    ba = torch.empty((tokens, 2 * num_value_heads), dtype=dtype, device=device).uniform_(
        -0.25, 0.25, generator=generator
    )
    state = torch.empty(
        (tokens + 1, num_value_heads, value_head_dim, key_head_dim),
        dtype=state_dtype,
        device=device,
    ).normal_(mean=0.0, std=0.01, generator=generator)
    norm_weight = torch.empty((value_head_dim,), dtype=dtype, device=device).uniform_(
        0.5, 1.5, generator=generator
    )
    return VerifyRecurrenceOperands(
        batch_size=batch_size,
        per_request=per_request,
        num_qk_heads=num_qk_heads,
        num_value_heads=num_value_heads,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        qkvz=qkvz,
        ba=ba,
        A_log=torch.zeros((num_value_heads,), dtype=state_dtype, device=device),
        dt_bias=torch.zeros((num_value_heads,), dtype=state_dtype, device=device),
        norm_weight=norm_weight,
        state=state,
        state_indices=torch.tensor(
            checkpoint_indices(batch_size, per_request), dtype=torch.int32, device=device
        ),
        num_accepted=torch.tensor(
            accepted_counts(batch_size, per_request), dtype=torch.int32, device=device
        ),
        cu_seqlens=torch.arange(0, tokens + 1, per_request, dtype=torch.int32, device=device),
    )
