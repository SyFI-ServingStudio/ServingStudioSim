"""Framework-independent shape and output oracle for MoE block alignment."""

from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_BLOCK_SIZES = frozenset({8, 16, 32, 48, 64})


@dataclass(frozen=True)
class AlignmentShape:
    num_tokens: int
    num_experts: int
    top_k: int
    block_size: int

    @property
    def num_routes(self) -> int:
        return self.num_tokens * self.top_k

    @property
    def expert_counts(self) -> tuple[int, ...]:
        quotient, remainder = divmod(self.num_routes, self.num_experts)
        return tuple(quotient + (expert < remainder) for expert in range(self.num_experts))

    @property
    def padded_routes(self) -> int:
        return sum(round_up(count, self.block_size) for count in self.expert_counts)


def validate_shape(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    block_size: int,
) -> AlignmentShape:
    """Validate the public four-field cache identity."""

    scalars = {
        "num_tokens": num_tokens,
        "num_experts": num_experts,
        "top_k": top_k,
        "block_size": block_size,
    }
    for name, value in scalars.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if top_k > num_experts:
        raise ValueError("top_k must not exceed num_experts")
    if block_size not in SUPPORTED_BLOCK_SIZES:
        raise ValueError(f"block_size must be one of {sorted(SUPPORTED_BLOCK_SIZES)}")

    return AlignmentShape(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        block_size=block_size,
    )


def build_topk_ids(shape: AlignmentShape) -> tuple[tuple[int, ...], ...]:
    """Construct the canonical balanced routing used for this compact schema."""

    return tuple(
        tuple((token * shape.top_k + slot) % shape.num_experts for slot in range(shape.top_k))
        for token in range(shape.num_tokens)
    )


def expected_block_owners(expert_counts: tuple[int, ...], block_size: int) -> tuple[int, ...]:
    if block_size not in SUPPORTED_BLOCK_SIZES:
        raise ValueError(f"block_size must be one of {sorted(SUPPORTED_BLOCK_SIZES)}")
    if any(type(count) is not int or count < 0 for count in expert_counts):
        raise ValueError("expert_counts entries must be non-negative integers")
    return tuple(
        expert
        for expert, count in enumerate(expert_counts)
        for _ in range(ceil_div(count, block_size))
    )


def logical_bytes(shape: AlignmentShape) -> int:
    """Visible int32 input/output traffic, excluding private kernel workspace."""

    blocks = len(expected_block_owners(shape.expert_counts, shape.block_size))
    int32_values = shape.num_routes + shape.padded_routes + blocks + 1
    return 4 * int32_values


def round_up(value: int, multiple: int) -> int:
    return ceil_div(value, multiple) * multiple


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor
