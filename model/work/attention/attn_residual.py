"""Necessary-work accounting for Kimi-K3's attention-residual stream."""

from __future__ import annotations

from dataclasses import dataclass

from ..core import LearnedWeightGroup, MatmulGroup, Workload
from .base import AttentionSemantic


@dataclass(frozen=True)
class KimiAttentionResidual:
    """Layer-indexed snapshot-bank work shared by KDA and MLA layers.

    ``layer_indices`` are zero-based production layer indices.  A grouped
    ``LayerStack`` supplies all indices represented by one attention archetype;
    the helper returns the per-layer average so the generic stack count restores
    the exact heterogeneous total.
    """

    hidden: int
    block_size: int
    layer_indices: tuple[int, ...]
    total_layers: int
    include_output: bool = False
    activation_dtype_bytes: float = 2.0  # The production bank is BF16.

    def __post_init__(self) -> None:
        if self.hidden <= 0:
            raise ValueError("attention-residual hidden size must be positive")
        if self.block_size <= 0:
            raise ValueError("attention-residual block size must be positive")
        if self.total_layers <= 0:
            raise ValueError("attention-residual layer count must be positive")
        if not self.layer_indices:
            raise ValueError("attention-residual requires at least one layer index")
        if any(index < 0 or index >= self.total_layers for index in self.layer_indices):
            raise ValueError("attention-residual layer indices must be zero-based and in range")

    def _num_valid_blocks(self, layer_index: int) -> int:
        return (layer_index + self.block_size - 1) // self.block_size

    @property
    def mean_num_valid_blocks(self) -> float:
        return sum(self._num_valid_blocks(index) for index in self.layer_indices) / len(
            self.layer_indices
        )

    @property
    def mean_write_layers(self) -> float:
        return sum(index % self.block_size == 0 for index in self.layer_indices) / len(
            self.layer_indices
        )

    @property
    def output_num_valid_blocks(self) -> int:
        return (self.total_layers + self.block_size - 1) // self.block_size

    @property
    def row_bytes_per_token(self) -> float:
        return self.hidden * self.activation_dtype_bytes

    def matmul_groups(self) -> list[MatmulGroup]:
        """The two per-layer scalar score projections."""
        return [
            MatmulGroup(
                "attn_res.self_attention_res_proj",
                n=1,
                k=self.hidden,
                bucket="attn_proj",
                module="self_attention_res_proj",
            ),
            MatmulGroup(
                "attn_res.mlp_res_proj",
                n=1,
                k=self.hidden,
                bucket="attn_proj",
                module="mlp_res_proj",
            ),
        ]

    def learned_weight_groups(self) -> list[LearnedWeightGroup]:
        """The two per-layer RMSNorm scale vectors."""
        return [
            LearnedWeightGroup("attn_res.self_attention_res_norm", self.hidden, 1, "norm"),
            LearnedWeightGroup("attn_res.mlp_res_norm", self.hidden, 1, "norm"),
        ]

    def output_matmul_groups(self) -> list[MatmulGroup]:
        """The model-level scalar score projection used after the last layer."""
        return [
            MatmulGroup(
                "output_attn_res_proj",
                n=1,
                k=self.hidden,
                bucket="attn_proj",
                module="output_attn_res_proj",
            )
        ]

    def output_learned_weight_groups(self) -> list[LearnedWeightGroup]:
        """The model-level output aggregation RMSNorm scale vector."""
        return [LearnedWeightGroup("output_attn_res_norm", self.hidden, 1, "norm")]

    def semantic_segments(self, wl: Workload) -> list[AttentionSemantic]:
        """Return bank traffic and score/mix FLOPs for this layer archetype."""
        tokens = wl.matmul_tokens
        row_bytes = tokens * self.row_bytes_per_token
        nvb = self.mean_num_valid_blocks
        write_bytes = self.mean_write_layers * row_bytes
        aggregate_flops = 2.0 * nvb * tokens * self.hidden
        aggregate_bytes = nvb * row_bytes
        rows = [
            AttentionSemantic(
                "attn_res.agg1",
                flops=aggregate_flops,
                bytes=aggregate_bytes + write_bytes,
            ),
            AttentionSemantic(
                "attn_res.agg2",
                flops=aggregate_flops,
                bytes=aggregate_bytes,
            ),
        ]
        if self.include_output:
            output_nvb = self.output_num_valid_blocks
            rows.append(
                AttentionSemantic(
                    "attn_res.output_agg",
                    flops=2.0 * output_nvb * tokens * self.hidden,
                    bytes=output_nvb * row_bytes,
                )
            )
        return rows
