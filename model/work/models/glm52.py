"""GLM-5.2 DSA/MoE model.work composition.

GLM-5.2 is heterogeneous in two independent ways: the first three layers use a
dense SwiGLU, and DSA full-index layers occur every fourth sparse layer after the
initial three index-share layers. The four stacks below preserve that schedule,
which is also the unified CostTree's stable archetype order.
"""

from __future__ import annotations

from ..attention.glm52_dsa import Glm52DsaAttention
from ..core import LayerStack, Model, NormWeightGroup, dtype_bytes
from ..ffn.dense import DenseSwiGLU
from ..ffn.moe import MoE


def _require_schedule(raw_config: dict) -> tuple[int, int, int, int]:
    layers = raw_config["num_hidden_layers"]
    mlp_types = raw_config.get("mlp_layer_types")
    indexer_types = raw_config.get("indexer_types")
    if mlp_types is None or indexer_types is None:
        raise ValueError("GLM-5.2 model.work requires explicit mlp_layer_types and indexer_types")
    if len(mlp_types) != layers or len(indexer_types) != layers:
        raise ValueError("GLM-5.2 layer schedules must each match num_hidden_layers")

    dense_full = sum(
        mlp == "dense" and indexer == "full" for mlp, indexer in zip(mlp_types, indexer_types)
    )
    # The model's explicit first six entries identify the initial three sparse
    # shared-index layers; use positional counts so duplicate values are harmless.
    sparse_initial_share = sum(
        1
        for layer, (mlp, indexer) in enumerate(zip(mlp_types, indexer_types))
        if 3 <= layer < 6 and mlp == "sparse" and indexer == "shared"
    )
    sparse_cycle_full = sum(
        1
        for layer, (mlp, indexer) in enumerate(zip(mlp_types, indexer_types))
        if layer >= 6 and mlp == "sparse" and indexer == "full"
    )
    sparse_cycle_share = sum(
        1
        for layer, (mlp, indexer) in enumerate(zip(mlp_types, indexer_types))
        if layer >= 6 and mlp == "sparse" and indexer == "shared"
    )
    expected = (3, 3, 18, 54)
    actual = (dense_full, sparse_initial_share, sparse_cycle_full, sparse_cycle_share)
    if actual != expected:
        raise ValueError(f"unsupported GLM-5.2 layer schedule: expected {expected}, got {actual}")
    if any(mlp != "dense" for mlp in mlp_types[:3]) or any(
        mlp != "sparse" for mlp in mlp_types[3:]
    ):
        raise ValueError("GLM-5.2 expects dense layers 0..2 followed by sparse layers")
    return actual


def _attention(raw_config: dict, *, full_index: bool) -> Glm52DsaAttention:
    weight_bytes = dtype_bytes(raw_config.get("dtype") or raw_config.get("torch_dtype", "bfloat16"))
    return Glm52DsaAttention(
        hidden=raw_config["hidden_size"],
        num_heads=raw_config["num_attention_heads"],
        q_lora_rank=raw_config["q_lora_rank"],
        kv_lora_rank=raw_config["kv_lora_rank"],
        qk_nope_head_dim=raw_config["qk_nope_head_dim"],
        qk_rope_head_dim=raw_config["qk_rope_head_dim"],
        v_head_dim=raw_config["v_head_dim"],
        index_n_heads=raw_config["index_n_heads"],
        index_head_dim=raw_config["index_head_dim"],
        index_topk=raw_config["index_topk"],
        full_index=full_index,
        weight_dtype_bytes=weight_bytes,
        mla_cache_dtype_bytes=weight_bytes,
    )


def build(raw_config: dict) -> Model:
    dense_full_count, sparse_initial_count, sparse_cycle_full_count, sparse_cycle_share_count = (
        _require_schedule(raw_config)
    )
    hidden = raw_config["hidden_size"]
    weight_bytes = dtype_bytes(raw_config.get("dtype") or raw_config.get("torch_dtype", "bfloat16"))
    dense_ffn = DenseSwiGLU(hidden=hidden, intermediate=raw_config["intermediate_size"])
    sparse_ffn = MoE(
        hidden=hidden,
        moe_intermediate=raw_config["moe_intermediate_size"],
        num_experts=raw_config["n_routed_experts"],
        top_k=raw_config["num_experts_per_tok"],
        shared_intermediate=raw_config.get("shared_expert_intermediate_size")
        or raw_config["moe_intermediate_size"],
    )

    layers = [
        LayerStack(
            attn=_attention(raw_config, full_index=True),
            ffn=dense_ffn,
            count=dense_full_count,
            tag="dense_full_index",
        ),
        LayerStack(
            attn=_attention(raw_config, full_index=False),
            ffn=sparse_ffn,
            count=sparse_initial_count,
            tag="sparse_initial_index_share",
        ),
        LayerStack(
            attn=_attention(raw_config, full_index=True),
            ffn=sparse_ffn,
            count=sparse_cycle_full_count,
            tag="sparse_cycle_full_index",
        ),
        LayerStack(
            attn=_attention(raw_config, full_index=False),
            ffn=sparse_ffn,
            count=sparse_cycle_share_count,
            tag="sparse_cycle_index_share",
        ),
    ]

    norm_weights: list[NormWeightGroup] = []
    for stack in layers:
        norm_weights.extend(
            [
                NormWeightGroup(f"{stack.tag}.input_norm", hidden, stack.count),
                NormWeightGroup(
                    f"{stack.tag}.q_a_norm", raw_config["q_lora_rank"], stack.count
                ),
                NormWeightGroup(
                    f"{stack.tag}.kv_a_norm", raw_config["kv_lora_rank"], stack.count
                ),
                NormWeightGroup(f"{stack.tag}.post_norm", hidden, stack.count),
            ]
        )
        if stack.attn.full_index:
            norm_weights.append(
                NormWeightGroup(
                    f"{stack.tag}.indexer_k_norm", raw_config["index_head_dim"], stack.count
                )
            )
    norm_weights.append(NormWeightGroup("final_norm", hidden, 1))

    return Model(
        name=raw_config["architectures"][0],
        hidden=hidden,
        vocab=raw_config["vocab_size"],
        weight_dtype_bytes=weight_bytes,
        tie_word_embeddings=raw_config.get("tie_word_embeddings", False),
        layers=layers,
        norm_weights=norm_weights,
    )
