"""Qwen3-MoE family builder: (GQA attention, MoE FFN, no shared expert).

Covers any ``Qwen3MoeForCausalLM`` checkpoint. Assumes every layer is MoE (Qwen3-MoE
has no dense-layer schedule — no ``mlp_only_layers`` / ``decoder_sparse_step``); a model
that mixes dense and MoE layers would need per-layer composition, a future extension.
"""

from __future__ import annotations

from ..attention.gqa import GQA
from ..core import Model, dtype_bytes
from ..ffn.moe import MoE
from ..quantization import parse_quantization_config


def build(raw_config: dict) -> Model:
    hidden = raw_config["hidden_size"]
    num_qo_heads = raw_config["num_attention_heads"]
    num_kv_heads = raw_config["num_key_value_heads"]
    head_dim = raw_config.get("head_dim", hidden // num_qo_heads)
    master_dtype = raw_config.get("dtype") or raw_config.get("torch_dtype", "bfloat16")
    weight_bytes = dtype_bytes(master_dtype)

    attn = GQA(
        hidden=hidden,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_dtype_bytes=weight_bytes,
    )
    ffn = MoE(
        hidden=hidden,
        moe_intermediate=raw_config["moe_intermediate_size"],
        num_experts=raw_config["num_experts"],
        top_k=raw_config["num_experts_per_tok"],
        shared_intermediate=raw_config.get("shared_expert_intermediate_size", 0) or 0,
        # Qwen2/Qwen3-MoE names the always-on expert in the singular.
        shared_module="shared_expert",
    )
    return Model.uniform(
        name=raw_config["architectures"][0],
        num_layers=raw_config["num_hidden_layers"],
        hidden=hidden,
        vocab=raw_config["vocab_size"],
        attn=attn,
        ffn=ffn,
        weight_dtype_bytes=weight_bytes,
        tie_word_embeddings=raw_config.get("tie_word_embeddings", False),
        master_dtype=master_dtype,
        quant=parse_quantization_config(raw_config),
    )
