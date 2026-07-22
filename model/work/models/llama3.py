"""Llama-3 family builder: (GQA attention, dense SwiGLU FFN).

Covers any ``LlamaForCausalLM`` checkpoint — the structure is fixed, the numbers come
from the config. Mistral (also GQA + dense) can register against this same builder.
"""

from __future__ import annotations

from ..attention.gqa import GQA
from ..core import Model, NormWeightGroup, dtype_bytes
from ..ffn.dense import DenseSwiGLU


def build(raw_config: dict) -> Model:
    hidden = raw_config["hidden_size"]
    num_qo_heads = raw_config["num_attention_heads"]
    num_kv_heads = raw_config["num_key_value_heads"]
    head_dim = raw_config.get("head_dim", hidden // num_qo_heads)
    intermediate = raw_config["intermediate_size"]
    weight_bytes = dtype_bytes(raw_config.get("torch_dtype", "bfloat16"))

    attn = GQA(
        hidden=hidden,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_dtype_bytes=weight_bytes,
    )
    ffn = DenseSwiGLU(hidden=hidden, intermediate=intermediate)
    return Model.uniform(
        name=raw_config["architectures"][0],
        num_layers=raw_config["num_hidden_layers"],
        hidden=hidden,
        vocab=raw_config["vocab_size"],
        attn=attn,
        ffn=ffn,
        weight_dtype_bytes=weight_bytes,
        tie_word_embeddings=raw_config.get("tie_word_embeddings", False),
        norm_weights=[
            NormWeightGroup("input_norm", hidden, raw_config["num_hidden_layers"]),
            NormWeightGroup("post_norm", hidden, raw_config["num_hidden_layers"]),
            NormWeightGroup("final_norm", hidden, 1),
        ],
    )
