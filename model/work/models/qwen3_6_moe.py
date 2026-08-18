"""Qwen3.6-MoE text accountant: 30 Gated DeltaNet + 10 gated-GQA layers.

The checkpoint is a multimodal wrapper, but model.work labels only its nested
``text_config`` causal language model.  The explicit layer schedule is part of
the architecture identity: silently deriving or homogenizing it could assign
full-attention KV work to the wrong layers.
"""

from __future__ import annotations

from ..attention.gqa import GQA
from ..attention.linear import GatedDeltaNet
from ..core import LayerStack, Model, NormWeightGroup, dtype_bytes
from ..ffn.moe import MoE
from ..quantization import parse_quantization_config

_NUM_LAYERS = 40


def _validate_schedule(text: dict) -> None:
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != _NUM_LAYERS:
        raise ValueError(
            f"Qwen3.6-MoE requires an explicit {_NUM_LAYERS}-entry layer_types schedule"
        )
    expected = [
        "full_attention" if index % 4 == 3 else "linear_attention" for index in range(_NUM_LAYERS)
    ]
    if layer_types != expected:
        raise ValueError("Qwen3.6-MoE layer_types must repeat three linear then one full attention")
    if text.get("num_hidden_layers") != _NUM_LAYERS:
        raise ValueError(f"Qwen3.6-MoE requires num_hidden_layers={_NUM_LAYERS}")


def build(raw_config: dict) -> Model:
    text = raw_config["text_config"]
    _validate_schedule(text)

    hidden = text["hidden_size"]
    master_dtype = text.get("dtype") or text.get("torch_dtype", "bfloat16")
    activation_bytes = dtype_bytes(master_dtype)
    state_bytes = dtype_bytes(text["mamba_ssm_dtype"])

    gdn = GatedDeltaNet(
        hidden=hidden,
        num_v_heads=text["linear_num_value_heads"],
        num_k_heads=text["linear_num_key_heads"],
        head_k_dim=text["linear_key_head_dim"],
        head_v_dim=text["linear_value_head_dim"],
        conv_kernel=text["linear_conv_kernel_dim"],
        state_dtype_bytes=state_bytes,
        activation_dtype_bytes=activation_bytes,
    )
    gated_gqa = GQA(
        hidden=hidden,
        num_qo_heads=text["num_attention_heads"],
        num_kv_heads=text["num_key_value_heads"],
        head_dim=text["head_dim"],
        kv_dtype_bytes=activation_bytes,
        output_gate=text["attn_output_gate"],
    )
    moe = MoE(
        hidden=hidden,
        moe_intermediate=text["moe_intermediate_size"],
        num_experts=text["num_experts"],
        top_k=text["num_experts_per_tok"],
        shared_intermediate=text["shared_expert_intermediate_size"],
        shared_module="shared_expert",
        shared_gate=True,
    )

    head_dim = text["head_dim"]
    norm_weights = [
        NormWeightGroup("gdn.input_norm", hidden, 30),
        NormWeightGroup("gdn.post_norm", hidden, 30),
        NormWeightGroup("gated_gqa.input_norm", hidden, 10),
        NormWeightGroup("gated_gqa.q_norm", head_dim, 10),
        NormWeightGroup("gated_gqa.k_norm", head_dim, 10),
        NormWeightGroup("gated_gqa.post_norm", hidden, 10),
        NormWeightGroup("final_norm", hidden, 1),
    ]
    return Model(
        name=raw_config["architectures"][0],
        hidden=hidden,
        vocab=text["vocab_size"],
        weight_dtype_bytes=activation_bytes,
        tie_word_embeddings=text.get(
            "tie_word_embeddings", raw_config.get("tie_word_embeddings", False)
        ),
        layers=[
            LayerStack(attn=gdn, ffn=moe, count=30, tag="gdn"),
            LayerStack(attn=gated_gqa, ffn=moe, count=10, tag="gated_gqa"),
        ],
        norm_weights=norm_weights,
        master_dtype=master_dtype,
        quant=parse_quantization_config(raw_config),
    )
