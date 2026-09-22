"""Kimi-K3 text-backbone necessary-work composition.

The top-level checkpoint is a VL wrapper. This builder deliberately labels only
the nested KimiLinear causal language model; the vision tower is outside the
text-serving workloads accepted by ``model.work``.
"""

from __future__ import annotations

from ..attention.linear import KimiDeltaAttention
from ..attention.mla import MLA
from ..core import LayerStack, Model, NormWeightGroup, dtype_bytes
from ..ffn.dense import DenseSwiGLU
from ..ffn.moe import MoE
from ..quantization import parse_quantization_config


def _text_config(raw_config: dict) -> dict:
    return raw_config.get("text_config", raw_config)


def _validate_schedule(text: dict) -> tuple[list[int], list[int]]:
    num_layers = text["num_hidden_layers"]
    if num_layers != 93:
        raise ValueError(f"Kimi-K3 model.work requires num_hidden_layers=93, got {num_layers}")

    linear = text.get("linear_attn_config") or {}
    kda_layers = linear.get("kda_layers")
    full_layers = linear.get("full_attn_layers")
    if not isinstance(kda_layers, list) or not isinstance(full_layers, list):
        raise ValueError(
            "Kimi-K3 requires explicit linear_attn_config.kda_layers and "
            "full_attn_layers lists"
        )
    all_layers = set(range(1, num_layers + 1))
    if set(kda_layers) | set(full_layers) != all_layers or set(kda_layers) & set(full_layers):
        raise ValueError("Kimi-K3 attention lists must partition 1-based layers 1..93")
    if len(kda_layers) != 69 or len(full_layers) != 24:
        raise ValueError(
            f"Kimi-K3 requires 69 KDA and 24 MLA layers, got "
            f"{len(kda_layers)} and {len(full_layers)}"
        )
    if text.get("first_k_dense_replace") != 1 or text.get("moe_layer_freq") != 1:
        raise ValueError("Kimi-K3 requires one dense layer followed by every-layer MoE")
    if text.get("use_grouped_topk") is not True or text.get("topk_method") != "noaux_tc":
        raise ValueError("Kimi-K3 routing must use grouped noaux_tc top-k")
    return kda_layers, full_layers


def _layer_norms(tag: str, hidden: int, count: int) -> list[NormWeightGroup]:
    return [
        NormWeightGroup(f"{tag}.input_layernorm", hidden, count),
        NormWeightGroup(f"{tag}.post_attention_layernorm", hidden, count),
    ]


def build(raw_config: dict) -> Model:
    text = _text_config(raw_config)
    kda_layers, full_layers = _validate_schedule(text)

    hidden = text["hidden_size"]
    master_dtype = text.get("dtype") or text.get("torch_dtype", "bfloat16")
    weight_bytes = dtype_bytes(master_dtype)
    state_bytes = dtype_bytes(text.get("mamba_ssm_dtype", "float32"))
    conv_state_bytes = dtype_bytes(text.get("mamba_conv_dtype", "bfloat16"))
    kv_cache_bytes = dtype_bytes(text.get("kv_cache_dtype", "bfloat16"))

    linear_config = text["linear_attn_config"]
    if linear_config.get("head_dim") != 128 or linear_config.get("num_heads") != 96:
        raise ValueError("Kimi-K3 KDA geometry must be 96 heads of head_dim 128")
    if not linear_config.get("use_full_rank_gate", False):
        raise ValueError("Kimi-K3 requires use_full_rank_gate=true")

    kda = KimiDeltaAttention(
        hidden=hidden,
        num_heads=linear_config["num_heads"],
        head_dim=linear_config["head_dim"],
        conv_kernel=linear_config["short_conv_kernel_size"],
        state_dtype_bytes=state_bytes,
        activation_dtype_bytes=conv_state_bytes,
    )
    mla = MLA(
        hidden=hidden,
        num_heads=text["num_attention_heads"],
        q_lora_rank=text["q_lora_rank"],
        kv_lora_rank=text["kv_lora_rank"],
        qk_nope_head_dim=text["qk_nope_head_dim"],
        qk_rope_head_dim=text["qk_rope_head_dim"],
        v_head_dim=text["v_head_dim"],
        kv_dtype_bytes=kv_cache_bytes,
        output_gate=text.get("mla_use_output_gate", False),
    )
    dense = DenseSwiGLU(hidden=hidden, intermediate=text["intermediate_size"])
    moe = MoE(
        hidden=hidden,
        moe_intermediate=text["moe_intermediate_size"],
        num_experts=text["num_experts"],
        top_k=text["num_experts_per_token"],
        shared_intermediate=text["moe_intermediate_size"] * text["num_shared_experts"],
        shared_module="shared_experts",
        expert_input_dim=text["routed_expert_hidden_size"],
        router_bias_elements=text["num_experts"] if text["topk_method"] == "noaux_tc" else 0,
        routed_norm_elements=(
            text["routed_expert_hidden_size"]
            if text.get("latent_moe_use_norm", False)
            else 0
        ),
    )

    kda_moe_count = sum(layer != 1 for layer in kda_layers)
    # Layer 1 is the only dense KDA layer; the remaining KDA layers are MoE.
    if 1 not in kda_layers or kda_moe_count != 68:
        raise ValueError("Kimi-K3 expects layer 1 to be the dense KDA layer")

    layers = [
        LayerStack(attn=kda, ffn=dense, count=1, tag="dense"),
        LayerStack(attn=kda, ffn=moe, count=kda_moe_count, tag="kda"),
        LayerStack(attn=mla, ffn=moe, count=len(full_layers), tag="mla"),
    ]

    norm_weights = [
        *_layer_norms("dense", hidden, 1),
        *_layer_norms("kda", hidden, kda_moe_count),
        *_layer_norms("mla", hidden, len(full_layers)),
        NormWeightGroup("mla.q_a_layernorm", text["q_lora_rank"], len(full_layers)),
        NormWeightGroup("mla.kv_a_layernorm", text["kv_lora_rank"], len(full_layers)),
        NormWeightGroup("final_norm", hidden, 1),
    ]

    return Model(
        name=raw_config["architectures"][0],
        hidden=hidden,
        vocab=text["vocab_size"],
        weight_dtype_bytes=weight_bytes,
        tie_word_embeddings=text.get(
            "tie_word_embeddings", raw_config.get("tie_word_embeddings", False)
        ),
        layers=layers,
        norm_weights=norm_weights,
        master_dtype=master_dtype,
        quant=parse_quantization_config(text),
    )
