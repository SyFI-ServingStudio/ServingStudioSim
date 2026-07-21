"""Qwen3.5 / Qwen3.6 builder: hybrid (linear + full) attention, dense SwiGLU FFN.

Qwen3.6-27B is dense (all params active) but its 64 layers are NOT uniform: a repeating
``[linear, linear, linear, full]`` pattern (``full_attention_interval=4``) gives 48
Gated DeltaNet linear-attention layers + 16 gated full-attention layers, every layer
paired with the same dense SwiGLU MLP. This is the first model here to use the hybrid
:class:`~model.work.core.LayerStack` path — two archetypes over one shared FFN.

The 27B checkpoint ships as a ``Qwen3_5ForConditionalGeneration`` (VL wrapper), so the
text fields live under ``text_config``; this builder reads whichever is present.
"""

from __future__ import annotations

from ..attention.gqa import GQA
from ..attention.linear import GatedDeltaNet
from ..core import LayerStack, Model, dtype_bytes
from ..ffn.dense import DenseSwiGLU


def _layer_types(text: dict) -> list[str]:
    """Per-layer ["linear_attention" | "full_attention"], explicit or interval-derived."""
    if text.get("layer_types"):
        return text["layer_types"]
    # HF Qwen3_5TextConfig default: layer i is full when (i+1) % interval == 0.
    interval = text.get("full_attention_interval", 4)
    num_layers = text["num_hidden_layers"]
    return [
        "linear_attention" if (i + 1) % interval else "full_attention"
        for i in range(num_layers)
    ]


def build(raw_config: dict) -> Model:
    text = raw_config.get("text_config", raw_config)
    hidden = text["hidden_size"]
    weight_bytes = dtype_bytes(text.get("torch_dtype") or text.get("dtype") or "bfloat16")
    state_bytes = dtype_bytes(text.get("mamba_ssm_dtype", "float32"))

    full_attn = GQA(
        hidden=hidden,
        num_qo_heads=text["num_attention_heads"],
        num_kv_heads=text["num_key_value_heads"],
        head_dim=text["head_dim"],
        kv_dtype_bytes=weight_bytes,
        output_gate=text.get("attn_output_gate", False),
    )
    linear_attn = GatedDeltaNet(
        hidden=hidden,
        num_v_heads=text["linear_num_value_heads"],
        num_k_heads=text["linear_num_key_heads"],
        head_k_dim=text["linear_key_head_dim"],
        head_v_dim=text["linear_value_head_dim"],
        conv_kernel=text["linear_conv_kernel_dim"],
        state_dtype_bytes=state_bytes,
    )
    ffn = DenseSwiGLU(hidden=hidden, intermediate=text["intermediate_size"])

    layer_types = _layer_types(text)
    num_linear = layer_types.count("linear_attention")
    num_full = layer_types.count("full_attention")
    layers = [
        LayerStack(attn=linear_attn, ffn=ffn, count=num_linear, tag="linear"),
        LayerStack(attn=full_attn, ffn=ffn, count=num_full, tag="full"),
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
    )
