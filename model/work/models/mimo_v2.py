"""MiMo-V2 text decoder model.work composition (``MiMoV2ForCausalLM``).

Sources: the checkpoint's own ``modeling_mimo_v2.py`` and safetensors headers
(XiaomiMiMo/MiMo-V2.6-Flash-RL, revision 5711b268), and vLLM's
``model_executor/models/mimo_v2.py``. Every number comes from the config.

Each decoder layer is ``input_layernorm -> attention -> post_attention_layernorm
-> FFN``:

- attention: ``hybrid_layer_pattern[i] == 0`` is global causal attention
  (``num_attention_heads`` / ``num_key_value_heads`` / ``head_dim`` /
  ``v_head_dim``), ``1`` is sliding-window attention (the ``swa_*`` geometry,
  ``sliding_window`` keys including the query's own). Query/key heads are wider than
  value heads (asymmetric :class:`~model.work.attention.gqa.GQA`). One fused
  ``qkv_proj`` and an ``o_proj`` of ``heads * v_head_dim`` inputs. A layer kind
  carries a per-head softmax sink when ``add_{swa,full}_attention_sink_bias`` says
  so. Partial RoPE, the value scale and the softmax are elementwise and stay out of
  the denominator (model.work convention 4);
- FFN: ``moe_layer_freq[i] == 0`` is a dense SwiGLU (``intermediate_size``), ``1``
  the shared :class:`~model.work.ffn.moe.MoE` (``n_routed_experts`` routed experts,
  top ``num_experts_per_tok``, no shared expert) whose sigmoid ``noaux_tc`` router
  adds an FP32 ``e_score_correction_bias`` per expert.

Layers group into stacks by (attention kind, FFN kind): ``ga_dense``, ``ga_moe``
and ``swa_moe`` for the released checkpoint.

Precision (``quantization_config``, vLLM ``Fp8Config``): Linear weights are FP8
E4M3 with one FP32 scale per 128x128 block; ``store_dtype: mxfp4`` stores the
routed experts as packed E2M1 plus one E8M0 scale per 32 weights, multiplied by
BF16 activations. ``ignored_layers`` keeps every layer's ``o_proj`` BF16. The
router (vLLM ``GateLinear``, built without a quant config) and ``lm_head`` are
BF16 in the checkpoint although no exclusion names them. The KV cache is BF16.

Out of scope, by construction: the vision and audio encoders and the MTP layers
(``model.mtp``) — the labelled deployment serves text without speculation.
"""

from __future__ import annotations

from dataclasses import replace

from ..attention.gqa import GQA
from ..attention.sliding_window import SlidingWindowGQA
from ..core import LayerStack, Model, NormWeightGroup, dtype_bytes
from ..ffn.dense import DenseSwiGLU
from ..ffn.moe import MoE
from ..quantization import parse_quantization_config, uniform_layer_exclusions

#: Checkpoint matrices stored at the master dtype that no exclusion lists.
_UNLISTED_MASTER_DTYPE = frozenset({"mlp.gate", "lm_head"})


def _check_supported(config: dict) -> None:
    expectations = {
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_projection_layout": "fused_qkv",
        "n_shared_experts": None,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "tie_word_embeddings": False,
    }
    for key, expected in expectations.items():
        if config.get(key) != expected:
            raise ValueError(
                f"MiMo-V2 model.work models {key}={expected!r}, got {config.get(key)!r}"
            )
    layers = config["num_hidden_layers"]
    for key in ("hybrid_layer_pattern", "moe_layer_freq"):
        schedule = config.get(key)
        if (
            not isinstance(schedule, list)
            or len(schedule) != layers
            or any(entry not in (0, 1) or isinstance(entry, bool) for entry in schedule)
        ):
            raise ValueError(f"MiMo-V2 {key} must hold one 0/1 entry per decoder layer")
    if config.get("kv_cache_dtype") not in (None, "auto"):
        raise ValueError("MiMo-V2 model.work models a master-dtype KV cache")


def _attention(config: dict, *, swa: bool, kv_bytes: float) -> GQA:
    prefix = "swa_" if swa else ""
    geometry = dict(
        hidden=config["hidden_size"],
        num_qo_heads=config[f"{prefix}num_attention_heads"],
        num_kv_heads=config[f"{prefix}num_key_value_heads"],
        head_dim=config[f"{prefix}head_dim"],
        v_head_dim=config[f"{prefix}v_head_dim"],
        kv_dtype_bytes=kv_bytes,
        attention_sinks=bool(
            config.get("add_swa_attention_sink_bias" if swa else "add_full_attention_sink_bias")
        ),
    )
    if swa:
        return SlidingWindowGQA(**geometry, window=config["sliding_window"])
    return GQA(**geometry)


def build(raw_config: dict) -> Model:
    _check_supported(raw_config)
    hidden = raw_config["hidden_size"]
    master_dtype = raw_config.get("dtype") or raw_config.get("torch_dtype", "bfloat16")
    master_bytes = dtype_bytes(master_dtype)
    num_layers = raw_config["num_hidden_layers"]

    schedule: dict[str, list[int]] = {}
    for layer, (swa, moe) in enumerate(
        zip(raw_config["hybrid_layer_pattern"], raw_config["moe_layer_freq"], strict=True)
    ):
        schedule.setdefault(f"{'swa' if swa else 'ga'}_{'moe' if moe else 'dense'}", []).append(
            layer
        )
    moe = MoE(
        hidden=hidden,
        moe_intermediate=raw_config["moe_intermediate_size"],
        num_experts=raw_config["n_routed_experts"],
        top_k=raw_config["num_experts_per_tok"],
    )
    dense = DenseSwiGLU(hidden=hidden, intermediate=raw_config["intermediate_size"])
    stacks = [
        LayerStack(
            attn=_attention(raw_config, swa=tag.startswith("swa_"), kv_bytes=master_bytes),
            ffn=moe if tag.endswith("_moe") else dense,
            count=len(layers),
            tag=tag,
            layers=tuple(layers),
        )
        for tag, layers in schedule.items()
    ]

    norm_weights = []
    for stack in stacks:
        norm_weights += [
            NormWeightGroup(f"{stack.tag}.input_norm", hidden, stack.count, stack=stack.tag),
            NormWeightGroup(
                f"{stack.tag}.post_attention_norm", hidden, stack.count, stack=stack.tag
            ),
        ]
        if stack.ffn is moe:
            # noaux_tc's learned FP32 per-expert selection bias.
            norm_weights.append(
                NormWeightGroup(
                    f"{stack.tag}.router_bias",
                    moe.num_experts,
                    stack.count,
                    "router",
                    dtype_bytes=4,
                    stack=stack.tag,
                )
            )
    norm_weights.append(NormWeightGroup("final_norm", hidden, 1))

    quant = parse_quantization_config(raw_config, layer_qualified_exclusions=True)
    if quant is not None:
        declared = raw_config["quantization_config"]
        excluded = declared.get("ignored_layers") or declared.get("modules_to_not_convert") or []
        quant = replace(
            quant,
            not_converted=uniform_layer_exclusions(excluded, num_layers) | _UNLISTED_MASTER_DTYPE,
        )
    return Model(
        name=raw_config["architectures"][0],
        hidden=hidden,
        vocab=raw_config["vocab_size"],
        weight_dtype_bytes=master_bytes,
        tie_word_embeddings=False,
        layers=stacks,
        norm_weights=norm_weights,
        master_dtype=master_dtype,
        quant=quant,
    )
