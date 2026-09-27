"""GLM-5.3-Flash text model.work composition (``Glm5NextForConditionalGeneration``).

45 decoder layers over a 4-stream mHC residual (``hc_mult``):

- attention: 34 KDA linear-attention layers and 11 no-rope MLA layers with the kpool
  DSA indexer, repeating ``[KDA, KDA, KDA, DSA]`` with a final KDA layer;
- FFN: dense SwiGLU (``intermediate_size``) for the first ``first_k_dense_replace``
  layers, then a 288-expert top-8 MoE with one shared expert.

All three dense layers are KDA layers and every DSA layer is MoE, so four archetypes
cover the model. Layer 0 is its own stack because its mHC attention boundary has no
incoming post (see ``mixers/mhc.py``); the stack tags are those archetypes.

Out of scope, by construction:

- the vision tower (``model.visual``) — the label describes the text decoder; the
  deployment serves text only (``--limit-mm-per-prompt image=0,video=0``);
- the MTP layer (``num_nextn_predict_layers``, checkpoint layer 45) — speculative
  decoding is off in the labelled deployment, so no stage runs it.

Precision is per module from the FP8 config's ``modules_to_not_convert``. Its entries
are layer-qualified, and one layer-relative path is BF16 on some layers and FP8 on
others (``self_attn.o_proj`` stays BF16 on KDA layers; DSA's ``o_proj`` carries a
``weight_scale_inv``). DSA-layer attention paths are therefore matched under a
``dsa.`` module prefix.
"""

from __future__ import annotations

import re
from dataclasses import replace

from ..attention.glm53_kpool_dsa import Glm53KpoolDsaAttention
from ..attention.kda import KimiDeltaAttention
from ..core import LayerStack, MatmulGroup, Model, NormWeightGroup, dtype_bytes
from ..ffn.dense import DenseSwiGLU
from ..ffn.moe import MoE
from ..mixers.mhc import ManifoldHyperConnections, MhcFinalPost
from ..quantization import QuantScheme, parse_quantization_config

_KDA = "linear_attention"
_DSA = "deepseek_sparse_attention"
_DSA_MODULE_PREFIX = "dsa"
_QUALIFIED = re.compile(r"^model\.(?:language_model\.)?layers\.(\d+)\.(.+)$")


def _schedule(config: dict) -> tuple[list[str], list[str]]:
    layers = config["num_hidden_layers"]
    attention = config.get("layer_types")
    mlp = config.get("mlp_layer_types")
    if not isinstance(attention, list) or not isinstance(mlp, list):
        raise ValueError(
            "GLM-5.3-Flash model.work requires explicit layer_types and mlp_layer_types"
        )
    if len(attention) != layers or len(mlp) != layers:
        raise ValueError("GLM-5.3-Flash layer schedules must each match num_hidden_layers")
    if set(attention) - {_KDA, _DSA} or set(mlp) - {"dense", "sparse"}:
        raise ValueError(f"unsupported GLM-5.3-Flash layer types {set(attention) | set(mlp)}")
    dense = config["first_k_dense_replace"]
    if mlp != ["dense"] * dense + ["sparse"] * (layers - dense):
        raise ValueError("GLM-5.3-Flash expects dense layers first, then MoE layers")
    linear = config["linear_attn_config"]
    if sorted(linear["kda_layers"]) != [i for i, t in enumerate(attention) if t == _KDA] or sorted(
        linear["full_attn_layers"]
    ) != [i for i, t in enumerate(attention) if t == _DSA]:
        raise ValueError("linear_attn_config layer lists disagree with layer_types")
    if any(attention[i] != _KDA for i in range(dense)):
        raise ValueError("GLM-5.3-Flash model.work expects every dense layer to be KDA")
    if config.get("indexer_types") and set(config["indexer_types"]) != {"full"}:
        raise ValueError("GLM-5.3-Flash model.work expects a full indexer on every DSA layer")
    if not config.get("mhc"):
        raise ValueError("GLM-5.3-Flash model.work models the mHC residual; config has mhc=false")
    return attention, mlp


def _quant(config: dict, attention: list[str]) -> QuantScheme | None:
    scheme = parse_quantization_config(config)
    if scheme is None:
        return None
    not_converted = set()
    for module in config["quantization_config"].get("modules_to_not_convert", ()):
        match = _QUALIFIED.match(module)
        if match is None:
            not_converted.add(module.removeprefix("model."))
            continue
        layer, path = int(match.group(1)), match.group(2)
        if layer >= len(attention):
            continue  # the MTP layer, out of scope
        if attention[layer] == _DSA and path.startswith("self_attn."):
            path = f"{_DSA_MODULE_PREFIX}.{path}"
        not_converted.add(path)
    return replace(scheme, not_converted=frozenset(not_converted))


def _prefixed(groups: list[MatmulGroup], prefix: str) -> list[MatmulGroup]:
    return [replace(group, module=f"{prefix}.{group.module}") for group in groups]


class _DsaAttention(Glm53KpoolDsaAttention):
    """DSA layer whose checkpoint paths are matched under the ``dsa.`` prefix."""

    def matmul_groups(self) -> list[MatmulGroup]:
        return _prefixed(super().matmul_groups(), _DSA_MODULE_PREFIX)


def build(raw_config: dict) -> Model:
    config = raw_config.get("text_config", raw_config)
    attention, _mlp = _schedule(config)
    hidden = config["hidden_size"]
    streams = config["hc_mult"]
    master_dtype = config.get("dtype") or config.get("torch_dtype", "bfloat16")
    master_bytes = dtype_bytes(master_dtype)
    quant_config = config.get("quantization_config", {})
    kv_cache_fp8 = quant_config.get("kv_cache_dtype") == "fp8_e4m3" or bool(
        quant_config.get("kv_cache_scheme")
    )
    linear = config["linear_attn_config"]
    if linear.get("gate_rank", linear["head_dim"]) != linear["head_dim"]:
        raise ValueError("GLM-5.3-Flash KDA gates are head_dim wide")

    kda = KimiDeltaAttention(
        hidden=hidden,
        num_v_heads=linear["num_heads"],
        num_k_heads=linear["num_heads"],
        head_k_dim=linear["head_dim"],
        head_v_dim=linear["head_dim"],
        conv_kernel=linear["short_conv_kernel_size"],
        # vLLM's kda_state_dtype keeps the recurrent state in fp32 by default.
        state_dtype_bytes=4.0,
        activation_dtype_bytes=master_bytes,
        gate_rank=linear["head_dim"],
    )
    dsa = _DsaAttention(
        hidden=hidden,
        num_heads=config["num_attention_heads"],
        q_lora_rank=config["q_lora_rank"],
        kv_lora_rank=config["kv_lora_rank"],
        qk_nope_head_dim=config["qk_nope_head_dim"],
        qk_rope_head_dim=config["qk_rope_head_dim"],
        v_head_dim=config["v_head_dim"],
        index_n_heads=config["index_n_heads"],
        index_head_dim=config["index_head_dim"],
        index_topk=config["index_topk"],
        index_kpool=config["index_kpool"],
        mla_cache_dtype_bytes=1.0 if kv_cache_fp8 else master_bytes,
    )
    dense = DenseSwiGLU(hidden=hidden, intermediate=config["intermediate_size"])
    moe = MoE(
        hidden=hidden,
        moe_intermediate=config["moe_intermediate_size"],
        num_experts=config["n_routed_experts"],
        top_k=config["num_experts_per_tok"],
        shared_intermediate=config["moe_intermediate_size"] * config["n_shared_experts"],
    )

    first_dense = config["first_k_dense_replace"]
    kda_moe = sum(t == _KDA for t in attention[first_dense:])
    dsa_moe = sum(t == _DSA for t in attention)
    mixer = ManifoldHyperConnections(hidden=hidden, streams=streams)
    layers = [
        LayerStack(
            attn=kda,
            ffn=dense,
            count=1,
            tag="first_kda_dense",
            mixer=replace(mixer, first_layer=True),
        ),
        LayerStack(attn=kda, ffn=dense, count=first_dense - 1, tag="kda_dense", mixer=mixer),
        LayerStack(attn=dsa, ffn=moe, count=dsa_moe, tag="dsa_moe", mixer=mixer),
        LayerStack(attn=kda, ffn=moe, count=kda_moe, tag="kda_moe", mixer=mixer),
    ]
    # noaux_tc routing adds a learned fp32 per-expert bias to the router scores.
    norm_weights = [
        NormWeightGroup(
            f"{stack.tag}.router_bias", config["n_routed_experts"], stack.count, "router"
        )
        for stack in layers
        if stack.ffn is moe and config.get("topk_method") == "noaux_tc"
    ]
    norm_weights.append(NormWeightGroup("final_norm", hidden, 1))
    return Model(
        name=raw_config["architectures"][0],
        hidden=hidden,
        vocab=config["vocab_size"],
        weight_dtype_bytes=master_bytes,
        tie_word_embeddings=config.get("tie_word_embeddings", False),
        layers=layers,
        norm_weights=norm_weights,
        master_dtype=master_dtype,
        quant=_quant(config, attention),
        epilogue=[MhcFinalPost(hidden=hidden, streams=streams)],
    )
