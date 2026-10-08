"""Inkling text model.work composition (``InklingForConditionalGeneration``).

Sources: the vLLM fork's ``vllm/models/inkling`` (``configs.py``,
``nvidia/{model,attention,moe,mlp}.py``) and the ModelOpt NVFP4 checkpoint's
safetensors shapes. Dimensions come from ``text_config``; the precision comes from
the ModelOpt ``hf_quant_config`` the config file carries verbatim.

Every decoder layer is ``attn_norm -> attention -> sconv + residual -> mlp_norm ->
FFN -> sconv + residual``:

- attention: :class:`~model.work.attention.inkling_rel.InklingRelAttention`,
  sliding-window (``local_layer_ids``, ``swa_*`` geometry, ``sliding_window_size``
  relative extent) or global (the rest; ``rel_extent``, log-scaled ``q``);
- the two sublayer-output short convolutions and the two norms:
  :class:`~model.work.mixers.short_conv.InklingSublayerShortConv`;
- FFN: a dense SwiGLU (``dense_intermediate_size``, learned ``global_scale``) on
  layers ``0..dense_mlp_idx``, then the shared :class:`~model.work.ffn.moe.MoE`:
  ``n_routed_experts`` routed experts, top ``num_experts_per_tok``, and
  ``n_shared_experts`` always-on sink experts. The sinks are the shared expert
  (``n_shared_experts * intermediate_size`` wide): each token weights them by its
  own sink gammas, an elementwise scale. The router scores routed and sink experts
  together (``n_routed + n_shared`` logits; vLLM pads its GEMM to a multiple of 8,
  which is not necessary work), with an FP32 selection bias and an FP32
  ``global_scale``.

Layers group into stacks by (attention kind, FFN kind and expert precision); the
tags (``local_dense``, ``local_bf16_moe``, ``local_nvfp4_moe``,
``global_nvfp4_moe``) are the simulator's layer-group labels.

Precision: the ModelOpt exclusions keep attention, the router, the sink experts,
the dense FFN, the embedding and the head in BF16 on every layer; the routed
experts are NVFP4 (packed E2M1 plus one FP8 scale per 16 weights) except on layers
whose ``mlp.experts`` is excluded (layer 2 of both released checkpoints), which
stay BF16. The KV cache and the short-conv state are BF16.

The embedding and head tables hold ``vocab_size`` (padded) rows, but the model
emits ``unpadded_vocab_size`` tokens: the padding rows count as parameters and
nothing else.

Out of scope, by construction: the vision encoder (``model.visual``), the audio
encoder/decoder (``model.audio``) and the MTP head (``model.mtp``) — the labelled
deployment serves text only without speculative decoding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..attention.inkling_rel import InklingRelAttention
from ..core import LayerStack, LearnedWeightGroup, Model, dtype_bytes
from ..ffn.dense import DenseSwiGLU
from ..ffn.moe import MoE
from ..mixers.short_conv import InklingSublayerShortConv
from ..quantization import QuantScheme

_LAYER_ENTRY = re.compile(r"^model\.llm\.layers\.(\d+)\.(.+)$")
#: Whole-model exclusions: the text path's embedding, norms and head, and the
#: unlabelled branches.
_MODEL_ENTRIES = frozenset(
    {"model.llm.embed", "model.llm.embed_norm", "model.llm.norm", "model.llm.unembed"}
)
_UNLABELLED_PREFIXES = ("model.audio.", "model.visual.", "model.mtp")
#: Per-layer modules that carry no quantizable GEMM.
_NO_GEMM = frozenset({"attn_norm", "attn_sconv", "mlp_norm", "mlp_sconv"})
_NVFP4_GROUP = 16


@dataclass
class _InklingDenseFFN(DenseSwiGLU):
    """Dense SwiGLU whose output is multiplied by a learned BF16 scalar."""

    def learned_weight_groups(self) -> list[LearnedWeightGroup]:
        return [LearnedWeightGroup("global_scale", 1, 1, "ffn")]


@dataclass
class _InklingMoE(MoE):
    """The MoE plus its gate's FP32 selection bias and FP32 ``global_scale``."""

    def learned_weight_groups(self) -> list[LearnedWeightGroup]:
        return [
            LearnedWeightGroup("gate_bias", self.num_experts, 1, "router", dtype_bytes=4),
            LearnedWeightGroup("gate_global_scale", 1, 1, "router", dtype_bytes=4),
        ]


def _check_supported(text: dict) -> None:
    expectations = {
        "use_sconv": True,
        "use_embed_norm": True,
        "shared_expert_sink": True,
        "gate_activation": "sigmoid",
        "q_bias": False,
        "o_bias": False,
        "use_global_scale": True,
        "use_gate_bias": True,
    }
    for key, expected in expectations.items():
        if text.get(key) != expected:
            raise ValueError(f"Inkling model.work models {key}={expected!r}, got {text.get(key)!r}")
    if text.get("tie_word_embeddings", False):
        raise ValueError("Inkling model.work models separate embed and unembed tables")


def _bf16_expert_layers(raw_config: dict, text: dict) -> set[int]:
    """Layers whose routed experts the ModelOpt NVFP4 checkpoint leaves BF16.

    Strict: every per-layer exclusion must be one the label models — attention,
    the router, the sink experts and the dense FFN excluded on every layer they
    exist, the embedding and head excluded — or quantization would move a
    matrix this builder keeps at BF16.
    """
    quant = (raw_config.get("hf_quant_config") or {}).get("quantization")
    if quant is None:
        raise ValueError(
            "Inkling model.work is modeled for the ModelOpt NVFP4 checkpoint; the config "
            "carries no hf_quant_config.quantization"
        )
    if quant.get("quant_algo") != "NVFP4" or quant.get("group_size", _NVFP4_GROUP) != 16:
        raise ValueError("Inkling model.work expects NVFP4 with 16-element groups")
    if quant.get("kv_cache_quant_algo") not in (None, "none"):
        raise ValueError("Inkling model.work expects a BF16 KV cache")
    layers = text["num_hidden_layers"]
    dense_layers = text["dense_mlp_idx"]
    excluded: list[set[str]] = [set() for _ in range(layers)]
    model_entries = set()
    for entry in quant["exclude_modules"]:
        match = _LAYER_ENTRY.match(entry)
        if match is not None:
            layer, module = int(match.group(1)), match.group(2)
            if layer >= layers:
                raise ValueError(f"exclusion {entry!r} is past the text body")
            excluded[layer].add(module)
        elif entry in _MODEL_ENTRIES:
            model_entries.add(entry)
        elif not entry.startswith(_UNLABELLED_PREFIXES):
            raise ValueError(f"Inkling model.work does not model exclusion {entry!r}")
    if model_entries != _MODEL_ENTRIES:
        raise ValueError(f"expected {sorted(_MODEL_ENTRIES)} excluded, got {sorted(model_entries)}")
    bf16_experts = set()
    for layer, modules in enumerate(excluded):
        if layer < dense_layers:
            required = {"attn", "mlp.w13_dn", "mlp.w2_md"}
            allowed = required | {"mlp.global_scale"}
        else:
            required = {"attn", "mlp.gate", "mlp.shared_experts"}
            allowed = required | {"mlp.experts"}
        if not required <= modules or not modules - _NO_GEMM <= allowed:
            raise ValueError(f"layer {layer}: unmodeled exclusion set {sorted(modules)}")
        if "mlp.experts" in modules:
            bf16_experts.add(layer)
    return bf16_experts


def build(raw_config: dict) -> Model:
    text = raw_config["text_config"]
    _check_supported(text)
    hidden = text["hidden_size"]
    master_dtype = text.get("dtype") or text.get("torch_dtype", "bfloat16")
    master_bytes = dtype_bytes(master_dtype)
    layers = text["num_hidden_layers"]
    local = set(text["local_layer_ids"])
    dense_layers = text["dense_mlp_idx"]
    bf16_experts = _bf16_expert_layers(raw_config, text)
    kernel = text["sconv_kernel_size"]

    def attention(is_local: bool) -> InklingRelAttention:
        prefix = "swa_" if is_local else ""
        return InklingRelAttention(
            hidden=hidden,
            num_heads=text[f"{prefix}num_attention_heads"],
            num_kv_heads=text[f"{prefix}num_key_value_heads"],
            head_dim=text[f"{prefix}head_dim"],
            d_rel=text["d_rel"],
            rel_extent=text["sliding_window_size"] if is_local else text["rel_extent"],
            window=text["sliding_window_size"] if is_local else None,
            sconv_kernel=kernel,
            kv_dtype_bytes=master_bytes,
            master_dtype_bytes=master_bytes,
        )

    def ffn(kind: str, tag: str) -> DenseSwiGLU | MoE:
        if kind == "dense":
            return _InklingDenseFFN(hidden=hidden, intermediate=text["dense_intermediate_size"])
        return _InklingMoE(
            hidden=hidden,
            moe_intermediate=text["intermediate_size"],
            num_experts=text["n_routed_experts"],
            top_k=text["num_experts_per_tok"],
            shared_intermediate=text["n_shared_experts"] * text["intermediate_size"],
            router_extra_logits=text["n_shared_experts"],
            # Stack-qualified paths let the quant scheme tell NVFP4 and BF16
            # expert layers apart.
            module_prefix=tag,
        )

    schedule: dict[str, list[int]] = {}
    for layer in range(layers):
        attn_kind = "local" if layer in local else "global"
        if layer < dense_layers:
            ffn_kind = "dense"
        else:
            ffn_kind = "bf16_moe" if layer in bf16_experts else "nvfp4_moe"
        schedule.setdefault(f"{attn_kind}_{ffn_kind}", []).append(layer)

    mixer = InklingSublayerShortConv(hidden=hidden, kernel=kernel, state_dtype_bytes=master_bytes)
    stacks = [
        LayerStack(
            attn=attention(tag.startswith("local_")),
            ffn=ffn(tag.split("_", 1)[1], tag),
            count=len(stack_layers),
            tag=tag,
            mixer=mixer,
            layers=tuple(stack_layers),
        )
        for tag, stack_layers in schedule.items()
    ]
    quant = QuantScheme(
        bytes_per_weight=0.5,
        compute_dtype="fp4",
        # One FP8 E4M3 scale per 16 weights along K.
        block_shape=(1, _NVFP4_GROUP),
        scale_dtype_bytes=1.0,
        not_converted=frozenset(),
        converted_prefixes=frozenset(
            f"{tag}.mlp.experts" for tag in schedule if tag.endswith("_nvfp4_moe")
        ),
    )
    vocab = text.get("unpadded_vocab_size") or text["vocab_size"]
    return Model(
        name=raw_config["architectures"][0],
        hidden=hidden,
        vocab=vocab,
        param_vocab=text["vocab_size"],
        weight_dtype_bytes=master_bytes,
        tie_word_embeddings=False,
        layers=stacks,
        norm_weights=[
            LearnedWeightGroup("embed_norm", hidden, 1),
            LearnedWeightGroup("final_norm", hidden, 1),
        ],
        master_dtype=master_dtype,
        quant=quant,
    )
