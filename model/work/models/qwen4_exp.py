"""Qwen4Exp (Qwen3.8-Flash-Next) model.work composition, language model only.

``Qwen4ExpForConditionalGeneration`` wraps a vision tower (not labelled) around a
text decoder (``text_config``) that the vLLM fork implements in
``models/qwen4_exp/nvidia/model.py``:

- the residual is ``hc_count`` streams: the embedding is repeated into them, every
  sublayer is fed and merged by a gated low-rank hyper-connection module
  (:mod:`..mixers.gated_hc`), and a final mixer collapses the streams before
  ``lm_head``; there are no per-layer input/post norms besides the modules';
- ``layer_types`` interleaves Gated DeltaNet (``linear_attention``) and QSA
  (``full_attention``: :mod:`..attention.qsa`) layers; every layer's FFN is the
  same router + routed experts + sigmoid-gated shared expert MoE;
- the layers in ``ple_layer_ids`` (1-based) add a per-layer n-gram embedding to the
  streams first (:mod:`..mixers.ngram_embedding`); its table lives in host memory;
- ``mtp_num_hidden_layers`` MTP layers: ``pre_fc_norm_embedding`` + ``fc_embedding``
  on the drafted token's embedding (the target's table), ``pre_fc_norm_hidden`` +
  ``fc_hidden`` on each carried stream, one QSA decoder layer whose attention module
  merges the projected embedding with unit weight, its own final mixer, and the
  target's ``lm_head`` (the checkpoint ships no MTP head or embedding).

Stacks follow the architecture: layer 0's attention module has no pending combine
and each PLE layer carries the embedding, so both are their own archetype; the rest
group by attention kind. The MTP layer is staged under the GLM-5.2 MTP stage names
(:mod:`..speculative` builds them from the logged geometry): the first pass over the
target's rows scores its own index; later passes reuse that selection.

Precision is the checkpoint's ModelOpt ``MIXED_PRECISION`` list: NVFP4 body experts
(per-16 FP8 scales), FP8 per-tensor PLE table, and FP8 128x128-block MTP experts with
dynamic per-group FP8 activations (``FP8_PB_WO``; block scales stored at the master
dtype). Everything else stays BF16.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..attention.base import AttentionSemantic
from ..attention.linear import GatedDeltaNet
from ..attention.qsa import QuerySparseAttention
from ..core import (
    LayerStack,
    LearnedWeightGroup,
    MatmulGroup,
    Model,
    NormWeightGroup,
    Workload,
    dtype_bytes,
)
from ..ffn.moe import MoE
from ..mixers.gated_hc import (
    GatedHcFinalMixer,
    GatedHyperConnections,
    decoder_layer_modules,
    final_mixer_module,
)
from ..mixers.ngram_embedding import NgramEmbeddingInjection, ngram_table_rows
from ..quantization import QuantScheme, parse_quantization_config
from .glm52 import DRAFT_FIRST_STAGE, DRAFT_RECURRENT_STAGE

_GDN = "linear_attention"
_QSA = "full_attention"
_TAGS = {_GDN: "gdn", _QSA: "qsa"}
_MTP_PREFIX = "mtp"

#: vLLM ``Qwen4ExpForCausalLM.packed_modules_mapping``.
_PACKED_MODULES = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
    "in_proj_qkvz": ("in_proj_qkv", "in_proj_z"),
    "in_proj_ba": ("in_proj_b", "in_proj_a"),
}


@dataclass
class _LayerWithNgramEmbedding:
    """A decoder layer's attention plus the n-gram embedding injected before it.

    The embedding's convolution history is per-request geometry, and only an
    attention spec may label exact geometry apart from the scalar rows, so the
    injection rides on the attention of its layer.
    """

    attn: object
    ple: NgramEmbeddingInjection

    split_attention_phases = False
    needs_request_geometry = True

    def matmul_groups(self) -> list[MatmulGroup]:
        return [*self.attn.matmul_groups(), *self.ple.matmul_groups()]

    def learned_weight_groups(self) -> list[LearnedWeightGroup]:
        return [*self.attn.learned_weight_groups(), *self.ple.learned_weight_groups()]

    def semantic_segments(self, wl: Workload) -> list[AttentionSemantic]:
        return [*self.attn.semantic_segments(wl), *self.ple.semantic_segments(wl)]

    def internal_flops(self, wl: Workload) -> float:
        return sum(row.flops for row in self.semantic_segments(wl) if row.bucket == "attn_internal")

    def kv_bytes(self, wl: Workload) -> float:
        return sum(row.bytes for row in self.semantic_segments(wl) if row.byte_kind == "kv")

    def cache_write_bytes(self, wl: Workload) -> float:
        return 0.0


def _schedule(text: dict) -> tuple[list[str], set[int]]:
    """``layer_types`` and the 0-based PLE layers, checked against the config."""
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != text["num_hidden_layers"]:
        raise ValueError("Qwen4Exp model.work requires an explicit per-layer layer_types")
    if set(layer_types) - set(_TAGS):
        raise ValueError(
            f"unsupported Qwen4Exp layer types {sorted(set(layer_types) - set(_TAGS))}"
        )
    if text.get("mlp_only_layers") or text.get("decoder_sparse_step", 1) != 1:
        raise ValueError("Qwen4Exp model.work models a MoE FFN on every layer")
    ple_layers = {int(layer_id) - 1 for layer_id in text.get("ple_layer_ids") or ()}
    if any(not 0 <= layer < len(layer_types) for layer in ple_layers):
        raise ValueError("ple_layer_ids are 1-based decoder layers")
    if 0 in ple_layers:
        raise ValueError("a PLE injection on layer 0 (no pending combine) is not modeled")
    return layer_types, ple_layers


def _quant(raw: dict, layer_types: list[str], ple_layers: set[int]) -> tuple[QuantScheme | None, float]:
    """The checkpoint's scheme and the PLE table's bytes per element.

    The PLE table is an embedding, not a Linear, and the MTP experts sit past the
    body, so both entries are read here; the rest is the shared MIXED_PRECISION
    reader over the body's own schedule.
    """
    master = dtype_bytes(raw["text_config"].get("dtype") or raw.get("dtype", "bfloat16"))
    quant = raw.get("quantization_config")
    if quant is None:
        return None, master
    if quant.get("quant_algo") != "MIXED_PRECISION":
        raise ValueError("Qwen4Exp model.work reads a ModelOpt MIXED_PRECISION checkpoint")
    groups = quant.get("config_groups", {}).values()
    body: dict[str, dict] = {}
    table_bytes = master
    mtp_scheme = None
    for path, entry in quant["quantized_layers"].items():
        algo = entry.get("quant_algo")
        if path.endswith(".ple.ple_embedding.ngram_embedding"):
            if algo != "FP8":
                raise ValueError(
                    f"PLE table {path!r}: only per-tensor FP8 is modeled, not {algo!r}"
                )
            table_bytes = 1.0
        elif path.startswith(f"{_MTP_PREFIX}."):
            if not path.endswith(".mlp.experts") or algo != "FP8_PB_WO":
                raise ValueError(f"MTP {path!r} {entry}: only FP8_PB_WO experts are modeled")
            block = int(entry["group_size"])
            targeting = [group for group in groups if path in group.get("targets", ())]
            if len(targeting) != 1:
                raise ValueError(f"MTP {path!r} needs exactly one config_groups entry")
            activations = targeting[0].get("input_activations") or {}
            mtp_scheme = QuantScheme(
                bytes_per_weight=1.0,
                # W8A8 when the group quantizes activations, else weight-only.
                compute_dtype="fp8" if activations.get("num_bits") == 8 else "bf16",
                block_shape=(block, block),
                scale_dtype_bytes=master,
                not_converted=frozenset(),
            )
        else:
            body[path] = entry
    # One schedule entry per layer type, with the PLE layers apart, so the reader's
    # per-type consistency check sees exactly the body's own Linears.
    types = [
        f"{kind}_ple" if layer in ple_layers else kind for layer, kind in enumerate(layer_types)
    ]
    scheme = parse_quantization_config(
        {**raw, "quantization_config": {**quant, "quantized_layers": body}},
        layer_types=types,
        packed_modules=_PACKED_MODULES,
    )
    if mtp_scheme is not None:
        prefix = f"{_MTP_PREFIX}.mlp.experts"
        scheme = replace(
            scheme,
            converted_prefixes=(scheme.converted_prefixes or frozenset()) | {prefix},
            module_schemes=(*scheme.module_schemes, (prefix, mtp_scheme)),
        )
    return scheme, table_bytes


def _qsa(text: dict, master: float, **fields) -> QuerySparseAttention:
    if text["indexer_kv_heads"] != 1:
        raise ValueError("the QSA indexer has one key head")
    if text.get("output_gate_type", "sigmoid") != "sigmoid":
        raise ValueError("QSA packs a sigmoid output gate next to Q")
    return QuerySparseAttention(
        hidden=text["hidden_size"],
        num_heads=text["num_attention_heads"],
        num_kv_heads=text["num_key_value_heads"],
        head_dim=text["head_dim"],
        index_n_heads=text["indexer_n_heads"],
        index_head_dim=text["indexer_head_dim"],
        budget=text["indexer_budget"],
        ratio=text["indexer_compress_ratio"],
        kv_dtype_bytes=master,
        index_cache_dtype_bytes=master,
        **fields,
    )


def _moe(text: dict, **fields) -> MoE:
    return MoE(
        hidden=text["hidden_size"],
        moe_intermediate=text["moe_intermediate_size"],
        num_experts=text["num_experts"],
        top_k=text["num_experts_per_tok"],
        shared_intermediate=text["shared_expert_intermediate_size"],
        shared_module="shared_expert",
        shared_gate=True,
        **fields,
    )


def _mtp_stacks(text: dict, master: float) -> tuple[list[LayerStack], list[NormWeightGroup]]:
    layers = text.get("mtp_num_hidden_layers") or 0
    if not layers:
        return [], []
    if layers != 1:
        raise ValueError(f"Qwen4Exp model.work models one MTP layer; got {layers}")
    if (text.get("mtp") or {}).get("layer_types", [_QSA]) != [_QSA]:
        raise ValueError("Qwen4Exp model.work models a QSA MTP layer")
    if text.get("mtp_use_dedicated_embeddings"):
        raise ValueError("a dedicated MTP embedding table is not modeled")
    hidden, streams, lowrank = text["hidden_size"], text["hc_count"], text["hc_lowrank"]
    mixer = GatedHyperConnections(
        (
            *decoder_layer_modules(
                hidden, streams, lowrank, attn_combine="unit", prefix=_MTP_PREFIX
            ),
            final_mixer_module(hidden, streams, lowrank, prefix=_MTP_PREFIX),
        )
    )
    fc = [
        MatmulGroup("fc_embedding", hidden, hidden, bucket="dense_ffn", module="mtp.fc_embedding"),
        # Shared across the carried streams: one row per (token, stream).
        MatmulGroup(
            "fc_hidden",
            hidden,
            hidden,
            bucket="dense_ffn",
            module="mtp.fc_hidden",
            rows=lambda wl: streams * wl.matmul_tokens,
        ),
    ]
    ffn = _moe(text, module_prefix=_MTP_PREFIX)
    stacks = [
        LayerStack(
            attn=_qsa(text, master, module_prefix=_MTP_PREFIX),
            ffn=ffn,
            count=1,
            tag="mtp_first",
            stage=DRAFT_FIRST_STAGE,
            extra_matmuls=fc,
            mixer=mixer,
        ),
        LayerStack(
            attn=_qsa(text, master, module_prefix=_MTP_PREFIX, index_share=True),
            ffn=ffn,
            count=1,
            tag="mtp_recurrent",
            stage=DRAFT_RECURRENT_STAGE,
            extra_matmuls=fc,
            mixer=mixer,
            param_count=0,
        ),
    ]
    norms = [
        NormWeightGroup(
            f"{stack.tag}.{name}",
            elements,
            1,
            stage=stack.stage,
            param_count=stack.param_count,
            stack=stack.tag,
        )
        for stack in stacks
        for name, elements in (
            ("pre_fc_norm_embedding", hidden),
            ("pre_fc_norm_hidden", streams * hidden),
        )
    ]
    return stacks, norms


def build(raw_config: dict) -> Model:
    text = raw_config["text_config"]
    layer_types, ple_layers = _schedule(text)
    hidden, streams, lowrank = text["hidden_size"], text["hc_count"], text["hc_lowrank"]
    master_dtype = text.get("dtype") or raw_config.get("dtype", "bfloat16")
    master = dtype_bytes(master_dtype)
    quant, table_bytes = _quant(raw_config, layer_types, ple_layers)

    attention = {
        _GDN: GatedDeltaNet(
            hidden=hidden,
            num_v_heads=text["linear_num_value_heads"],
            num_k_heads=text["linear_num_key_heads"],
            head_k_dim=text["linear_key_head_dim"],
            head_v_dim=text["linear_value_head_dim"],
            conv_kernel=text["linear_conv_kernel_dim"],
            state_dtype_bytes=dtype_bytes(text["mamba_ssm_dtype"]),
            activation_dtype_bytes=master,
        ),
        _QSA: _qsa(text, master),
    }
    ngram_heads = (text["ngram_size"] - 1) * text["heads_per_ngram"]
    ple_ids = sorted(ple_layers)
    moe = _moe(text)

    def mixer(attn_combine: str) -> GatedHyperConnections:
        return GatedHyperConnections(
            decoder_layer_modules(hidden, streams, lowrank, attn_combine=attn_combine)
        )

    # (tag, layers, attention, mixer), in decoder order of first appearance.
    archetypes: list[tuple[str, tuple[int, ...], object, GatedHyperConnections]] = [
        (f"{_TAGS[layer_types[0]]}_first", (0,), attention[layer_types[0]], mixer("none"))
    ]
    for dense_id, layer in enumerate(ple_ids):
        ple = NgramEmbeddingInjection(
            hidden=hidden,
            streams=streams,
            embed_dim=text["ple_embed_dim"],
            ngram_heads=ngram_heads,
            table_rows=ngram_table_rows(
                text["ngram_vocab_size_base"],
                ngram_heads,
                text["make_ngram_vocab_size_divisible_by"],
                dense_id,
            ),
            table_dtype_bytes=table_bytes,
            conv_kernel=text["ple_conv_kernel_size"],
            # The fork dilates the convolution by the n-gram size.
            conv_dilation=text["ngram_size"],
            state_dtype_bytes=master,
        )
        tag = f"{_TAGS[layer_types[layer]]}_ple" + (f"{dense_id}" if len(ple_ids) > 1 else "")
        archetypes.append(
            (
                tag,
                (layer,),
                _LayerWithNgramEmbedding(attention[layer_types[layer]], ple),
                mixer("injected"),
            )
        )
    for kind in sorted(set(layer_types), key=layer_types.index):
        layers = tuple(
            layer
            for layer, layer_kind in enumerate(layer_types)
            if layer_kind == kind and layer != 0 and layer not in ple_layers
        )
        if layers:
            archetypes.append((_TAGS[kind], layers, attention[kind], mixer("injected")))

    stacks = [
        LayerStack(attn=attn, ffn=moe, count=len(layers), tag=tag, mixer=hc, layers=layers)
        for tag, layers, attn, hc in archetypes
    ]
    mtp, mtp_norms = _mtp_stacks(text, master)
    final_mixer = GatedHcFinalMixer(final_mixer_module(hidden, streams, lowrank), master)
    return Model(
        name=raw_config["architectures"][0],
        hidden=hidden,
        vocab=text["vocab_size"],
        weight_dtype_bytes=master,
        tie_word_embeddings=text.get(
            "tie_word_embeddings", raw_config.get("tie_word_embeddings", False)
        ),
        layers=[*stacks, *mtp],
        norm_weights=[*final_mixer.weight_groups(), *mtp_norms],
        master_dtype=master_dtype,
        quant=quant,
        epilogue=[final_mixer],
    )
