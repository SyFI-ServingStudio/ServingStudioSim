"""DeepSeek-V4.1 (``DeepseekV41ForCausalLM``) model.work composition.

Sources: DeepSeek's reference ``inference/model.py`` / ``engram.py`` and the
checkpoint's safetensors dtypes. Every one of the 40 layers is
``mHC -> [engram] -> attention -> mHC -> MoE``:

- attention: :class:`~model.work.attention.deepseek_v41.DeepseekV41Attention`,
  whose role per layer comes from ``compress_ratios``, ``kv_source_layer_ids``,
  ``index_source_layer_ids`` and ``candidate_source_layer_id``;
- FFN: the shared :class:`~model.work.ffn.moe.MoE` (384 routed MXFP4 experts,
  top-6, one FP8 shared expert, BF16 router plus an FP32 selection bias);
- mHC (``hc_mult`` residual copies): before each sublayer one FP32
  ``[(2+hc)*hc, hc*hidden]`` mixing projection of the flattened stream, plus
  learned base/scale vectors. Sinkhorn, pre-collapse and post-expand are
  activation arithmetic and stay out of the pinned FLOPs;
- engram at ``engram_layer_ids``: each token gathers ``(max_ngram-1)*n_heads``
  hashed rows of an FP8 table (one E8M0 scale per 32 values), projects them with
  the FP8 ``wkv`` into ``hc+1`` hidden-wide vectors, and gates them with learned
  query/key vectors. The table is a gathered weight: only the hashed rows are
  read, and only they count as activated parameters.

The layer stacks are the schedule's runs of identical roles, split where the
unified CostTree keeps separate bodies (consumers of the engram KV source, and
the ``dspark_target_layer_ids`` consumers whose hidden state feeds the draft
head); both splits add no work.

Excluded, deliberately: the ``num_nextn_predict_layers`` DSpark draft layers
(``mtp.*``) and the vision tower/aligner/image-token embeddings, which only a
speculative or multimodal deployment runs, and the router's ``bias_vl``, which
only image-span tokens read.
"""

from __future__ import annotations

from dataclasses import replace

from ..attention.deepseek_v41 import (
    INDEX_CANDIDATE_CONSUMER,
    INDEX_NONE,
    INDEX_OWNER,
    DeepseekV41Attention,
)
from ..core import LayerStack, LearnedWeightGroup, MatmulGroup, Model, dtype_bytes
from ..ffn.moe import MoE
from ..quantization import parse_quantization_config

#: Checkpoint tensors stored BF16 although the config declares an FP8 scheme and
#: no ``modules_to_not_convert`` (layer-relative, as MatmulGroup.module names them).
_BF16_MODULES = frozenset(
    {
        "mlp.gate",
        "attn.compressor",
        "attn.indexer.wk",
        "attn.indexer.weights_proj",
        "lm_head",
    }
)

#: (tag, layers). Validated against the config's own role fields in `_roles`.
_STACKS = (
    ("l0_swa", (0,)),
    ("l1_swa_engram", (1,)),
    ("c2_source", (2, 8)),
    ("c2_share", (3, 4, 5, 6, 7, 9, 10, 11, 12, 13)),
    ("c2_source_engram", (14,)),
    ("c2_share_after_engram", (15, 16, 17, 18, 19)),
    ("c1_source_candidate", (20,)),
    ("c1_share", (21, 22, 23, 25, 26, 27, 29, 30, 31, 33, 34, 35)),
    ("c1_candidate_index", (24, 28, 32, 36)),
    ("c1_share_dspark_target", (37, 38, 39)),
)


def _roles(raw_config: dict) -> dict[int, tuple]:
    """Each body layer's role, read from the config's schedule fields."""
    layers = raw_config["num_hidden_layers"]
    ratios = raw_config["compress_ratios"]
    if len(ratios) < layers:
        raise ValueError("compress_ratios must cover every body layer")
    kv_sources = set(raw_config["kv_source_layer_ids"])
    index_sources = set(raw_config["index_source_layer_ids"])
    candidate_source = raw_config.get("candidate_source_layer_id", -1)
    engram_layers = set(raw_config.get("engram_layer_ids") or ())
    roles = {}
    for layer in range(layers):
        if layer in index_sources and layer in kv_sources:
            index_role = INDEX_OWNER
        elif layer in index_sources:
            if not 0 <= candidate_source < layer:
                raise ValueError(f"index source {layer} owns no keys and has no candidates")
            index_role = INDEX_CANDIDATE_CONSUMER
        else:
            index_role = INDEX_NONE
        roles[layer] = (ratios[layer], layer in kv_sources, index_role, layer in engram_layers)
    return roles


def _schedule(raw_config: dict) -> list[tuple[str, tuple[int, ...], tuple]]:
    roles = _roles(raw_config)
    covered = sorted(layer for _tag, layers in _STACKS for layer in layers)
    if covered != list(range(raw_config["num_hidden_layers"])):
        raise ValueError("DeepSeek-V4.1 stacks must cover each body layer exactly once")
    schedule = []
    for tag, layers in _STACKS:
        stack_roles = {roles[layer] for layer in layers}
        if len(stack_roles) != 1:
            raise ValueError(f"unsupported DeepSeek-V4.1 schedule: stack {tag} mixes {stack_roles}")
        schedule.append((tag, layers, stack_roles.pop()))
    return schedule


def build(raw_config: dict) -> Model:
    hidden = raw_config["hidden_size"]
    hc = raw_config["hc_mult"]
    master = raw_config.get("dtype") or raw_config.get("torch_dtype", "bfloat16")
    weight_bytes = dtype_bytes(master)
    quant = parse_quantization_config(raw_config)
    if quant is not None:
        quant = replace(quant, not_converted=quant.not_converted | _BF16_MODULES)

    moe = MoE(
        hidden=hidden,
        moe_intermediate=raw_config["moe_intermediate_size"],
        num_experts=raw_config["n_routed_experts"],
        top_k=raw_config["num_experts_per_tok"],
        shared_intermediate=raw_config["n_shared_experts"] * raw_config["moe_intermediate_size"],
    )
    mix_width = (2 + hc) * hc
    mhc_matmuls = [
        MatmulGroup(
            f"mhc_{sublayer}.fn",
            n=mix_width,
            k=hc * hidden,
            bucket="mhc",
            module=f"hc_{sublayer}_fn",
            storage_dtype="fp32",
            # The FP32 mixing GEMM runs on TF32 tensor cores; gpu/spec.json has no
            # TF32 peak, and the BF16 peak is a lower time bound on it.
            compute_dtype="bf16",
        )
        for sublayer in ("attn", "ffn")
    ]
    engram_hash_columns = (raw_config.get("engram_max_ngram_size", 1) - 1) * raw_config.get(
        "engram_n_heads", 0
    )
    engram_head_dim = raw_config.get("engram_head_dim", 0)
    engram_layers = list(raw_config.get("engram_layer_ids") or ())
    engram_rows = dict(zip(engram_layers, raw_config.get("engram_num_embeddings") or ()))

    layers: list[LayerStack] = []
    weights: list[LearnedWeightGroup] = []
    for tag, stack_layers, (ratio, kv_source, index_role, engram) in _schedule(raw_config):
        count = len(stack_layers)
        attention = DeepseekV41Attention(
            hidden=hidden,
            num_heads=raw_config["num_attention_heads"],
            head_dim=raw_config["head_dim"],
            q_lora_rank=raw_config["q_lora_rank"],
            o_lora_rank=raw_config["o_lora_rank"],
            o_groups=raw_config["o_groups"],
            window=raw_config["sliding_window"],
            compress_ratio=ratio,
            kv_source=kv_source,
            index_role=index_role,
            index_n_heads=raw_config["index_n_heads"],
            index_head_dim=raw_config["index_head_dim"],
            index_topk=raw_config["index_topk"],
            candidate_positions=raw_config.get("candidate_topk_blocks", 0)
            * raw_config.get("candidate_block_size", 0),
        )
        extra = list(mhc_matmuls)
        if engram:
            extra.append(
                MatmulGroup(
                    "engram.wkv",
                    n=hidden * (hc + 1),
                    k=engram_hash_columns * engram_head_dim,
                    bucket="engram",
                    module="engram.wkv",
                )
            )
        layers.append(
            LayerStack(attn=attention, ffn=moe, count=count, tag=tag, extra_matmuls=extra)
        )

        def weight(name: str, elements: int, **kwargs) -> LearnedWeightGroup:
            return LearnedWeightGroup(f"{tag}.{name}", elements, count, **kwargs)

        weights.extend(
            [
                weight("attn_norm", hidden),
                weight("ffn_norm", hidden),
                # Base (one per mix) and the three pre/post/comb scales, FP32.
                weight("mhc_attn.mix", mix_width + 3, breakdown="mhc", dtype_bytes=4),
                weight("mhc_ffn.mix", mix_width + 3, breakdown="mhc", dtype_bytes=4),
                # The FP32 correction bias chooses experts; it is read every token.
                weight("router_bias", moe.num_experts, breakdown="router", dtype_bytes=4),
            ]
        )
        if engram:
            if count != 1:
                raise ValueError("each engram layer owns its own table")
            rows = engram_rows[stack_layers[0]]
            lookups = engram_hash_columns

            def gathered(workload, rows=rows, lookups=lookups) -> float:
                return min(workload.matmul_tokens * lookups, rows) * engram_head_dim

            weights.extend(
                [
                    weight("engram.gate_weights", 2 * hc * hidden, breakdown="engram"),
                    # FP8 values plus one E8M0 byte per 32 values, per gathered row.
                    weight(
                        "engram.table",
                        rows * engram_head_dim,
                        breakdown="engram",
                        dtype_bytes=1 + 1 / 32,
                        read_elements=gathered,
                        activated_elements=lookups * engram_head_dim,
                    ),
                ]
            )
    weights.append(LearnedWeightGroup("final_norm", hidden, 1))

    return Model(
        name=raw_config["architectures"][0],
        hidden=hidden,
        vocab=raw_config["vocab_size"],
        weight_dtype_bytes=weight_bytes,
        tie_word_embeddings=raw_config.get("tie_word_embeddings", False),
        layers=layers,
        norm_weights=weights,
        master_dtype=master,
        quant=quant,
    )
