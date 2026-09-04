"""GLM-5.2 DSA/MoE model.work composition.

GLM-5.2 is heterogeneous in two independent ways: the first three layers use a
dense SwiGLU, and DSA full-index layers occur every fourth sparse layer after the
initial three index-share layers. The four body stacks below preserve that
schedule, which is also the unified CostTree's stable archetype order.

The checkpoint also carries ``num_nextn_predict_layers`` multi-token-prediction
layers past the body (layer 78: ``enorm``/``hnorm``/``eh_proj``, one full decoder
block, and a ``shared_head`` tied to the model's own output projection — see
vLLM's ``Glm4MoeLiteMultiTokenPredictorLayer``). Only a speculating deployment
loads and runs them, so they are staged: the two MTP stacks contribute nothing
unless the workload says a draft stage ran.
"""

from __future__ import annotations

from ..attention.glm52_dsa import Glm52DsaAttention
from ..core import LayerStack, MatmulGroup, Model, NormWeightGroup, dtype_bytes
from ..ffn.dense import DenseSwiGLU
from ..ffn.moe import MoE
from ..quantization import parse_quantization_config

#: :attr:`Workload.stages` key for the proposer's first draft pass. It forwards
#: the target's whole scheduled batch, because it is the pass that fills the MTP
#: layer's own attention state.
DRAFT_FIRST_STAGE = "mtp_first"
#: …and for every drafted position after the first: one row per request, over a
#: context the first pass already indexed.
DRAFT_RECURRENT_STAGE = "mtp_recurrent"


def _require_schedule(raw_config: dict) -> tuple[int, int, int, int]:
    layers = raw_config["num_hidden_layers"]
    mlp_types = raw_config.get("mlp_layer_types")
    indexer_types = raw_config.get("indexer_types")
    if mlp_types is None or indexer_types is None:
        raise ValueError("GLM-5.2 model.work requires explicit mlp_layer_types and indexer_types")
    if len(mlp_types) != layers or len(indexer_types) != layers:
        raise ValueError("GLM-5.2 layer schedules must each match num_hidden_layers")

    dense_full = sum(
        mlp == "dense" and indexer == "full" for mlp, indexer in zip(mlp_types, indexer_types)
    )
    # The model's explicit first six entries identify the initial three sparse
    # shared-index layers; use positional counts so duplicate values are harmless.
    sparse_initial_share = sum(
        1
        for layer, (mlp, indexer) in enumerate(zip(mlp_types, indexer_types))
        if 3 <= layer < 6 and mlp == "sparse" and indexer == "shared"
    )
    sparse_cycle_full = sum(
        1
        for layer, (mlp, indexer) in enumerate(zip(mlp_types, indexer_types))
        if layer >= 6 and mlp == "sparse" and indexer == "full"
    )
    sparse_cycle_share = sum(
        1
        for layer, (mlp, indexer) in enumerate(zip(mlp_types, indexer_types))
        if layer >= 6 and mlp == "sparse" and indexer == "shared"
    )
    expected = (3, 3, 18, 54)
    actual = (dense_full, sparse_initial_share, sparse_cycle_full, sparse_cycle_share)
    if actual != expected:
        raise ValueError(f"unsupported GLM-5.2 layer schedule: expected {expected}, got {actual}")
    if any(mlp != "dense" for mlp in mlp_types[:3]) or any(
        mlp != "sparse" for mlp in mlp_types[3:]
    ):
        raise ValueError("GLM-5.2 expects dense layers 0..2 followed by sparse layers")
    return actual


def _attention(raw_config: dict, *, full_index: bool) -> Glm52DsaAttention:
    weight_bytes = dtype_bytes(raw_config.get("dtype") or raw_config.get("torch_dtype", "bfloat16"))
    quant = raw_config.get("quantization_config", {})
    kv_cache_is_fp8 = quant.get("kv_cache_dtype") == "fp8_e4m3" or bool(
        quant.get("kv_cache_scheme")
    )
    return Glm52DsaAttention(
        hidden=raw_config["hidden_size"],
        num_heads=raw_config["num_attention_heads"],
        q_lora_rank=raw_config["q_lora_rank"],
        kv_lora_rank=raw_config["kv_lora_rank"],
        qk_nope_head_dim=raw_config["qk_nope_head_dim"],
        qk_rope_head_dim=raw_config["qk_rope_head_dim"],
        v_head_dim=raw_config["v_head_dim"],
        index_n_heads=raw_config["index_n_heads"],
        index_head_dim=raw_config["index_head_dim"],
        index_topk=raw_config["index_topk"],
        full_index=full_index,
        weight_dtype_bytes=weight_bytes,
        mla_cache_dtype_bytes=1.0 if kv_cache_is_fp8 else weight_bytes,
    )


def _mtp_stacks(raw_config: dict, hidden: int) -> list[LayerStack]:
    """The proposer's two draft archetypes, or nothing if the checkpoint has none.

    One MTP layer, entered twice per speculative iteration under two different
    index regimes, so it is two stacks rather than one:

    - the first pass forwards the target's whole batch and must build the DSA
      index itself, because it is the pass that fills the index buffer vLLM
      allocates per MTP layer (``Glm4MoeLiteMultiTokenPredictorLayer.__init__``);
    - every pass after it advances one token per request over a context the first
      already indexed, so it reuses those indices.

    Splitting them is not cosmetic: an index-share floor for the first pass would
    claim its index build is free, and a full-index floor for the recurrent passes
    would put the floor above the work a sharing implementation actually does.

    Only the first stack owns the layer's parameters (``param_count=0`` on the
    second). Both still read its weights, because the draft steps are serially
    dependent — step j+1 consumes step j's output — so no fusion can collapse the
    reads the way :meth:`Model.label`'s read-once convention collapses a single
    fused pass.
    """
    if not raw_config.get("num_nextn_predict_layers"):
        return []
    if raw_config["num_nextn_predict_layers"] != 1:
        raise ValueError(
            "GLM-5.2 model.work models a single MTP layer re-entered per drafted "
            f"position; got num_nextn_predict_layers={raw_config['num_nextn_predict_layers']}"
        )
    # Layer-relative module paths cannot distinguish the MTP layer's submodules
    # from a body layer's — both are `mlp.experts` — so the NVFP4 checkpoint's
    # `routed_experts_only` rule would sweep the MTP experts in with the body's.
    # It must not: the checkpoint quantizes the 78 body layers' routed experts and
    # ships the MTP layer's in BF16. Naming this stack's modules under `nextn` is
    # what keeps the two apart.
    mtp_ffn = MoE(
        hidden=hidden,
        moe_intermediate=raw_config["moe_intermediate_size"],
        num_experts=raw_config["n_routed_experts"],
        top_k=raw_config["num_experts_per_tok"],
        shared_intermediate=raw_config.get("shared_expert_intermediate_size")
        or raw_config["moe_intermediate_size"],
        module_prefix="nextn",
    )
    # eh_proj fuses the drafted token's embedding with the carried hidden state:
    # Linear(2*hidden -> hidden), no bias.
    eh_proj = MatmulGroup(
        "eh_proj",
        n=hidden,
        k=2 * hidden,
        bucket="dense_ffn",
        module="nextn.eh_proj",
    )
    return [
        LayerStack(
            attn=_attention(raw_config, full_index=True),
            ffn=mtp_ffn,
            count=1,
            tag="mtp_first_index",
            stage=DRAFT_FIRST_STAGE,
            extra_matmuls=[eh_proj],
        ),
        LayerStack(
            attn=_attention(raw_config, full_index=False),
            ffn=mtp_ffn,
            count=1,
            tag="mtp_recurrent",
            stage=DRAFT_RECURRENT_STAGE,
            extra_matmuls=[eh_proj],
            param_count=0,
        ),
    ]


def build(raw_config: dict) -> Model:
    dense_full_count, sparse_initial_count, sparse_cycle_full_count, sparse_cycle_share_count = (
        _require_schedule(raw_config)
    )
    hidden = raw_config["hidden_size"]
    weight_bytes = dtype_bytes(raw_config.get("dtype") or raw_config.get("torch_dtype", "bfloat16"))
    dense_ffn = DenseSwiGLU(hidden=hidden, intermediate=raw_config["intermediate_size"])
    sparse_ffn = MoE(
        hidden=hidden,
        moe_intermediate=raw_config["moe_intermediate_size"],
        num_experts=raw_config["n_routed_experts"],
        top_k=raw_config["num_experts_per_tok"],
        shared_intermediate=raw_config.get("shared_expert_intermediate_size")
        or raw_config["moe_intermediate_size"],
    )

    layers = [
        LayerStack(
            attn=_attention(raw_config, full_index=True),
            ffn=dense_ffn,
            count=dense_full_count,
            tag="dense_full_index",
        ),
        LayerStack(
            attn=_attention(raw_config, full_index=False),
            ffn=sparse_ffn,
            count=sparse_initial_count,
            tag="sparse_initial_index_share",
        ),
        LayerStack(
            attn=_attention(raw_config, full_index=True),
            ffn=sparse_ffn,
            count=sparse_cycle_full_count,
            tag="sparse_cycle_full_index",
        ),
        LayerStack(
            attn=_attention(raw_config, full_index=False),
            ffn=sparse_ffn,
            count=sparse_cycle_share_count,
            tag="sparse_cycle_index_share",
        ),
        *_mtp_stacks(raw_config, hidden),
    ]

    norm_weights: list[NormWeightGroup] = []
    for stack in layers:

        def norm(name: str, elements: int, stack: LayerStack = stack) -> NormWeightGroup:
            return NormWeightGroup(
                f"{stack.tag}.{name}",
                elements,
                stack.count,
                stage=stack.stage,
                param_count=stack.param_count,
            )

        norm_weights.extend(
            [
                norm("input_norm", hidden),
                norm("q_a_norm", raw_config["q_lora_rank"]),
                norm("kv_a_norm", raw_config["kv_lora_rank"]),
                norm("post_norm", hidden),
            ]
        )
        if stack.attn.full_index:
            norm_weights.append(norm("indexer_k_norm", raw_config["index_head_dim"]))
        if stack.stage is not None:
            # The MTP layer normalizes the drafted embedding and the carried
            # hidden state separately before fusing them, then normalizes once
            # more in front of its (tied) output projection.
            norm_weights.extend(
                [norm("enorm", hidden), norm("hnorm", hidden), norm("shared_head_norm", hidden)]
            )
    norm_weights.append(NormWeightGroup("final_norm", hidden, 1))

    return Model(
        name=raw_config["architectures"][0],
        hidden=hidden,
        vocab=raw_config["vocab_size"],
        weight_dtype_bytes=weight_bytes,
        tie_word_embeddings=raw_config.get("tie_word_embeddings", False),
        layers=layers,
        norm_weights=norm_weights,
        master_dtype=raw_config.get("dtype") or raw_config.get("torch_dtype", "bfloat16"),
        quant=parse_quantization_config(raw_config),
    )
