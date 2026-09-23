"""Kimi-K3 text-backbone necessary-work composition.

The top-level checkpoint is a VL wrapper. This builder deliberately labels only
the nested KimiLinear causal language model; the vision tower is outside the
text-serving workloads accepted by ``model.work``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..attention.attn_residual import KimiAttentionResidual
from ..attention.linear import KimiDeltaAttention
from ..attention.mla import MLA
from ..core import LayerStack, Model, NormWeightGroup, dtype_bytes
from ..ffn.dense import DenseSwiGLU
from ..ffn.moe import MoE
from ..quantization import parse_quantization_config


@dataclass(frozen=True)
class KimiK3WorkScope:
    """The rank/stage shape emitted by the Kimi-K3 simulator arch.

    The simulator's manifest is a physical rank-local view, while the model
    config describes the complete checkpoint.  Keeping this bridge explicit
    prevents the accountant from silently labeling the 93-layer, 96-head,
    896-expert model for a one-layer or one-rank prediction.
    """

    dense_layers: int
    kda_layers: int
    mla_layers: int
    heads_per_rank: int
    local_experts: int
    kda_state_dtype: str | None = None
    include_model_io: bool = True
    work_scale: float = 1.0
    global_heads: int = 96
    global_experts: int = 896
    pp_size: int = 1
    pp_stage_layer_counts: tuple[tuple[int, int, int], ...] = ()

    @classmethod
    def from_dict(cls, raw: dict) -> "KimiK3WorkScope":
        if raw.get("schema_version") != 1:
            raise ValueError("unsupported Kimi-K3 model.work scope schema")
        counts = raw.get("layer_counts")
        if not isinstance(counts, dict):
            raise ValueError("Kimi-K3 model.work scope requires layer_counts")

        def positive_int(name: str, value: object) -> int:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Kimi-K3 scope {name} must be a non-negative integer")
            return value

        dense_layers = positive_int("dense", counts.get("dense", 0))
        kda_layers = positive_int("kda", counts.get("kda", 0))
        mla_layers = positive_int("mla", counts.get("mla", 0))
        heads = positive_int("heads_per_rank", raw.get("heads_per_rank"))
        local_experts = positive_int("local_experts", raw.get("local_experts"))
        global_heads = positive_int("global_heads", raw.get("global_heads", 96))
        global_experts = positive_int("global_experts", raw.get("global_experts", 896))
        pp_size = positive_int("pp_size", raw.get("pp_size", 1))
        if pp_size == 0:
            raise ValueError("Kimi-K3 scope pp_size must be positive")
        if not heads or not local_experts:
            raise ValueError("Kimi-K3 scope rank-local heads and experts must be positive")
        if global_heads != 96 or global_experts != 896:
            raise ValueError("Kimi-K3 scope global geometry does not match the checkpoint")
        if global_heads % heads or global_experts % local_experts:
            raise ValueError("Kimi-K3 scope rank-local geometry must divide global geometry")
        state_dtype = raw.get("kda_state_dtype")
        if state_dtype is not None and state_dtype not in {"bf16", "fp32"}:
            raise ValueError("Kimi-K3 scope kda_state_dtype must be bf16 or fp32")
        work_scale = raw.get("work_scale", 1.0)
        if isinstance(work_scale, bool) or not isinstance(work_scale, (int, float)):
            raise ValueError("Kimi-K3 scope work_scale must be positive")
        work_scale = float(work_scale)
        if work_scale <= 0.0:
            raise ValueError("Kimi-K3 scope work_scale must be positive")
        include_model_io = raw.get("include_model_io", True)
        if not isinstance(include_model_io, bool):
            raise ValueError("Kimi-K3 scope include_model_io must be boolean")
        raw_stage_counts = raw.get("pp_stage_layer_counts")
        stage_counts: tuple[tuple[int, int, int], ...] = ()
        if raw_stage_counts is not None:
            if not isinstance(raw_stage_counts, list) or len(raw_stage_counts) != pp_size:
                raise ValueError("Kimi-K3 scope pp_stage_layer_counts must match pp_size")
            parsed_stages = []
            for index, stage in enumerate(raw_stage_counts):
                if not isinstance(stage, dict):
                    raise ValueError(f"Kimi-K3 scope PP stage {index} must be an object")
                parsed_stages.append(
                    (
                        positive_int("stage dense", stage.get("dense", 0)),
                        positive_int("stage kda", stage.get("kda", 0)),
                        positive_int("stage mla", stage.get("mla", 0)),
                    )
                )
            stage_counts = tuple(parsed_stages)
            stage_totals = tuple(
                sum(stage[index] for stage in stage_counts) for index in range(3)
            )
            if stage_totals != (
                dense_layers,
                kda_layers,
                mla_layers,
            ):
                raise ValueError("Kimi-K3 PP stage layer counts do not sum to layer_counts")
        return cls(
            dense_layers=dense_layers,
            kda_layers=kda_layers,
            mla_layers=mla_layers,
            heads_per_rank=heads,
            local_experts=local_experts,
            kda_state_dtype=state_dtype,
            include_model_io=include_model_io,
            work_scale=work_scale,
            global_heads=global_heads,
            global_experts=global_experts,
            pp_size=pp_size,
            pp_stage_layer_counts=stage_counts,
        )


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
            "Kimi-K3 requires explicit linear_attn_config.kda_layers and full_attn_layers lists"
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


def build(raw_config: dict, scope: dict | None = None) -> Model:
    text = _text_config(raw_config)
    kda_layers, full_layers = _validate_schedule(text)
    work_scope = KimiK3WorkScope.from_dict(scope) if scope is not None else None

    hidden = text["hidden_size"]
    master_dtype = text.get("dtype") or text.get("torch_dtype", "bfloat16")
    weight_bytes = dtype_bytes(master_dtype)
    state_bytes = dtype_bytes(
        work_scope.kda_state_dtype
        if work_scope is not None and work_scope.kda_state_dtype is not None
        else text.get("mamba_ssm_dtype", "float32")
    )
    conv_state_bytes = dtype_bytes(text.get("mamba_conv_dtype", "bfloat16"))
    kv_cache_bytes = dtype_bytes(text.get("kv_cache_dtype", "bfloat16"))

    linear_config = text["linear_attn_config"]
    if linear_config.get("head_dim") != 128 or linear_config.get("num_heads") != 96:
        raise ValueError("Kimi-K3 KDA geometry must be 96 heads of head_dim 128")
    if not linear_config.get("use_full_rank_gate", False):
        raise ValueError("Kimi-K3 requires use_full_rank_gate=true")

    attn_res_block_size = text.get("attn_res_block_size")
    layer_count = text["num_hidden_layers"]
    kda_indices = tuple(layer - 1 for layer in kda_layers)
    full_indices = tuple(layer - 1 for layer in full_layers)
    kda_moe_indices = tuple(index for index in kda_indices if index != 0)

    def residual(
        indices: tuple[int, ...], *, include_output: bool = False
    ) -> KimiAttentionResidual | None:
        if attn_res_block_size is None:
            return None
        return KimiAttentionResidual(
            hidden=hidden,
            block_size=attn_res_block_size,
            layer_indices=indices,
            total_layers=layer_count,
            include_output=include_output,
        )

    heads = work_scope.heads_per_rank if work_scope is not None else linear_config["num_heads"]
    moe_local_experts = work_scope.local_experts if work_scope is not None else None
    moe_routing_scale = (
        (moe_local_experts / text["num_experts"])
        if moe_local_experts is not None
        else 1.0
    )

    def make_kda(attn_residual: KimiAttentionResidual | None) -> KimiDeltaAttention:
        return KimiDeltaAttention(
            hidden=hidden,
            num_heads=heads,
            head_dim=linear_config["head_dim"],
            conv_kernel=linear_config["short_conv_kernel_size"],
            state_dtype_bytes=state_bytes,
            activation_dtype_bytes=conv_state_bytes,
            attn_residual=attn_residual,
        )

    dense_kda_residual = residual((0,), include_output=True)
    kda_moe_residual = residual(kda_moe_indices)
    mla_residual = residual(full_indices)
    dense_kda = make_kda(dense_kda_residual)
    kda_moe = make_kda(kda_moe_residual)
    mla = MLA(
        hidden=hidden,
        num_heads=heads,
        q_lora_rank=text["q_lora_rank"],
        kv_lora_rank=text["kv_lora_rank"],
        qk_nope_head_dim=text["qk_nope_head_dim"],
        qk_rope_head_dim=text["qk_rope_head_dim"],
        v_head_dim=text["v_head_dim"],
        kv_dtype_bytes=kv_cache_bytes,
        output_gate=text.get("mla_use_output_gate", False),
        attn_residual=mla_residual,
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
            text["routed_expert_hidden_size"] if text.get("latent_moe_use_norm", False) else 0
        ),
        local_experts=moe_local_experts,
        routing_scale=moe_routing_scale,
    )

    dense_count = work_scope.dense_layers if work_scope is not None else 1
    kda_moe_count = work_scope.kda_layers if work_scope is not None else len(kda_moe_indices)
    mla_count = work_scope.mla_layers if work_scope is not None else len(full_indices)
    # Layer 1 is the only dense KDA layer; the remaining KDA layers are MoE.
    if work_scope is None and (1 not in kda_layers or kda_moe_count != 68):
        raise ValueError("Kimi-K3 expects layer 1 to be the dense KDA layer")

    layers = []
    if dense_count:
        layers.append(
            LayerStack(
                attn=dense_kda,
                ffn=dense,
                count=dense_count,
                tag="dense",
                extra_matmuls=(
                    dense_kda_residual.output_matmul_groups()
                    if dense_kda_residual is not None
                    else []
                ),
            )
        )
    if kda_moe_count:
        layers.append(LayerStack(attn=kda_moe, ffn=moe, count=kda_moe_count, tag="kda"))
    if mla_count:
        layers.append(LayerStack(attn=mla, ffn=moe, count=mla_count, tag="mla"))

    norm_weights = []
    if dense_count:
        norm_weights.extend(_layer_norms("dense", hidden, dense_count))
    if kda_moe_count:
        norm_weights.extend(_layer_norms("kda", hidden, kda_moe_count))
    if mla_count:
        norm_weights.extend(_layer_norms("mla", hidden, mla_count))
        norm_weights.extend(
            [
                NormWeightGroup("mla.q_a_layernorm", text["q_lora_rank"], mla_count),
                NormWeightGroup("mla.kv_a_layernorm", text["kv_lora_rank"], mla_count),
            ]
        )
    if work_scope is None or work_scope.include_model_io:
        norm_weights.append(NormWeightGroup("final_norm", hidden, 1))
    if dense_kda_residual is not None and dense_count and (
        work_scope is None or work_scope.include_model_io
    ):
        norm_weights.extend(dense_kda_residual.output_learned_weight_groups())

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
        include_model_io=work_scope is None or work_scope.include_model_io,
        work_scale=work_scope.work_scale if work_scope is not None else 1.0,
    )
