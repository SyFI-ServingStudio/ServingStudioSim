"""DeepSeek-V4.1 DSpark block proposer: the checkpoint's ``mtp.*`` draft stages.

Sources: DeepSeek's reference ``inference/model.py`` (``DSparkBlock``,
``DSparkAttention``, ``DSparkMarkovHead``, ``Transformer.forward_spec``) and the
checkpoint's safetensors dtypes; vLLM's ``deepseek_v41/nvidia/dspark.py`` and the
DSpark speculator only confirm the semantics. One speculative step drafts
``k = dspark_block_size`` tokens per request in one block pass, then fixes them
by a short sequential Markov chain. Three stages with different row counts:

- **context** (``dspark_context``), over the ``C`` target rows the step
  scheduled: the target's ``dspark_target_layer_ids`` layers hand down the mean
  of their ``hc_mult`` residual copies (activation arithmetic), concatenated;
  ``main_proj`` (FP8, ``len(ids)·hidden -> hidden``) and ``main_norm`` turn them
  into ``main_x``. Each of the ``num_nextn_predict_layers`` draft layers then
  projects ``main_x`` with its own ``wkv`` and ``kv_norm`` into its own
  sliding-window cache (:class:`~model.work.attention.dspark.DSparkContextKv`).
- **draft** (``dspark_draft``), over ``R·k`` block rows (the anchor token plus
  ``k - 1`` noise tokens per request): ``num_nextn_predict_layers`` DeepSeek-V4.1
  sliding-window layers -- mHC around both sublayers, the block attention
  (:class:`~model.work.attention.dspark.DSparkBlockAttention`), a MoE of
  ``dspark_n_routed_experts`` MXFP4 experts with top-``dspark_num_experts_per_tok``
  routing and one FP8 shared expert -- then the last layer's ``norm`` and the
  target's ``lm_head`` over every block row.
- **markov** (``dspark_markov``), ``k`` sequential steps over ``R`` rows each:
  ``markov_head.embed`` gathers the previous token's rank-``dspark_markov_rank``
  row, ``markov_head.head`` projects it to a full-vocabulary logit bias, which is
  added to that position's draft logits before greedy sampling.

Excluded, deliberately: the ``confidence_head`` (only adaptive verification runs
it; the served config does not) and each draft router's ``bias_vl`` (drafts are
text). The draft embeds its block through the target's table and samples
through the target's head, whose reads the target's own rows already own. hc
expand/collapse, norms, RoPE, softmax, the logit-bias add and argmax carry no
FLOPs. Each stage reads its weights once (the accountant's read-once
convention), so ``wkv``/``kv_norm`` are read by both the context and the draft
stage but owned (counted as parameters) once.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..attention.base import AttentionSemantic
from ..attention.deepseek_v41 import INDEX_NONE
from ..attention.dspark import DSparkBlockAttention, DSparkContextKv
from ..core import AttnInteraction, LayerStack, LearnedWeightGroup, MatmulGroup, Workload
from ..ffn.moe import MoE

CONTEXT_STAGE = "dspark_context"
DRAFT_STAGE = "dspark_draft"
MARKOV_STAGE = "dspark_markov"
STAGES = (CONTEXT_STAGE, DRAFT_STAGE, MARKOV_STAGE)

#: The markov tables are BF16 in the checkpoint.
_MARKOV_DTYPE_BYTES = 2


@dataclass
class _NoAttention:
    """The attention slot of a stack that is only projections (``main_proj``)."""

    split_attention_phases = False

    def matmul_groups(self) -> list[MatmulGroup]:
        return []

    def internal_flops(self, wl: Workload) -> float:
        return 0.0

    def kv_bytes(self, wl: Workload) -> float:
        return 0.0

    def cache_write_bytes(self, wl: Workload) -> float:
        return 0.0

    def semantic_segments(self, wl: Workload) -> list[AttentionSemantic]:
        return []


@dataclass
class _NoFFN:
    def matmul_groups(self) -> list[MatmulGroup]:
        return []


@dataclass
class MarkovHead:
    """``DSparkMarkovHead`` over the ``R·k`` drafted positions of one step.

    Each position gathers one ``[vocab, rank]`` embedding row (its predecessor:
    the anchor, then the previous draft) and multiplies it by the ``[vocab, rank]``
    head, ``2·rank·vocab`` FLOPs. Distinct embedding rows are read once each,
    capped at the vocabulary; the head is read once for the stage.
    """

    vocab: int
    rank: int

    split_attention_phases = False

    def matmul_groups(self) -> list[MatmulGroup]:
        return []

    def learned_weight_groups(self) -> list[LearnedWeightGroup]:
        return [
            LearnedWeightGroup(
                "markov_embed",
                self.vocab * self.rank,
                1,
                "embedding",
                gathered=True,
                activated_elements=self.rank,
            ),
            LearnedWeightGroup("markov_head", self.vocab * self.rank, 1, "lm_head", gathered=True),
        ]

    def semantic_segments(self, wl: Workload) -> list[AttentionSemantic]:
        positions = wl.matmul_tokens
        return [
            AttentionSemantic(
                "markov_embed",
                bucket="embedding",
                byte_kind="weights",
                bytes=min(positions, self.vocab) * self.rank * _MARKOV_DTYPE_BYTES,
            ),
            AttentionSemantic(
                "markov_head",
                bucket="lm_head",
                byte_kind="weights",
                flops=2.0 * positions * self.rank * self.vocab,
                bytes=self.vocab * self.rank * _MARKOV_DTYPE_BYTES,
                compute_dtype="bf16",
            ),
        ]

    def internal_flops(self, wl: Workload) -> float:
        return 0.0

    def kv_bytes(self, wl: Workload) -> float:
        return 0.0

    def cache_write_bytes(self, wl: Workload) -> float:
        return 0.0


def enabled(raw_config: dict) -> bool:
    """Does this checkpoint ship DSpark draft layers?"""
    return bool(raw_config.get("num_nextn_predict_layers")) and bool(
        raw_config.get("dspark_block_size")
    )


def stacks(
    raw_config: dict, mhc_matmuls: list[MatmulGroup]
) -> tuple[list[LayerStack], list[LearnedWeightGroup]]:
    """The DSpark draft's staged layer stacks and learned vectors.

    ``mhc_matmuls`` are the target's per-layer FP32 mixing projections, which a
    draft layer carries unchanged (``hc_attn_fn``/``hc_ffn_fn``).
    """
    hidden = raw_config["hidden_size"]
    hc = raw_config["hc_mult"]
    layers = raw_config["num_nextn_predict_layers"]
    body = raw_config["num_hidden_layers"]
    block = raw_config["dspark_block_size"]
    targets = raw_config["dspark_target_layer_ids"]
    ratios = raw_config["compress_ratios"]
    if len(ratios) != body + layers or any(ratios[body + i] for i in range(layers)):
        raise ValueError("DSpark draft layers must be sliding-window only (compress ratio 0)")
    if not targets or any(not 0 <= layer < body for layer in targets):
        raise ValueError(f"DSpark target layers {targets} must be body layers")
    window = raw_config["sliding_window"]
    head_dim = raw_config["head_dim"]
    experts = raw_config.get("dspark_n_routed_experts") or raw_config["n_routed_experts"]

    attention = DSparkBlockAttention(
        hidden=hidden,
        num_heads=raw_config["num_attention_heads"],
        head_dim=head_dim,
        q_lora_rank=raw_config["q_lora_rank"],
        o_lora_rank=raw_config["o_lora_rank"],
        o_groups=raw_config["o_groups"],
        window=window,
        compress_ratio=0,
        kv_source=False,
        index_role=INDEX_NONE,
        index_n_heads=raw_config["index_n_heads"],
        index_head_dim=raw_config["index_head_dim"],
        index_topk=raw_config["index_topk"],
        candidate_positions=0,
        block_size=block,
    )
    moe = MoE(
        hidden=hidden,
        moe_intermediate=raw_config["moe_intermediate_size"],
        num_experts=experts,
        top_k=raw_config.get("dspark_num_experts_per_tok") or raw_config["num_experts_per_tok"],
        shared_intermediate=raw_config["n_shared_experts"] * raw_config["moe_intermediate_size"],
    )
    layer_stacks = [
        LayerStack(
            attn=_NoAttention(),
            ffn=_NoFFN(),
            count=1,
            tag=CONTEXT_STAGE,
            stage=CONTEXT_STAGE,
            extra_matmuls=[
                MatmulGroup(
                    "main_proj",
                    n=hidden,
                    k=len(targets) * hidden,
                    bucket="dense_ffn",
                    module="main_proj",
                )
            ],
        ),
        LayerStack(
            attn=DSparkContextKv(hidden=hidden, head_dim=head_dim, window=window),
            ffn=_NoFFN(),
            count=layers,
            tag=f"{CONTEXT_STAGE}_kv",
            stage=CONTEXT_STAGE,
            # The draft layers own these wkv weights and kv_norm vectors.
            param_count=0,
        ),
        LayerStack(
            attn=attention,
            ffn=moe,
            count=layers,
            tag=DRAFT_STAGE,
            stage=DRAFT_STAGE,
            extra_matmuls=list(mhc_matmuls),
        ),
        LayerStack(
            attn=MarkovHead(vocab=raw_config["vocab_size"], rank=raw_config["dspark_markov_rank"]),
            ffn=_NoFFN(),
            count=1,
            tag=MARKOV_STAGE,
            stage=MARKOV_STAGE,
        ),
    ]

    mix_width = (2 + hc) * hc

    def per_layer(name: str, elements: int, **kwargs) -> LearnedWeightGroup:
        return LearnedWeightGroup(
            f"{DRAFT_STAGE}.{name}",
            elements,
            layers,
            stage=DRAFT_STAGE,
            stack=DRAFT_STAGE,
            **kwargs,
        )

    weights = [
        LearnedWeightGroup(f"{CONTEXT_STAGE}.main_norm", hidden, 1, stage=CONTEXT_STAGE),
        per_layer("attn_norm", hidden),
        per_layer("ffn_norm", hidden),
        per_layer("mhc_attn.mix", mix_width + 3, breakdown="mhc", dtype_bytes=4),
        per_layer("mhc_ffn.mix", mix_width + 3, breakdown="mhc", dtype_bytes=4),
        per_layer("router_bias", moe.num_experts, breakdown="router", dtype_bytes=4),
        LearnedWeightGroup(f"{DRAFT_STAGE}.final_norm", hidden, 1, stage=DRAFT_STAGE),
    ]
    return layer_stacks, weights


def stage_workloads(
    target: Workload, endpoints: dict[int, float], depth: int
) -> dict[str, Workload]:
    """The three DSpark stages of one step.

    ``target`` is the step's verification workload (its per-request causal
    interactions give each request's appended rows); ``endpoints`` maps the
    context each request ends the step with to how many requests end there.
    Every scheduled request drafts, prefilling or verifying alike, one block of
    ``depth`` rows against that context.
    """
    requests = sum(endpoints.values())
    rows = requests * depth
    blocks = [
        AttnInteraction(depth, context + depth, context, "full", None, count)
        for context, count in sorted(endpoints.items())
    ]
    return {
        CONTEXT_STAGE: Workload(
            matmul_tokens=target.matmul_tokens,
            head_positions=0,
            attn=list(target.attn),
            attention_step_count=target.num_attention_steps,
        ),
        DRAFT_STAGE: Workload(
            matmul_tokens=rows,
            head_positions=rows,
            attn=blocks,
            attention_step_count=requests,
        ),
        MARKOV_STAGE: Workload(matmul_tokens=rows, head_positions=0, attention_step_count=requests),
    }
