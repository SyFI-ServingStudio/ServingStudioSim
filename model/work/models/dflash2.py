"""DFlash2 block-parallel proposer, attached to a target model.

A DFlash2 draft (``DFlash2DraftModel``; vLLM ``qwen3_dflash2`` on top of
``qwen3_dflash``) is its own small checkpoint. One speculative step runs it
once, in three stages whose row counts differ:

- **context** (``dflash2_context``), over the ``C`` target rows the step
  scheduled: ``fc`` fuses the concatenated hidden states of the
  ``target_layer_ids`` layers (``len(ids)·hidden -> hidden``), ``hidden_norm``
  normalizes them, and ``precompute_and_store_context_kv`` projects every draft
  layer's K and V (the K/V halves of its ``qkv_proj``) and writes them into the
  draft's cache (:class:`~model.work.attention.dflash2.ContextKvProjection`).
- **draft** (``dflash2_draft``), over ``R·(k+1)`` query rows (bonus token plus
  ``k`` mask tokens per request): ``num_hidden_layers`` Qwen3 decoder layers,
  each wrapped by two grouped dynamic convolutions (``attention_conv`` and
  ``mlp_conv``: a ``hidden -> 2·taps·hidden/group`` coefficient projection plus
  a learned ``[2, taps, hidden]`` base kernel), attending non-causally inside a
  sliding window (:class:`~model.work.attention.dflash2.BlockWindowAttention`),
  then the final ``norm``.
- **select** (``dflash2_select``), over the ``R·k`` drafted positions:
  ``compute_candidates`` runs the target's ``lm_head`` (the draft ships none,
  vLLM shares the target's) and keeps ``selector_top_k`` candidates per
  position; ``CandidateSelector`` projects the hidden state to
  ``selector_rank`` and scores every predecessor/successor candidate edge
  through two ``[vocab, rank]`` codebooks (:class:`CandidateSelector`).

The draft also ships no embedding; its query rows embed through the target's
table, whose read the target's own ``embedding`` row already owns.

Convolution, norm, RoPE, top-k and the selector walk are elementwise or
selection work and stay out of the denominator; their learned weights do not.
Each stage reads its weights once (the accountant's read-once convention, as
for the MTP proposer), so the K/V projection weights are read by both the
context and the draft stage, but owned (counted as parameters) once.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..attention.base import AttentionSemantic
from ..attention.dflash2 import BlockWindowAttention, ContextKvProjection
from ..core import LayerStack, LearnedWeightGroup, MatmulGroup, Model, Workload, dtype_bytes
from ..ffn.dense import DenseSwiGLU

CONTEXT_STAGE = "dflash2_context"
DRAFT_STAGE = "dflash2_draft"
SELECT_STAGE = "dflash2_select"
STAGES = (CONTEXT_STAGE, DRAFT_STAGE, SELECT_STAGE)

#: The draft checkpoint is BF16 and separate from the target, so its modules
#: are named apart: no target quantization rule may match them.
MODULE_PREFIX = "dflash2"


@dataclass
class _NoAttention:
    """The attention slot of a stack that is only projections (``fc``)."""

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
class _Weights:
    """An FFN slot that carries only learned vectors (norms)."""

    weights: list[LearnedWeightGroup]

    def matmul_groups(self) -> list[MatmulGroup]:
        return []

    def learned_weight_groups(self) -> list[LearnedWeightGroup]:
        return self.weights


@dataclass
class _ConvWrappedMlp(DenseSwiGLU):
    """The draft layer's MLP half: post-attention norm, ``mlp_conv``, SwiGLU."""

    conv_projection_n: int = 0
    conv_base_elements: int = 0

    def matmul_groups(self) -> list[MatmulGroup]:
        return [
            MatmulGroup(
                "mlp_conv_proj",
                n=self.conv_projection_n,
                k=self.hidden,
                bucket="dense_ffn",
                module=self._module("mlp_conv.kernel_projection"),
            ),
            *super().matmul_groups(),
        ]

    def learned_weight_groups(self) -> list[LearnedWeightGroup]:
        return [
            LearnedWeightGroup("post_norm", self.hidden, 1, "norm"),
            LearnedWeightGroup("mlp_conv_base", self.conv_base_elements, 1, "ffn"),
        ]


@dataclass
class _ConvWrappedAttention(BlockWindowAttention):
    """The draft layer's attention half: input norm, ``attention_conv``, attention."""

    conv_projection_n: int = 0
    conv_base_elements: int = 0

    def matmul_groups(self) -> list[MatmulGroup]:
        return [
            MatmulGroup(
                "attention_conv_proj",
                n=self.conv_projection_n,
                k=self.hidden,
                bucket="attn_proj",
                module=self._module("attention_conv.kernel_projection"),
            ),
            *super().matmul_groups(),
        ]

    def learned_weight_groups(self) -> list[LearnedWeightGroup]:
        return [
            LearnedWeightGroup("input_norm", self.hidden, 1, "norm"),
            LearnedWeightGroup("attention_conv_base", self.conv_base_elements, 1, "attn"),
            *super().learned_weight_groups(),
        ]


@dataclass
class CandidateSelector:
    """``CandidateSelector`` over the drafted positions of one step.

    Per drafted position: ``hidden_projection`` (``hidden -> rank``), then
    ``einsum("blpr,blcr->blpc", predecessors * hidden, successors)`` -- a
    ``top_k x top_k`` edge matrix of rank-``rank`` dot products, ``2·top_k²·rank``
    FLOPs. The codebooks are read by row: a request's predecessors are its
    anchor token plus the candidates of its first ``k - 1`` positions, its
    successors the candidates of all ``k``. Each distinct row is read once
    (the embedding's convention), capped at the vocabulary.

    The workload is the stage's: ``matmul_tokens`` is the drafted positions and
    ``attention_step_count`` the requests.
    """

    split_attention_phases = False

    hidden: int
    vocab: int
    rank: int
    top_k: int
    weight_dtype_bytes: float

    def matmul_groups(self) -> list[MatmulGroup]:
        return [
            MatmulGroup(
                "hidden_projection",
                n=self.rank,
                k=self.hidden,
                bucket="dense_ffn",
                module=f"{MODULE_PREFIX}.candidate_selector.hidden_projection",
            )
        ]

    def learned_weight_groups(self) -> list[LearnedWeightGroup]:
        return [
            LearnedWeightGroup(
                "predecessor_codebook", self.vocab * self.rank, 1, "lm_head", gathered=True
            ),
            LearnedWeightGroup(
                "successor_codebook", self.vocab * self.rank, 1, "lm_head", gathered=True
            ),
        ]

    def internal_flops(self, wl: Workload) -> float:
        return 2.0 * wl.matmul_tokens * self.top_k * self.top_k * self.rank

    def kv_bytes(self, wl: Workload) -> float:
        return 0.0

    def cache_write_bytes(self, wl: Workload) -> float:
        return 0.0

    def semantic_segments(self, wl: Workload) -> list[AttentionSemantic]:
        positions = wl.matmul_tokens
        requests = wl.num_attention_steps
        if positions and (requests <= 0 or positions % requests):
            raise ValueError("selector positions must be a whole number per request")
        predecessors = requests + (positions - requests) * self.top_k
        successors = positions * self.top_k
        row_bytes = self.rank * self.weight_dtype_bytes
        return [
            AttentionSemantic("edge_scores", bucket="lm_head", flops=self.internal_flops(wl)),
            AttentionSemantic(
                "codebook_gather",
                bucket="embedding",
                byte_kind="weights",
                bytes=(min(predecessors, self.vocab) + min(successors, self.vocab)) * row_bytes,
            ),
        ]


def attach(target: Model, draft_config: dict, *, kv_dtype_bytes: float) -> Model:
    """``target`` with its own staged stacks (an MTP layer) replaced by a DFlash2
    proposer built from the draft checkpoint's ``config.json``.

    ``kv_dtype_bytes`` is the served KV-cache element size: vLLM's cache dtype
    is engine-wide, so the draft's layers share the target's.
    """
    if draft_config.get("architectures") != ["DFlash2DraftModel"]:
        raise ValueError(f"not a DFlash2 draft config: {draft_config.get('architectures')!r}")
    dflash = draft_config["dflash_config"]
    hidden = draft_config["hidden_size"]
    if hidden != target.hidden or draft_config["vocab_size"] != target.vocab:
        raise ValueError(
            "the DFlash2 draft shares the target's embedding and lm_head, so its "
            f"hidden/vocab must match: draft {hidden}/{draft_config['vocab_size']}, "
            f"target {target.hidden}/{target.vocab}"
        )
    target_hidden = draft_config.get("target_hidden_size") or hidden
    layers = draft_config["num_hidden_layers"]
    layer_types = draft_config.get("layer_types") or []
    if draft_config.get("is_causal", True) or any(t != "sliding_attention" for t in layer_types):
        raise ValueError("model.work models non-causal sliding-window DFlash2 layers only")
    if draft_config.get("attention_bias") or dflash.get("attention_sink_bias"):
        raise ValueError("model.work does not model DFlash2 attention biases or sinks")
    window = dflash.get("swa_window_size") or draft_config["sliding_window"]
    head_dim = draft_config["head_dim"]
    kv_heads = draft_config["num_key_value_heads"]
    taps = dflash["conv_kernel_size"]
    group = dflash["conv_group_size"]
    if hidden % group:
        raise ValueError("conv_group_size must divide hidden_size")
    conv_projection_n = 2 * taps * (hidden // group)
    conv_base_elements = 2 * taps * hidden
    weight_bytes = dtype_bytes(draft_config.get("dtype") or draft_config["torch_dtype"])
    if weight_bytes != target.weight_dtype_bytes:
        raise ValueError("model.work assumes the draft and target share a master dtype")

    def norm(name: str, elements: int = hidden) -> LearnedWeightGroup:
        return LearnedWeightGroup(name, elements, 1, "norm")

    stacks = [
        LayerStack(
            attn=_NoAttention(),
            ffn=_Weights([norm("hidden_norm")]),
            count=1,
            tag=CONTEXT_STAGE,
            stage=CONTEXT_STAGE,
            extra_matmuls=[
                MatmulGroup(
                    "fc",
                    n=hidden,
                    k=len(dflash["target_layer_ids"]) * target_hidden,
                    bucket="dense_ffn",
                    module=f"{MODULE_PREFIX}.fc",
                )
            ],
        ),
        LayerStack(
            attn=ContextKvProjection(
                hidden=hidden,
                num_kv_heads=kv_heads,
                head_dim=head_dim,
                kv_dtype_bytes=kv_dtype_bytes,
                module_prefix=MODULE_PREFIX,
            ),
            ffn=_NoFFN(),
            count=layers,
            tag=f"{CONTEXT_STAGE}_kv",
            stage=CONTEXT_STAGE,
            # The draft layers own these K/V weights and k_norm vectors.
            param_count=0,
        ),
        LayerStack(
            attn=_ConvWrappedAttention(
                hidden=hidden,
                num_qo_heads=draft_config["num_attention_heads"],
                num_kv_heads=kv_heads,
                head_dim=head_dim,
                sliding_window=window,
                kv_dtype_bytes=kv_dtype_bytes,
                compute_dtype="fp8" if kv_dtype_bytes == 1.0 else None,
                module_prefix=MODULE_PREFIX,
                conv_projection_n=conv_projection_n,
                conv_base_elements=conv_base_elements,
            ),
            ffn=_ConvWrappedMlp(
                hidden=hidden,
                intermediate=draft_config["intermediate_size"],
                module_prefix=MODULE_PREFIX,
                conv_projection_n=conv_projection_n,
                conv_base_elements=conv_base_elements,
            ),
            count=layers,
            tag=DRAFT_STAGE,
            stage=DRAFT_STAGE,
        ),
        LayerStack(
            attn=_NoAttention(),
            ffn=_Weights([norm("final_norm")]),
            count=1,
            tag=f"{DRAFT_STAGE}_head",
            stage=DRAFT_STAGE,
        ),
        LayerStack(
            attn=CandidateSelector(
                hidden=hidden,
                vocab=draft_config["vocab_size"],
                rank=dflash["selector_rank"],
                top_k=dflash["selector_top_k"],
                weight_dtype_bytes=weight_bytes,
            ),
            ffn=_NoFFN(),
            count=1,
            tag=SELECT_STAGE,
            stage=SELECT_STAGE,
        ),
    ]
    return replace(
        target,
        layers=[stack for stack in target.layers if stack.stage is None] + stacks,
        norm_weights=[weight for weight in target.norm_weights if weight.stage is None],
    )


def draft_sliding_window(draft_config: dict) -> int:
    return draft_config["dflash_config"].get("swa_window_size") or draft_config["sliding_window"]
