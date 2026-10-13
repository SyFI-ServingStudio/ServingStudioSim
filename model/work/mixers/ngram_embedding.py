"""Per-layer n-gram embedding injected into a multi-stream residual (Qwen4Exp "PLE").

At a few decoder layers (``ple_layer_ids``) the vLLM fork
(``models/qwen4_exp/nvidia/{ple_layer,ngram_embedding}.py``, ``ops/ple.py``) adds a
hashed n-gram embedding to the residual streams before the attention module mixes:

- each token hashes its 2..``ngram_size`` token suffixes into ``ngram_heads`` heads
  (``(ngram_size - 1) * heads_per_ngram``), one row per head of a single table whose
  per-head vocabularies are consecutive primes from ``ngram_vocab_size_base``,
  padded to ``make_ngram_vocab_size_divisible_by``; the rows concatenate into a
  ``ple_embed_dim`` vector (``head_dim = ple_embed_dim / ngram_heads``);
- ``kv_proj`` maps it to a key (``S * hidden``, one per stream) and a value
  (``hidden``);
- per stream, ``d = RMSNorm(key_s) . RMSNorm(X_s) / sqrt(hidden)`` and the gate
  ``g = sigmoid(sign(d) sqrt(|d|))``; ``g_s * value`` is the gated output;
- a depthwise causal convolution (``kernel`` taps, ``dilation``) over the
  normalized gated output, through SiLU, plus the outer residual, is added back.

Necessary work per token: the ``kv_proj`` and the depthwise conv as
``MatmulGroup`` s, and the per-stream key/query dot, a contraction worth
``2 * S * hidden`` FLOPs. Hashing, norms, the gate nonlinearity, the value
broadcast and SiLU are elementwise (convention 4).

Bytes:

- the table is a parameter, read by row gather like the token embedding: a batch
  touches at most ``tokens * ngram_heads`` rows, capped at one pass of the table.
  The deployment keeps the table in host memory (pinned, read over the
  host link); those bytes are the same rows, whichever memory serves them;
- the convolution's persistent history is its last ``(kernel - 1) * dilation``
  inputs per request: a request reads ``min(prefix, history)`` cached rows and
  leaves ``min(append, history)`` new ones (``S * hidden`` wide, model dtype). That
  is per-request geometry, so the owner that carries these rows must label exact
  causal interactions.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..attention.base import AttentionSemantic
from ..core import LearnedWeightGroup, MatmulGroup, Workload


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def ngram_table_rows(
    vocab_size_base: int, ngram_heads: int, divisible_by: int, dense_layer_id: int = 0
) -> int:
    """Rows of one PLE layer's table: the sum of its heads' prime vocabularies (the
    ``global_head + 1``-th prime above ``vocab_size_base - 1``), padded."""
    primes: list[int] = []
    candidate = vocab_size_base - 1
    while len(primes) < (dense_layer_id + 1) * ngram_heads:
        candidate += 1
        if _is_prime(candidate):
            primes.append(candidate)
    rows = sum(primes[dense_layer_id * ngram_heads :])
    return -(-rows // divisible_by) * divisible_by


@dataclass(frozen=True)
class NgramEmbeddingInjection:
    hidden: int
    streams: int
    embed_dim: int
    ngram_heads: int
    table_rows: int
    table_dtype_bytes: float
    conv_kernel: int
    conv_dilation: int
    state_dtype_bytes: float
    #: Layer-relative checkpoint path of the module.
    path: str = "ple"

    def __post_init__(self) -> None:
        if self.embed_dim % self.ngram_heads:
            raise ValueError("ple_embed_dim must be divisible by the n-gram head count")

    @property
    def width(self) -> int:
        return self.streams * self.hidden

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.ngram_heads

    @property
    def history(self) -> int:
        return (self.conv_kernel - 1) * self.conv_dilation

    def matmul_groups(self) -> list[MatmulGroup]:
        return [
            # key_proj (S*hidden) and value_proj (hidden), packed by vLLM.
            MatmulGroup(
                "ple.kv_proj",
                n=self.width + self.hidden,
                k=self.embed_dim,
                bucket="engram",
                module=f"{self.path}.kv_proj",
            ),
            # Depthwise: params width*kernel, 2*width*kernel FLOPs per token.
            MatmulGroup(
                "ple.conv1d",
                n=self.width,
                k=self.conv_kernel,
                bucket="engram",
                module=f"{self.path}.conv1d",
            ),
        ]

    def learned_weight_groups(self) -> list[LearnedWeightGroup]:
        return [
            LearnedWeightGroup("ple.norm_key", self.width, 1),
            LearnedWeightGroup("ple.norm_query", self.width, 1),
            LearnedWeightGroup("ple.norm_conv", self.width, 1),
            LearnedWeightGroup(
                "ple.ngram_embedding",
                self.table_rows * self.head_dim,
                1,
                "engram",
                gathered=True,
                dtype_bytes=self.table_dtype_bytes,
                activated_elements=self.embed_dim,
            ),
        ]

    def semantic_segments(self, wl: Workload) -> list[AttentionSemantic]:
        state_rows = 0.0
        for interaction in wl.attn:
            if interaction.mask != "causal":
                raise ValueError(
                    "the n-gram embedding's convolution history needs exact per-request "
                    f"causal geometry; got a {interaction.mask!r} interaction"
                )
            if interaction.num_query <= 0:
                continue
            state_rows += interaction.multiplicity * (
                min(interaction.num_cached_key, self.history)
                + min(interaction.num_query, self.history)
            )
        gathered_rows = min(wl.matmul_tokens * self.ngram_heads, self.table_rows)
        return [
            AttentionSemantic(
                "ple.lookup",
                bucket="embedding",
                byte_kind="weights",
                bytes=gathered_rows * self.head_dim * self.table_dtype_bytes,
            ),
            AttentionSemantic(
                "ple.gate", bucket="engram", flops=2.0 * self.width * wl.matmul_tokens
            ),
            AttentionSemantic(
                "ple.conv_state",
                bucket="embedding",
                bytes=state_rows * self.width * self.state_dtype_bytes,
            ),
        ]
