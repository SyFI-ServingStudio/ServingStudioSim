"""Independent DeepSeek-V4.1 sliding-window + compressed sparse attention accountant.

Derived from DeepSeek's reference ``inference/model.py`` (``Attention``,
``Compressor``, ``Indexer``, ``select_candidate_blocks``) and the config, never
from simulator shapes. Every layer projects a low-rank query
(``wq_a`` -> ``q_norm`` -> ``wq_b``: 64 heads x 512) and one shared 512-wide
KV head (``wkv`` -> ``kv_norm``); keys and values are the same vector, so each
(query, key, head) costs a 512-wide dot product plus a 512-wide value update.
The output projection is block-diagonal over ``o_groups`` (``wo_a``) followed
by ``wo_b``.

Each query attends to the union of two key sets in one ``sparse_attn`` call:

- its own layer's sliding window of the last ``sliding_window`` raw KV entries
  (every layer owns a window cache), and
- when ``compress_ratio > 0``, up to ``index_topk`` entries of a compressed KV
  cache holding one latent per ``compress_ratio`` tokens. Ratio 1 is a plain
  per-token projection; ratio 2 pools two tokens with a learned softmax gate.

Only the KV-source layers compress (and own the compressed cache); the others
read their source's cache. Only index-source layers run an indexer; the rest
reuse the published top-k. An index *owner* also projects the index keys
(``wk`` over each new compressed latent) into its own index cache. The other
index sources score with their own query against the candidate-source layer's
index keys, restricted to the ``candidate_topk_blocks`` blocks of
``candidate_block_size`` positions that the candidate source selected — logits
outside those blocks are masked away, so they are not necessary work.

Cache storage follows the reference quantize-dequantize numerics: window KV is
FP8 E4M3 with one UE8M0 scale per 32 values (528 B/entry), compressed KV is
E2M1 with one E4M3 scale per 16 values (288 B/entry), and index keys are MXFP4
(64 B + 4 UE8M0 scales = 68 B/entry). Index logits multiply MXFP4 queries by
MXFP4 keys; their floor uses the FP8 peak, which every target GPU has and which
vLLM's indexer uses (an FP4 MMA could halve this small compute term, which is
dwarfed by the index-cache reads). Attention runs BF16 queries.

Pinned conventions (see ``model/work/README.md``): norms, RoPE, softmax, the
attention sink logit, ReLU/weighting of index scores, top-k, candidate block
selection and masking carry zero FLOPs. Persistent state traffic is compulsory:
cached-key reads, cache appends, and the ratio-2 compressor's carried partial
group (FP32 KV + score, read or written across steps).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from ..core import LearnedWeightGroup, MatmulGroup, Workload
from .base import AttentionSemantic

# Bytes per cache entry, from the reference quantizers (see module docstring).
WINDOW_ENTRY_BYTES = 512 + 512 // 32  # 528
COMPRESSED_ENTRY_BYTES = 512 // 2 + 512 // 16  # 288
INDEX_ENTRY_BYTES = 128 // 2 + 128 // 32  # 68
FP32_BYTES = 4

INDEX_NONE = "none"
INDEX_OWNER = "owner"  # compresses its own index keys; scores every visible entry
INDEX_CANDIDATE_CONSUMER = "candidate_consumer"  # scores only inside candidate blocks


@dataclass(frozen=True)
class _Request:
    """A request's attention geometry: ``count`` equal requests appending
    ``append`` queries after ``prefix`` cached tokens."""

    count: float
    append: float
    prefix: float


def _interpolate(function: Callable[[int], float], x: float) -> float:
    """Evaluate an integer-domain function at a real point by linear interpolation.

    Exact inputs are integers and hit the function itself. Analyzer-collapsed
    geometry reconstructs a mean request whose lengths may be fractional.
    """
    low = math.floor(x)
    value = function(low)
    fraction = x - low
    if fraction:
        value += fraction * (function(low + 1) - value)
    return float(value)


def _window_cumulative(n: int, window: int) -> int:
    """Sum over the first ``n`` query positions of min(position, window)."""
    if n <= window:
        return n * (n + 1) // 2
    return window * (window + 1) // 2 + (n - window) * window


def _floor_sum(n: int, ratio: int) -> int:
    """Sum over j = 1..n of floor(j / ratio): compressed entries visible per query."""
    m = n // ratio
    return ratio * m * (m - 1) // 2 + m * (n - m * ratio + 1)


def _capped_floor_sum(n: int, ratio: int, cap: int) -> int:
    """Sum over j = 1..n of min(cap, floor(j / ratio))."""
    saturation = cap * ratio  # first position whose visible count reaches cap
    if n < saturation:
        return _floor_sum(n, ratio)
    return _floor_sum(saturation - 1, ratio) + (n - saturation + 1) * cap


@dataclass
class DeepseekV41Attention:
    """One DeepSeek-V4.1 attention layer role."""

    hidden: int
    num_heads: int
    head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    window: int
    compress_ratio: int
    kv_source: bool
    index_role: str
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    candidate_positions: int  # candidate_topk_blocks * candidate_block_size

    split_attention_phases = True

    def __post_init__(self) -> None:
        if self.compress_ratio == 0 and (self.kv_source or self.index_role != INDEX_NONE):
            raise ValueError("a sliding-window-only layer neither compresses nor indexes")
        if self.index_role not in (INDEX_NONE, INDEX_OWNER, INDEX_CANDIDATE_CONSUMER):
            raise ValueError(f"unknown index role {self.index_role!r}")
        if self.index_role == INDEX_OWNER and not self.kv_source:
            raise ValueError("an index owner projects keys from its own compressed latent")

    # ------------------------------------------------------------------ geometry

    def _requests(self, phase: str | None, workload: Workload) -> list[_Request]:
        """Per-request geometry of one phase.

        ``Workload.causal_lm`` keeps one causal interaction per request, which is
        exact. The analyzer instead sends one collapsed ``full`` interaction per
        phase carrying only sums (see ``floors._aggregate_workload``); a mean
        request is then reconstructed from the phase's request and token counts.
        That is exact for a single request and whenever every request sits on
        the same side of the window/top-k caps; its collapsed decode aggregate
        reports each request's current token as cached, which is removed here.
        """
        if not workload.attn:
            return []
        if all(interaction.mask == "causal" for interaction in workload.attn):
            return [
                _Request(
                    float(interaction.multiplicity),
                    float(interaction.num_query),
                    float(interaction.num_cached_key),
                )
                for interaction in workload.attn
            ]
        if any(interaction.mask == "causal" for interaction in workload.attn):
            raise ValueError("DeepSeek-V4.1 cannot mix exact and collapsed attention geometry")
        requests = float(workload.num_attention_steps)
        tokens = float(workload.matmul_tokens)
        if requests <= 0 or tokens <= 0:
            return []
        cached = sum(
            interaction.num_cached_key * interaction.multiplicity for interaction in workload.attn
        )
        prefix = cached / requests
        if phase == "decode":
            prefix = max(prefix - tokens / requests, 0.0)
        return [_Request(requests, tokens / requests, prefix)]

    def _latents(self, tokens: float) -> float:
        """Compressed entries formed after ``tokens`` tokens."""
        ratio = self.compress_ratio
        return _interpolate(lambda n: n // ratio, tokens)

    def _new_latents(self, request: _Request) -> float:
        return self._latents(request.prefix + request.append) - self._latents(request.prefix)

    def _cumulative_delta(self, function: Callable[[int], float], request: _Request) -> float:
        end = _interpolate(function, request.prefix + request.append)
        return end - _interpolate(function, request.prefix)

    def _selected_pairs(self, request: _Request) -> float:
        """(query, key) pairs attended: window keys plus top-k compressed keys."""
        pairs = self._cumulative_delta(lambda n: _window_cumulative(n, self.window), request)
        if self.compress_ratio:
            ratio, topk = self.compress_ratio, self.index_topk
            pairs += self._cumulative_delta(lambda n: _capped_floor_sum(n, ratio, topk), request)
        return pairs

    def _index_pairs(self, request: _Request) -> float:
        """(query, compressed entry) logits the indexer must compute."""
        ratio = self.compress_ratio
        if self.index_role == INDEX_CANDIDATE_CONSUMER:
            cap = self.candidate_positions
            return self._cumulative_delta(lambda n: _capped_floor_sum(n, ratio, cap), request)
        return self._cumulative_delta(lambda n: _floor_sum(n, ratio), request)

    def _attention_read_bytes(self, request: _Request) -> float:
        """Cached window and selected compressed entries read from HBM.

        A query's window reaches back ``window - 1`` tokens before its own; its
        chunk's own keys are fused activations. Selected compressed entries are
        read once per request, bounded by both the cache and ``index_topk`` per
        query.
        """
        window_entries = min(request.prefix, self.window - 1)
        read = window_entries * WINDOW_ENTRY_BYTES
        if self.compress_ratio:
            cached = self._latents(request.prefix)
            read += min(cached, request.append * self.index_topk) * COMPRESSED_ENTRY_BYTES
        return read

    def _index_read_bytes(self, request: _Request) -> float:
        cached = self._latents(request.prefix)
        if self.index_role == INDEX_CANDIDATE_CONSUMER:
            cached = min(cached, request.append * self.candidate_positions)
        return cached * INDEX_ENTRY_BYTES

    def _compressor_state_transactions(self, request: _Request) -> float:
        """Tokens moved through the ratio>1 compressor's carried partial group.

        A group completing in this step reads the tokens an earlier step left
        pending; tokens of a group still incomplete at the end are written. A
        decode token therefore always makes exactly one transaction at ratio 2.
        For a fractional (collapsed) request the pending phase is averaged.
        """
        ratio = self.compress_ratio

        def exact(prefix: int, append: int) -> int:
            pending = prefix % ratio
            completes = (prefix + append) // ratio > prefix // ratio
            read = pending if completes else 0
            write = (prefix + append) % ratio if completes else append
            return read + write

        if float(request.prefix).is_integer() and float(request.append).is_integer():
            return float(exact(int(request.prefix), int(request.append)))
        append = max(1, round(request.append))
        base = math.floor(request.prefix)
        base -= base % ratio
        return sum(exact(base + phase, append) for phase in range(ratio)) / ratio

    def _all_requests(self, workload: Workload) -> list[_Request]:
        requests: list[_Request] = []
        for phase, phase_workload in workload.attention_phases():
            requests.extend(self._requests(phase, phase_workload))
        return requests

    def _new_latent_rows(self, workload: Workload) -> float:
        return sum(
            request.count * self._new_latents(request) for request in self._all_requests(workload)
        )

    # ------------------------------------------------------------ AttentionSpec

    def matmul_groups(self) -> list[MatmulGroup]:
        head_width = self.num_heads * self.head_dim
        groups = [
            MatmulGroup(
                "wq_a", n=self.q_lora_rank, k=self.hidden, bucket="attn_proj", module="attn.wq_a"
            ),
            MatmulGroup(
                "wq_b", n=head_width, k=self.q_lora_rank, bucket="attn_proj", module="attn.wq_b"
            ),
            MatmulGroup(
                "wkv", n=self.head_dim, k=self.hidden, bucket="attn_proj", module="attn.wkv"
            ),
            # Block-diagonal: each group maps its own heads' outputs to o_lora_rank.
            MatmulGroup(
                "wo_a",
                n=self.o_lora_rank,
                k=head_width // self.o_groups,
                activated_mult=self.o_groups,
                total_count=self.o_groups,
                bucket="attn_proj",
                module="attn.wo_a",
            ),
            MatmulGroup(
                "wo_b",
                n=self.hidden,
                k=self.o_groups * self.o_lora_rank,
                bucket="attn_proj",
                module="attn.wo_b",
            ),
        ]
        if self.kv_source:
            groups.append(
                MatmulGroup(
                    "compressor.wkv",
                    n=self.head_dim,
                    k=self.hidden,
                    bucket="attn_proj",
                    module="attn.compressor.wkv",
                )
            )
            if self.compress_ratio > 1:
                groups.append(
                    MatmulGroup(
                        "compressor.wgate",
                        n=self.head_dim,
                        k=self.hidden,
                        bucket="attn_proj",
                        module="attn.compressor.wgate",
                    )
                )
        if self.index_role != INDEX_NONE:
            groups.extend(
                [
                    MatmulGroup(
                        "indexer.wq_b",
                        n=self.index_n_heads * self.index_head_dim,
                        k=self.q_lora_rank,
                        bucket="attn_proj",
                        module="attn.indexer.wq_b",
                    ),
                    MatmulGroup(
                        "indexer.weights_proj",
                        n=self.index_n_heads,
                        k=self.hidden,
                        bucket="attn_proj",
                        module="attn.indexer.weights_proj",
                    ),
                ]
            )
        if self.index_role == INDEX_OWNER:
            # Index keys are projected from each new compressed latent, not per token.
            groups.append(
                MatmulGroup(
                    "indexer.wk",
                    n=self.index_head_dim,
                    k=self.head_dim,
                    bucket="attn_proj",
                    module="attn.indexer.wk",
                    rows=self._new_latent_rows,
                )
            )
        return groups

    def learned_weight_groups(self) -> list[LearnedWeightGroup]:
        # The stack fold multiplies count / param_count; these describe one layer.
        weights = [
            LearnedWeightGroup("q_norm", self.q_lora_rank, 1),
            LearnedWeightGroup("kv_norm", self.head_dim, 1),
            LearnedWeightGroup("attn_sink", self.num_heads, 1, breakdown="attn", dtype_bytes=4),
        ]
        if self.kv_source:
            weights.append(LearnedWeightGroup("compressor.norm", self.head_dim, 1))
        if self.index_role == INDEX_OWNER:
            weights.append(LearnedWeightGroup("indexer.k_norm", self.index_head_dim, 1))
        return weights

    def semantic_segments(self, wl: Workload) -> list[AttentionSemantic]:
        """Indexer and attention rows per phase, then persistent-state rows.

        Every row is always emitted, even at zero work, so one iteration's
        semantic row set does not depend on its batch composition.
        """
        rows: list[AttentionSemantic] = []
        all_requests: list[_Request] = []
        for phase, phase_workload in wl.attention_phases():
            suffix = f".{phase}" if phase is not None else ""
            requests = self._requests(phase, phase_workload)
            all_requests.extend(requests)
            if self.index_role != INDEX_NONE:
                index_pairs = sum(r.count * self._index_pairs(r) for r in requests)
                rows.append(
                    AttentionSemantic(
                        name=f"indexer{suffix}",
                        flops=2.0 * self.index_n_heads * self.index_head_dim * index_pairs,
                        bytes=sum(r.count * self._index_read_bytes(r) for r in requests),
                        compute_dtype="fp8",
                    )
                )
            selected = sum(r.count * self._selected_pairs(r) for r in requests)
            rows.append(
                AttentionSemantic(
                    name=f"attn{suffix}",
                    # q.k over head_dim plus p.v over the same head_dim-wide value.
                    flops=2.0 * self.num_heads * (2 * self.head_dim) * selected,
                    bytes=sum(r.count * self._attention_read_bytes(r) for r in requests),
                    compute_dtype="bf16",
                )
            )

        # Only the last `window` tokens of a chunk survive in the ring buffer.
        rows.append(
            AttentionSemantic(
                name="window_cache_append",
                bucket="embedding",
                bytes=sum(r.count * min(r.append, self.window) for r in all_requests)
                * WINDOW_ENTRY_BYTES,
            )
        )
        if self.kv_source:
            new_latents = sum(r.count * self._new_latents(r) for r in all_requests)
            rows.append(
                AttentionSemantic(
                    name="compressed_cache_append",
                    bucket="embedding",
                    bytes=new_latents * COMPRESSED_ENTRY_BYTES,
                )
            )
            if self.compress_ratio > 1:
                state_bytes_per_token = 2 * self.head_dim * FP32_BYTES  # kv + score
                rows.append(
                    AttentionSemantic(
                        name="compressor_state",
                        bucket="embedding",
                        bytes=sum(
                            r.count * self._compressor_state_transactions(r) for r in all_requests
                        )
                        * state_bytes_per_token,
                    )
                )
            if self.index_role == INDEX_OWNER:
                rows.append(
                    AttentionSemantic(
                        name="index_cache_append",
                        bucket="embedding",
                        bytes=new_latents * INDEX_ENTRY_BYTES,
                    )
                )
        return rows

    # Historical AttentionSpec surface; Model.label uses semantic_segments.
    def internal_flops(self, wl: Workload) -> float:
        return sum(row.flops for row in self.semantic_segments(wl) if row.bucket == "attn_internal")

    def kv_bytes(self, wl: Workload) -> float:
        return sum(row.bytes for row in self.semantic_segments(wl) if row.bucket == "attn_internal")

    def cache_write_bytes(self, wl: Workload) -> float:
        return sum(row.bytes for row in self.semantic_segments(wl) if row.bucket == "embedding")
