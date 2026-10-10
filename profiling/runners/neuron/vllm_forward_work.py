"""Per-rank graph contractions and persistent operand estimates, not counters.

Derived from stock model/FX shapes and nkilib attention_cte causal tile loops;
see doc/trainium2_vllm_inventory.md. These intentionally do not use model.work.
Nonmatmul arithmetic, temporaries, collectives, spills and residency are omitted.
"""


def estimate_work(phase: str, token_bucket: int, max_model_len: int) -> dict[str, int]:
    tokens = token_bucket
    sampled_rows = 1 if phase == "prefill" else tokens
    dense = 2 * 32 * tokens * (4096 * 1536 + 1024 * 4096 + 3 * 4096 * 3584)
    dense += 2 * sampled_rows * 4096 * 32064
    if phase == "prefill":
        qk_pairs = pv_pairs = 0
        for query_start in range(0, tokens, 128):
            query_rows = min(128, tokens - query_start)
            qk_pairs += query_rows * min(tokens, 512 * (query_start // 512 + 1))
            pv_pairs += query_rows * min(tokens, query_start + 128)
        attention = 32 * 2 * 8 * 128 * (qk_pairs + pv_pairs)
        cache_bytes = 32768 * tokens
    elif phase == "decode":
        attention = 32 * 4 * tokens * 8 * max_model_len * 128
        cache_bytes = 32768 * tokens * (max_model_len + 1)
    else:
        raise ValueError("unknown full-forward phase")
    return {
        "contraction_flops_per_rank": dense + attention,
        "attention_flops_per_rank": attention,
        "persistent_operand_bytes_per_rank": 3752861696 + 2 * tokens * 4096 + cache_bytes,
    }


LM_HEAD_BYTES_PER_RANK = 2 * 32064 * 4096  # BF16 vocabulary shard read by the projection.


def estimate_region_work(
    region: str, phase: str, token_bucket: int, max_model_len: int
) -> dict[str, int]:
    """Split ``estimate_work`` at the public LlamaModel return; same caveats.

    ``head`` is the lm_head contraction over the sampled rows and its weight
    shard; ``model`` is the whole-forward estimate minus ``head``.
    """
    whole = estimate_work(phase, token_bucket, max_model_len)
    sampled_rows = 1 if phase == "prefill" else token_bucket
    head = {
        "contraction_flops_per_rank": 2 * sampled_rows * 4096 * 32064,
        "attention_flops_per_rank": 0,
        "persistent_operand_bytes_per_rank": LM_HEAD_BYTES_PER_RANK,
    }
    if region == "head":
        return head
    if region == "model":
        return {name: whole[name] - head[name] for name in whole}
    raise ValueError("region must be model or head")


NORM_BYTES_PER_RANK = 2 * 4096  # One replicated BF16 RMSNorm weight.


def estimate_segment_work(
    segment: str, phase: str, token_bucket: int, max_model_len: int
) -> dict[str, int]:
    """Split ``estimate_work`` by the semantic owner of each collective segment.

    ``embedding``: gathered BF16 embedding rows, no contraction. ``attention_block``
    and ``mlp_block`` are ONE layer: its input or post-attention RMSNorm weight,
    QKV/O (with 1/32 of the attention contractions and KV-cache bytes) or the gated
    MLP. ``head``: final norm plus the lm_head contraction and shard. Then
    embedding + 32 x (attention + MLP) + head equals the whole-forward estimate.
    Same caveats: estimates, not counters.
    """
    tokens = token_bucket
    whole = estimate_work(phase, token_bucket, max_model_len)
    head = estimate_region_work("head", phase, token_bucket, max_model_len)
    cache_bytes = whole["persistent_operand_bytes_per_rank"] - 3752861696 - 2 * tokens * 4096
    attention = whole["attention_flops_per_rank"] // 32
    rows = {
        "embedding": (0, 0, 2 * tokens * 4096),
        "attention_block": (
            2 * tokens * (4096 * 1536 + 1024 * 4096) + attention,
            attention,
            2 * (4096 * 1536 + 1024 * 4096) + NORM_BYTES_PER_RANK + cache_bytes // 32,
        ),
        "mlp_block": (
            2 * tokens * 3 * 4096 * 3584, 0, 2 * 3 * 4096 * 3584 + NORM_BYTES_PER_RANK
        ),
        "head": (
            head["contraction_flops_per_rank"],
            0,
            head["persistent_operand_bytes_per_rank"] + NORM_BYTES_PER_RANK,
        ),
    }
    if segment not in rows:
        raise ValueError("segment must be embedding, attention_block, mlp_block or head")
    flops, attention_flops, persistent = rows[segment]
    return {
        "contraction_flops_per_rank": flops,
        "attention_flops_per_rank": attention_flops,
        "persistent_operand_bytes_per_rank": persistent,
    }
