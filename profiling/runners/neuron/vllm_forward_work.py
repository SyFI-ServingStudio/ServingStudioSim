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
