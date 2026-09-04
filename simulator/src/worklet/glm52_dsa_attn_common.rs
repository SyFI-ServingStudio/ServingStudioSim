//! Provider-neutral GLM-5.2 DSA attention input normalization.
//!
//! This module owns request-shape semantics only. Provider-specific worklets
//! continue to own their measured kernel identities and launch graphs.

use crate::op::attention::DsaIndexerDecodeInput;
use crate::timing::Dim;

#[derive(Clone, Debug)]
pub(crate) struct Glm52DsaAttnTpPartition {
    pub attention_heads_per_rank: Dim,
    pub fused_qkv_a_n: Dim,
    pub q_b_n: Dim,
    pub o_proj_k: Dim,
}

pub(crate) fn resolve_glm52_dsa_attn_tp_partition(
    num_attention_heads: &Dim,
    q_lora_rank: &Dim,
    kv_lora_rank: &Dim,
    qk_nope_head_dim: &Dim,
    rope_dim: &Dim,
    v_head_dim: &Dim,
    tp_size: u16,
) -> Result<Glm52DsaAttnTpPartition, String> {
    if tp_size == 0 {
        return Err("tp_size must be positive".to_string());
    }
    let tp = u32::from(tp_size);
    if num_attention_heads.get() % tp != 0 {
        return Err(format!(
            "num_attention_heads {} must be divisible by tp_size {tp}",
            num_attention_heads
        ));
    }
    let attention_heads_per_rank = num_attention_heads.clone() / Dim::param("attn_tp", tp);
    Ok(Glm52DsaAttnTpPartition {
        fused_qkv_a_n: q_lora_rank.clone() + kv_lora_rank.clone() + rope_dim.clone(),
        q_b_n: attention_heads_per_rank.clone() * (qk_nope_head_dim.clone() + rope_dim.clone()),
        o_proj_k: attention_heads_per_rank.clone() * v_head_dim.clone(),
        attention_heads_per_rank,
    })
}

#[derive(Clone, Debug)]
pub struct Glm52DsaAttnLocalDecodeInput {
    pub batch_size: u32,
    pub context_len: u32,
    /// Exact per-request KV lengths when production metadata is available.
    pub context_lens: Option<Vec<u32>>,
    pub requires_padding: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Glm52DsaAttnLocalInput {
    pub num_new_tokens: u32,
    pub prefill_query_cache_pairs: Vec<(u32, u32)>,
    pub decode: Option<Glm52DsaAttnLocalDecodeInput>,
}

#[derive(Debug)]
pub(crate) struct Glm52DsaAttnNormalizedInput {
    pub active_rows: u32,
    pub indexer_decode: Option<DsaIndexerDecodeInput>,
    pub sparse_decode: Option<(u32, u32)>,
    pub sparse_decode_context_lens: Option<Vec<u32>>,
}

pub(crate) fn normalize_glm52_dsa_attn_input(
    input: &Glm52DsaAttnLocalInput,
    decode_next_n: u32,
    max_model_len: u32,
) -> Result<Glm52DsaAttnNormalizedInput, String> {
    // The width only scales the decode row count below. It selects a measured
    // sparse-MLA cache identity, not a code path here, so any positive value is
    // a shape this section can bill.
    if decode_next_n == 0 {
        return Err("decode_next_n must be positive".to_string());
    }

    let mut active_rows = 0_u32;
    for (index, &(num_queries, num_cache_tokens)) in
        input.prefill_query_cache_pairs.iter().enumerate()
    {
        if num_queries == 0 || num_cache_tokens == 0 {
            return Err(format!(
                "prefill pair {index} must have nonzero Q and S, got ({num_queries}, {num_cache_tokens})"
            ));
        }
        if num_queries > num_cache_tokens {
            return Err(format!(
                "prefill pair {index} requires Q<=S, got ({num_queries}, {num_cache_tokens})"
            ));
        }
        active_rows = active_rows
            .checked_add(num_queries)
            .ok_or_else(|| "active-row sum overflows u32".to_string())?;
    }

    let (indexer_decode, sparse_decode, sparse_decode_context_lens) = match &input.decode {
        Some(decode) => {
            if decode.batch_size == 0 || decode.context_len == 0 {
                return Err(format!(
                    "decode requires positive batch_size and context_len, got ({}, {})",
                    decode.batch_size, decode.context_len
                ));
            }
            if decode.context_len > max_model_len {
                return Err(format!(
                    "decode context_len {} exceeds max_model_len {max_model_len}",
                    decode.context_len
                ));
            }
            let mut context_len_sum: u64 = 0;
            if let Some(context_lens) = &decode.context_lens {
                if context_lens.len() != decode.batch_size as usize {
                    return Err(format!(
                        "decode context_lens has {} entries, expected batch_size {}",
                        context_lens.len(),
                        decode.batch_size
                    ));
                }
                for (request, &context) in context_lens.iter().enumerate() {
                    if context == 0 || context > max_model_len {
                        return Err(format!(
                            "decode request {request} context {context} must be in 1..={max_model_len}"
                        ));
                    }
                    context_len_sum += u64::from(context);
                }
            }
            let decode_rows = decode
                .batch_size
                .checked_mul(decode_next_n)
                .ok_or_else(|| "decode row count overflows u32".to_string())?;
            active_rows = active_rows
                .checked_add(decode_rows)
                .ok_or_else(|| "active-row sum overflows u32".to_string())?;
            (
                Some(DsaIndexerDecodeInput {
                    batch_size: decode.batch_size,
                    context_len: indexer_decode_context_len(decode, context_len_sum),
                    requires_padding: decode.requires_padding,
                }),
                Some((decode_rows, decode.context_len)),
                decode.context_lens.clone(),
            )
        }
        None => (None, None, None),
    };

    if input.num_new_tokens != active_rows {
        return Err(format!(
            "num_new_tokens {} must equal active query rows {active_rows}",
            input.num_new_tokens
        ));
    }

    Ok(Glm52DsaAttnNormalizedInput {
        active_rows,
        indexer_decode,
        sparse_decode,
        sparse_decode_context_lens,
    })
}

/// The uniform context the indexer's decode leaves are billed against.
///
/// Both leaves are profiled with `context_mode: uniform` -- one length for the
/// whole batch -- so a mixed batch has to be collapsed onto that axis. The
/// collapse that preserves the work is the one that preserves the KV the batch
/// reads, `ceil(sum / batch)`, not the longest request: a serving batch mixes a
/// request on its first output token with one near the context limit, and
/// charging every request the longest one's context bills the kernel for KV no
/// request holds.
///
/// The sparse-MLA attention next door does not need this -- it is handed the
/// per-request lengths themselves and evaluates each one.
///
/// Without the per-request lengths there is nothing to average, so the caller's
/// own scalar stands. `context_lens` is validated non-empty and positive above,
/// so the sum is positive whenever it is present and the batch is non-empty.
fn indexer_decode_context_len(decode: &Glm52DsaAttnLocalDecodeInput, context_len_sum: u64) -> u32 {
    if decode.context_lens.is_none() || decode.batch_size == 0 {
        return decode.context_len;
    }
    let batch_size = u64::from(decode.batch_size);
    let mean = context_len_sum.div_ceil(batch_size);
    // Bounded by the caller's own scalar, so it inherits the `max_model_len`
    // check that scalar already passed and can never widen the shape.
    u32::try_from(mean)
        .unwrap_or(decode.context_len)
        .min(decode.context_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_MODEL_LEN: u32 = 1_048_576;

    fn decode_input(batch_size: u32, decode_next_n: u32) -> Glm52DsaAttnLocalInput {
        Glm52DsaAttnLocalInput {
            num_new_tokens: batch_size * decode_next_n,
            prefill_query_cache_pairs: Vec::new(),
            decode: Some(Glm52DsaAttnLocalDecodeInput {
                batch_size,
                context_len: 4096,
                context_lens: None,
                requires_padding: false,
            }),
        }
    }

    #[test]
    fn decode_rows_scale_with_the_verify_width() {
        // A verify step submits one group of `decode_next_n` rows per request.
        // The width is a multiplier on the sparse-MLA query coordinate and
        // nothing else: the indexer still sees one entry per request, because
        // index selection is per request, not per verified row.
        for decode_next_n in [1, 2, 3, 6] {
            let normalized = normalize_glm52_dsa_attn_input(
                &decode_input(8, decode_next_n),
                decode_next_n,
                MAX_MODEL_LEN,
            )
            .expect("any positive width is a billable shape");

            assert_eq!(normalized.active_rows, 8 * decode_next_n);
            assert_eq!(normalized.sparse_decode, Some((8 * decode_next_n, 4096)));
            assert_eq!(
                normalized.indexer_decode.as_ref().map(|d| d.batch_size),
                Some(8)
            );
        }
    }

    #[test]
    fn indexer_decode_is_billed_the_kv_the_batch_holds_not_the_longest_request() {
        // The indexer's decode leaves are profiled at one context for the whole
        // batch, so a mixed batch has to collapse onto that axis. Charging every
        // request the longest one's context bills KV no request holds: here the
        // four requests hold 404 tokens between them, not 4 x 190.
        let input = Glm52DsaAttnLocalInput {
            num_new_tokens: 4,
            prefill_query_cache_pairs: Vec::new(),
            decode: Some(Glm52DsaAttnLocalDecodeInput {
                batch_size: 4,
                context_len: 190,
                context_lens: Some(vec![12, 190, 12, 190]),
                requires_padding: false,
            }),
        };
        let normalized = normalize_glm52_dsa_attn_input(&input, 1, MAX_MODEL_LEN).unwrap();

        let decode = normalized.indexer_decode.unwrap();
        assert_eq!(decode.batch_size, 4);
        // ceil(404 / 4) == 101.
        assert_eq!(decode.context_len, 101);
        // The sparse-MLA path is unaffected: it keeps the caller's scalar as
        // its cache coordinate and receives the exact lengths besides.
        assert_eq!(normalized.sparse_decode, Some((4, 190)));
        assert_eq!(
            normalized.sparse_decode_context_lens,
            Some(vec![12, 190, 12, 190])
        );
    }

    #[test]
    fn indexer_decode_keeps_the_caller_scalar_without_per_request_lengths() {
        let decode = normalize_glm52_dsa_attn_input(&decode_input(4, 1), 1, MAX_MODEL_LEN)
            .unwrap()
            .indexer_decode
            .unwrap();
        assert_eq!(decode.context_len, 4096);
    }

    #[test]
    fn zero_width_is_the_only_rejected_width() {
        assert_eq!(
            normalize_glm52_dsa_attn_input(&decode_input(8, 1), 0, MAX_MODEL_LEN).unwrap_err(),
            "decode_next_n must be positive"
        );
    }
}
