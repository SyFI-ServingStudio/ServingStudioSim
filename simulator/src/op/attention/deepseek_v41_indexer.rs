//! DeepSeek-V4.1 sparse-indexer scoring: logits, top-k and candidate masks.
//!
//! This is the eager `indexer_op` of an index-source layer (fork
//! `model_executor/layers/sparse_attn_indexer.py:726` logits, `:770` top-k),
//! everything after the indexer Q/K preparation. The preparation GEMMs and
//! stores run on the attention worklet's stream fan-out, so they stay there.
//!
//! Leaves, in launch order:
//!
//! 1. `prefill_k_gather` (placeholder): `cp_gather_indexer_k_quant_cache`
//!    gathers each prefill request's compressed index keys (128 FP8 bytes plus
//!    one fp32 scale per key) into a contiguous buffer.
//! 2. `prefill_logits`: DeepGEMM `sm100_mqa_logits`, reusing
//!    `dsa_mqa_logits_prefill` (B200 rows exist at 32 heads x 128, FP8,
//!    `single_causal_tail`). One `num_sequences = 1` call per request, summed;
//!    a request whose logits pass vLLM's 512 MiB budget runs as query
//!    sub-chunks, one call each (see [`INDEXER_MAX_LOGITS_ELEMS`]).
//! 3. `prefill_topk` (placeholder): `topKPerRowPrefill` (top-512). No B200
//!    `dsa_topk_prefill` rows exist at `top_k = 512` (only 2048).
//! 4. `decode_logits`: DeepGEMM `sm100_paged_mqa_logits`, reusing
//!    `dsa_paged_mqa_logits_decode` at `page_block_size = 128 / ratio`.
//! 5. `candidates` (placeholder, layer 20 and 24/28/32/36 only): layer 20 runs
//!    the block-score -> `torch.topk(2048)` -> `_store_candidates` chain; the
//!    non-owning index layers run `_candidate_flags` -> `_mask_candidates`
//!    (fork `model_executor/kernels/attention/dsa/candidate_blocks.py:164,180,
//!    181,208,221`).
//! 6. `decode_topk` (placeholder): `cooperative_topk_cs*<512>`. No B200
//!    `dsa_persistent_topk_decode` rows exist at `top_k = 512` (only 2048).
//!
//! Top-k and candidate placeholders are sized by the logits they stream: one
//! elementwise "token" is a 4 KiB tile (1024 fp32 logits). A row with `c`
//! compressed keys reads `4c` bytes and writes its 512 int32 indices, so the
//! placeholder token count is `ceil(sum(4c + 2048) / 4096)`. The elementwise
//! kind cannot see a context length, so the tile is the unit that carries it.
//!
//! A long prefill context sizes the prefill placeholders past the elementwise
//! sweep (65536 tokens: a 2048-token chunk at a 128K-key context already
//! streams 134K tiles). Past it they hold the edge's bandwidth, see
//! [`held_at_grid_edge`].

use std::sync::Arc;

use crate::timing::bridge::DType;
use crate::timing::kernels::{
    DsaMqaLogitsPrefillKernel, DsaMqaLogitsPrefillKernelConfig, DsaMqaLogitsPrefillKernelInput,
    DsaPagedMqaLogitsDecodeKernel, DsaPagedMqaLogitsDecodeKernelConfig,
    DsaPagedMqaLogitsDecodeKernelInput, ElementwiseKernel, ElementwiseKernelConfig,
    ElementwiseKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, CoverageFlags, Dim, Evaluator, LeafMetrics,
    PerfApiBridge, Probe, SlotInput,
};

/// Bytes of one placeholder tile: 1024 fp32 logits.
pub const LOGIT_TILE_BYTES: u32 = 4096;
/// Write side of a read-only tile (top-k, candidate writer). The per-row index
/// write is already folded into the tile count; the `elementwise` runner
/// rejects a zero-byte output, so these tiles write one int32.
pub const READ_TILE_OUTPUT_BYTES: u32 = 4;
/// Largest input:output byte ratio a V4.1 placeholder keeps as written.
pub const MAX_PLACEHOLDER_FAN_IN: u32 = 8;

/// The `(input, output)` bytes per token a V4.1 elementwise placeholder is
/// profiled at.
///
/// A placeholder stands for a launch by the bytes it streams. The `triton`
/// elementwise runner is a fan-in reduce: one lane per output byte loops over
/// `input / output` inputs, so a launch with a tiny output (a 4 KiB logit tile
/// feeding a top-k, the hc-prenorm's 96-byte mix) runs on a handful of CTAs
/// and measures loop latency, not bandwidth (72 us for 48 hc-prenorm rows,
/// ~180 us per decode top-k in the first B200 prediction). Above
/// [`MAX_PLACEHOLDER_FAN_IN`] the same total bytes are split evenly between
/// read and write, which keeps the byte count and restores full-grid
/// streaming.
pub fn byte_rate_placeholder_shape(input: u32, output: u32) -> (u32, u32) {
    if output > 0 && input > MAX_PLACEHOLDER_FAN_IN * output {
        let half = (input + output).div_ceil(2);
        (half, half)
    } else {
        (input, output)
    }
}
/// One index-cache key: 128 FP8 bytes plus one fp32 scale.
const INDEX_KEY_BYTES: u32 = 132;
/// fp32 logits one prefill logits call may hold: vLLM's
/// `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` default 512 (fork `envs.py:61`).
/// `_split_indexer_prefill_chunks` (fork `v1/attention/backends/mla/
/// indexer.py:1208`) sub-chunks a request whose `queries x keys` exceeds it on
/// the query axis, `elems / keys` queries per call; the keys are gathered
/// once (`skip_kv_gather` for every later sub-chunk). A 2048-token chunk
/// splits past 65536 compressed keys.
const INDEXER_MAX_LOGITS_ELEMS: u32 = 512 * 1024 * 1024 / 4;

/// Where a layer sits in the layer-20 candidate-block scheme.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeepseekV41CandidateRole {
    None,
    /// Layer 20: scores blocks and stores the top candidate blocks.
    Writer,
    /// Layers 24/28/32/36: mask their logits with layer 20's candidates.
    Consumer,
}

#[derive(Clone, Debug)]
pub struct DeepseekV41IndexerOpConfig {
    pub gpu_name: String,
    /// The index source's compress ratio (1 or 2).
    pub compress_ratio: u32,
    /// Index heads per rank (the indexer is replicated: 32).
    pub num_heads: Dim,
    pub head_dim: Dim,
    pub index_topk: u32,
    /// Compressed keys per index-cache page: `kv_block_size / ratio`.
    pub page_block_size: u32,
    /// The model's `max_model_len`; the indexer sees `max_model_len / ratio`.
    pub max_model_len: u32,
    pub candidate: DeepseekV41CandidateRole,
    pub logits_prefill_backends: Vec<&'static str>,
    pub logits_decode_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
}

#[derive(Clone, Debug, Default)]
pub struct DeepseekV41IndexerOpInput {
    /// `(query_len, context_len)` per prefill request.
    pub prefill_query_context_pairs: Vec<(u32, u32)>,
    /// Resident KV length before each one-token decode row.
    pub decode_kv_lens: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct DeepseekV41IndexerOpResolved {
    pub prefill_k_gather: ElementwiseKernelConfig,
    pub prefill_logits: DsaMqaLogitsPrefillKernelConfig,
    pub prefill_topk: ElementwiseKernelConfig,
    pub decode_logits: DsaPagedMqaLogitsDecodeKernelConfig,
    pub candidates: Option<ElementwiseKernelConfig>,
    pub decode_topk: ElementwiseKernelConfig,
    pub compress_ratio: u32,
}

pub struct DeepseekV41IndexerOp {
    pub name: String,
    pub prefill_k_gather: Arc<ElementwiseKernel>,
    pub prefill_logits: Arc<DsaMqaLogitsPrefillKernel>,
    pub prefill_topk: Arc<ElementwiseKernel>,
    pub decode_logits: Arc<DsaPagedMqaLogitsDecodeKernel>,
    pub candidates: Option<Arc<ElementwiseKernel>>,
    pub decode_topk: Arc<ElementwiseKernel>,
    compress_ratio: u32,
}

impl DeepseekV41IndexerOp {
    /// Pure config expansion into the six sub-kernel configs.
    pub fn resolve(cfg: &DeepseekV41IndexerOpConfig) -> DeepseekV41IndexerOpResolved {
        assert!(
            matches!(cfg.compress_ratio, 1 | 2),
            "V4.1 index sources are ratio 1 or 2, got {}",
            cfg.compress_ratio
        );
        assert!(
            matches!(cfg.page_block_size, 64 | 128),
            "index page block must be 64 or 128, got {}",
            cfg.page_block_size
        );
        let elementwise = |input: u32, output: u32| {
            let (input, output) = byte_rate_placeholder_shape(input, output);
            ElementwiseKernelConfig {
                backends: cfg.elementwise_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: input.into(),
                output_bytes_per_token: output.into(),
            }
        };
        DeepseekV41IndexerOpResolved {
            prefill_k_gather: elementwise(INDEX_KEY_BYTES, INDEX_KEY_BYTES),
            prefill_logits: DsaMqaLogitsPrefillKernelConfig {
                backends: cfg.logits_prefill_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_sequences: 1,
                num_heads: cfg.num_heads.clone(),
                head_dim: cfg.head_dim.clone(),
                q_dtype: DType::Fp8E4m3,
                k_dtype: DType::Fp8E4m3,
                k_scale_dtype: DType::Fp32,
                weight_dtype: DType::Fp32,
                output_dtype: DType::Fp32,
                span_mode: "single_causal_tail".to_string(),
                clean_logits: false,
            },
            prefill_topk: elementwise(LOGIT_TILE_BYTES, READ_TILE_OUTPUT_BYTES),
            decode_logits: DsaPagedMqaLogitsDecodeKernelConfig {
                backends: cfg.logits_decode_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                next_n: 1,
                max_model_len: (cfg.max_model_len / cfg.compress_ratio).into(),
                num_heads: cfg.num_heads.clone(),
                head_dim: cfg.head_dim.clone(),
                block_size: cfg.page_block_size,
                q_dtype: DType::Fp8E4m3,
                cache_dtype: DType::Fp8E4m3,
                scale_dtype: DType::Fp32,
                weight_dtype: DType::Fp32,
                output_dtype: DType::Fp32,
                context_mode: "uniform".to_string(),
                page_mapping: "unique_scattered".to_string(),
                cache_format: "page_planar_fp8_fp32_scale".to_string(),
                clean_logits: false,
            },
            candidates: match cfg.candidate {
                DeepseekV41CandidateRole::None => None,
                // Block scores read the logits once; the candidate top-k and
                // store are small next to that read.
                DeepseekV41CandidateRole::Writer => {
                    Some(elementwise(LOGIT_TILE_BYTES, READ_TILE_OUTPUT_BYTES))
                }
                // The mask rewrites non-candidate logits in place.
                DeepseekV41CandidateRole::Consumer => {
                    Some(elementwise(LOGIT_TILE_BYTES, LOGIT_TILE_BYTES))
                }
            },
            decode_topk: elementwise(LOGIT_TILE_BYTES, READ_TILE_OUTPUT_BYTES),
            compress_ratio: cfg.compress_ratio,
        }
    }

    pub fn build(
        name: String,
        resolved: DeepseekV41IndexerOpResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let slot = |suffix: &str| format!("{name}.{suffix}");
        Ok(Self {
            prefill_k_gather: Arc::new(ElementwiseKernel::build(
                slot("prefill_k_gather"),
                resolved.prefill_k_gather,
                bridge,
            )?),
            prefill_logits: Arc::new(DsaMqaLogitsPrefillKernel::build(
                slot("prefill_logits"),
                resolved.prefill_logits,
                bridge,
            )?),
            prefill_topk: Arc::new(ElementwiseKernel::build(
                slot("prefill_topk"),
                resolved.prefill_topk,
                bridge,
            )?),
            decode_logits: Arc::new(DsaPagedMqaLogitsDecodeKernel::build(
                slot("decode_logits"),
                resolved.decode_logits,
                bridge,
            )?),
            candidates: resolved
                .candidates
                .map(|config| ElementwiseKernel::build(slot("candidates"), config, bridge))
                .transpose()?
                .map(Arc::new),
            decode_topk: Arc::new(ElementwiseKernel::build(
                slot("decode_topk"),
                resolved.decode_topk,
                bridge,
            )?),
            compress_ratio: resolved.compress_ratio,
            name,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let mut leaves = vec![
            leaf(builder, &self.name, "prefill_k_gather", &*self.prefill_k_gather),
            leaf(builder, &self.name, "prefill_logits", &*self.prefill_logits),
            leaf(builder, &self.name, "prefill_topk", &*self.prefill_topk),
            leaf(builder, &self.name, "decode_logits", &*self.decode_logits),
        ];
        if let Some(candidates) = &self.candidates {
            leaves.push(leaf(builder, &self.name, "candidates", &**candidates));
        }
        leaves.push(leaf(builder, &self.name, "decode_topk", &*self.decode_topk));
        CostNode::Sum(leaves)
    }

    pub fn eval(&self, input: &DeepseekV41IndexerOpInput, ev: &mut Evaluator) {
        let work = derive_work(input, self.compress_ratio);

        push(&self.prefill_k_gather, work.prefill_k_gather, ev);

        let mut prefill_logits = LeafMetrics::ZERO;
        for shape in &work.prefill_logits {
            prefill_logits.add_fanin(self.prefill_logits.eval(shape));
        }
        let logged = work.prefill_logits.first().cloned().unwrap_or(
            DsaMqaLogitsPrefillKernelInput {
                num_queries: 0,
                num_keys: 0,
            },
        );
        ev.push(prefill_logits, || logged.into());

        push(&self.prefill_topk, work.prefill_topk, ev);

        let decode_logits = match &work.decode_logits {
            Some(shape) => self.decode_logits.eval(shape),
            None => LeafMetrics::ZERO,
        };
        let logged = work
            .decode_logits
            .clone()
            .unwrap_or(DsaPagedMqaLogitsDecodeKernelInput {
                batch_size: 0,
                context_len: 0,
            });
        ev.push(decode_logits, || logged.into());

        if let Some(candidates) = &self.candidates {
            push(candidates, work.candidates, ev);
        }
        push(&self.decode_topk, work.decode_topk, ev);
    }
}

fn leaf<K: Probe>(builder: &mut CostTreeBuilder, name: &str, suffix: &str, kernel: &K) -> CostNode {
    builder.leaf(
        format!("{name}.{suffix}"),
        kernel.kind(),
        kernel.describe_config(),
    )
}

fn push(kernel: &ElementwiseKernel, input: ElementwiseKernelInput, ev: &mut Evaluator) {
    let metrics = if input.num_tokens == 0 {
        LeafMetrics::ZERO
    } else {
        held_at_grid_edge(input.num_tokens, |num_tokens| {
            kernel.eval(&ElementwiseKernelInput { num_tokens })
        })
    };
    ev.push(metrics, || SlotInput::from(input));
}

/// The last token count of the `elementwise` sweep (`Axis::token_axis`).
const ELEMENTWISE_GRID_EDGE: u32 = 65_536;

/// A placeholder of `tokens` tiles or keys. Past the elementwise grid the
/// kind's cache extends its last segment, and at these few-microsecond sizes
/// that slope is launch-bound noise: the 132 B/key gather rows rise only
/// 2.50 -> 2.85 us from 32768 to 65536 keys (a 25 TB/s marginal rate), so a
/// 526336-key gather came out at 7.7 us, 18 TB/s, past B200's HBM. These
/// placeholders only stream bytes, so they hold the edge's bandwidth instead
/// (6.1 TB/s for that gather), as `compressed_sparse_mla_*` do past their
/// grids: the edge's metrics scaled by `tokens / edge`, flagged extrapolated.
fn held_at_grid_edge(tokens: u32, eval: impl Fn(u32) -> LeafMetrics) -> LeafMetrics {
    if tokens <= ELEMENTWISE_GRID_EDGE {
        return eval(tokens);
    }
    let mut metrics = eval(ELEMENTWISE_GRID_EDGE);
    metrics.scale(tokens as f32 / ELEMENTWISE_GRID_EDGE as f32);
    metrics.coverage |= CoverageFlags::EXTRAPOLATED;
    metrics
}

/// Per-call shapes of the six leaves.
pub(crate) struct IndexerWork {
    pub prefill_k_gather: ElementwiseKernelInput,
    pub prefill_logits: Vec<DsaMqaLogitsPrefillKernelInput>,
    pub prefill_topk: ElementwiseKernelInput,
    pub decode_logits: Option<DsaPagedMqaLogitsDecodeKernelInput>,
    pub candidates: ElementwiseKernelInput,
    pub decode_topk: ElementwiseKernelInput,
}

fn tiles(bytes: u64) -> u32 {
    u32::try_from(bytes.div_ceil(u64::from(LOGIT_TILE_BYTES)))
        .expect("indexer placeholder tile count must fit u32")
}

/// Pure input collapse.
///
/// - Prefill request `(q, ctx)` scores `q` queries against `ctx / ratio`
///   compressed keys (at least one).
/// - Decode rows collapse to one uniform `(batch, mean compressed context)`
///   cell: the paged kernel's scheduler splits the summed context evenly over
///   SMs, so total work follows the sum, which the mean preserves.
pub(crate) fn derive_work(input: &DeepseekV41IndexerOpInput, ratio: u32) -> IndexerWork {
    let topk_output = u64::from(512_u32 * 4);
    let mut gathered_keys = 0_u32;
    let mut prefill_logits = Vec::with_capacity(input.prefill_query_context_pairs.len());
    let mut prefill_tile_bytes = 0_u64;
    for &(queries, context) in &input.prefill_query_context_pairs {
        let keys = (context / ratio).max(1);
        gathered_keys = gathered_keys
            .checked_add(keys)
            .expect("gathered index keys must fit u32");
        let per_call = (INDEXER_MAX_LOGITS_ELEMS / keys).max(1);
        prefill_logits.extend((0..queries).step_by(per_call as usize).map(|first| {
            DsaMqaLogitsPrefillKernelInput {
                num_queries: per_call.min(queries - first),
                num_keys: keys,
            }
        }));
        prefill_tile_bytes += u64::from(queries) * (4 * u64::from(keys) + topk_output);
    }

    let rows = input.decode_kv_lens.len() as u64;
    let decode_keys: u64 = input
        .decode_kv_lens
        .iter()
        .map(|&kv| (u64::from(kv) + 1) / u64::from(ratio))
        .map(|keys| keys.max(1))
        .sum();
    let decode_logits = (rows > 0).then(|| DsaPagedMqaLogitsDecodeKernelInput {
        batch_size: rows as u32,
        context_len: u32::try_from(decode_keys.div_ceil(rows))
            .expect("mean decode context must fit u32"),
    });
    let decode_tile_bytes = 4 * decode_keys + rows * topk_output;

    IndexerWork {
        prefill_k_gather: ElementwiseKernelInput {
            num_tokens: gathered_keys,
        },
        prefill_logits,
        prefill_topk: ElementwiseKernelInput {
            num_tokens: tiles(prefill_tile_bytes),
        },
        decode_logits,
        candidates: ElementwiseKernelInput {
            num_tokens: tiles(prefill_tile_bytes + decode_tile_bytes),
        },
        decode_topk: ElementwiseKernelInput {
            num_tokens: tiles(decode_tile_bytes),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(ratio: u32, candidate: DeepseekV41CandidateRole) -> DeepseekV41IndexerOpConfig {
        DeepseekV41IndexerOpConfig {
            gpu_name: "NVIDIA B200".into(),
            compress_ratio: ratio,
            num_heads: 32.into(),
            head_dim: 128.into(),
            index_topk: 512,
            page_block_size: 128 / ratio,
            max_model_len: 131_072,
            candidate,
            logits_prefill_backends: vec!["deepgemm_fp8"],
            logits_decode_backends: vec!["deepgemm_fp8"],
            elementwise_backends: vec!["triton"],
        }
    }

    #[test]
    fn decode_rows_collapse_to_mean_compressed_context_and_prefill_stays_per_request() {
        let work = derive_work(
            &DeepseekV41IndexerOpInput {
                prefill_query_context_pairs: vec![(128, 128), (3, 1000)],
                decode_kv_lens: vec![1303, 4243],
            },
            2,
        );
        let decode = work.decode_logits.unwrap();
        assert_eq!((decode.batch_size, decode.context_len), (2, 1387));
        let shapes: Vec<_> = work
            .prefill_logits
            .iter()
            .map(|s| (s.num_queries, s.num_keys))
            .collect();
        assert_eq!(shapes, vec![(128, 64), (3, 500)]);
        assert_eq!(work.prefill_k_gather.num_tokens, 564);
        // decode: (4 * 2774 + 2 * 2048) / 4096 -> 4 tiles.
        assert_eq!(work.decode_topk.num_tokens, 4);
    }

    #[test]
    fn prefill_logits_past_the_budget_split_on_the_query_axis() {
        let shapes = |queries, context, ratio| {
            derive_work(
                &DeepseekV41IndexerOpInput {
                    prefill_query_context_pairs: vec![(queries, context)],
                    decode_kv_lens: vec![],
                },
                ratio,
            )
            .prefill_logits
            .iter()
            .map(|s| (s.num_queries, s.num_keys))
            .collect::<Vec<_>>()
        };
        // 2048 x 65536 is exactly the budget: one call.
        assert_eq!(shapes(2048, 131_072, 2), [(2048, 65_536)]);
        assert_eq!(shapes(2048, 131_072, 1), [(1024, 131_072); 2]);
        // A 1M-key chunk runs 16 calls of 128 queries; a ragged tail keeps
        // its remainder.
        assert_eq!(shapes(2048, 1_048_576, 1), [(128, 1_048_576); 16]);
        assert_eq!(shapes(200, 1_048_576, 1), [(128, 1_048_576), (72, 1_048_576)]);
    }

    #[test]
    fn ratio_sets_index_domain_and_page_block_and_roles_add_one_leaf() {
        let r2 = DeepseekV41IndexerOp::resolve(&cfg(2, DeepseekV41CandidateRole::None));
        assert_eq!(
            (r2.decode_logits.max_model_len.get(), r2.decode_logits.block_size),
            (65_536, 64)
        );
        assert!(r2.candidates.is_none());
        let r1 = DeepseekV41IndexerOp::resolve(&cfg(1, DeepseekV41CandidateRole::Consumer));
        assert_eq!(
            (r1.decode_logits.max_model_len.get(), r1.decode_logits.block_size),
            (131_072, 128)
        );
        assert_eq!(
            r1.candidates.unwrap().output_bytes_per_token.get(),
            LOGIT_TILE_BYTES
        );
    }

    #[test]
    fn placeholders_past_the_elementwise_grid_hold_the_edge_bandwidth() {
        use crate::timing::sweep::Axis;
        use crate::timing::Metrics4;
        assert_eq!(
            Axis::token_axis().last().copied(),
            Some(f64::from(ELEMENTWISE_GRID_EDGE))
        );
        // A stand-in cache: 1 us per 1000 tokens plus a 2 us floor.
        let eval = |tokens: u32| LeafMetrics {
            m: Metrics4 {
                time_ms: 0.002 + tokens as f32 * 1e-6,
                flops: 0.0,
                bytes: tokens as f32,
                energy_j: 0.0,
            },
            ..LeafMetrics::ZERO
        };
        let inside = held_at_grid_edge(4096, eval);
        assert_eq!(inside.m.bytes, 4096.0);
        assert!(inside.coverage.is_empty());
        let past = held_at_grid_edge(4 * ELEMENTWISE_GRID_EDGE, eval);
        assert_eq!(past.m.bytes, 4.0 * 65_536.0);
        assert!((past.m.time_ms - 4.0 * eval(ELEMENTWISE_GRID_EDGE).m.time_ms).abs() < 1e-6);
        assert!(past.coverage.contains(CoverageFlags::EXTRAPOLATED));
    }

    #[test]
    fn high_fan_in_placeholders_stream_the_same_bytes_evenly() {
        assert_eq!(byte_rate_placeholder_shape(4096, 4), (2050, 2050));
        assert_eq!(byte_rate_placeholder_shape(40960, 96), (20528, 20528));
        assert_eq!(byte_rate_placeholder_shape(8256, 4224), (8256, 4224));
        assert_eq!(byte_rate_placeholder_shape(4, 10240), (4, 10240));
        let r = DeepseekV41IndexerOp::resolve(&cfg(2, DeepseekV41CandidateRole::None));
        assert_eq!(r.decode_topk.input_bytes_per_token.get(), 2050);
        assert_eq!(r.decode_topk.output_bytes_per_token.get(), 2050);
    }
}
