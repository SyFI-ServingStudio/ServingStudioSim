//! Sparse-GQA indexer compound op: one step's token selection for a sparse
//! GQA attention layer whose indexer scores compressed keys (one pooled key per
//! `compress_ratio` tokens) and keeps the top `token_topk / compress_ratio`
//! groups per query row.
//!
//! Six fixed leaves, in launch order:
//!
//! ```text
//! prepare           sparse_indexer_prepare   all rows: q norm + RoPE, raw-key ring, pooling
//! logits_decode     sparse_indexer_logits_decode   the uniform decode/verify batch, one launch
//! logits_prefill    sparse_indexer_logits_prefill  aggregating: one cell per prefill chunk
//! topk_cooperative  sparse_indexer_topk      aggregating: calls of at most 64 rows
//! topk_persistent   sparse_indexer_topk      aggregating: every other call
//! expand            sparse_index_expand      all rows: groups to token indices + tail
//! ```
//!
//! The decode rows lead the batch and share one query length; the prefill rows
//! follow (`indexer_qsa.py` `forward`). Each selection call picks its top-k
//! backend by its own row count, as the framework does: the cooperative kernel
//! for at most 64 rows when the platform has it and the logits row stride is a
//! multiple of 4, the persistent kernel otherwise
//! (`model_executor/layers/indexer_topk.py:215-254`, mirrored by
//! `models/qwen4_exp/nvidia/ops/qsa_indexer.py` `_topk`). The backend is never
//! the faster of the two.
//!
//! A row at 0-based position `p` sees `(p + 1) / compress_ratio` complete
//! groups (`common/qsa_cache.py` `visible_blocks`). The decode logits launch
//! scores every request over the padded page-table row, `logits_width` columns:
//! `ceil(max_model_len / kv_block_size) · kv_block_size / compress_ratio`
//! (32800 at 131072 tokens and 3200-token blocks). Its key count is the mean
//! complete-group count of the requests. The decode selection is priced at that
//! mean on the persistent kernel and at the longest request on the cooperative
//! kernel, which runs all rows in one wave and finishes with its longest row. The prefill selection is one call over
//! all prefill rows; its rows split at `block_topk` visible groups: a row with
//! no more candidates than it keeps selects trivially, so the short rows are
//! priced as one cell at their own mean context and the rest as another
//! (decision D5 of the Flash-Next design; geometry only, no new rows). The
//! framework further cuts a prefill call at `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`
//! (at least 4096 rows at the widest row); the persistent kernel is linear in
//! rows there, so that cut is not modeled.

use std::sync::Arc;

use crate::timing::bridge::DType;
use crate::timing::kernels::sparse_indexer_topk::COOPERATIVE_MAX_ROWS;
use crate::timing::kernels::{
    DsaPersistentTopkDecodeKernelConfig, DsaPersistentTopkDecodeKernelInput,
    SparseIndexExpandKernel, SparseIndexExpandKernelConfig, SparseIndexExpandKernelInput,
    SparseIndexerLogitsDecodeKernel, SparseIndexerLogitsDecodeKernelConfig,
    SparseIndexerLogitsDecodeKernelInput, SparseIndexerLogitsPrefillKernel,
    SparseIndexerLogitsPrefillKernelConfig, SparseIndexerLogitsPrefillKernelInput,
    SparseIndexerPrepareKernel, SparseIndexerPrepareKernelConfig, SparseIndexerPrepareKernelInput,
    SparseIndexerTopkKernel,
};
use crate::timing::slot_input::DsaIndexerPrefillLog;
use crate::timing::{
    AttnPrefillLog, BuildError, CostNode, CostTreeBuilder, Evaluator, LeafMetrics, PerfApiBridge,
    Probe,
};

/// The top-k selection scores and indices, as the framework allocates them.
const LOGITS_DTYPE: DType = DType::Fp32;
const INDEX_DTYPE: &str = "int32";
/// Independent normal scores, like a live indexer's (see `sparse_indexer_topk`).
const TOPK_CONTEXT_MODE: &str = "uniform_random";

#[derive(Clone, Debug)]
pub struct SparseGqaIndexerOpConfig {
    pub gpu_name: String,
    /// Indexer query heads; the indexer has one key head.
    pub num_heads: u32,
    pub head_dim: u32,
    /// Tokens pooled into one compressed key.
    pub compress_ratio: u32,
    /// Tokens selected per row (`indexer_budget`); `token_topk / compress_ratio`
    /// groups are kept.
    pub token_topk: u32,
    /// Raw keys each request keeps between steps.
    pub ring_size: u32,
    /// Query rows of every decode request (1 + draft tokens for a verify step).
    pub decode_query_len: u32,
    /// The model's position range; its compressed range spaces the top-k rows.
    pub max_position_embeddings: u32,
    /// The deployment's `max_model_len` and KV page size: they set the padded
    /// decode logits row.
    pub max_model_len: u32,
    pub kv_block_size: u32,
    pub input_dtype: DType,
    /// Compressed-key cache dtype; the queries and logits operands share it.
    pub cache_dtype: DType,
    pub prepare_backends: Vec<&'static str>,
    pub logits_decode_backends: Vec<&'static str>,
    pub logits_prefill_backends: Vec<&'static str>,
    pub topk_backends: Vec<&'static str>,
    /// Empty when the platform has no cooperative top-k (below SM90, or SM12x).
    pub cooperative_topk_backends: Vec<&'static str>,
    pub expand_backends: Vec<&'static str>,
}

/// One step as the indexer sees it.
#[derive(Clone, Debug, Default)]
pub struct SparseGqaIndexerOpInput {
    /// Rows the indexer launches over (graph-padded under a FULL graph).
    pub num_rows: u32,
    /// Decode requests the decode launch covers (graph-padded under a FULL
    /// graph); each owns `decode_query_len` rows.
    pub decode_requests: u32,
    /// Context of each real decode request, its query rows included.
    pub decode_kv_lens: Vec<u32>,
    /// `(cached prefix, new tokens)` of each prefill chunk.
    pub prefill_chunks: Vec<(u32, u32)>,
}

#[derive(Clone, Debug)]
pub struct SparseGqaIndexerOpResolved {
    pub prepare: SparseIndexerPrepareKernelConfig,
    pub logits_decode: SparseIndexerLogitsDecodeKernelConfig,
    pub logits_prefill: SparseIndexerLogitsPrefillKernelConfig,
    pub topk_decode: DsaPersistentTopkDecodeKernelConfig,
    pub topk_prefill: DsaPersistentTopkDecodeKernelConfig,
    pub cooperative_topk_decode: Option<DsaPersistentTopkDecodeKernelConfig>,
    pub cooperative_topk_prefill: Option<DsaPersistentTopkDecodeKernelConfig>,
    pub expand: SparseIndexExpandKernelConfig,
    pub decode_query_len: u32,
    pub compress_ratio: u32,
    pub block_topk: u32,
    pub logits_width: u32,
}

/// Which top-k kernel a selection call runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopkBackend {
    Cooperative,
    Persistent,
}

/// One top-k cell: `batch` requests of the kernel's `next_n` rows at
/// `context` visible groups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopkCell {
    pub batch: u32,
    pub context: u32,
}

/// The leaf inputs of one step; `None` is a launch that does not happen.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SparseGqaIndexerPlan {
    pub rows: u32,
    pub logits_decode: Option<(u32, u32)>,
    pub prefill_chunks: Vec<(u32, u32)>,
    pub decode_topk: Option<(TopkBackend, TopkCell)>,
    pub prefill_topk: Option<(TopkBackend, Vec<TopkCell>)>,
}

pub struct SparseGqaIndexerOp {
    pub name: String,
    prepare: Arc<SparseIndexerPrepareKernel>,
    logits_decode: Arc<SparseIndexerLogitsDecodeKernel>,
    logits_prefill: Arc<SparseIndexerLogitsPrefillKernel>,
    cooperative_topk: Option<(Arc<SparseIndexerTopkKernel>, Arc<SparseIndexerTopkKernel>)>,
    topk_decode: Arc<SparseIndexerTopkKernel>,
    topk_prefill: Arc<SparseIndexerTopkKernel>,
    expand: Arc<SparseIndexExpandKernel>,
    resolved: SparseGqaIndexerOpResolved,
}

impl SparseGqaIndexerOp {
    /// Expand the op config into its kernel configs. Pure.
    pub fn resolve_config(cfg: &SparseGqaIndexerOpConfig) -> SparseGqaIndexerOpResolved {
        assert!(cfg.compress_ratio > 0, "compress_ratio must be positive");
        assert_eq!(
            cfg.token_topk % cfg.compress_ratio,
            0,
            "token_topk {} must be whole groups of {}",
            cfg.token_topk,
            cfg.compress_ratio
        );
        assert_eq!(
            cfg.kv_block_size % cfg.compress_ratio,
            0,
            "KV page {} must hold whole groups of {}",
            cfg.kv_block_size,
            cfg.compress_ratio
        );
        assert!(
            cfg.decode_query_len > 0,
            "decode_query_len must be positive"
        );
        let gpu = &cfg.gpu_name;
        let compressed_range = cfg.max_position_embeddings / cfg.compress_ratio;
        let topk = |backends: &[&'static str], next_n: u32| DsaPersistentTopkDecodeKernelConfig {
            backends: backends.to_vec(),
            gpu_name: gpu.clone(),
            next_n,
            max_model_len: compressed_range.into(),
            top_k: cfg.token_topk / cfg.compress_ratio,
            logits_row_stride: compressed_range.into(),
            logits_dtype: LOGITS_DTYPE,
            index_dtype: INDEX_DTYPE.into(),
            context_mode: TOPK_CONTEXT_MODE.into(),
        };
        let cooperative = (!cfg.cooperative_topk_backends.is_empty())
            .then_some(cfg.cooperative_topk_backends.as_slice());
        SparseGqaIndexerOpResolved {
            prepare: SparseIndexerPrepareKernelConfig {
                backends: cfg.prepare_backends.clone(),
                gpu_name: gpu.clone(),
                num_heads: cfg.num_heads.into(),
                head_dim: cfg.head_dim.into(),
                compress_ratio: cfg.compress_ratio,
                ring_size: cfg.ring_size,
                input_dtype: cfg.input_dtype,
                cache_dtype: cfg.cache_dtype,
            },
            logits_decode: SparseIndexerLogitsDecodeKernelConfig {
                backends: cfg.logits_decode_backends.clone(),
                gpu_name: gpu.clone(),
                next_n: cfg.decode_query_len,
                num_heads: cfg.num_heads.into(),
                head_dim: cfg.head_dim.into(),
                dtype: cfg.cache_dtype,
            },
            logits_prefill: SparseIndexerLogitsPrefillKernelConfig {
                backends: cfg.logits_prefill_backends.clone(),
                gpu_name: gpu.clone(),
                num_heads: cfg.num_heads.into(),
                head_dim: cfg.head_dim.into(),
                compress_ratio: cfg.compress_ratio,
                dtype: cfg.cache_dtype,
            },
            topk_decode: topk(&cfg.topk_backends, cfg.decode_query_len),
            topk_prefill: topk(&cfg.topk_backends, 1),
            cooperative_topk_decode: cooperative.map(|b| topk(b, cfg.decode_query_len)),
            cooperative_topk_prefill: cooperative.map(|b| topk(b, 1)),
            expand: SparseIndexExpandKernelConfig {
                backends: cfg.expand_backends.clone(),
                gpu_name: gpu.clone(),
                token_topk: cfg.token_topk,
                compress_ratio: cfg.compress_ratio,
            },
            decode_query_len: cfg.decode_query_len,
            compress_ratio: cfg.compress_ratio,
            block_topk: cfg.token_topk / cfg.compress_ratio,
            logits_width: decode_logits_width(
                cfg.max_model_len,
                cfg.kv_block_size,
                cfg.compress_ratio,
            ),
        }
    }

    pub fn build(
        name: String,
        resolved: SparseGqaIndexerOpResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let cooperative = match (r.cooperative_topk_decode, r.cooperative_topk_prefill) {
            (Some(decode), Some(prefill)) => Some((
                Arc::new(SparseIndexerTopkKernel::build(
                    format!("{name}.topk_cooperative.decode"),
                    decode,
                    bridge,
                )?),
                Arc::new(SparseIndexerTopkKernel::build(
                    format!("{name}.topk_cooperative.prefill"),
                    prefill,
                    bridge,
                )?),
            )),
            _ => None,
        };
        Ok(Self {
            prepare: Arc::new(SparseIndexerPrepareKernel::build(
                format!("{name}.prepare"),
                r.prepare,
                bridge,
            )?),
            logits_decode: Arc::new(SparseIndexerLogitsDecodeKernel::build(
                format!("{name}.logits_decode"),
                r.logits_decode,
                bridge,
            )?),
            logits_prefill: Arc::new(SparseIndexerLogitsPrefillKernel::build(
                format!("{name}.logits_prefill"),
                r.logits_prefill,
                bridge,
            )?),
            cooperative_topk: cooperative,
            topk_decode: Arc::new(SparseIndexerTopkKernel::build(
                format!("{name}.topk_persistent.decode"),
                r.topk_decode,
                bridge,
            )?),
            topk_prefill: Arc::new(SparseIndexerTopkKernel::build(
                format!("{name}.topk_persistent.prefill"),
                r.topk_prefill,
                bridge,
            )?),
            expand: Arc::new(SparseIndexExpandKernel::build(
                format!("{name}.expand"),
                r.expand,
                bridge,
            )?),
            name,
            resolved,
        })
    }

    /// Fixed leaves; the cooperative leaf exists only where the platform has
    /// the kernel.
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let mut slots = vec![
            leaf(builder, &self.name, "prepare", &*self.prepare),
            leaf(builder, &self.name, "logits_decode", &*self.logits_decode),
            leaf(builder, &self.name, "logits_prefill", &*self.logits_prefill),
        ];
        if let Some((decode, _)) = &self.cooperative_topk {
            slots.push(leaf(builder, &self.name, "topk_cooperative", &**decode));
        }
        slots.push(leaf(
            builder,
            &self.name,
            "topk_persistent",
            &*self.topk_decode,
        ));
        slots.push(leaf(builder, &self.name, "expand", &*self.expand));
        CostNode::Sum(slots)
    }

    pub fn eval(&self, input: &SparseGqaIndexerOpInput, ev: &mut Evaluator) {
        let plan = plan(&self.resolved, input, self.cooperative_topk.is_some());

        let prepare = SparseIndexerPrepareKernelInput {
            num_tokens: plan.rows,
        };
        let metrics = if plan.rows == 0 {
            LeafMetrics::ZERO
        } else {
            self.prepare.eval(&prepare)
        };
        ev.push(metrics, || prepare.clone().into());

        let logits_decode =
            plan.logits_decode.map(
                |(batch_size, num_keys)| SparseIndexerLogitsDecodeKernelInput {
                    batch_size,
                    num_keys,
                    logits_width: self.resolved.logits_width,
                },
            );
        let metrics = logits_decode
            .as_ref()
            .map_or(LeafMetrics::ZERO, |shape| self.logits_decode.eval(shape));
        ev.push(metrics, || {
            logits_decode
                .clone()
                .unwrap_or(SparseIndexerLogitsDecodeKernelInput {
                    batch_size: 0,
                    num_keys: 0,
                    logits_width: self.resolved.logits_width,
                })
                .into()
        });

        let mut metrics = LeafMetrics::ZERO;
        for &(prefix_len, num_queries) in &plan.prefill_chunks {
            metrics.add_fanin(
                self.logits_prefill
                    .eval(&SparseIndexerLogitsPrefillKernelInput {
                        num_queries,
                        prefix_len,
                    }),
            );
        }
        ev.push(metrics, || {
            AttnPrefillLog {
                prefill_chunk_pairs: plan.prefill_chunks.clone(),
            }
            .into()
        });

        if let Some((coop_decode, coop_prefill)) = &self.cooperative_topk {
            self.push_topk(
                &plan,
                TopkBackend::Cooperative,
                coop_decode,
                coop_prefill,
                ev,
            );
        }
        self.push_topk(
            &plan,
            TopkBackend::Persistent,
            &self.topk_decode,
            &self.topk_prefill,
            ev,
        );

        let expand = SparseIndexExpandKernelInput {
            num_rows: plan.rows,
        };
        let metrics = if plan.rows == 0 {
            LeafMetrics::ZERO
        } else {
            self.expand.eval(&expand)
        };
        ev.push(metrics, || expand.clone().into());
    }

    /// One top-k leaf: the decode call and the prefill call that chose
    /// `backend`, each through the kernel of its own `next_n`.
    fn push_topk(
        &self,
        plan: &SparseGqaIndexerPlan,
        backend: TopkBackend,
        decode: &SparseIndexerTopkKernel,
        prefill: &SparseIndexerTopkKernel,
        ev: &mut Evaluator,
    ) {
        let mut metrics = LeafMetrics::ZERO;
        let mut cells = Vec::new();
        if let Some((chosen, cell)) = plan.decode_topk {
            if chosen == backend {
                metrics.add_fanin(decode.eval(&topk_input(cell)));
                cells.push((cell.batch, cell.context));
            }
        }
        if let Some((chosen, prefill_cells)) = &plan.prefill_topk {
            if *chosen == backend {
                for &cell in prefill_cells {
                    metrics.add_fanin(prefill.eval(&topk_input(cell)));
                    cells.push((cell.batch, cell.context));
                }
            }
        }
        ev.push(metrics, || {
            DsaIndexerPrefillLog {
                prefill_query_key_pairs: cells,
            }
            .into()
        });
    }
}

fn topk_input(cell: TopkCell) -> DsaPersistentTopkDecodeKernelInput {
    DsaPersistentTopkDecodeKernelInput {
        batch_size: cell.batch,
        context_len: cell.context,
    }
}

fn leaf<K: Probe>(builder: &mut CostTreeBuilder, name: &str, slot: &str, kernel: &K) -> CostNode {
    builder.leaf(
        format!("{name}.{slot}"),
        kernel.kind(),
        kernel.describe_config(),
    )
}

/// Columns of the padded decode logits row: the page-table width times the
/// compressed keys one page holds.
pub fn decode_logits_width(max_model_len: u32, kv_block_size: u32, compress_ratio: u32) -> u32 {
    max_model_len.div_ceil(kv_block_size) * (kv_block_size / compress_ratio)
}

/// `Σ_{x=1..n} ⌊x / r⌋`: complete groups summed over rows whose context runs
/// 1 to `n`.
fn groups_prefix_sum(n: u64, r: u64) -> u64 {
    let (a, b) = (n / r, n % r);
    r * a * a.saturating_sub(1) / 2 + a * (b + 1)
}

/// Complete groups summed over rows whose context runs `lo..=hi`.
fn groups_sum(lo: u64, hi: u64, r: u64) -> u64 {
    if hi < lo {
        return 0;
    }
    groups_prefix_sum(hi, r) - groups_prefix_sum(lo.saturating_sub(1), r)
}

fn rounded_mean(sum: u64, count: u64) -> u32 {
    ((sum + count / 2) / count) as u32
}

/// The prefill selection call's cells: rows with at most `block_topk` visible
/// groups, then the rest, each at its own mean visible-group count.
pub fn prefill_topk_cells(
    chunks: &[(u32, u32)],
    compress_ratio: u32,
    block_topk: u32,
) -> Vec<TopkCell> {
    let r = u64::from(compress_ratio);
    // A row is short while its context stays below (block_topk + 1) groups.
    let short_max_context = (u64::from(block_topk) + 1) * r - 1;
    let (mut short_rows, mut short_sum, mut long_rows, mut long_sum) = (0u64, 0u64, 0u64, 0u64);
    for &(prefix, new) in chunks {
        if new == 0 {
            continue;
        }
        let (lo, hi) = (u64::from(prefix) + 1, u64::from(prefix) + u64::from(new));
        let short_hi = hi.min(short_max_context);
        if short_hi >= lo {
            short_rows += short_hi - lo + 1;
            short_sum += groups_sum(lo, short_hi, r);
        }
        let long_lo = lo.max(short_max_context + 1);
        if hi >= long_lo {
            long_rows += hi - long_lo + 1;
            long_sum += groups_sum(long_lo, hi, r);
        }
    }
    [(short_rows, short_sum), (long_rows, long_sum)]
        .into_iter()
        .filter(|&(rows, _)| rows > 0)
        .map(|(rows, sum)| TopkCell {
            batch: rows as u32,
            context: rounded_mean(sum, rows),
        })
        .collect()
}

/// The framework's top-k dispatch for one call of `rows` rows.
pub fn topk_backend(rows: u32, row_stride: u32, cooperative_available: bool) -> TopkBackend {
    if cooperative_available && rows <= COOPERATIVE_MAX_ROWS && row_stride.is_multiple_of(4) {
        TopkBackend::Cooperative
    } else {
        TopkBackend::Persistent
    }
}

/// Every leaf input of one step. Pure.
pub fn plan(
    r: &SparseGqaIndexerOpResolved,
    input: &SparseGqaIndexerOpInput,
    cooperative_available: bool,
) -> SparseGqaIndexerPlan {
    let q = r.decode_query_len;
    let real_decodes = input.decode_kv_lens.len() as u32;
    assert!(
        input.decode_requests >= real_decodes,
        "decode launch covers {} requests but {} are real",
        input.decode_requests,
        real_decodes
    );
    let (logits_decode, decode_topk) = if input.decode_requests == 0 {
        (None, None)
    } else {
        let request_groups = input.decode_kv_lens.iter().map(|&kv| kv / r.compress_ratio);
        let groups: u64 = request_groups.clone().map(u64::from).sum();
        let keys = if real_decodes == 0 {
            0
        } else {
            rounded_mean(groups, u64::from(real_decodes))
        };
        let backend = topk_backend(
            input.decode_requests * q,
            r.logits_width,
            cooperative_available,
        );
        // The cooperative kernel runs every row in one wave of CTA clusters
        // (`cooperative_topk.cu`), so the call lasts as long as its longest row;
        // the persistent kernel streams rows, so it follows the mean.
        let selected = match backend {
            TopkBackend::Cooperative => request_groups.max().unwrap_or(0),
            TopkBackend::Persistent => keys,
        };
        // The top-k template needs every row of a request non-negative.
        let context = selected.max(q - 1);
        (
            Some((input.decode_requests, keys)),
            Some((
                backend,
                TopkCell {
                    batch: input.decode_requests,
                    context,
                },
            )),
        )
    };
    let prefill_chunks: Vec<(u32, u32)> = input
        .prefill_chunks
        .iter()
        .copied()
        .filter(|&(_, new)| new > 0)
        .collect();
    let prefill_rows: u32 = prefill_chunks.iter().map(|&(_, new)| new).sum();
    let prefill_topk = (prefill_rows > 0).then(|| {
        // The prefill logits row is rounded up to 64 columns, so its stride
        // always admits the cooperative kernel.
        (
            topk_backend(prefill_rows, 64, cooperative_available),
            prefill_topk_cells(&prefill_chunks, r.compress_ratio, r.block_topk),
        )
    });
    SparseGqaIndexerPlan {
        rows: input.num_rows,
        logits_decode,
        prefill_chunks,
        decode_topk,
        prefill_topk,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::kernels::sparse_indexer_topk::COOPERATIVE_BACKEND;

    fn config() -> SparseGqaIndexerOpConfig {
        SparseGqaIndexerOpConfig {
            gpu_name: "NVIDIA B200".into(),
            num_heads: 4,
            head_dim: 128,
            compress_ratio: 4,
            token_topk: 2048,
            ring_size: 8,
            decode_query_len: 4,
            max_position_embeddings: 262144,
            max_model_len: 131072,
            kv_block_size: 3200,
            input_dtype: DType::Bf16,
            cache_dtype: DType::Fp8E4m3,
            prepare_backends: vec!["vllm_triton"],
            logits_decode_backends: vec!["vllm_triton"],
            logits_prefill_backends: vec!["vllm_triton"],
            topk_backends: vec!["vllm_cuda"],
            cooperative_topk_backends: vec![COOPERATIVE_BACKEND],
            expand_backends: vec!["vllm_triton"],
        }
    }

    #[test]
    fn configs_key_the_profiled_rows() {
        let r = SparseGqaIndexerOp::resolve_config(&config());
        assert_eq!(r.logits_width, 32800);
        assert_eq!(r.block_topk, 512);
        assert_eq!(r.topk_decode.next_n, 4);
        assert_eq!(r.topk_prefill.next_n, 1);
        assert_eq!(r.topk_decode.top_k, 512);
        assert_eq!(r.topk_decode.max_model_len.get(), 65536);
        assert_eq!(r.topk_decode.logits_row_stride.get(), 65536);
        assert_eq!(r.topk_decode.context_mode, "uniform_random");
        let coop = r.cooperative_topk_prefill.as_ref().unwrap();
        assert_eq!(coop.backends, vec![COOPERATIVE_BACKEND]);
        assert_eq!(r.logits_decode.next_n, 4);
    }

    #[test]
    fn group_sums_match_a_direct_count() {
        for r in [1u64, 4, 8] {
            for lo in 1..40u64 {
                for hi in lo..60 {
                    let direct: u64 = (lo..=hi).map(|x| x / r).sum();
                    assert_eq!(groups_sum(lo, hi, r), direct, "r {r} {lo}..={hi}");
                }
            }
        }
    }

    /// Verify steps: the cooperative kernel up to 64 rows (16 requests of 4),
    /// the persistent one past it; the backend is never a min.
    #[test]
    fn decode_topk_switches_backend_at_64_rows() {
        let r = SparseGqaIndexerOp::resolve_config(&config());
        let verify = |requests: u32| SparseGqaIndexerOpInput {
            num_rows: 4 * requests,
            decode_requests: requests,
            decode_kv_lens: vec![8192; requests as usize],
            prefill_chunks: vec![],
        };
        let at16 = plan(&r, &verify(16), true);
        assert_eq!(
            at16.decode_topk,
            Some((
                TopkBackend::Cooperative,
                TopkCell {
                    batch: 16,
                    context: 2048
                }
            ))
        );
        assert_eq!(at16.logits_decode, Some((16, 2048)));
        assert!(at16.prefill_topk.is_none());
        let at17 = plan(&r, &verify(17), true);
        assert_eq!(at17.decode_topk.unwrap().0, TopkBackend::Persistent);
        // No cooperative kernel on the platform: always persistent.
        assert_eq!(
            plan(&r, &verify(1), false).decode_topk.unwrap().0,
            TopkBackend::Persistent
        );
    }

    /// One cooperative wave lasts as long as its longest row; the persistent
    /// kernel and the logits launch follow the mean.
    #[test]
    fn cooperative_decode_topk_is_priced_at_the_longest_request() {
        let r = SparseGqaIndexerOp::resolve_config(&config());
        let mixed = |requests: u32| SparseGqaIndexerOpInput {
            num_rows: 4 * requests,
            decode_requests: requests,
            decode_kv_lens: (0..requests)
                .map(|i| if i == 0 { 32768 } else { 4096 })
                .collect(),
            prefill_chunks: vec![],
        };
        let coop = plan(&r, &mixed(4), true);
        assert_eq!(coop.decode_topk.unwrap().1.context, 8192);
        assert_eq!(coop.logits_decode, Some((4, 2816)));
        let persistent = plan(&r, &mixed(20), true);
        assert_eq!(persistent.decode_topk.unwrap().0, TopkBackend::Persistent);
        assert_eq!(persistent.decode_topk.unwrap().1.context, 1382);
    }

    #[test]
    fn prefill_topk_switches_backend_at_64_rows() {
        let r = SparseGqaIndexerOp::resolve_config(&config());
        let prefill = |chunks: Vec<(u32, u32)>| SparseGqaIndexerOpInput {
            num_rows: chunks.iter().map(|c| c.1).sum(),
            decode_requests: 0,
            decode_kv_lens: vec![],
            prefill_chunks: chunks,
        };
        let small = plan(&r, &prefill(vec![(0, 40), (100, 24)]), true);
        assert_eq!(small.prefill_topk.unwrap().0, TopkBackend::Cooperative);
        // Two chunks of one call: the call's rows decide, not each chunk's.
        let split = plan(&r, &prefill(vec![(0, 40), (100, 25)]), true);
        assert_eq!(split.prefill_topk.unwrap().0, TopkBackend::Persistent);
        assert!(split.logits_decode.is_none() && split.decode_topk.is_none());
    }

    /// D5: rows with at most 512 visible groups form their own cell.
    #[test]
    fn prefill_rows_split_at_block_topk() {
        // 2048@0: every row sees at most 512 groups.
        let cells = prefill_topk_cells(&[(0, 2048)], 4, 512);
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].batch, 2048);
        assert!(cells[0].context <= 512);
        // 8192@0: rows at contexts 1..=2051 are short, 2052..=8192 long.
        let cells = prefill_topk_cells(&[(0, 8192)], 4, 512);
        assert_eq!(
            cells.iter().map(|c| c.batch).collect::<Vec<_>>(),
            [2051, 6141]
        );
        assert!(cells[0].context <= 512 && cells[1].context > 512);
        let direct: u64 = (2052..=8192u64).map(|x| x / 4).sum();
        assert_eq!(cells[1].context, rounded_mean(direct, 6141));
        // A long prefix: one long cell at the mean visible groups.
        let cells = prefill_topk_cells(&[(126976, 2048)], 4, 512);
        assert_eq!(
            cells,
            [TopkCell {
                batch: 2048,
                context: (126976 + 1024) / 4
            }]
        );
    }

    /// A mixed step runs both the decode and the prefill selection.
    #[test]
    fn mixed_step_runs_both_branches() {
        let r = SparseGqaIndexerOp::resolve_config(&config());
        let p = plan(
            &r,
            &SparseGqaIndexerOpInput {
                num_rows: 1536 + 4 * 128,
                decode_requests: 128,
                decode_kv_lens: vec![4096; 128],
                prefill_chunks: vec![(0, 1536)],
            },
            true,
        );
        assert_eq!(p.rows, 2048);
        assert_eq!(p.logits_decode, Some((128, 1024)));
        assert_eq!(p.decode_topk.unwrap().0, TopkBackend::Persistent);
        assert_eq!(p.prefill_chunks, [(0, 1536)]);
        assert_eq!(p.prefill_topk.unwrap().0, TopkBackend::Persistent);
    }
}
