//! Shared infrastructure construction for whole-iteration worker recipes.
//!
//! Concrete `build_*_worker` files choose the semantic composition — KV,
//! admission lifecycle, selection policy, execution adapter, shell. This helper
//! only owns the repeated allocation → KV-capacity registration → sampler →
//! cost-buffer choreography, and it takes plain model *facts* rather than a
//! model handle so it stays free of any L4 trait bound. That is what lets an
//! `IterwiseUnifiedModel` recipe and a `SpeculativeUnifiedModel` recipe share
//! it: neither trait is named here, and the caller — not this file — decides
//! which execution adapter wraps the returned `CostBuffers`.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::kv_layout::KvCacheLayout;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::log::KvSampler;
use crate::log::PrefixCacheLogger;
use crate::timing::{CostManifest, CostManifestDoc};
use crate::worker::config::PrefillChunkAlignment;
use crate::worker::cost_buffers::CostBuffers;
use crate::worker::gpu_cluster::{KvPoolFacts, SharedGpuCluster};
use crate::worker::kv::{
    KvAccounting, KvAccountingOptions, KvCacheStore, SharedKvAccounting, SharedTiersReports,
    TierBytes, TierStateOptions,
};
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::WorkerConfig;

pub(super) struct IterBuildEssentials {
    pub(super) context: WorkerContext,
    pub(super) allocation_base: u16,
    pub(super) sampler: Option<KvSampler>,
    /// Reusable cost/slot buffers, already wired to this worker's cost log. The
    /// recipe hands them to whichever execution adapter it selected.
    pub(super) cost: CostBuffers,
    /// Where the KV store leaves its tiers' end-of-run report.
    pub(super) tiers_reports: SharedTiersReports,
}

/// One KV partition's accounting: the arch's layout against the worker's
/// per-GPU budget over the partition's `num_attn_shards` GPUs, before a family
/// sets any slice aside for its own cadence, such as PD decode's pull backlog.
/// Every worker family sizes its KV here.
pub(crate) fn resolve_kv_accounting(
    layout: &KvCacheLayout,
    num_attn_shards: u16,
    config: &WorkerConfig,
) -> SharedKvAccounting {
    let budget_bytes = config
        .attn_kv_bytes
        .saturating_mul(u64::from(num_attn_shards.max(1)));
    let options = KvAccountingOptions {
        overhead: config.kv_overhead,
        part_padding: config.kv_part_padding.clone(),
        block_tokens: config.kv_block_tokens,
        resume_points: config.prefix_resume_points,
        resume_interval: config.prefix_resume_interval,
        state_retention: config.state_retention,
        max_in_flight_tokens: config.kv_max_in_flight_tokens.or_else(|| {
            config
                .max_batch_tokens
                .map(|tokens| tokens.saturating_mul(2))
        }),
        prefix_caching: config.prefix_cache.enabled(),
    };
    layout
        .check_placement(num_attn_shards)
        .unwrap_or_else(|error| panic!("the arch's KV cache layout is misplaced: {error:#}"));
    let accounting = KvAccounting::resolve(layout, options, budget_bytes)
        .unwrap_or_else(|error| panic!("cannot account the arch's KV cache layout: {error:#}"));
    Arc::new(accounting.over_attention_shards(num_attn_shards))
}

/// The KV store of `num_partitions` partitions under `accounting`, with the
/// worker's prefix cache resolved into its units.
pub(crate) fn kv_cache_store(
    accounting: SharedKvAccounting,
    num_partitions: usize,
    num_attn_shards: u16,
    config: &WorkerConfig,
    sampler: Option<KvSampler>,
    prefix_cache_logger: Option<PrefixCacheLogger>,
) -> KvCacheStore {
    let prefix_cache = config
        .prefix_cache
        .resolve_units(&accounting, num_attn_shards);
    KvCacheStore::new(
        num_partitions,
        accounting,
        prefix_cache,
        sampler,
        prefix_cache_logger,
    )
    .with_external_decode(config.external_decode)
}

/// `kv_store` with the config's host tiers behind each partition's prefix
/// cache, each GPU storing and reading `tier_bytes`, reads staged through
/// loading areas when the config has one, and the tiers' end-of-run report
/// left in `tiers_reports`. Without tiers the store is unchanged but for the
/// report. A loading area stages reads of recurrent state, so a layout whose
/// tiers store none (full attention) cannot take one.
pub(crate) fn with_host_tiers(
    kv_store: KvCacheStore,
    tier_bytes: TierBytes,
    config: &WorkerConfig,
    tiers_reports: SharedTiersReports,
    pool_tag: &str,
    id: WorkerId,
) -> KvCacheStore {
    let area = config.prefix_tier_loading_area;
    assert!(
        area == 0.0 || tier_bytes.state > 0,
        "{pool_tag} worker {}: prefix_tier_loading_area ({area}) needs an arch with recurrent \
         state in its host tiers; this one stores none per GPU (full attention)",
        id.0
    );
    let kv_store = kv_store.with_prefix_tiers(
        &config.prefix_tiers,
        tier_bytes,
        TierStateOptions {
            compression_ratio: config.tier_state_compression_ratio,
            anchors_in_place: config.recurrent_state_anchors_in_place,
        },
    );
    let kv_store = if area > 0.0 {
        kv_store.with_loading_area(area)
    } else {
        kv_store
    };
    if config.prefix_tier_warm_start {
        kv_store.with_tier_warm_start()
    } else {
        kv_store
    }
    .with_tiers_reports(tiers_reports)
}

/// The token boundary non-final prefill chunks end on, if any: the layout's
/// checkpoint block when it keeps a resumable recurrent state, prefix caching
/// is on, and the worker aligns chunks (vLLM's Mamba `align` mode).
pub(crate) fn prefill_chunk_end_quantum(
    accounting: &KvAccounting,
    config: &WorkerConfig,
) -> Option<u32> {
    let aligns = match config.prefill_chunk_alignment {
        Some(alignment) => alignment == PrefillChunkAlignment::Checkpoint,
        None => accounting.aligns_chunks_by_default(),
    };
    accounting
        .checkpoint_block_tokens()
        .filter(|_| aligns && config.prefix_cache.enabled())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_iter_build_essentials(
    id: WorkerId,
    pool_tag: &'static str,
    requests: SharedRequests,
    config: &WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: &SharedGpuCluster,
    gpus_per_replica: u16,
    cost_manifest: CostManifest,
    kv_accounting: &KvAccounting,
    num_partitions: usize,
) -> IterBuildEssentials {
    let num_partitions = num_partitions.max(1);

    let allocation_base =
        cluster
            .borrow_mut()
            .allocate(pool.0, id.0, gpus_per_replica, gpu_name, pool_tag);

    let facts = KvPoolFacts::of(kv_accounting, config.max_model_len);
    let tiers_reports = facts.tiers.clone();
    {
        let mut cluster = cluster.borrow_mut();
        for partition in 0..num_partitions as u16 {
            cluster.register_kv_pool(pool_tag, pool.0, id.0, partition, facts.clone());
        }
    }

    let sampler = KvSampler::open_opt(
        cost_log_dir.as_deref(),
        pool_tag,
        id,
        num_partitions,
        config.kv_log_stride,
    );
    // Every iter-family worker evaluates one whole iteration, so its manifest is
    // the degenerate single-`iter` document.
    let cost = CostBuffers::new(
        cost_log_dir,
        pool_tag,
        id,
        &CostManifestDoc::single("iter", cost_manifest),
        config.gpu_time_multiplier,
    )
    .with_prefill_gpu_time_multiplier(config.prefill_gpu_time_multiplier);
    let context = WorkerContext {
        id,
        pool,
        requests,
        log_output_token_times: config.log_output_token_times,
        log_stage_transitions: config.log_stage_transitions,
    };

    IterBuildEssentials {
        context,
        allocation_base,
        sampler,
        cost,
        tiers_reports,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::kv::{TierSpec, TierWrites, TieredPrefixKv};

    /// A one-partition store with a DRAM tier and a loading area of `area`,
    /// its tiers storing `state` bytes per context.
    fn staged(area: f64, state: u64) -> KvCacheStore {
        let config = WorkerConfig {
            prefix_tiers: vec![TierSpec {
                name: "dram",
                capacity_gb: 1e-3,
                read_gb_per_s: 1e-3,
                writes: TierWrites::EveryContext,
            }],
            prefix_tier_loading_area: area,
            ..WorkerConfig::default()
        };
        let kv_store = kv_cache_store(
            KvAccounting::tokens_for_test(1_000),
            1,
            1,
            &config,
            None,
            None,
        );
        with_host_tiers(
            kv_store,
            TierBytes {
                per_token: 545,
                state,
            },
            &config,
            SharedTiersReports::default(),
            "stage",
            WorkerId(3),
        )
    }

    #[test]
    fn a_loading_area_needs_an_arch_with_recurrent_state() {
        assert!(staged(0.1, 1_000).stages_tier_reads());
        assert!(!staged(0.0, 0).stages_tier_reads());
        let Err(full_attention) = std::panic::catch_unwind(|| staged(0.1, 0)) else {
            panic!("a full-attention layout cannot take a loading area");
        };
        let message = full_attention
            .downcast_ref::<String>()
            .expect("a formatted panic");
        assert!(
            message.starts_with("stage worker 3: prefix_tier_loading_area (0.1)"),
            "{message}"
        );
    }
}
