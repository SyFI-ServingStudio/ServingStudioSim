//! Production pipeline-head recipe.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::arch::kv_layout::KvCacheLayout;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::log::PrefixCacheLogger;
use crate::worker::admission::{
    ChunkEndAlignment, PendingOrder, PipelinedChunkedPrefillAdmission, PrefixFetch,
    SessionPrefixTiers,
};
use crate::worker::config::MicrobatchSizing;
use crate::worker::execution::UnifiedIterExecution;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::types::WorkerConfig;
use crate::worker::workers::iter_build_essentials::{
    kv_cache_store, prefill_chunk_end_quantum, prepare_iter_build_essentials, resolve_kv_accounting,
};

use super::pipeline_head_worker::{PipelineHeadWorker, PipelineLayout};
use super::PipelineHead;

/// `stage_model` is stage 0's layers. KV capacity comes from
/// `kv_cache_layout`, not the stage model: every stage holds the same
/// requests, so the stage with the most KV per token bounds the pipeline. With
/// prefix caching on and a resumable recurrent state in the layout (vLLM's
/// Mamba `align` mode), every non-final chunk ends on the layout's block, and
/// a DRAM/SSD tier hit counts in that block.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_pipeline_head_worker<M: IterwiseUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    stage_model: Arc<M>,
    layout: PipelineLayout,
    kv_cache_layout: &KvCacheLayout,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> PipelineHead<M> {
    let max_batch_tokens = config
        .max_batch_tokens
        .expect("pipeline head requires max_batch_tokens");
    assert_eq!(
        stage_model.gpus_per_replica(),
        1,
        "a pipeline stage runs on one GPU"
    );
    let prefix_cache_logger = PrefixCacheLogger::open_opt(cost_log_dir.as_deref(), pool_tag, id);
    let kv_accounting = resolve_kv_accounting(kv_cache_layout, 1, &config);
    let chunk_end_quantum = prefill_chunk_end_quantum(&kv_accounting, &config);
    let prefix_tiers = head_prefix_tiers(
        &config,
        &layout,
        chunk_end_quantum.unwrap_or(1),
        cost_log_dir.as_deref(),
        id,
    );
    tracing::info!(
        worker = id.0,
        pool_tag,
        kv_capacity_units = kv_accounting.capacity_units(),
        kv_bytes_per_unit = kv_accounting.bytes_per_unit(),
        tier_kv_bytes_per_token = layout.tier_kv_bytes_per_token,
        tier_state_bytes_per_request = layout.tier_state_bytes_per_request,
        chunk_end_quantum,
        max_batch_tokens,
        "pipeline head KV, sized by the most constrained stage"
    );
    let essentials = prepare_iter_build_essentials(
        id,
        pool_tag,
        requests,
        &config,
        cost_log_dir,
        pool,
        gpu_name,
        &cluster,
        1,
        stage_model.cost_log_manifest(),
        &kv_accounting,
        1,
    );
    let send_gid =
        cluster
            .borrow_mut()
            .register_comm_group(essentials.allocation_base, 1, pool_tag, id.0);
    let kv_store = kv_cache_store(
        kv_accounting,
        1,
        1,
        &config,
        essentials.sampler,
        prefix_cache_logger,
    );
    let admission = PipelinedChunkedPrefillAdmission::new(
        PendingOrder::new(config.pending_order),
        (),
        max_batch_tokens,
    );
    let mut admission = match chunk_end_quantum {
        Some(quantum) => admission.with_chunk_end_alignment(ChunkEndAlignment::on_blocks(quantum)),
        None => admission,
    };
    if config.balance_decode_microbatches {
        admission = admission.with_balanced_decodes(layout.depth);
    }
    if let MicrobatchSizing::Even { min_tokens } = config.microbatch_sizing {
        admission = admission.with_even_split(layout.depth, min_tokens);
    }
    if let Some(threshold) = config.long_prefill_token_threshold {
        admission = admission.with_long_prefill_threshold(threshold);
    }
    if let Some(load) = config.load_budget {
        admission = admission.with_load_budget(load);
    }
    if config.srpt {
        admission = admission.with_srpt();
    }
    if config.force_schedule_after_ms > 0.0 {
        admission = admission.with_force_after(config.force_schedule_after_ms);
    }
    let has_tiers = prefix_tiers.is_some();
    if let Some(tiers) = prefix_tiers {
        admission = admission.with_prefix_fetch(
            PrefixFetch::new(vec![tiers], config.prefix_tier_warm_start).with_max_read_wait_ms(
                config.prefix_tier_max_read_wait_ms,
                config.prefix_tier_read_wait_policy,
            ),
        );
    }

    let head = PipelineHeadWorker::from_components(
        essentials.context,
        kv_store,
        admission,
        UnifiedIterExecution::new(stage_model, essentials.cost),
        layout,
        send_gid,
    );
    let head = if has_tiers {
        head.with_prefix_tiers()
    } else {
        head
    };
    if config.external_decode {
        head.with_external_decode()
    } else {
        head
    }
}

/// The DRAM/SSD tiers `config` asks for, in the pipeline's per-GPU token units.
/// A context's recurrent state is charged as `layout.tier_state_bytes_per_request`
/// in those units.
fn head_prefix_tiers(
    config: &WorkerConfig,
    layout: &PipelineLayout,
    hit_quantum: u32,
    log_dir: Option<&std::path::Path>,
    id: WorkerId,
) -> Option<SessionPrefixTiers> {
    let specs: Vec<_> = config.prefix_tiers.iter().flatten().copied().collect();
    (!specs.is_empty()).then(|| {
        SessionPrefixTiers::new(
            &specs,
            layout.tier_kv_bytes_per_token,
            layout
                .tier_state_bytes_per_request
                .div_ceil(layout.tier_kv_bytes_per_token.max(1)),
            hit_quantum,
            log_dir,
            &format!("w{}", id.0),
        )
    })
}
