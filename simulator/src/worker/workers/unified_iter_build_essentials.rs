//! Shared infrastructure construction for `UnifiedIterExecution` recipes.
//!
//! Concrete `build_*_worker` files still choose the semantic composition. This
//! helper only owns the repeated allocation → KV-capacity registration → sampler
//! → cost-buffer choreography.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::log::KvSampler;
use crate::worker::cost_buffers::CostBuffers;
use crate::worker::execution::UnifiedIterExecution;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::WorkerConfig;

pub(super) struct UnifiedIterBuildEssentials<M: IterwiseUnifiedModel> {
    pub(super) context: WorkerContext,
    pub(super) allocation_base: u16,
    pub(super) kv_capacity: u64,
    pub(super) sampler: Option<KvSampler>,
    pub(super) execution: UnifiedIterExecution<M>,
}

/// Full-attention token capacity before a family reserves any slice for its
/// own cadence, such as PD decode's pull backlog.
pub(super) fn full_attention_token_capacity<M: IterwiseUnifiedModel>(
    model: &M,
    config: &WorkerConfig,
) -> u64 {
    let partition_kv_bytes = config
        .attn_kv_bytes
        .saturating_mul(u64::from(model.num_attn_shards().max(1)));
    (partition_kv_bytes / model.total_kv_bytes_per_token().max(1)).max(1)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_unified_iter_build_essentials<M: IterwiseUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    requests: SharedRequests,
    config: &WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: &SharedGpuCluster,
    kv_capacity: u64,
    num_partitions: usize,
) -> UnifiedIterBuildEssentials<M> {
    let num_partitions = num_partitions.max(1);

    let allocation_base =
        cluster
            .borrow_mut()
            .allocate(pool.0, id.0, model.gpus_per_replica(), gpu_name, pool_tag);

    {
        let mut cluster = cluster.borrow_mut();
        #[allow(
            clippy::cast_possible_truncation,
            reason = "num_partitions is a GPU/rank partition count, always small"
        )]
        for partition in 0..num_partitions as u16 {
            cluster.register_kv_capacity(pool_tag, pool.0, id.0, partition, kv_capacity);
        }
    }

    let sampler = KvSampler::open_opt(
        cost_log_dir.as_deref(),
        pool_tag,
        id,
        num_partitions,
        config.kv_log_stride,
    );
    let cost = CostBuffers::new_iter(
        cost_log_dir,
        pool_tag,
        id,
        model.as_ref(),
        config.gpu_time_multiplier,
    );
    let context = WorkerContext {
        id,
        pool,
        requests,
        log_output_token_times: config.log_output_token_times,
        log_stage_transitions: config.log_stage_transitions,
    };

    UnifiedIterBuildEssentials {
        context,
        allocation_base,
        kv_capacity,
        sampler,
        execution: UnifiedIterExecution::new(model, cost),
    }
}
