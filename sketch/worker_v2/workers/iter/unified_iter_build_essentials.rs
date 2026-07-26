use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::log::KvSampler;
use crate::worker::cost_buffers::CostBuffers;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::types::WorkerConfig;

use super::super::super::execution::UnifiedIterExecution;
use super::super::super::shared::context::WorkerContext;

/// Family-private construction essentials shared by every recipe that uses
/// `UnifiedIterExecution`.
///
/// Concrete builders still own the semantic choices (KV wrapper, admission
/// lifecycle, policy, and config knobs). This helper owns only the repeated
/// infrastructure translation from worker config to allocated resources.
pub(super) struct UnifiedIterBuildEssentials<M: IterwiseUnifiedModel> {
    pub(super) context: WorkerContext,
    pub(super) allocation_base: u16,
    pub(super) kv_capacity: u64,
    pub(super) sampler: Option<KvSampler>,
    pub(super) execution: UnifiedIterExecution<M>,
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
    num_partitions: usize,
) -> UnifiedIterBuildEssentials<M> {
    let num_partitions = num_partitions.max(1);
    let allocation_base =
        cluster
            .borrow_mut()
            .allocate(pool.0, id.0, model.gpus_per_replica(), gpu_name, pool_tag);
    let group_kv_bytes = config
        .attn_kv_bytes
        .saturating_mul(model.num_attn_shards().max(1) as u64);
    let kv_capacity = (group_kv_bytes / model.total_kv_bytes_per_token().max(1)).max(1);
    {
        let mut cluster = cluster.borrow_mut();
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
