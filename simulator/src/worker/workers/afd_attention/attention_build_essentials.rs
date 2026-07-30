//! Family-local construction of AFD-attention resource and execution components.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::AttnLayerwiseModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::log::KvSampler;
use crate::worker::cost_buffers::CostBuffers;
use crate::worker::execution::AttentionLayerExecutionAdapter;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::kv::FullAttnKv;
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::WorkerConfig;

pub(super) struct AttentionBuildEssentials<M: AttnLayerwiseModel> {
    pub(super) context: WorkerContext,
    pub(super) kv_store: FullAttnKv,
    pub(super) execution: AttentionLayerExecutionAdapter<M>,
    pub(super) cluster: SharedGpuCluster,
    pub(super) receive_group_id: u16,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_attention_build_essentials<M: AttnLayerwiseModel>(
    id: WorkerId,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
    cost_log_dir: Option<PathBuf>,
    pool_tag: &'static str,
) -> AttentionBuildEssentials<M> {
    let receive_group_id = {
        let mut cluster = cluster.borrow_mut();
        let allocation_base =
            cluster.allocate(pool.0, id.0, model.gpus_per_replica(), gpu_name, pool_tag);
        cluster.register_comm_group(
            allocation_base,
            model.num_attn_shards().max(1),
            pool_tag,
            id.0,
        )
    };
    let shard_bytes = config
        .attn_kv_bytes
        .saturating_mul(model.num_attn_shards().max(1) as u64);
    let kv_capacity = (shard_bytes / model.total_kv_bytes_per_token().max(1)).max(1);
    cluster
        .borrow_mut()
        .register_kv_capacity(pool_tag, pool.0, id.0, 0, kv_capacity);
    let sampler = KvSampler::open_opt(
        cost_log_dir.as_deref(),
        pool_tag,
        id,
        1,
        config.kv_log_stride,
    );
    let cost = CostBuffers::new(
        cost_log_dir,
        pool_tag,
        id,
        &model.cost_log_manifest(),
        config.gpu_time_multiplier,
    );
    let context = WorkerContext {
        id,
        pool,
        requests,
        log_output_token_times: config.log_output_token_times,
        log_stage_transitions: config.log_stage_transitions,
    };

    AttentionBuildEssentials {
        context,
        kv_store: FullAttnKv::new(1, kv_capacity, sampler),
        execution: AttentionLayerExecutionAdapter::new(model, cost),
        cluster,
        receive_group_id,
    }
}
