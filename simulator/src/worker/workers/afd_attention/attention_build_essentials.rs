//! Family-local construction of AFD-attention resource and execution components.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::AttnLayerwiseModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::log::{KvSampler, PrefixCacheLogger};
use crate::worker::cost_buffers::CostBuffers;
use crate::worker::execution::AttentionLayerExecutionAdapter;
use crate::worker::gpu_cluster::{KvPoolFacts, SharedGpuCluster};
use crate::worker::kv::KvCacheStore;
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::WorkerConfig;
use crate::worker::workers::iter_build_essentials::{kv_cache_store, resolve_kv_accounting};

pub(super) struct AttentionBuildEssentials<M: AttnLayerwiseModel> {
    pub(super) context: WorkerContext,
    pub(super) kv_store: KvCacheStore,
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
    let kv_accounting =
        resolve_kv_accounting(&model.kv_cache_layout(), model.num_attn_shards(), &config);
    cluster.borrow_mut().register_kv_pool(
        pool_tag,
        pool.0,
        id.0,
        0,
        KvPoolFacts::of(&kv_accounting, config.max_model_len),
    );
    let sampler = KvSampler::open_opt(
        cost_log_dir.as_deref(),
        pool_tag,
        id,
        1,
        config.kv_log_stride,
    );
    let prefix_cache_logger = PrefixCacheLogger::open_opt(cost_log_dir.as_deref(), pool_tag, id);
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
        kv_store: kv_cache_store(
            kv_accounting,
            1,
            model.num_attn_shards(),
            &config,
            sampler,
            prefix_cache_logger,
        ),
        execution: AttentionLayerExecutionAdapter::new(model, cost),
        cluster,
        receive_group_id,
    }
}
