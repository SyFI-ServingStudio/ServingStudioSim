//! Production pipeline-head recipe.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::log::PrefixCacheLogger;
use crate::worker::admission::{PendingOrder, PipelinedPrefillAdmission};
use crate::worker::execution::UnifiedIterExecution;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::kv::FullAttnKv;
use crate::worker::types::WorkerConfig;
use crate::worker::workers::iter_build_essentials::{
    full_attention_token_capacity, prepare_iter_build_essentials,
};

use super::pipeline_head_worker::{PipelineHeadWorker, PipelineLayout};
use super::PipelineHead;

/// `stage_model` is stage 0's layers. KV capacity comes from `layout`, not the
/// stage model: every stage holds the same tokens, so the stage with the most KV
/// bytes per token bounds the pipeline.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_pipeline_head_worker<M: IterwiseUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    stage_model: Arc<M>,
    layout: PipelineLayout,
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
    let kv_capacity = full_attention_token_capacity(1, layout.kv_bytes_per_token, &config);
    let prefix_cache =
        config
            .prefix_cache
            .resolve_tokens(kv_capacity, layout.kv_bytes_per_token, 1);
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
        kv_capacity,
        1,
    );
    let send_gid =
        cluster
            .borrow_mut()
            .register_comm_group(essentials.allocation_base, 1, pool_tag, id.0);
    let kv_store = FullAttnKv::with_prefix_cache(
        1,
        essentials.kv_capacity,
        prefix_cache,
        essentials.sampler,
        prefix_cache_logger,
    );
    let admission = PipelinedPrefillAdmission::new(
        PendingOrder::new(config.pending_order),
        (),
        max_batch_tokens,
    );

    PipelineHeadWorker::from_components(
        essentials.context,
        kv_store,
        admission,
        UnifiedIterExecution::new(stage_model, essentials.cost),
        layout,
        send_gid,
    )
}
