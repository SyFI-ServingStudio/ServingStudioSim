//! Production pipeline follower-stage recipe.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::timing::CostManifestDoc;
use crate::worker::cost_buffers::CostBuffers;
use crate::worker::execution::UnifiedIterExecution;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::WorkerConfig;

use super::pipeline_stage_worker::PipelineStageWorker;
use super::PipelineStage;

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_pipeline_stage_worker<M: IterwiseUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    stage_model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> PipelineStage<M> {
    assert_eq!(
        stage_model.gpus_per_replica(),
        1,
        "a pipeline stage runs on one GPU"
    );
    let communication_group_id = {
        let mut cluster = cluster.borrow_mut();
        let allocation_base = cluster.allocate(pool.0, id.0, 1, gpu_name, pool_tag);
        cluster.register_comm_group(allocation_base, 1, pool_tag, id.0)
    };
    // Followers cost the same whole-stage iteration as the head.
    let cost = CostBuffers::new(
        cost_log_dir,
        pool_tag,
        id,
        &CostManifestDoc::single("iter", stage_model.cost_log_manifest()),
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
    PipelineStageWorker::from_components(
        context,
        UnifiedIterExecution::new(stage_model, cost),
        cluster,
        communication_group_id,
    )
}
