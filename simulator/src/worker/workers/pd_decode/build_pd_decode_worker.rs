//! Production PD-decode composition recipe.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::worker::admission::LoadBalance;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::kv::FullAttnKv;
use crate::worker::types::WorkerConfig;
use crate::worker::workers::unified_iter_build_essentials::{
    full_attention_token_capacity, prepare_unified_iter_build_essentials,
};

use super::pull_decode_worker::{PdDecodeWorker, PullDecodeWorker, PULL_BUDGET_FRACTION};

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_pd_decode_worker<M: IterwiseUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> PdDecodeWorker<M> {
    let num_partitions = model.num_attn_dp_groups().max(1) as usize;
    let num_attn_shards = model.num_attn_shards().max(1);
    let kv_bytes_per_token = model.total_kv_bytes_per_token();
    let total_partition_tokens = full_attention_token_capacity(model.as_ref(), &config);
    let pull_budget_tokens = ((total_partition_tokens as f64 * PULL_BUDGET_FRACTION) as u64).max(1);
    let active_kv_capacity = total_partition_tokens
        .saturating_sub(pull_budget_tokens)
        .max(1);

    let essentials = prepare_unified_iter_build_essentials(
        id,
        pool_tag,
        model,
        requests,
        &config,
        cost_log_dir,
        pool,
        gpu_name,
        &cluster,
        active_kv_capacity,
        num_partitions,
    );
    let receive_group_id = cluster.borrow_mut().register_comm_group(
        essentials.allocation_base,
        num_attn_shards,
        pool_tag,
        id.0,
    );
    let balance = match config.balance {
        LoadBalance::Single if num_partitions > 1 => LoadBalance::RoundRobin { next: 0 },
        configured => configured,
    };
    let kv_store = FullAttnKv::new(num_partitions, essentials.kv_capacity, essentials.sampler);

    PullDecodeWorker::from_components(
        essentials.context,
        kv_store,
        essentials.execution,
        balance,
        cluster,
        receive_group_id,
        pull_budget_tokens,
        kv_bytes_per_token,
    )
}
