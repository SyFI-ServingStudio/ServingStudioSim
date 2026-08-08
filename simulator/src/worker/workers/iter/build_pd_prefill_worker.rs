//! PD-prefill whole-iteration worker recipe.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::worker::admission::{FifoOrder, PrefillHandoffAdmission};
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::kv::FullAttnKv;
use crate::worker::types::WorkerConfig;

use super::iter_batch_worker::{IterBatchWorker, PdPrefillWorker};
use crate::worker::workers::unified_iter_build_essentials::{
    full_attention_token_capacity, prepare_unified_iter_build_essentials,
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_pd_prefill_worker<M: IterwiseUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> PdPrefillWorker<M> {
    let num_attn_shards = model.num_attn_shards().max(1);
    let kv_capacity = full_attention_token_capacity(model.as_ref(), &config);
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
        kv_capacity,
        1,
    );
    let send_group_id = cluster.borrow_mut().register_comm_group(
        essentials.allocation_base,
        num_attn_shards,
        pool_tag,
        id.0,
    );
    let kv_store = FullAttnKv::new(1, essentials.kv_capacity, essentials.sampler);
    let admission = PrefillHandoffAdmission::new(FifoOrder::new(), (), send_group_id);

    IterBatchWorker::from_components(
        essentials.context,
        kv_store,
        admission,
        essentials.execution,
    )
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::*;
    use crate::common::{RequestId, Time};
    use crate::test_helpers::{shared_with, test_cluster, FakeModel};
    use crate::worker::types::{PdPrefillEvent, PdPrefillMsg};

    #[test]
    fn prefill_emits_handoff() {
        let store = shared_with(&[(0, 16, 3)]);
        let mut worker = build_pd_prefill_worker(
            WorkerId(0),
            "prefill",
            Arc::new(FakeModel::for_ms(1.0)),
            Rc::clone(&store),
            WorkerConfig::default(),
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        );
        worker.enqueue(PdPrefillMsg::Request(RequestId(0)));
        let mut events = Vec::new();
        for step in 0..20 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        assert_eq!(
            events,
            vec![PdPrefillEvent::PrefillDone {
                worker: WorkerId(0),
                req: RequestId(0),
                send_gid: 0,
                kv_tokens: 16,
            }]
        );
        assert_eq!(
            store.borrow()[RequestId(0)].progress.output_tokens_emitted,
            1
        );
    }

    #[test]
    fn one_token_request_completes_at_prefill() {
        let store = shared_with(&[(0, 16, 1)]);
        let mut worker = build_pd_prefill_worker(
            WorkerId(0),
            "prefill",
            Arc::new(FakeModel::for_ms(1.0)),
            Rc::clone(&store),
            WorkerConfig::default(),
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        );
        worker.enqueue(PdPrefillMsg::Request(RequestId(0)));
        let mut events = Vec::new();
        for step in 0..20 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        assert_eq!(
            events,
            vec![PdPrefillEvent::RequestComplete {
                worker: WorkerId(0),
                req: RequestId(0),
            }]
        );
        assert!(store.borrow()[RequestId(0)].lifecycle.completed);
    }
}
