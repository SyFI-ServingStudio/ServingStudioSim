//! Test-only helpers shared across worker / orchestrator / sim unit tests.
//! Each test module used to roll its own `FakeModel`, `test_cluster`, and
//! request-store builders — identical bodies, copy-paste drift. Centralized
//! here so a trait change (e.g. renaming `kv_bytes_per_token` →
//! `total_kv_bytes_per_token`) touches one impl, not seven.

use std::cell::RefCell;
use std::rc::Rc;

use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::common::{Request, RequestId, RequestStore, SharedRequests, Time};
use crate::timing::cache::interp::{CoverageFlags, Metrics4};
use crate::timing::LeafMetrics;
use crate::worker::gpu_cluster::{CostSource, GpuCluster, SharedGpuCluster};

/// Empty GPU cluster with an analytic 1 GB/s transfer cost. Workers' `new`
/// always calls `allocate`/`register_comm_group` against the cluster; tests
/// that don't transfer never observe the cost source value.
pub(crate) fn test_cluster() -> SharedGpuCluster {
    Rc::new(RefCell::new(GpuCluster::new(CostSource::analytic(1.0))))
}

/// Fixed-cost stand-in for an L4 model. `eval_iter` returns `time_ms = ms`
/// (no cost-tree, empty slot vector) and asserts the worker fed exactly one
/// group per DP shard. `dp_groups` doubles as `gpus_per_replica` and
/// `num_attn_dp_groups`; default `for_ms` uses `dp_groups: 1`.
pub(crate) struct FakeModel {
    pub ms: f64,
    pub dp_groups: u16,
}

impl FakeModel {
    pub fn for_ms(ms: f64) -> Self {
        Self { ms, dp_groups: 1 }
    }
}

impl IterwiseUnifiedModel for FakeModel {
    fn eval_iter(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        _scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        slots.clear();
        assert_eq!(batch.groups.len(), self.dp_groups as usize);
        LeafMetrics {
            m: Metrics4 {
                time_ms: self.ms as f32,
                flops: 0.0,
                bytes: 0.0,
                energy_j: 0.0,
            },
            coverage: CoverageFlags::EMPTY,
        }
    }
    fn total_kv_bytes_per_token(&self) -> u64 {
        1
    }
    fn gpus_per_replica(&self) -> u16 {
        self.dp_groups
    }
    fn num_attn_dp_groups(&self) -> u16 {
        self.dp_groups
    }
}

/// Build a `SharedRequests` from `(id, prompt, decode)` tuples. Used by
/// prefill / unified / hp_unified tests that drive prefill themselves.
pub(crate) fn shared_with(reqs: &[(u32, u32, u32)]) -> SharedRequests {
    let store = Rc::new(RefCell::new(RequestStore::new()));
    for &(id, prompt, decode) in reqs {
        store
            .borrow_mut()
            .insert(&Request::new(RequestId(id), prompt, decode, Time::ZERO));
    }
    store
}

/// Like `shared_with`, but marks every request as already prefilled (handoff
/// state: `prefill_processed`, first token emitted). Used by `PdDecodeWorker`
/// tests that admit via `WorkerMsg::Request` to bypass the pull path.
pub(crate) fn prefilled_store(reqs: &[(u32, u32, u32)]) -> SharedRequests {
    let store = shared_with(reqs);
    {
        let mut s = store.borrow_mut();
        for &(id, prompt, _decode) in reqs {
            let rec = &mut s[RequestId(id)];
            rec.prefill_processed = prompt;
            rec.tokens_emitted = 1;
            rec.first_token_time = Some(Time::ZERO);
            rec.last_token_time = Some(Time::ZERO);
        }
    }
    store
}
