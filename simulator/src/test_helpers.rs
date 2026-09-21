//! Test-only helpers shared across worker / orchestrator / sim unit tests.
//! Each test module used to roll its own `FakeModel`, `test_cluster`, and
//! request-store builders — identical bodies, copy-paste drift. Centralized
//! here so a trait change (e.g. renaming `kv_bytes_per_token` →
//! `total_kv_bytes_per_token`) touches one impl, not seven.

use std::cell::RefCell;
use std::rc::Rc;

use crate::arch::contract::{
    AttnArchInput, AttnLayerwiseModel, FfnArchInput, IterwiseUnifiedModel, UnifiedArchInput,
};
use crate::common::{
    DecodingStrategy, PlacementDirective, Request, RequestCore, RequestId, RequestStore,
    SchedulingContract, SessionInput, SharedRequests, SloContract, TextGenerationDefinition, Time,
    WorkerId,
};
use crate::timing::cache::interp::{CoverageFlags, Metrics4};
use crate::timing::LeafMetrics;
use crate::worker::gpu_cluster::{CostSource, GpuCluster, SharedGpuCluster};

/// Fixed-cost [`LeafMetrics`] (`time_ms = ms`, no real cost-tree leaves). The
/// shared building block for the fake L4 models below.
pub(crate) fn lm(ms: f64) -> LeafMetrics {
    LeafMetrics {
        m: Metrics4 {
            time_ms: ms as f32,
            flops: 0.0,
            bytes: 0.0,
            energy_j: 0.0,
        },
        coverage: CoverageFlags::EMPTY,
        backend_index: LeafMetrics::NO_BACKEND,
    }
}

/// Empty GPU cluster with an analytic 1 GB/s transfer cost. Workers' `new`
/// always calls `allocate`/`register_comm_group` against the cluster; tests
/// that don't transfer never observe the cost source value.
pub(crate) fn test_cluster() -> SharedGpuCluster {
    Rc::new(RefCell::new(GpuCluster::new(CostSource::analytic(1.0))))
}

/// Build the default text request used by worker/orchestrator tests. Legacy
/// four-column trace compatibility is tested in the frontend parser; this
/// helper is deliberately test-only and does not define a production API.
pub(crate) const fn text_request(
    request_id: RequestId,
    prompt_tokens: u32,
    target_output_tokens: u32,
    arrival_time: Time,
) -> Request<TextGenerationDefinition> {
    Request::new(
        RequestCore {
            id: request_id,
            arrival_time,
            slo: SloContract {
                ttft_slo: None,
                tpot_slo: None,
                e2e_slo: None,
            },
            scheduling: SchedulingContract { priority: 0 },
            placement: PlacementDirective { worker: None },
        },
        TextGenerationDefinition {
            prompt_tokens,
            target_output_tokens,
            session: SessionInput::Standalone,
            decoding: DecodingStrategy::Standard,
        },
    )
}

/// [`text_request`] plus the trace-declared worker a `trace-directed` pool must
/// obey. Separate helper so the common case keeps its four arguments.
pub(crate) const fn text_request_on(
    request_id: RequestId,
    prompt_tokens: u32,
    target_output_tokens: u32,
    arrival_time: Time,
    target_worker: WorkerId,
) -> Request<TextGenerationDefinition> {
    let mut request = text_request(
        request_id,
        prompt_tokens,
        target_output_tokens,
        arrival_time,
    );
    request.core.placement = PlacementDirective {
        worker: Some(target_worker),
    };
    request
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
            backend_index: LeafMetrics::NO_BACKEND,
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

/// Fixed-cost stand-in for an AFD attn-side L4 model: every layer's attention
/// costs `ms`, the handoff is 2 bytes/token, one shard. Shared by the disagg attn
/// worker tests and the AFD orchestrator tests.
pub(crate) struct FakeAttn {
    pub ms: f64,
    pub layers: u32,
}

impl AttnLayerwiseModel for FakeAttn {
    fn num_layers(&self) -> u32 {
        self.layers
    }
    fn gpus_per_replica(&self) -> u16 {
        1
    }
    fn total_kv_bytes_per_token(&self) -> u64 {
        1
    }
    fn attn_to_ffn_bytes_per_token(&self) -> u64 {
        2
    }
    fn attn_cost(
        &self,
        _layer: usize,
        batch: &AttnArchInput,
        slots: &mut Vec<LeafMetrics>,
        _scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        assert_eq!(batch.groups.len(), 1, "attn worker feeds exactly one group");
        slots.clear();
        lm(self.ms)
    }
}

/// Fixed-cost stand-in for an AFD ffn-side L4 model: each section costs `ms`, the
/// QKV handoff is 2 bytes/token, one DP group. Shared by the disagg ffn worker
/// tests and the AFD orchestrator tests.
pub(crate) struct FakeFfn {
    pub ms: f64,
    pub layers: u32,
}

impl crate::arch::contract::FfnLayerwiseModel for FakeFfn {
    fn num_layers(&self) -> u32 {
        self.layers
    }
    fn gpus_per_replica(&self) -> u16 {
        1
    }
    fn num_dp_groups(&self) -> u16 {
        1
    }
    fn ffn_to_attn_bytes_per_token(&self) -> u64 {
        2
    }
    fn pre_attn_cost(
        &self,
        _l: usize,
        _b: &FfnArchInput,
        s: &mut Vec<LeafMetrics>,
        _sc: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        s.clear();
        lm(self.ms)
    }
    fn post_attn_cost(
        &self,
        _l: usize,
        _b: &FfnArchInput,
        s: &mut Vec<LeafMetrics>,
        _sc: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        s.clear();
        lm(self.ms)
    }
    fn prologue_cost(
        &self,
        _b: &FfnArchInput,
        s: &mut Vec<LeafMetrics>,
        _sc: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        s.clear();
        lm(self.ms)
    }
    fn epilogue_cost(
        &self,
        _b: &FfnArchInput,
        s: &mut Vec<LeafMetrics>,
        _sc: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        s.clear();
        lm(self.ms)
    }
}

/// Build a `SharedRequests` from `(id, prompt, decode)` tuples. Used by
/// prefill / unified / hp_unified tests that drive prefill themselves.
pub(crate) fn shared_with(reqs: &[(u32, u32, u32)]) -> SharedRequests {
    let store = Rc::new(RefCell::new(RequestStore::new()));
    for &(id, prompt, decode) in reqs {
        store
            .borrow_mut()
            .insert(text_request(RequestId(id), prompt, decode, Time::ZERO));
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
            rec.progress.prefill_tokens_processed = prompt;
            rec.progress.output_tokens_emitted = 1;
            rec.telemetry.first_output_time = Some(Time::ZERO);
            rec.telemetry.last_output_time = Some(Time::ZERO);
        }
    }
    store
}
