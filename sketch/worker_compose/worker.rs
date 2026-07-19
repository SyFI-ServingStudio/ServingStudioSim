//! `Worker` — the composition container + the unchanged L6-facing surface.
//!
//! M1 concrete container: generic only over the model `M`. The public surface
//! (`new` 9-arg signature, `enqueue` / `tick` / `status` / `release_request`, and
//! the `IterWorker` impl) is byte-for-byte the old `BareboneWorker`, so the
//! deployment / factory / selector (`orchestrator/common.rs`, `config.rs`) are
//! untouched and the goldens run through this code.
//!
//! Construction choreography (a strict cross-axis sequence — review point):
//!   1. `cluster.allocate` the GPU block.
//!   2. derive `kv_capacity` from the model's KV layout (needs the model BEFORE
//!      it moves into `UnifiedArch`).
//!   3. `register_kv_capacity` into the run-meta registry.
//!   4. open the `KvSampler` (borrows `cost_log_dir` before it is moved).
//!   5. build `CostBuffers` (consumes `cost_log_dir` + `gpu_time_multiplier`).
//! Only THEN are the three reusable components assembled; `WorkerConfig` is
//! distributed, not stored whole (policy → Admission, logging flags → ctx, the
//! rest consumed here). The concrete worker owns both its short iter-wise state
//! machine and the cross-component `build_arch_input` glue.
//!
//! Reading order: types → construction → message handling (the `IterWorker` entry
//! points) → the tick / pipeline loop → its helpers, each following the function
//! that calls it.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::{ArchGroupInput, IterwiseUnifiedModel, UnifiedArchInput};
use crate::common::{PoolId, RequestId, SharedRequests, Time, WorkerId};
use crate::log::KvSampler;
use crate::worker::cost_buffers::CostBuffers;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::iter_worker::IterWorker;
use crate::worker::types::{
    BatchFsmState, IterCursor, WorkerConfig, WorkerEventCommon, WorkerFsmState, WorkerMsgCommon,
    WorkerStatus,
};

use super::admission::PrefillDecode;
use super::arch_unified::UnifiedArch;
use super::ctx::WorkerCtx;
use super::kv::FullAttnKv;

/// Private execution state for this worker protocol. This is deliberately data,
/// not a replaceable component: another protocol gets another short Worker.
struct IterState {
    worker_state: WorkerFsmState,
    batch_state: BatchFsmState,
    iter_counter: u32,
}

impl IterState {
    fn new() -> Self {
        Self {
            worker_state: WorkerFsmState::Idle,
            batch_state: BatchFsmState {
                cursor: IterCursor::Done,
                compute_end: Time::ZERO,
            },
            iter_counter: 0,
        }
    }
}

pub struct Worker<M: IterwiseUnifiedModel> {
    ctx: WorkerCtx,
    kv: FullAttnKv,
    admission: PrefillDecode,
    arch: UnifiedArch<M>,
    iter: IterState,
}

/// The name the rest of the tree still imports. The alias preserves the current
/// public surface while this concrete worker keeps its own protocol and input glue.
pub type BareboneWorker<M> = Worker<M>;

impl<M: IterwiseUnifiedModel> Worker<M> {
    /// SAME 9-arg signature as `BareboneWorker::new` today (`WorkerBuildFn`,
    /// orchestrator/common.rs:38). Only the body changes: it wires three components
    /// (distributing `config`) instead of filling one flat struct.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: WorkerId,
        pool_tag: &'static str,
        model: Arc<M>,
        requests: SharedRequests,
        config: WorkerConfig,
        cost_log_dir: Option<PathBuf>,
        pool: PoolId,
        gpu_name: &str,
        cluster: SharedGpuCluster,
    ) -> Self {
        // (1) allocate this worker's GPU block in the shared cluster.
        cluster
            .borrow_mut()
            .allocate(pool.0, id.0, model.gpus_per_replica(), gpu_name, pool_tag);
        // (2) partition memory = num_attn_shards × attn_kv_bytes; ÷ per-token wire size.
        let partition_kv_bytes = config
            .attn_kv_bytes
            .saturating_mul(model.num_attn_shards().max(1) as u64);
        let kv_capacity = (partition_kv_bytes / model.total_kv_bytes_per_token().max(1)).max(1);
        // (3) report the pool's static token capacity (barebone = partition 0).
        cluster
            .borrow_mut()
            .register_kv_capacity(pool_tag, pool.0, id.0, 0, kv_capacity);
        // (4) open the KV occupancy sampler, borrowing cost_log_dir before move.
        let sampler = KvSampler::open_opt(
            cost_log_dir.as_deref(),
            pool_tag,
            id,
            1,
            config.kv_log_stride,
        );
        // (5) build the cost buffers (consumes cost_log_dir + gpu_time_multiplier).
        let cost = CostBuffers::new_iter(
            cost_log_dir,
            pool_tag,
            id,
            model.as_ref(),
            config.gpu_time_multiplier,
        );
        Self {
            ctx: WorkerCtx {
                id,
                pool,
                requests,
                log_output_token_times: config.log_output_token_times,
                log_stage_transitions: config.log_stage_transitions,
            },
            kv: FullAttnKv::new(1, kv_capacity, sampler), // barebone = 1 partition
            admission: PrefillDecode::new(config.admission, config.max_batch_tokens),
            arch: UnifiedArch::new(model, cost),
            iter: IterState::new(),
        }
    }

    pub fn enqueue(&mut self, msg: WorkerMsgCommon) {
        self.admission.accept(msg, &self.ctx);
    }

    /// Concrete barebone state machine. The worker owns sequencing because it is
    /// the only place that knows how Admission, KV facts, and this model's
    /// ArchInput fit together.
    pub fn tick(&mut self, now: Time, events: &mut Vec<WorkerEventCommon>) -> Option<Time> {
        use IterCursor::{Computing, Done, NotStarted};
        use WorkerFsmState::{Active, Idle};

        loop {
            match self.iter.worker_state {
                Idle => {
                    if !self.admission.form_batch(&mut self.kv, &self.ctx, now) {
                        break;
                    }
                    self.iter.iter_counter += 1;
                    self.iter.worker_state = Active;
                    self.iter.batch_state.cursor = NotStarted;
                    self.iter.batch_state.compute_end = Time::ZERO;
                }
                Active => match self.iter.batch_state.cursor {
                    NotStarted => {
                        let compute_end = self.start_iter(now);
                        self.iter.batch_state.cursor = Computing;
                        self.iter.batch_state.compute_end = compute_end;
                        break;
                    }
                    Computing if now < self.iter.batch_state.compute_end => break,
                    Computing => {
                        self.iter.batch_state.cursor = Done;
                    }
                    Done => {
                        self.admission
                            .on_iter_complete(&mut self.kv, &self.ctx, events, now);
                        self.iter.worker_state = Idle;
                    }
                },
            }
        }
        self.next_wakeup(now)
    }

    /// Build the concrete model input, evaluate it, and arm this iteration's end.
    /// Input construction stays on the worker: another worker may reuse
    /// `FullAttnKv` while using speculative or model-specific input semantics.
    fn start_iter(&mut self, now: Time) -> Time {
        let input = self.build_arch_input();
        let cost = self
            .arch
            .eval_iter(&input, self.iter.iter_counter as u64, now);
        now + cost
    }

    /// Translate this worker's current batch into its Arch vocabulary (ref
    /// `BareboneWorker::build_arch_input`). KV supplies resident lengths and
    /// membership facts; it does not know this model's input type.
    fn build_arch_input(&self) -> UnifiedArchInput {
        let batch = self.kv.batch(0);
        let store = self.ctx.requests.borrow();
        let mut group = ArchGroupInput::default();

        for &request in &batch.prefill_admits {
            let record = &store[request];
            group
                .prefill_chunk_pairs
                .push((record.prefix_kv, record.active_chunk_len));
            group.prefill_tokens += record.active_chunk_len;
        }
        for (_, decode_state) in batch.iter_decoding() {
            group.decode_kv_lens.push(decode_state.current_kv as u32);
            group.total_kv_len += decode_state.current_kv as u32;
        }
        group.decode_tokens = group.decode_kv_lens.len() as u32;
        group.batch_tokens = group.prefill_tokens + group.decode_tokens;

        UnifiedArchInput {
            groups: vec![group],
            tokens_per_source_rank: Vec::new(),
        }
    }

    fn next_wakeup(&self, now: Time) -> Option<Time> {
        match self.iter.worker_state {
            WorkerFsmState::Idle => {
                let has_work = self.admission.status_queued() > 0 || self.kv.status_active(0) > 0;
                has_work.then_some(now)
            }
            WorkerFsmState::Active => match self.iter.batch_state.cursor {
                IterCursor::NotStarted | IterCursor::Done => Some(now),
                IterCursor::Computing => Some(self.iter.batch_state.compute_end),
            },
        }
    }

    /// Cross-axis read (queued from Admission + active from Kv) → lives on the
    /// container, not on any single axis trait (reconciles the §9 `status` home).
    pub fn status(&self) -> WorkerStatus {
        WorkerStatus {
            queued_requests: self.admission.status_queued(),
            active_requests: self.kv.status_active(0),
        }
    }

    /// External release (cancellation): pending queue (Admission) then the
    /// admitted state (Kv). Cross-axis, so it composes two axis-local cancellation
    /// methods on the container — the §9 interface must expose both (review: this
    /// was a trait-surface hole).
    pub fn release_request(&mut self, rid: RequestId, current_kv: u64) -> Option<u16> {
        if self.admission.remove_pending(rid) {
            return None;
        }
        self.kv.release_external(rid, current_kv)
    }

    pub fn id(&self) -> WorkerId {
        self.ctx.id
    }
}

/// Unchanged from today — delegates to the inherent methods above.
impl<M: IterwiseUnifiedModel> IterWorker for Worker<M> {
    type Msg = WorkerMsgCommon;
    type Event = WorkerEventCommon;

    fn id(&self) -> WorkerId {
        self.ctx.id
    }
    fn enqueue(&mut self, msg: Self::Msg) {
        Worker::enqueue(self, msg)
    }
    fn tick(&mut self, now: Time, events: &mut Vec<Self::Event>) -> Option<Time> {
        Worker::tick(self, now, events)
    }
    fn status(&self) -> WorkerStatus {
        Worker::status(self)
    }
}
