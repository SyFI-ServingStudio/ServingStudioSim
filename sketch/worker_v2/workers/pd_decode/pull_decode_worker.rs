//! `PullDecodeWorker<K, E>` — the PD decode shell (matrix, iter-family cadence + a KV
//! pull front-end). Two independent timelines: a KV-transfer PULL pipeline (a
//! handed-off request's prompt KV is pulled from its prefill worker before it can
//! decode) and the decode iteration FSM. `IterBatchWorker`'s single-compute FSM does not
//! model the transfer timeline, so PD decode needs its own shell — but it reuses
//! `FullAttnKv` (K) + `UnifiedIterExecution` (E) unchanged, and the decode lifecycle is thin
//! enough to inline (like AFD-ffn), so there is no separate admission axis here.
//!
//! Mirrors `PdDecodeWorker` (pd_decode.rs): at most one pull in flight (NCCL is
//! serial), a token backlog gate throttling fetches, `PullComplete` acking the
//! prefill side (→ L6 → `ReleaseKv`), and a direct admit into the decode set (the
//! request is already prefilled — `commit_resident` with no prior reserve). External
//! calls verified: `submit_transfer`, `register_comm_group`, `Batch` via KvStore.

use std::collections::VecDeque;

use crate::common::{PdStage, RequestId, Time, WorkerId};
use crate::worker::admission_helpers::LoadBalance;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::iter_worker::IterWorker;
use crate::worker::types::{
    BatchFsmState, IterCursor, PdDecodeEvent, PdDecodeMsg, TransferPlan, WorkerFsmState,
    WorkerStatus,
};

use super::super::super::execution::IterModelExecution;
use super::super::super::kv::{IterWorkerKv, KvStore};
use super::super::super::shared::advance_scope::AdvanceScope;
use super::super::super::shared::context::WorkerContext;

/// Pull backlog ceiling as a fraction of shard token capacity (matches the real).
pub(super) const PULL_BUDGET_FRAC: f64 = 0.05;

#[derive(Clone, Copy)]
struct ActiveKvPull {
    req: RequestId,
    pull_end: Time,
    prefill_worker: WorkerId,
    tokens: u64,
}

fn earliest_wakeup(first: Option<Time>, second: Option<Time>) -> Option<Time> {
    match (first, second) {
        (Some(first), Some(second)) => Some(first.min(second)),
        (Some(wakeup), None) | (None, Some(wakeup)) => Some(wakeup),
        (None, None) => None,
    }
}

pub struct PullDecodeWorker<K, E>
where
    K: KvStore + IterWorkerKv,
    E: IterModelExecution<K>,
{
    context: WorkerContext,
    kv_store: K,
    execution: E,
    input: <E as IterModelExecution<K>>::Input,
    // decode iteration FSM (same shape as IterBatchWorker's).
    worker_state: WorkerFsmState,
    batch_state: BatchFsmState,
    iter_counter: u32,
    balance: LoadBalance,
    // pull pipeline (comm seam on the shell).
    cluster: SharedGpuCluster,
    receive_group_id: u16,
    /// Landed-but-not-yet-active handoffs awaiting a decode slot.
    pending_decodes: VecDeque<RequestId>,
    /// Backlog sum (tokens) for `pending_decodes` + in-flight; the fetch throttle.
    pending_decode_kv_tokens: u64,
    /// Handoffs awaiting submission (not yet fetching → not in the backlog).
    pending_pulls: VecDeque<(RequestId, TransferPlan)>,
    active_pull: Option<ActiveKvPull>,
    pull_budget_tokens: u64,
    kv_bytes_per_token: u64,
}

impl<K, E> PullDecodeWorker<K, E>
where
    K: KvStore + IterWorkerKv,
    E: IterModelExecution<K>,
{
    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_components(
        context: WorkerContext,
        kv_store: K,
        execution: E,
        balance: LoadBalance,
        cluster: SharedGpuCluster,
        receive_group_id: u16,
        pull_budget_tokens: u64,
        kv_bytes_per_token: u64,
    ) -> Self {
        Self {
            context,
            kv_store,
            execution,
            input: Default::default(),
            worker_state: WorkerFsmState::Idle,
            batch_state: BatchFsmState {
                cursor: IterCursor::Done,
                compute_end: Time::ZERO,
            },
            iter_counter: 0,
            balance,
            cluster,
            receive_group_id,
            pending_decodes: VecDeque::new(),
            pending_decode_kv_tokens: 0,
            pending_pulls: VecDeque::new(),
            active_pull: None,
            pull_budget_tokens,
            kv_bytes_per_token,
        }
    }

    /// Direct-admit test/cluster-free path: treat the transfer as instant.
    fn on_msg_request(&mut self, rid: RequestId) {
        let (tokens, arrival) = {
            let store = self.context.requests.borrow();
            let request = &store[rid];
            (
                u64::from(request.prompt_len + request.prefix_kv),
                request.arrival_time,
            )
        };
        self.pending_decodes.push_back(rid);
        self.pending_decode_kv_tokens += tokens;
        let mut store = self.context.requests.borrow_mut();
        self.context
            .stamp_stage(&mut store[rid], arrival, PdStage::PendingDecode as u16);
    }

    /// Queue a PD handoff for the pull FSM (ref `on_enqueue`).
    fn on_msg_handoff(
        &mut self,
        req: RequestId,
        send_group_id: u16,
        tokens: u64,
        prefill_worker: WorkerId,
    ) {
        self.pending_pulls.push_back((
            req,
            TransferPlan {
                send_gid: send_group_id,
                recv_gid: self.receive_group_id,
                tokens,
                prefill_worker,
            },
        ));
    }

    /// Pull FSM: promote a landed in-flight pull into `pending_decodes` (+ ack the
    /// prefill side), then submit the next queued handoff if the backlog gate allows.
    fn drive_kv_pulls(&mut self, now: Time, events: &mut Vec<PdDecodeEvent>) {
        loop {
            if let Some(pull) = self.active_pull {
                if now >= pull.pull_end {
                    self.pending_decodes.push_back(pull.req);
                    self.pending_decode_kv_tokens += pull.tokens;
                    self.active_pull = None;
                    {
                        let mut store = self.context.requests.borrow_mut();
                        self.context.stamp_stage(
                            &mut store[pull.req],
                            now,
                            PdStage::PendingDecode as u16,
                        );
                    }
                    events.push(PdDecodeEvent::PullComplete {
                        worker: self.context.id,
                        req: pull.req,
                        prefill_worker: pull.prefill_worker,
                    });
                } else {
                    return;
                }
            }
            let Some(&(_, next)) = self.pending_pulls.front() else {
                return;
            };
            let head_tokens = next.tokens;
            // `active_pull` is None here (a non-landed pull returns above).
            let backlog = self.pending_decode_kv_tokens;
            let fits = backlog + head_tokens <= self.pull_budget_tokens;
            let single_req_exception = backlog == 0 && head_tokens > self.pull_budget_tokens;
            if !fits && !single_req_exception {
                return;
            }
            let (rid, transfer) = self.pending_pulls.pop_front().unwrap();
            let bytes = transfer.tokens.saturating_mul(self.kv_bytes_per_token);
            let pull_end = self.cluster.borrow_mut().submit_transfer(
                now,
                transfer.send_gid,
                transfer.recv_gid,
                bytes,
                "pd_kv_pull",
                "",
            );
            self.active_pull = Some(ActiveKvPull {
                req: rid,
                pull_end,
                prefill_worker: transfer.prefill_worker,
                tokens: transfer.tokens,
            });
            {
                let mut store = self.context.requests.borrow_mut();
                self.context
                    .stamp_stage(&mut store[rid], now, PdStage::Transfer as u16);
            }
            // Loop back: an instant transfer (pull_end <= now) promotes this tick.
        }
    }

    fn drive_decode_worker(&mut self, now: Time, events: &mut Vec<PdDecodeEvent>) -> Option<Time> {
        self.drive_kv_pulls(now, events);
        use IterCursor::{Computing, Done, NotStarted};
        use WorkerFsmState::{Active, Idle};
        loop {
            match self.worker_state {
                Idle => {
                    if !self.form_batch(now) {
                        break;
                    }
                    self.iter_counter += 1;
                    self.worker_state = Active;
                    self.batch_state.cursor = NotStarted;
                }
                Active => match self.batch_state.cursor {
                    NotStarted => {
                        self.batch_state.compute_end = self.start_iteration(now);
                        self.batch_state.cursor = Computing;
                        break;
                    }
                    Computing if now < self.batch_state.compute_end => break,
                    Computing => self.batch_state.cursor = Done,
                    Done => {
                        self.complete_iteration(now, events);
                        self.worker_state = Idle;
                    }
                },
            }
        }
        self.next_wakeup(now)
    }

    /// Admit ONE landed handoff directly into a decode group (already prefilled →
    /// no prefill compute; `commit_resident` with no prior reserve).
    fn form_batch(&mut self, now: Time) -> bool {
        let num_partitions = self.kv_store.num_partitions();
        let had_live_decode =
            (0..num_partitions as u16).any(|partition| self.kv_store.has_live_decode(partition));
        if let Some(&rid) = self.pending_decodes.front() {
            let (prompt_kv, remaining) = {
                let store = self.context.requests.borrow();
                let record = &store[rid];
                (
                    u64::from(record.prompt_len + record.prefix_kv),
                    record.decode_len.saturating_sub(record.tokens_emitted),
                )
            };
            let partition_id = self.balance.choose(num_partitions) as u16;
            let footprint = self.kv_store.footprint(rid, prompt_kv as u32, remaining);
            if remaining == 0 || self.kv_store.fits(partition_id, &footprint) {
                self.pending_decodes.pop_front();
                self.pending_decode_kv_tokens =
                    self.pending_decode_kv_tokens.saturating_sub(prompt_kv);
                self.kv_store
                    .commit_resident(rid, partition_id, prompt_kv, remaining);
                let mut store = self.context.requests.borrow_mut();
                self.context
                    .stamp_stage(&mut store[rid], now, PdStage::Decode as u16);
            }
        }
        let still_has_live_decode =
            (0..num_partitions as u16).any(|partition| self.kv_store.has_live_decode(partition));
        had_live_decode || still_has_live_decode
    }

    fn start_iteration(&mut self, now: Time) -> Time {
        self.execution.build_iteration_input(
            &self.kv_store,
            &self.context.requests,
            &mut self.input,
        );
        now + self
            .execution
            .evaluate_iteration(&self.input, self.iter_counter as u64, now)
    }

    fn complete_iteration(&mut self, now: Time, events: &mut Vec<PdDecodeEvent>) {
        let num_partitions = self.kv_store.num_partitions();
        for partition in 0..num_partitions as u16 {
            let mut completed: Vec<RequestId> = Vec::new();
            {
                let log_tokens = self.context.log_tokens();
                let members = self.kv_store.decode_members(partition);
                let mut store = self.context.requests.borrow_mut();
                for (rid, _current_kv) in members {
                    let record = &mut store[rid];
                    let emitted = 1u32.min(record.decode_len.saturating_sub(record.tokens_emitted));
                    for _ in 0..emitted {
                        record.record_token(now, log_tokens);
                    }
                    if record.is_complete() {
                        self.context.stamp_stage(record, now, PdStage::Done as u16);
                        completed.push(rid);
                    }
                }
            }
            self.kv_store
                .advance(AdvanceScope::WholePartition(partition), 1);
            for rid in completed {
                self.kv_store.release(rid, partition);
                events.push(PdDecodeEvent::RequestComplete {
                    worker: self.context.id,
                    req: rid,
                });
            }
            self.kv_store.sample_submit(partition, now);
        }
    }

    fn next_wakeup(&self, now: Time) -> Option<Time> {
        let pull_wakeup = self.active_pull.map(|pull| pull.pull_end);
        let num_partitions = self.kv_store.num_partitions();
        let compute_wakeup = match self.worker_state {
            WorkerFsmState::Idle => {
                let has_decode = (0..num_partitions as u16)
                    .any(|partition| self.kv_store.has_live_decode(partition));
                if !self.pending_decodes.is_empty() || has_decode {
                    Some(now)
                } else {
                    None
                }
            }
            WorkerFsmState::Active => match self.batch_state.cursor {
                IterCursor::NotStarted | IterCursor::Done => Some(now),
                IterCursor::Computing => Some(self.batch_state.compute_end),
            },
        };
        earliest_wakeup(pull_wakeup, compute_wakeup)
    }
}

impl<K, E> IterWorker for PullDecodeWorker<K, E>
where
    K: KvStore + IterWorkerKv,
    E: IterModelExecution<K>,
{
    type Msg = PdDecodeMsg;
    type Event = PdDecodeEvent;

    fn id(&self) -> WorkerId {
        self.context.id
    }

    fn enqueue(&mut self, msg: PdDecodeMsg) {
        match msg {
            PdDecodeMsg::Request(rid) => self.on_msg_request(rid),
            PdDecodeMsg::Handoff {
                req,
                send_gid: send_group_id,
                tokens,
                prefill_worker,
            } => self.on_msg_handoff(req, send_group_id, tokens, prefill_worker),
        }
    }

    fn tick(&mut self, now: Time, events: &mut Vec<PdDecodeEvent>) -> Option<Time> {
        self.drive_decode_worker(now, events)
    }

    fn status(&self) -> WorkerStatus {
        let num_partitions = self.kv_store.num_partitions();
        let live_decodes: u32 = (0..num_partitions as u16)
            .map(|partition| self.kv_store.live_decode_count(partition))
            .sum();
        let queued = self.pending_decodes.len()
            + self.pending_pulls.len()
            + usize::from(self.active_pull.is_some());
        WorkerStatus {
            queued_requests: queued as u32,
            active_requests: live_decodes,
        }
    }
}
