//! `PdDecodeWorker` — the decode half of a PD (prefill/decode disaggregation)
//! deployment. Iter-wise, but its admitted requests have **already been
//! prefilled** by a sibling prefill pool (handed off via `WorkerEvent::PrefillDone`
//! → L6 → `enqueue` here): the request's prompt KV is treated as already resident,
//! so a fresh admit enters the decode set directly (no prefill compute, no
//! prefill→decode transition). The worker then only runs decode iterations.
//!
//! Multi-group like `HpUnifiedWorker`: it keeps **N independent `Batch` containers**
//! (`N = model.num_attn_dp_groups()`), one per attention DP shard, and round-robins
//! handed-off requests across them. With a non-DP arch (`N == 1`) it degenerates to
//! a single decode group. The iteration is costed jointly (one `eval_iter`; the
//! arch's `Max` fan-out over the per-group attention slots supplies the DP
//! wallclock).
//!
//! v1 simplification: the prompt KV is assumed present at admit time (the
//! prefill→decode transfer is not modeled); the shared `RequestStore` carries the
//! prefill state (prompt_len, first token already emitted) across the handoff.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::{ArchGroupInput, IterwiseUnifiedModel, UnifiedArchInput};
use crate::common::{PoolId, RequestId, SharedRequests, Time, WorkerId};
use crate::log::{CostLogEntry, CostLogger, GroupInputLog};
use crate::timing::{LeafMetrics, SlotInput};
use crate::worker::admission_helpers::{Batch, LoadBalance};
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::iter_worker::IterWorker;
use crate::worker::types::{
    BatchFsmState, IterCursor, TransferPlan, WorkerConfig, WorkerEvent, WorkerFsmState, WorkerMsg,
    WorkerStatus,
};

/// One pull actively flowing through the shared cluster. Carries everything
/// the worker needs at promote time: the request id (→ `pending_decodes`),
/// the cluster-reported `pull_end` (→ wake-up scheduling), and the prefill
/// worker that still holds this request's KV (→ ack target once the pull
/// lands, so the prefill side can free its held capacity).
/// Pull backlog ceiling as a fraction of `attn_kv_bytes`. The "backlog" is
/// in-transit bytes + landed-but-not-yet-active bytes; above this, the worker
/// stops fetching new handoffs so the active decode batch can drain before
/// more KV piles up. A single request whose KV alone exceeds the budget is
/// allowed in only when the backlog is empty (single-req exception — prevents
/// starvation of big requests).
const PULL_BUDGET_FRAC: f64 = 0.05;

#[derive(Clone, Copy, Debug)]
struct InFlightPull {
    req: RequestId,
    pull_end: Time,
    prefill_worker: WorkerId,
    /// KV tokens this pull is bringing in — counted toward the backlog the
    /// instant it lands in `pending_decodes`. Token-based to match `KvPool`'s
    /// accounting; wire bytes are computed on the fly at `submit_transfer`.
    tokens: u64,
}

struct DecodeRuntime {
    /// Handoff queue: requests whose prefill is done AND KV is resident, awaiting
    /// a decode slot. Direct-`Request` admits land here too (test path / cluster-
    /// free runs treat the transfer as instant).
    pending_decodes: VecDeque<RequestId>,
    /// Sum of per-request KV tokens for everything in `pending_decodes`. Kept
    /// incrementally — paired with each push/pop — so the fetch-throttle gate
    /// is O(1). Decoupled from `Batch`'s own admission accounting (which is
    /// per-shard); this is the worker-level *backlog* metric. Token-based so
    /// it shares units with `pull_budget_tokens` and the `KvPool` capacity.
    pending_decodes_tokens: u64,
    /// Handoffs newly delivered to this worker, awaiting submission to the shared
    /// cluster. Submitted at the next `tick_inner` (no `now` at `enqueue` time).
    /// **Not counted** toward the backlog: a queued handoff isn't fetching yet,
    /// so it isn't occupying KV cache on this worker. Backpressure is delivered
    /// implicitly — once the backlog hits the budget, nothing in `pending_pulls`
    /// moves until the active batch drains and frees room.
    pending_pulls: VecDeque<(RequestId, TransferPlan)>,
    /// The single pull currently flowing through the cluster (0 or 1). A NCCL
    /// collective is strictly sequential at one comm, so the worker keeps at
    /// most one in flight; subsequent handoffs wait in `pending_pulls` until
    /// the active one's `pull_end` is reached. Promoted to `pending_decodes`
    /// once `now >= pull_end`. Its `tokens` count toward the backlog (KV is
    /// physically landing during the transfer).
    in_transit: Option<InFlightPull>,
    /// Backlog ceiling in tokens (`PULL_BUDGET_FRAC * total_shard_tokens`).
    /// Computed once at construction.
    pull_budget_tokens: u64,
    /// Shared run-level transfer oracle. Always present (handed in at `new`);
    /// the previous `Option` / `KvPuller::set_cluster` two-phase wiring was
    /// dropped now that every worker takes the cluster at construction.
    cluster: SharedGpuCluster,
    request_to_group: HashMap<RequestId, u16>,
    iter_counter: u32,
    iter_compute_start: Time,
    worker_fsm_state: WorkerFsmState,
    batch_fsm_state: BatchFsmState,
    /// Cursor for routing handed-off requests across the N decode groups.
    balance: LoadBalance,
}

impl DecodeRuntime {
    fn new(balance: LoadBalance, cluster: SharedGpuCluster, pull_budget_tokens: u64) -> Self {
        Self {
            pending_decodes: VecDeque::new(),
            pending_decodes_tokens: 0,
            pending_pulls: VecDeque::new(),
            in_transit: None,
            pull_budget_tokens,
            cluster,
            request_to_group: HashMap::new(),
            iter_counter: 0,
            iter_compute_start: Time::ZERO,
            worker_fsm_state: WorkerFsmState::Idle,
            batch_fsm_state: BatchFsmState {
                cursor: IterCursor::Done,
                compute_end: Time::ZERO,
            },
            balance,
        }
    }

    /// Current backlog: KV tokens that this worker has already pulled (landed
    /// in `pending_decodes`) or is actively pulling (`in_transit`). A handoff
    /// queued in `pending_pulls` is **not** part of the backlog — its tokens
    /// haven't started arriving yet.
    fn backlog_tokens(&self) -> u64 {
        self.pending_decodes_tokens + self.in_transit.map(|p| p.tokens).unwrap_or(0)
    }
}

pub struct PdDecodeWorker<M: IterwiseUnifiedModel> {
    pub id: WorkerId,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    runtime: DecodeRuntime,
    /// One container per attention DP shard; length = `model.num_attn_dp_groups()`.
    batches: Vec<Batch>,
    cost_logger: Option<CostLogger>,
    cost_slots: Vec<LeafMetrics>,
    cost_scratch: Vec<LeafMetrics>,
    cost_groups: Vec<GroupInputLog>,
    cost_slot_inputs: Vec<SlotInput>,
    /// Reused per-iteration arch input. Refilled in place each forward pass
    /// (`fill_arch_input`) so the hot decode loop allocates ~nothing — the model
    /// reads it by reference; the cost log snapshots scalars out of it.
    arch_buf: UnifiedArchInput,
    /// Destination side of every incoming PD handoff = this worker's own comm
    /// group id (registered once at construction; covers the attn-shard prefix
    /// of its GPU range, sized by the decode model's `num_attn_shards()`).
    /// Stamped into each `TransferPlan` enqueued by `enqueue(Handoff)`.
    recv_gid: u16,
}

impl<M: IterwiseUnifiedModel> PdDecodeWorker<M> {
    /// Decode self-registers its GPU block in the shared cluster, pre-resolves
    /// its destination block (the first `num_attn_shards` GPUs of its own
    /// range), and **keeps the cluster handle** for runtime `submit_transfer`.
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
        let recv_gid = {
            let mut c = cluster.borrow_mut();
            let gpu_base = c.allocate(pool.0, id.0, model.gpus_per_replica(), gpu_name);
            // Arch invariant: `num_attn_shards() ≤ gpus_per_replica`, so the
            // attn-shard prefix is the comm group used as recv endpoint.
            c.register_comm_group(gpu_base, model.num_attn_shards().max(1))
        };
        let num_groups = model.num_attn_dp_groups().max(1) as usize;
        // Per-attn-shard physical memory (worker has `num_attn_dp_groups` of
        // these). KV bytes are summed across the `num_attn_shards` GPUs that
        // make up one shard set, so dividing by the model-level
        // `total_kv_bytes_per_token` yields shard capacity in tokens.
        let group_kv_bytes =
            config.attn_kv_bytes.saturating_mul(model.num_attn_shards().max(1) as u64);
        let total_shard_tokens =
            (group_kv_bytes / model.total_kv_bytes_per_token().max(1)).max(1);
        // Carve the backlog slice out of the active-decode budget so the two
        // accounts don't double-spend the same KV: `pull_budget_tokens` is the
        // backlog cap; `Batch` sees only the complement.
        let pull_budget_tokens =
            ((total_shard_tokens as f64 * PULL_BUDGET_FRAC) as u64).max(1);
        let kv_capacity = total_shard_tokens.saturating_sub(pull_budget_tokens).max(1);
        let batches: Vec<Batch> = (0..num_groups)
            .map(|g| Batch::new(g as u16, kv_capacity))
            .collect();
        // Round-robin handed-off requests across the groups (config.balance is a
        // hint; a single-group degenerate still works since choose(1) == 0).
        let balance = match config.balance {
            LoadBalance::Single if num_groups > 1 => LoadBalance::RoundRobin { next: 0 },
            other => other,
        };
        let cost_logger = match cost_log_dir {
            Some(dir) => match CostLogger::open(&dir, pool_tag, id, &model.cost_log_manifest()) {
                Ok(logger) => Some(logger),
                Err(e) => {
                    tracing::warn!("cost_log disabled: failed to open writer: {e}");
                    None
                }
            },
            None => None,
        };
        Self {
            id,
            model,
            requests,
            config,
            runtime: DecodeRuntime::new(balance, cluster, pull_budget_tokens),
            batches,
            cost_logger,
            cost_slots: Vec::new(),
            cost_scratch: Vec::new(),
            cost_groups: Vec::new(),
            cost_slot_inputs: Vec::new(),
            arch_buf: UnifiedArchInput::default(),
            recv_gid,
        }
    }

    /// Pull FSM: drive at most one transfer through the cluster at a time. Each
    /// call (a) promotes the in-flight pull to `pending_decodes` if its KV is
    /// now resident, then (b) if no pull is in flight, submits the next pending
    /// one. The loop wraps both so an instant transfer (zero-byte / zero-cost
    /// path) doesn't park itself in `in_transit` for a tick. Keeping at most one
    /// in flight mirrors the cluster's serialization (a comm group's `recv_free`
    /// queues subsequent transfers anyway) and reads more directly: a glance at
    /// `in_transit.is_some()` answers "am I waiting on a pull".
    ///
    /// Runs at the top of every tick — independent of the decode iteration FSM,
    /// so a pull can complete mid-iter and admit at the next iteration boundary.
    fn advance_pulls(&mut self, now: Time, events: &mut Vec<WorkerEvent>) {
        loop {
            if let Some(pull) = self.runtime.in_transit {
                if now >= pull.pull_end {
                    self.runtime.pending_decodes.push_back(pull.req);
                    self.runtime.pending_decodes_tokens += pull.tokens;
                    self.runtime.in_transit = None;
                    // Ack the prefill side: its held reservation can drop now
                    // that the KV has fully landed here. L6 routes this back
                    // by `prefill_worker` (not placement-chosen).
                    events.push(WorkerEvent::PullComplete {
                        worker: self.id,
                        req: pull.req,
                        prefill_worker: pull.prefill_worker,
                    });
                } else {
                    return;
                }
            }
            // Slot free — peek the next handoff and apply the backlog gate
            // before consuming it (so a held-back pull stays at the front of
            // the queue for the next tick to retry).
            let Some(&(_, ref next)) = self.runtime.pending_pulls.front() else {
                return;
            };
            let head_tokens = next.tokens;
            // Submission gate: keep the pull backlog (in-transit + landed-but-
            // not-yet-active) under the configured fraction of shard token
            // capacity. Single-request exception: if the backlog is empty and
            // one request alone exceeds the budget, still fetch it — otherwise
            // big requests would starve permanently.
            let backlog = self.runtime.backlog_tokens();
            let budget = self.runtime.pull_budget_tokens;
            let fits_normally = backlog + head_tokens <= budget;
            let single_req_exception = backlog == 0 && head_tokens > budget;
            if !fits_normally && !single_req_exception {
                return;
            }
            let (rid, transfer) = self.runtime.pending_pulls.pop_front().unwrap();
            // Convert tokens → wire bytes for the cluster's transfer cost model.
            // This is the only place bytes appear inside the worker.
            let bytes = transfer
                .tokens
                .saturating_mul(self.model.total_kv_bytes_per_token());
            let pull_end = self.runtime.cluster.borrow_mut().submit_transfer(
                now,
                transfer.send_gid,
                transfer.recv_gid,
                bytes,
            );
            self.runtime.in_transit = Some(InFlightPull {
                req: rid,
                pull_end,
                prefill_worker: transfer.prefill_worker,
                tokens: transfer.tokens,
            });
            // Loop back: if pull_end <= now (instant transfer), promote it this
            // same tick rather than parking a finished pull in `in_transit`.
        }
    }

    fn tick_inner(&mut self, now: Time, events: &mut Vec<WorkerEvent>) -> Option<Time> {
        self.advance_pulls(now, events);
        use IterCursor::*;
        use WorkerFsmState::*;
        loop {
            let should_continue = match self.runtime.worker_fsm_state {
                Idle => self.tick_idle(),
                Active => match self.runtime.batch_fsm_state.cursor {
                    NotStarted => self.tick_not_started(now),
                    Computing => self.tick_computing(now),
                    Done => self.tick_done(now, events),
                },
            };
            if !should_continue {
                break;
            }
        }
        self.next_wakeup(now)
    }

    fn next_wakeup(&self, now: Time) -> Option<Time> {
        // Two independent pipelines run in this worker — the pull side
        // (`advance_pulls`) and the decode FSM (`tick_idle`/`tick_*`). Compute
        // each side's "next interesting time" alone, then take the earlier.

        // Pull side: `advance_pulls` already drained everything it could at
        // `now`, so by here `pending_pulls` is non-empty only when the slot is
        // occupied. The only future event is the in-flight pull becoming
        // resident; once it does, `advance_pulls` can both promote it and submit
        // the next queued handoff.
        let pull_wake = self.runtime.in_transit.map(|p| p.pull_end);

        // Decode FSM side: when does the compute pipeline want to be re-ticked?
        let compute_wake = match self.runtime.worker_fsm_state {
            // Idle only needs to wake if there's a resident request ready to
            // form_batch; bare pulls-in-flight are handled by `pull_wake`.
            WorkerFsmState::Idle => {
                if self.runtime.pending_decodes.is_empty() {
                    None
                } else {
                    Some(now)
                }
            }
            WorkerFsmState::Active => match self.runtime.batch_fsm_state.cursor {
                IterCursor::NotStarted | IterCursor::Done => Some(now),
                IterCursor::Computing => Some(self.runtime.batch_fsm_state.compute_end),
            },
        };

        // Earliest of the two; None only when both pipelines are quiet.
        match (pull_wake, compute_wake) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(t), None) | (None, Some(t)) => Some(t),
            (None, None) => None,
        }
    }

    fn tick_idle(&mut self) -> bool {
        if !self.form_batch() {
            return false;
        }
        self.runtime.worker_fsm_state = WorkerFsmState::Active;
        self.runtime.batch_fsm_state.cursor = IterCursor::NotStarted;
        self.runtime.batch_fsm_state.compute_end = Time::ZERO;
        true
    }

    fn tick_not_started(&mut self, now: Time) -> bool {
        let compute_end = self.start_iter(now);
        self.runtime.batch_fsm_state.cursor = IterCursor::Computing;
        self.runtime.batch_fsm_state.compute_end = compute_end;
        false
    }

    fn tick_computing(&mut self, now: Time) -> bool {
        if now < self.runtime.batch_fsm_state.compute_end {
            return false;
        }
        self.runtime.batch_fsm_state.cursor = IterCursor::Done;
        true
    }

    fn tick_done(&mut self, now: Time, events: &mut Vec<WorkerEvent>) -> bool {
        self.complete_iter(now, events);
        self.runtime.worker_fsm_state = WorkerFsmState::Idle;
        true
    }

    // ── Stage 1: form_batch — admit one handed-off request straight into decode ─

    fn form_batch(&mut self) -> bool {
        let had_decode = self
            .batches
            .iter()
            .any(|b| b.iter_decoding().next().is_some());

        // A handed-off request is already prefilled: its prompt KV is resident and
        // its first token was emitted by the prefill pool. Admit it directly into a
        // balance-chosen decode group (no prefill compute), sizing KV from the
        // prompt + the remaining decode budget.
        if let Some(&rid) = self.runtime.pending_decodes.front() {
            let (prompt_kv, remaining) = {
                let store = self.requests.borrow();
                let r = &store[rid];
                let prompt_kv = (r.prompt_len + r.prefix_kv) as u64;
                let remaining = r.decode_len.saturating_sub(r.tokens_emitted);
                (prompt_kv, remaining)
            };
            let gid = self.runtime.balance.choose(self.batches.len()) as u16;
            // Admission gate: the request's peak occupancy is prompt + remaining
            // decode tokens. `try_admit` takes (prefill, decode) token counts; here
            // "prefill" is the already-resident prompt KV that loading reserves.
            if remaining == 0
                || self.config.admission.try_admit(
                    &self.batches[gid as usize],
                    0,
                    prompt_kv as u32,
                    remaining,
                )
            {
                self.runtime.pending_decodes.pop_front();
                // Drain this request's tokens from the backlog counter: it
                // moved from `pending_decodes` into an active `Batch`, where
                // `KvPool` admission tracks it now.
                self.runtime.pending_decodes_tokens =
                    self.runtime.pending_decodes_tokens.saturating_sub(prompt_kv);
                self.batches[gid as usize].finalize_to_decode(rid, prompt_kv, remaining);
                self.runtime.request_to_group.insert(rid, gid);
            }
        }

        let still_decoding = self
            .batches
            .iter()
            .any(|b| b.iter_decoding().next().is_some());
        if !had_decode && !still_decoding {
            return false;
        }
        self.runtime.iter_counter += 1;
        true
    }

    fn start_iter(&mut self, now: Time) -> Time {
        self.fill_arch_input();
        let logging_cost = self.cost_logger.is_some();
        let agg = if logging_cost {
            self.model.eval_iter_with_inputs(
                &self.arch_buf,
                &mut self.cost_slots,
                &mut self.cost_scratch,
                &mut self.cost_slot_inputs,
            )
        } else {
            self.model
                .eval_iter(&self.arch_buf, &mut self.cost_slots, &mut self.cost_scratch)
        };
        let cost_time = Time::from_ms(agg.m.time_ms as f64);
        if self.cost_logger.is_some() {
            // Snapshot scalars out of the reused `arch_buf` into the reused
            // `cost_groups` buffer (refilled in place — no per-iter `Vec` alloc);
            // the per-slot time/coverage/input breakdowns are appended into the
            // logger's flat chunk buffers by `record`, so the entry owns no `Vec`.
            // `prefill_chunk_pairs` is empty on a decode worker, so building each
            // `GroupInputLog` does not allocate.
            self.cost_groups.clear();
            for g in &self.arch_buf.groups {
                self.cost_groups.push(GroupInputLog {
                    batch_tokens: g.batch_tokens,
                    prefill_tokens: g.prefill_tokens,
                    decode_request_count: g.decode_tokens,
                    decode_kv_total: g.total_kv_len,
                    prefill_chunk_pairs: g.prefill_chunk_pairs.clone(),
                });
            }
            let entry = CostLogEntry {
                worker_id: self.id.0,
                iter_id: self.runtime.iter_counter as u64,
                batch_id: 0,
                wall_start_ms: now.as_ms(),
                total_time_ms: agg.m.time_ms as f64,
                energy_j: agg.m.energy_j as f64,
                group_len: 0,
                slot_len: 0,
                slot_input_len: 0,
            };
            if let Some(logger) = self.cost_logger.as_mut() {
                if let Err(e) = logger.record(
                    entry,
                    &self.cost_slots,
                    &mut self.cost_groups,
                    &mut self.cost_slot_inputs,
                ) {
                    tracing::warn!("cost_log record failed: {e}");
                }
            }
        }
        self.runtime.iter_compute_start = now;
        now + cost_time
    }

    // ── Stage 3: complete_iter — decode bookkeeping only (no prefill phase) ──────

    fn complete_iter(&mut self, now: Time, events: &mut Vec<WorkerEvent>) {
        let log_tokens = self.config.log_output_token_times;
        for gid in 0..self.batches.len() {
            let mut completed: Vec<RequestId> = Vec::new();

            // (a) live decodes produced one token.
            {
                let mut store = self.requests.borrow_mut();
                for (rid, _) in self.batches[gid].iter_decoding() {
                    let r = &mut store[rid];
                    r.record_token(now, log_tokens);
                    if r.is_complete() {
                        completed.push(rid);
                    }
                }
            }

            // (b) bump KV for every live decode.
            self.batches[gid].advance_decodes();

            // (c) emit + release completed requests.
            for rid in completed {
                let current_kv = self.batches[gid]
                    .decodes
                    .iter()
                    .find(|(r, _)| *r == rid)
                    .map(|(_, s)| s.current_kv)
                    .unwrap_or(0);
                self.batches[gid].release(rid, current_kv);
                self.runtime.request_to_group.remove(&rid);
                events.push(WorkerEvent::RequestComplete {
                    worker: self.id,
                    req: rid,
                });
            }
        }
    }

    // ── build_arch_input — one decode-only ArchGroupInput per DP shard ──────────

    /// Refill the reused `arch_buf` in place from the current batches. `groups`
    /// stays at `batches.len()` (stable for a worker) so `resize_with` is a no-op
    /// after warmup; each group's `decode_kv_lens` keeps its capacity via `clear`.
    fn fill_arch_input(&mut self) {
        let n = self.batches.len();
        self.arch_buf.groups.resize_with(n, ArchGroupInput::default);
        self.arch_buf.tokens_per_source_rank.clear();
        for (g, b) in self.arch_buf.groups.iter_mut().zip(self.batches.iter()) {
            g.clear();
            let mut total_kv = 0u64;
            for (_, s) in b.iter_decoding() {
                g.decode_kv_lens.push(s.current_kv as u32);
                total_kv += s.current_kv;
            }
            let decode_tokens = g.decode_kv_lens.len() as u32;
            g.batch_tokens = decode_tokens;
            g.decode_tokens = decode_tokens;
            g.total_kv_len = total_kv as u32;
        }
    }

    pub fn release_request(&mut self, rid: RequestId, current_kv: u64) -> Option<u16> {
        if let Some(pos) = self.runtime.pending_decodes.iter().position(|&x| x == rid) {
            self.runtime.pending_decodes.remove(pos);
            // Mirror the increment in `enqueue(Request)` / `advance_pulls`'s
            // promote step: a request leaving the backlog should free its
            // share of the budget.
            let tokens = {
                let store = self.requests.borrow();
                let r = &store[rid];
                r.prompt_len as u64 + r.prefix_kv as u64
            };
            self.runtime.pending_decodes_tokens =
                self.runtime.pending_decodes_tokens.saturating_sub(tokens);
            return None;
        }
        if let Some(gid) = self.runtime.request_to_group.remove(&rid) {
            self.batches[gid as usize].release(rid, current_kv);
            return Some(gid);
        }
        None
    }
}

impl<M: IterwiseUnifiedModel> IterWorker for PdDecodeWorker<M> {
    fn id(&self) -> WorkerId {
        self.id
    }

    fn enqueue(&mut self, msg: WorkerMsg) {
        match msg {
            // Direct admit (e.g. tests): KV treated as instantly resident.
            // Still count it toward the backlog so the gate's accounting is
            // consistent — tests use tiny token counts so the budget never bites.
            WorkerMsg::Request(rid) => {
                let tokens = {
                    let store = self.requests.borrow();
                    let r = &store[rid];
                    r.prompt_len as u64 + r.prefix_kv as u64
                };
                self.runtime.pending_decodes.push_back(rid);
                self.runtime.pending_decodes_tokens += tokens;
            }
            // PD handoff: the wire message only carries the sender side; fill
            // in this worker's own destination block (pre-resolved at
            // construction) to assemble the full TransferPlan. Deferred submit
            // happens at the next `tick_inner` (no `now` available here);
            // `advance_pulls` then routes it to the cluster or, cluster-free,
            // straight to `pending_decodes`.
            WorkerMsg::Handoff { req, send_gid, tokens, prefill_worker } => {
                self.runtime.pending_pulls.push_back((
                    req,
                    TransferPlan {
                        send_gid,
                        recv_gid: self.recv_gid,
                        tokens,
                        prefill_worker,
                    },
                ));
            }
            WorkerMsg::ReleaseKv { .. } => {
                unreachable!("decode worker holds no KV at the source side; ack is its emitter")
            }
        }
    }

    fn tick(&mut self, now: Time, events: &mut Vec<WorkerEvent>) -> Option<Time> {
        self.tick_inner(now, events)
    }

    fn status(&self) -> WorkerStatus {
        let live_decodes: u32 = self
            .batches
            .iter()
            .map(|b| b.iter_decoding().count() as u32)
            .sum();
        // Both pulls (pending-submit + the at-most-one in-flight) count as
        // queued — placement sees them as load just like ready-to-admit decodes.
        let queued = self.runtime.pending_decodes.len()
            + self.runtime.pending_pulls.len()
            + usize::from(self.runtime.in_transit.is_some());
        WorkerStatus {
            queued_requests: queued as u32,
            active_requests: live_decodes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{PoolId, Request, RequestStore};
    use crate::timing::cache::interp::{CoverageFlags, Metrics4};
    use crate::worker::gpu_cluster::{CostSource, GpuCluster, SharedGpuCluster};
    use std::cell::RefCell;
    use std::rc::Rc;

    fn test_cluster() -> SharedGpuCluster {
        Rc::new(RefCell::new(GpuCluster::new(CostSource::analytic(1.0))))
    }

    struct FakeModel {
        ms: f64,
        dp_groups: u16,
    }
    impl IterwiseUnifiedModel for FakeModel {
        fn eval_iter(
            &self,
            b: &UnifiedArchInput,
            slots: &mut Vec<LeafMetrics>,
            _scratch: &mut Vec<LeafMetrics>,
        ) -> LeafMetrics {
            slots.clear();
            // The worker must feed exactly one decode group per DP shard.
            assert_eq!(b.groups.len(), self.dp_groups as usize);
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

    /// A request whose prefill is "already done": prompt set, first token emitted.
    fn prefilled_store(reqs: &[(u32, u32, u32)]) -> SharedRequests {
        let store = Rc::new(RefCell::new(RequestStore::new()));
        for &(id, prompt, decode) in reqs {
            let r = Request::new(RequestId(id), prompt, decode, Time::ZERO);
            store.borrow_mut().insert(&r);
            // Mimic the prefill pool's handoff state on the store record: prefill
            // processed, first token already emitted.
            let mut s = store.borrow_mut();
            let rec = &mut s[RequestId(id)];
            rec.prefill_processed = prompt;
            rec.tokens_emitted = 1;
            rec.first_token_time = Some(Time::ZERO);
            rec.last_token_time = Some(Time::ZERO);
        }
        store
    }

    fn worker(store: SharedRequests) -> PdDecodeWorker<FakeModel> {
        worker_dp(store, 1)
    }

    fn worker_dp(store: SharedRequests, dp_groups: u16) -> PdDecodeWorker<FakeModel> {
        PdDecodeWorker::new(
            WorkerId(0),
            "decode",
            Arc::new(FakeModel { ms: 1.0, dp_groups }),
            store,
            WorkerConfig::default(),
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        )
    }

    #[test]
    fn handed_off_request_decodes_to_completion() {
        // prompt 16, decode 3 → prefill already emitted token 1, decode emits 2 more.
        let store = prefilled_store(&[(0, 16, 3)]);
        let mut w = worker(Rc::clone(&store));
        w.enqueue(WorkerMsg::Request(RequestId(0)));
        let mut events = Vec::new();
        for step in 0..50u64 {
            w.tick(Time::from_ms(step as f64), &mut events);
        }
        assert_eq!(
            events,
            vec![WorkerEvent::RequestComplete {
                worker: WorkerId(0),
                req: RequestId(0)
            }]
        );
        let s = store.borrow();
        let r = &s[RequestId(0)];
        assert!(r.completed);
        assert_eq!(r.tokens_emitted, 3);
    }

    #[test]
    fn multiple_handed_off_requests_all_complete() {
        let store = prefilled_store(&[(0, 8, 2), (1, 8, 2), (2, 8, 2)]);
        let mut w = worker(Rc::clone(&store));
        for id in [0, 1, 2] {
            w.enqueue(WorkerMsg::Request(RequestId(id)));
        }
        let mut events = Vec::new();
        for step in 0..200u64 {
            w.tick(Time::from_ms(step as f64), &mut events);
        }
        assert_eq!(events.len(), 3);
        let s = store.borrow();
        for id in [0, 1, 2] {
            assert!(s[RequestId(id)].completed, "req {id} should complete");
        }
    }

    #[test]
    fn builds_one_decode_group_per_dp_shard() {
        let w = worker_dp(prefilled_store(&[]), 2);
        assert_eq!(w.batches.len(), 2);
        // fill_arch_input emits one group per shard (also asserted in eval_iter).
        let mut w = w;
        w.fill_arch_input();
        assert_eq!(w.arch_buf.groups.len(), 2);
    }

    #[test]
    fn dp_decode_round_robins_handoffs_across_shards() {
        // Two long-decode handed-off requests, two DP shards: RR routes one to each.
        let store = prefilled_store(&[(0, 4, 50), (1, 4, 50)]);
        let mut w = worker_dp(store, 2);
        w.enqueue(WorkerMsg::Request(RequestId(0)));
        w.enqueue(WorkerMsg::Request(RequestId(1)));
        let mut events = Vec::new();
        for step in 0..10u64 {
            w.tick(Time::from_ms(step as f64), &mut events);
        }
        let g0 = w.batches[0].iter_decoding().count();
        let g1 = w.batches[1].iter_decoding().count();
        assert_eq!(
            (g0, g1),
            (1, 1),
            "RoundRobin must place one decode in each DP shard"
        );
    }

    /// Register a 1-link sender comm group in `cluster` for use as the `send_gid`
    /// in a synthetic `Handoff`. Allocates one dummy GPU first so the registered
    /// group's `base` lines up with a real GPU id.
    fn register_test_sender(cluster: &SharedGpuCluster) -> u16 {
        let mut c = cluster.borrow_mut();
        c.allocate(99, 99, 1, "sender-gpu");
        c.register_comm_group(0, 1)
    }

    /// Pull backlog gate: a second handoff that would push (in-flight + landed-
    /// but-not-yet-active) tokens above 5% × shard capacity must stay in
    /// `pending_pulls` until the first drains. With `attn_kv_bytes=1000` and
    /// `total_kv_bytes_per_token=1` → 1000-token shard → 50-token budget; a
    /// 40-token + 20-token pair triggers the gate: 40 in-flight + 20 head > 50.
    #[test]
    fn pull_backlog_gate_holds_second_handoff_until_first_drains() {
        let store = prefilled_store(&[(0, 40, 1), (1, 20, 1)]);
        let cluster = test_cluster();
        let sender_gid = register_test_sender(&cluster);
        let mut w = PdDecodeWorker::new(
            WorkerId(0),
            "decode",
            Arc::new(FakeModel { ms: 1.0, dp_groups: 1 }),
            Rc::clone(&store),
            WorkerConfig { attn_kv_bytes: 1000, ..WorkerConfig::default() },
            None,
            PoolId(0),
            "test-gpu",
            Rc::clone(&cluster),
        );
        for &id in &[0u32, 1] {
            w.enqueue(WorkerMsg::Handoff {
                req: RequestId(id),
                send_gid: sender_gid,
                tokens: if id == 0 { 40 } else { 20 },
                prefill_worker: WorkerId(99),
            });
        }
        let mut events = Vec::new();
        // First tick at t=0: req0 submits (40 ≤ 50). req1 would push backlog to
        // 60 → gated, stays in pending_pulls.
        w.tick(Time::ZERO, &mut events);
        assert!(
            w.runtime.in_transit.is_some_and(|p| p.req == RequestId(0)),
            "req0 should be the in-flight pull"
        );
        assert_eq!(
            w.runtime.pending_pulls.len(),
            1,
            "req1 must stay in pending_pulls — gated by 40-token backlog"
        );
    }

    /// Single-request exception: a handoff whose KV alone exceeds the 5% budget
    /// would otherwise starve forever. When the backlog is empty, the gate lets
    /// it through.
    #[test]
    fn pull_backlog_single_req_exception_when_backlog_empty() {
        let store = prefilled_store(&[(0, 200, 1)]); // 200 tokens > budget (50).
        let cluster = test_cluster();
        let sender_gid = register_test_sender(&cluster);
        let mut w = PdDecodeWorker::new(
            WorkerId(0),
            "decode",
            Arc::new(FakeModel { ms: 1.0, dp_groups: 1 }),
            Rc::clone(&store),
            WorkerConfig { attn_kv_bytes: 1000, ..WorkerConfig::default() },
            None,
            PoolId(0),
            "test-gpu",
            Rc::clone(&cluster),
        );
        w.enqueue(WorkerMsg::Handoff {
            req: RequestId(0),
            send_gid: sender_gid,
            tokens: 200,
            prefill_worker: WorkerId(99),
        });
        let mut events = Vec::new();
        w.tick(Time::ZERO, &mut events);
        assert!(
            w.runtime.in_transit.is_some_and(|p| p.req == RequestId(0)),
            "single big req must go through the exception path"
        );
        assert!(w.runtime.pending_pulls.is_empty());
    }

    #[test]
    fn dp_decode_all_complete() {
        let store = prefilled_store(&[(0, 8, 2), (1, 8, 2), (2, 8, 2), (3, 8, 2)]);
        let mut w = worker_dp(Rc::clone(&store), 2);
        for id in 0..4u32 {
            w.enqueue(WorkerMsg::Request(RequestId(id)));
        }
        let mut events = Vec::new();
        for step in 0..200u64 {
            w.tick(Time::from_ms(step as f64), &mut events);
        }
        assert_eq!(events.len(), 4);
        let s = store.borrow();
        for id in 0..4u32 {
            assert!(s[RequestId(id)].completed, "req {id} should complete");
        }
    }
}
