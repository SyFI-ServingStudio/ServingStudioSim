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
use crate::common::{RequestId, SharedRequests, Time, WorkerId};
use crate::log::{CostLogEntry, CostLogger, GroupInputLog};
use crate::timing::{LeafMetrics, SlotInput};
use crate::worker::admission_helpers::{Batch, LoadBalance};
use crate::worker::iter_worker::IterWorker;
use crate::worker::types::{
    BatchFsmState, IterCursor, WorkerConfig, WorkerEvent, WorkerFsmState, WorkerMsg, WorkerStatus,
};

struct DecodeRuntime {
    /// Handoff queue: requests whose prefill is done, awaiting a decode slot.
    pending_decodes: VecDeque<RequestId>,
    request_to_group: HashMap<RequestId, u16>,
    iter_counter: u32,
    iter_compute_start: Time,
    worker_fsm_state: WorkerFsmState,
    batch_fsm_state: BatchFsmState,
    /// Cursor for routing handed-off requests across the N decode groups.
    balance: LoadBalance,
}

impl DecodeRuntime {
    fn new(balance: LoadBalance) -> Self {
        Self {
            pending_decodes: VecDeque::new(),
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
}

impl<M: IterwiseUnifiedModel> PdDecodeWorker<M> {
    pub fn new(
        id: WorkerId,
        pool_tag: &'static str,
        model: Arc<M>,
        requests: SharedRequests,
        config: WorkerConfig,
        cost_log_dir: Option<PathBuf>,
    ) -> Self {
        let num_groups = model.num_attn_dp_groups().max(1) as usize;
        // Each DP shard is an independent attention TP group with its own KV cache;
        // the per-GPU memory allowance sizes each group's pool identically.
        let kv_capacity = (config.attn_kv_bytes / model.kv_bytes_per_token().max(1)).max(1);
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
            runtime: DecodeRuntime::new(balance),
            batches,
            cost_logger,
            cost_slots: Vec::new(),
            cost_scratch: Vec::new(),
            cost_groups: Vec::new(),
            cost_slot_inputs: Vec::new(),
            arch_buf: UnifiedArchInput::default(),
        }
    }

    fn tick_inner(&mut self, now: Time, events: &mut Vec<WorkerEvent>) -> Option<Time> {
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
        match self.runtime.worker_fsm_state {
            WorkerFsmState::Idle => {
                let status = self.status();
                if status.queued_requests > 0 || status.active_requests > 0 {
                    Some(now)
                } else {
                    None
                }
            }
            WorkerFsmState::Active => match self.runtime.batch_fsm_state.cursor {
                IterCursor::NotStarted | IterCursor::Done => Some(now),
                IterCursor::Computing => Some(self.runtime.batch_fsm_state.compute_end),
            },
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

    // ── build_arch_input — one group, decode-only (no prefill on a decode worker)

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
            WorkerMsg::Request(rid) => self.runtime.pending_decodes.push_back(rid),
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
        WorkerStatus {
            queued_requests: self.runtime.pending_decodes.len() as u32,
            active_requests: live_decodes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{Request, RequestStore};
    use crate::timing::cache::interp::{CoverageFlags, Metrics4};
    use std::cell::RefCell;
    use std::rc::Rc;

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
        fn kv_bytes_per_token(&self) -> u64 {
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
