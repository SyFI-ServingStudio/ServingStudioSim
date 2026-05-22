//! `BareboneWorker` — the minimal viable iter-wise unified worker (L5 design.md
//! §3.4). One request group, `Strict` admission, whole-prefill-in-one-iter, an
//! iter-wise three-state `BatchFsmState` cursor.
//!
//! Reconciliations vs the design example (see plan):
//!   - event-driven surface (`enqueue`/`tick`/`drain_events`/`status`) so the L6
//!     `simple_dp` pool can drive it; `complete_iter` pushes
//!     `WorkerEvent::RequestComplete` to an outbox instead of returning a `Vec`.
//!   - the request slab is the shared `RequestStore`, injected at construction as
//!     `SharedRequests` and borrowed transiently inside each method (no per-tick
//!     `&mut RequestStore` parameter). Logger is deferred to L7.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use crate::arch::contract::{ArchGroupInput, IterwiseUnifiedModel, UnifiedArchInput};
use crate::common::{RequestId, SharedRequests, Time, WorkerId};
use crate::worker::admission_helpers::{Batch, KvAdmission, LoadBalance};

// ── FSM types ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerFsmState {
    Idle,
    Active,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IterCursor {
    NotStarted,
    Computing,
    Done,
}

#[derive(Clone, Copy, Debug)]
pub struct BatchFsmState {
    pub cursor: IterCursor,
    pub compute_end: Time,
}

// ── Messages / events / status (L6 interface) ─────────────────────────────────

#[derive(Clone, Debug)]
pub enum WorkerMsg {
    Request(RequestId),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkerEvent {
    RequestComplete { req: RequestId },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WorkerStatus {
    pub queued_requests: u32,
    pub active_requests: u32,
}

// ── Config / runtime ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct WorkerConfig {
    pub admission: KvAdmission,
    pub balance: LoadBalance,
    /// This worker's KV-cache memory allowance in bytes (its GPU's attention
    /// budget). Same per-worker tier as `gpu_name`; the worker divides it by the
    /// model's `kv_bytes_per_token` to size its `KvPool`.
    pub attn_kv_bytes: u64,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            admission: KvAdmission::Strict,
            balance: LoadBalance::Single,
            attn_kv_bytes: 80_000_000_000, // 80 GB
        }
    }
}

struct WorkerRuntime {
    pending_prefills: VecDeque<RequestId>,
    promised: HashMap<RequestId, (u16, u64)>,
    request_to_group: HashMap<RequestId, u16>,
    iter_counter: u32,
    iter_compute_start: Time,
    worker_fsm_state: WorkerFsmState,
    batch_fsm_state: BatchFsmState,
}

impl WorkerRuntime {
    fn new() -> Self {
        Self {
            pending_prefills: VecDeque::new(),
            promised: HashMap::new(),
            request_to_group: HashMap::new(),
            iter_counter: 0,
            iter_compute_start: Time::ZERO,
            worker_fsm_state: WorkerFsmState::Idle,
            batch_fsm_state: BatchFsmState {
                cursor: IterCursor::Done,
                compute_end: Time::ZERO,
            },
        }
    }
}

// ── Worker ─────────────────────────────────────────────────────────────────────

pub struct BareboneWorker<M: IterwiseUnifiedModel> {
    pub id: WorkerId,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    runtime: WorkerRuntime,
    batches: Vec<Batch>, // length 1 in barebone
    events: Vec<WorkerEvent>,
}

impl<M: IterwiseUnifiedModel> BareboneWorker<M> {
    pub fn new(
        id: WorkerId,
        model: Arc<M>,
        requests: SharedRequests,
        config: WorkerConfig,
    ) -> Self {
        // The worker sizes its own KvPool: memory allowance ÷ the model's
        // per-token KV footprint (L5 owns the division; arch owns the footprint).
        let kv_capacity = (config.attn_kv_bytes / model.kv_bytes_per_token().max(1)).max(1);
        Self {
            id,
            model,
            requests,
            config,
            runtime: WorkerRuntime::new(),
            batches: vec![Batch::new(0, kv_capacity)],
            events: Vec::new(),
        }
    }

    // ── L6-facing surface ────────────────────────────────────────────────────

    pub fn enqueue(&mut self, msg: WorkerMsg) {
        match msg {
            WorkerMsg::Request(rid) => self.runtime.pending_prefills.push_back(rid),
        }
    }

    pub fn drain_events(&mut self) -> Vec<WorkerEvent> {
        std::mem::take(&mut self.events)
    }

    pub fn status(&self) -> WorkerStatus {
        let live_decodes = self.batches[0].iter_decoding().count() as u32;
        let active = live_decodes
            + self.runtime.promised.len() as u32
            + self.batches[0].prefill_admits.len() as u32;
        WorkerStatus {
            queued_requests: self.runtime.pending_prefills.len() as u32,
            active_requests: active,
        }
    }

    // ── tick: state-forwarding loop (§3.2.1) ──────────────────────────────────

    pub fn tick(&mut self, now: Time) {
        use IterCursor::*;
        use WorkerFsmState::*;
        loop {
            match self.runtime.worker_fsm_state {
                Idle => {
                    if !self.form_batch() {
                        break; // no work this tick; quiesce
                    }
                    // Idle → Active/NotStarted (transition owned here, not in form_batch).
                    self.runtime.worker_fsm_state = Active;
                    self.runtime.batch_fsm_state.cursor = NotStarted;
                    self.runtime.batch_fsm_state.compute_end = Time::ZERO;
                    continue; // forward into the Active arm
                }
                Active => match self.runtime.batch_fsm_state.cursor {
                    NotStarted => {
                        let compute_end = self.start_iter(now);
                        // NotStarted → Computing.
                        self.runtime.batch_fsm_state.cursor = Computing;
                        self.runtime.batch_fsm_state.compute_end = compute_end;
                        break; // compute armed; wait for it to elapse
                    }
                    Computing => {
                        if now < self.runtime.batch_fsm_state.compute_end {
                            break; // still computing
                        }
                        self.runtime.batch_fsm_state.cursor = Done; // Computing → Done
                        continue; // forward into the Done arm
                    }
                    Done => {
                        self.complete_iter(now);
                        self.runtime.worker_fsm_state = Idle; // Active → Idle
                        continue; // forward into the Idle arm to form the next batch
                    }
                },
            }
        }
    }

    // ── Stage 1: form_batch (Phase A admit + Phase B drain) ────────────────────

    fn form_batch(&mut self) -> bool {
        let had_decode = self.batches[0].iter_decoding().next().is_some();

        // Phase A: admit one fresh prefill from the queue (barebone: one/iter).
        if let Some(&rid) = self.runtime.pending_prefills.front() {
            let (p, d) = {
                let store = self.requests.borrow();
                let r = &store[rid];
                (r.prompt_len, r.decode_len)
            };
            let group_promised = self.group_promised_kv(0);
            if self
                .config
                .admission
                .try_admit(&self.batches[0], group_promised, p, d)
            {
                self.runtime.pending_prefills.pop_front();
                self.promise(0, rid, p, d, 0);
            }
        }

        // Phase B: drain ready promises into the iter's prefill_admits.
        self.drain_promises_into_admits();
        let had_prefill = !self.batches[0].prefill_admits.is_empty();

        if !had_decode && !had_prefill {
            return false; // nothing admitted; stay Idle
        }

        self.runtime.iter_counter += 1;
        true
    }

    // ── Stage 2: start_iter (build ArchInput, cost query) → compute_end ────────

    fn start_iter(&mut self, now: Time) -> Time {
        let arch_input = self.build_arch_input();
        let cost = self.model.cost_whole_iter(&arch_input);
        self.runtime.iter_compute_start = now;
        now + cost.time
    }

    // ── Stage 3: complete_iter (token bookkeeping + KV transitions) ────────────

    fn complete_iter(&mut self, now: Time) {
        let admits: Vec<RequestId> = self.batches[0].prefill_admits.clone();
        let decode_ids: Vec<RequestId> =
            self.batches[0].iter_decoding().map(|(r, _)| r).collect();

        let mut completed: Vec<RequestId> = Vec::new();

        // (a) live decodes produced one token.
        {
            let mut store = self.requests.borrow_mut();
            for &rid in &decode_ids {
                let r = &mut store[rid];
                r.record_token(now);
                if r.is_complete() {
                    completed.push(rid);
                }
            }
        }

        // (b) bump KV for every live decode.
        self.batches[0].advance_decodes();

        // (c) prefills resolved → first token; complete now or enter decode set.
        let mut to_finalize: Vec<(RequestId, u64, u32)> = Vec::new();
        {
            let mut store = self.requests.borrow_mut();
            for &rid in &admits {
                let r = &mut store[rid];
                r.prefill_processed = r.prompt_len;
                r.record_first_token(now);
                if r.is_complete() {
                    completed.push(rid);
                } else {
                    let kv = (r.prompt_len + r.prefix_kv) as u64;
                    let remaining = r.decode_len.saturating_sub(r.tokens_emitted);
                    to_finalize.push((rid, kv, remaining));
                }
            }
        }
        for (rid, kv, remaining) in to_finalize {
            self.batches[0].finalize_to_decode(rid, kv, remaining);
        }
        self.batches[0].prefill_admits.clear();

        // (d) emit + release completed requests.
        for rid in completed {
            let current_kv = self.batches[0]
                .decodes
                .iter()
                .find(|(r, _)| *r == rid)
                .map(|(_, s)| s.current_kv)
                .unwrap_or(0);
            self.batches[0].release(rid, current_kv);
            self.runtime.request_to_group.remove(&rid);
            self.events.push(WorkerEvent::RequestComplete { req: rid });
        }
    }

    // ── build_arch_input (§3.10) ───────────────────────────────────────────────

    fn build_arch_input(&self) -> UnifiedArchInput {
        let b = &self.batches[0];
        let store = self.requests.borrow();

        let mut prefill_chunk_pairs = Vec::new();
        let mut prefill_tokens = 0u32;
        for &rid in &b.prefill_admits {
            let r = &store[rid];
            // (prefix_len, append_len) — matches FlashInfer attention input.
            prefill_chunk_pairs.push((r.prefix_kv, r.active_chunk_len));
            prefill_tokens += r.active_chunk_len;
        }

        let mut decode_kv_lens = Vec::new();
        let mut total_kv = 0u64;
        for (_, s) in b.iter_decoding() {
            decode_kv_lens.push(s.current_kv as u32);
            total_kv += s.current_kv;
        }
        let decode_tokens = decode_kv_lens.len() as u32;

        let group = ArchGroupInput {
            batch_tokens: prefill_tokens + decode_tokens,
            prefill_tokens,
            decode_tokens,
            prefill_chunk_pairs,
            decode_kv_lens,
            total_kv_len: total_kv as u32,
        };
        UnifiedArchInput {
            groups: vec![group],
            tokens_per_source_rank: Vec::new(),
        }
    }

    // ── lifecycle helpers ──────────────────────────────────────────────────────

    fn promise(&mut self, gid: u16, rid: RequestId, p: u32, d: u32, prefix: u32) {
        self.runtime
            .promised
            .insert(rid, (gid, (p + prefix + d) as u64));
        self.runtime.request_to_group.insert(rid, gid);
        let mut store = self.requests.borrow_mut();
        let r = &mut store[rid];
        r.active_chunk_len = p;
        r.prefix_kv = prefix;
    }

    /// Barebone readiness predicate is always true (KV is local; no remote pulls).
    fn drain_promises_into_admits(&mut self) {
        let drained: Vec<(u16, RequestId)> = self
            .runtime
            .promised
            .iter()
            .map(|(&rid, &(gid, _))| (gid, rid))
            .collect();
        for (gid, rid) in drained {
            self.runtime.promised.remove(&rid);
            self.batches[gid as usize].prefill_admits.push(rid);
        }
    }

    #[inline]
    fn group_promised_kv(&self, gid: u16) -> u64 {
        self.runtime
            .promised
            .values()
            .filter(|(g, _)| *g == gid)
            .map(|(_, t)| *t)
            .sum()
    }

    /// External release (e.g. cancellation). Cleans up wherever the request sits.
    pub fn release_request(&mut self, rid: RequestId, current_kv: u64) -> Option<u16> {
        if let Some(pos) = self
            .runtime
            .pending_prefills
            .iter()
            .position(|&x| x == rid)
        {
            self.runtime.pending_prefills.remove(pos);
            return None;
        }
        if let Some(gid) = self.runtime.request_to_group.remove(&rid) {
            let b = &mut self.batches[gid as usize];
            b.release(rid, current_kv);
            if let Some(p) = b.prefill_admits.iter().position(|&x| x == rid) {
                b.prefill_admits.swap_remove(p);
            }
            self.runtime.promised.remove(&rid);
            return Some(gid);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::time::Time;
    use crate::common::{Request, RequestStore};
    use crate::timing::LookupResult;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Fixed-cost stand-in for an L4 model — drives the FSM without Python/bridge.
    struct FakeModel {
        ms: f64,
    }
    impl IterwiseUnifiedModel for FakeModel {
        fn cost_whole_iter(&self, _batch: &UnifiedArchInput) -> LookupResult {
            LookupResult::leaf("fake", Time::from_ms(self.ms), 0, 0, 0.0, Vec::new())
        }
        // 1 byte/token → KvPool capacity == config.attn_kv_bytes (easy to size).
        fn kv_bytes_per_token(&self) -> u64 {
            1
        }
    }

    fn shared_with(reqs: &[(u32, u32, u32)]) -> crate::common::SharedRequests {
        let store = Rc::new(RefCell::new(RequestStore::new()));
        for &(id, prompt, decode) in reqs {
            store
                .borrow_mut()
                .insert(&Request::new(RequestId(id), prompt, decode, Time::ZERO));
        }
        store
    }

    /// Drive ticks at coarse 1ms steps until no work remains or a step cap hits.
    fn run_to_quiescence<M: IterwiseUnifiedModel>(
        w: &mut BareboneWorker<M>,
        max_steps: u64,
    ) -> Vec<WorkerEvent> {
        let mut all = Vec::new();
        for step in 0..max_steps {
            w.tick(Time::from_ms(step as f64));
            all.extend(w.drain_events());
        }
        all
    }

    #[test]
    fn single_request_prefill_then_decode_completes() {
        let store = shared_with(&[(1, 16, 3)]); // prompt 16, 3 decode tokens
        let model = Arc::new(FakeModel { ms: 1.0 });
        let mut w = BareboneWorker::new(
            WorkerId(0),
            model,
            Rc::clone(&store),
            WorkerConfig::default(),
        );
        w.enqueue(WorkerMsg::Request(RequestId(1)));

        let events = run_to_quiescence(&mut w, 50);
        assert_eq!(
            events,
            vec![WorkerEvent::RequestComplete {
                req: RequestId(1)
            }]
        );
        let s = store.borrow();
        let r = &s[RequestId(1)];
        assert!(r.completed);
        assert_eq!(r.tokens_emitted, 3); // first token + 2 decode tokens
        assert!(r.first_token_time.is_some());
    }

    #[test]
    fn three_requests_all_complete() {
        let store = shared_with(&[(1, 8, 2), (2, 8, 2), (3, 8, 2)]);
        let model = Arc::new(FakeModel { ms: 1.0 });
        let mut w = BareboneWorker::new(
            WorkerId(0),
            model,
            Rc::clone(&store),
            WorkerConfig::default(),
        );
        for id in [1, 2, 3] {
            w.enqueue(WorkerMsg::Request(RequestId(id)));
        }
        let events = run_to_quiescence(&mut w, 200);
        assert_eq!(events.len(), 3);
        let s = store.borrow();
        for id in [1, 2, 3] {
            assert!(s[RequestId(id)].completed, "req {id} should complete");
        }
    }
}
