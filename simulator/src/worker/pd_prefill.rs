//! `PdPrefillWorker` — the prefill half of a PD (prefill/decode disaggregation)
//! deployment. Iter-wise like the barebone worker, and admits + costs prefills the
//! same way, but at iter end it does **not** keep the request to decode: it emits
//! `WorkerEvent::PrefillDone` so L6 hands the request (its prompt KV already
//! computed) off to a decode pool, and releases its own transient state.
//!
//! v1 simplification: the prefill→decode KV transfer is not modeled (the handoff
//! is a control-plane event only; the shared `RequestStore` carries the request's
//! prefill state across pools). The local KvPool therefore stays empty — a prefill
//! worker reserves KV only transiently during admission, never finalizes a decode.
//!
//! Copied from `BareboneWorker` (build-first; the shared iter-wise core will be
//! factored once the four iter-wise workers exist). Only `complete_iter` differs.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::{ArchGroupInput, IterwiseUnifiedModel, UnifiedArchInput};
use crate::common::{RequestId, SharedRequests, Time, WorkerId};
use crate::log::{CostLogEntry, CostLogger, GroupInputLog};
use crate::timing::{LeafMetrics, SlotInput};
use crate::worker::admission_helpers::Batch;
use crate::worker::iter_worker::IterWorker;
use crate::worker::types::{
    BatchFsmState, IterCursor, WorkerConfig, WorkerEvent, WorkerFsmState, WorkerMsg, WorkerStatus,
};

struct PrefillRuntime {
    pending_prefills: VecDeque<RequestId>,
    promised: HashMap<RequestId, (u16, u64)>,
    request_to_group: HashMap<RequestId, u16>,
    iter_counter: u32,
    iter_compute_start: Time,
    worker_fsm_state: WorkerFsmState,
    batch_fsm_state: BatchFsmState,
}

impl PrefillRuntime {
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

pub struct PdPrefillWorker<M: IterwiseUnifiedModel> {
    pub id: WorkerId,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    runtime: PrefillRuntime,
    batches: Vec<Batch>, // length 1 (prefill is single-group in v1)
    events: Vec<WorkerEvent>,
    cost_logger: Option<CostLogger>,
    cost_slots: Vec<LeafMetrics>,
    cost_slot_inputs: Vec<SlotInput>,
}

impl<M: IterwiseUnifiedModel> PdPrefillWorker<M> {
    pub fn new(
        id: WorkerId,
        model: Arc<M>,
        requests: SharedRequests,
        config: WorkerConfig,
        cost_log_dir: Option<PathBuf>,
    ) -> Self {
        let kv_capacity = (config.attn_kv_bytes / model.kv_bytes_per_token().max(1)).max(1);
        let cost_logger = match cost_log_dir {
            Some(dir) => match CostLogger::open(&dir, &model.cost_log_manifest()) {
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
            runtime: PrefillRuntime::new(),
            batches: vec![Batch::new(0, kv_capacity)],
            events: Vec::new(),
            cost_logger,
            cost_slots: Vec::new(),
            cost_slot_inputs: Vec::new(),
        }
    }

    // ── tick: state-forwarding loop (identical to barebone) ────────────────────

    fn tick_inner(&mut self, now: Time) {
        use IterCursor::*;
        use WorkerFsmState::*;
        loop {
            match self.runtime.worker_fsm_state {
                Idle => {
                    if !self.form_batch() {
                        break;
                    }
                    self.runtime.worker_fsm_state = Active;
                    self.runtime.batch_fsm_state.cursor = NotStarted;
                    self.runtime.batch_fsm_state.compute_end = Time::ZERO;
                    continue;
                }
                Active => match self.runtime.batch_fsm_state.cursor {
                    NotStarted => {
                        let compute_end = self.start_iter(now);
                        self.runtime.batch_fsm_state.cursor = Computing;
                        self.runtime.batch_fsm_state.compute_end = compute_end;
                        break;
                    }
                    Computing => {
                        if now < self.runtime.batch_fsm_state.compute_end {
                            break;
                        }
                        self.runtime.batch_fsm_state.cursor = Done;
                        continue;
                    }
                    Done => {
                        self.complete_iter(now);
                        self.runtime.worker_fsm_state = Idle;
                        continue;
                    }
                },
            }
        }
    }

    // ── Stage 1: form_batch — admit one fresh prefill (same as barebone) ────────

    fn form_batch(&mut self) -> bool {
        if let Some(&rid) = self.runtime.pending_prefills.front() {
            // PD prefill admits by PROMPT only: the request hands off after prefill
            // and never decodes locally (its local KvPool stays empty), so its peak
            // KV occupancy here is the prompt — the decode budget is the decode
            // pool's concern. (Passing decode_len, as the barebone copy-source does,
            // would over-reserve and wrongly throttle prefill admission.)
            let p = {
                let store = self.requests.borrow();
                store[rid].prompt_len
            };
            let group_promised = self.group_promised_kv(0);
            if self
                .config
                .admission
                .try_admit(&self.batches[0], group_promised, p, 0)
            {
                self.runtime.pending_prefills.pop_front();
                self.promise(0, rid, p, 0, 0);
            }
        }
        self.drain_promises_into_admits();
        let had_prefill = !self.batches[0].prefill_admits.is_empty();
        if !had_prefill {
            return false;
        }
        self.runtime.iter_counter += 1;
        true
    }

    // ── Stage 2: start_iter (build ArchInput, cost query) → compute_end ─────────

    fn start_iter(&mut self, now: Time) -> Time {
        let arch_input = self.build_arch_input();
        let logging_cost = self.cost_logger.is_some();
        let agg = if logging_cost {
            self.model.eval_iter_with_inputs(
                &arch_input,
                &mut self.cost_slots,
                &mut self.cost_slot_inputs,
            )
        } else {
            self.model.eval_iter(&arch_input, &mut self.cost_slots)
        };
        let cost_time = Time::from_ms(agg.m.time_ms as f64);
        if self.cost_logger.is_some() {
            let groups = arch_input
                .groups
                .into_iter()
                .map(|g| GroupInputLog {
                    batch_tokens: g.batch_tokens,
                    prefill_tokens: g.prefill_tokens,
                    decode_request_count: g.decode_tokens,
                    decode_kv_total: g.total_kv_len,
                    prefill_chunk_pairs: g.prefill_chunk_pairs,
                })
                .collect();
            let entry = CostLogEntry {
                worker_id: self.id.0,
                iter_id: self.runtime.iter_counter as u64,
                batch_id: 0,
                wall_start_ms: now.as_ms(),
                total_time_ms: agg.m.time_ms as f64,
                energy_j: agg.m.energy_j as f64,
                groups,
                slot_time_ms: self.cost_slots.iter().map(|l| l.m.time_ms).collect(),
                slot_coverage: self.cost_slots.iter().map(|l| l.coverage.bits()).collect(),
                slot_input_len: 0,
            };
            if let Some(logger) = self.cost_logger.as_mut() {
                if let Err(e) = logger.record(entry, &mut self.cost_slot_inputs) {
                    tracing::warn!("cost_log record failed: {e}");
                }
            }
        }
        self.runtime.iter_compute_start = now;
        now + cost_time
    }

    // ── Stage 3: complete_iter — emit first token, then hand off to decode ──────

    fn complete_iter(&mut self, now: Time) {
        let log_tokens = self.config.log_output_token_times;
        // The prefill produced the request's first output token (TTFT). Then,
        // instead of entering a local decode set, the request is handed off to a
        // decode pool: emit RequestComplete if it needed only that one token,
        // else PrefillDone (L6 routes it to the decode pool).
        let mut done: Vec<(RequestId, bool)> = Vec::new(); // (rid, is_complete)
        {
            let mut store = self.requests.borrow_mut();
            for &rid in &self.batches[0].prefill_admits {
                let r = &mut store[rid];
                r.prefill_processed = r.prompt_len;
                r.record_first_token(now, log_tokens);
                done.push((rid, r.is_complete()));
            }
        }
        self.batches[0].prefill_admits.clear();
        for (rid, complete) in done {
            self.runtime.request_to_group.remove(&rid);
            if complete {
                self.events.push(WorkerEvent::RequestComplete { req: rid });
            } else {
                self.events.push(WorkerEvent::PrefillDone { req: rid });
            }
        }
    }

    // ── build_arch_input — one group, prefill-only (no decodes on a prefill worker)

    fn build_arch_input(&self) -> UnifiedArchInput {
        let b = &self.batches[0];
        let store = self.requests.borrow();
        let mut prefill_chunk_pairs = Vec::new();
        let mut prefill_tokens = 0u32;
        for &rid in &b.prefill_admits {
            let r = &store[rid];
            prefill_chunk_pairs.push((r.prefix_kv, r.active_chunk_len));
            prefill_tokens += r.active_chunk_len;
        }
        let group = ArchGroupInput {
            batch_tokens: prefill_tokens,
            prefill_tokens,
            decode_tokens: 0,
            prefill_chunk_pairs,
            decode_kv_lens: Vec::new(),
            total_kv_len: 0,
        };
        UnifiedArchInput {
            groups: vec![group],
            tokens_per_source_rank: Vec::new(),
        }
    }

    // ── lifecycle helpers (same as barebone) ────────────────────────────────────

    fn promise(&mut self, gid: u16, rid: RequestId, p: u32, d: u32, prefix: u32) {
        self.runtime
            .promised
            .insert(rid, (gid, (p + prefix + d) as u64));
        self.runtime.request_to_group.insert(rid, gid);
        let mut store = self.requests.borrow_mut();
        store.mark_admitted(rid);
        let r = &mut store[rid];
        r.active_chunk_len = p;
        r.prefix_kv = prefix;
    }

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

    pub fn release_request(&mut self, rid: RequestId, _current_kv: u64) -> Option<u16> {
        if let Some(pos) = self.runtime.pending_prefills.iter().position(|&x| x == rid) {
            self.runtime.pending_prefills.remove(pos);
            return None;
        }
        if let Some(gid) = self.runtime.request_to_group.remove(&rid) {
            let b = &mut self.batches[gid as usize];
            if let Some(p) = b.prefill_admits.iter().position(|&x| x == rid) {
                b.prefill_admits.swap_remove(p);
            }
            self.runtime.promised.remove(&rid);
            return Some(gid);
        }
        None
    }
}

impl<M: IterwiseUnifiedModel> IterWorker for PdPrefillWorker<M> {
    fn id(&self) -> WorkerId {
        self.id
    }

    fn enqueue(&mut self, msg: WorkerMsg) {
        match msg {
            WorkerMsg::Request(rid) => self.runtime.pending_prefills.push_back(rid),
        }
    }

    fn tick(&mut self, now: Time) {
        self.tick_inner(now);
    }

    fn drain_events(&mut self) -> Vec<WorkerEvent> {
        std::mem::take(&mut self.events)
    }

    fn status(&self) -> WorkerStatus {
        let active =
            self.runtime.promised.len() as u32 + self.batches[0].prefill_admits.len() as u32;
        WorkerStatus {
            queued_requests: self.runtime.pending_prefills.len() as u32,
            active_requests: active,
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
    }
    impl IterwiseUnifiedModel for FakeModel {
        fn eval_iter(&self, _b: &UnifiedArchInput, slots: &mut Vec<LeafMetrics>) -> LeafMetrics {
            slots.clear();
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
            1
        }
    }

    fn shared_with(reqs: &[(u32, u32, u32)]) -> SharedRequests {
        let store = Rc::new(RefCell::new(RequestStore::new()));
        for &(id, prompt, decode) in reqs {
            store
                .borrow_mut()
                .insert(&Request::new(RequestId(id), prompt, decode, Time::ZERO));
        }
        store
    }

    fn worker(store: SharedRequests) -> PdPrefillWorker<FakeModel> {
        PdPrefillWorker::new(
            WorkerId(0),
            Arc::new(FakeModel { ms: 1.0 }),
            store,
            WorkerConfig::default(),
            None,
        )
    }

    #[test]
    fn prefill_emits_prefilldone_handoff() {
        // A multi-token request: prefill emits the first token then hands off.
        let store = shared_with(&[(0, 16, 3)]);
        let mut w = worker(Rc::clone(&store));
        w.enqueue(WorkerMsg::Request(RequestId(0)));
        let mut events = Vec::new();
        for step in 0..20u64 {
            w.tick(Time::from_ms(step as f64));
            events.extend(w.drain_events());
        }
        assert_eq!(events, vec![WorkerEvent::PrefillDone { req: RequestId(0) }]);
        let s = store.borrow();
        let r = &s[RequestId(0)];
        assert_eq!(r.tokens_emitted, 1, "prefill emits exactly the first token");
        assert!(r.first_token_time.is_some());
        assert!(!r.completed, "multi-token request is not complete after prefill");
    }

    #[test]
    fn single_token_request_completes_at_prefill() {
        // decode_len == 1: the prefill's first token is the whole output.
        let store = shared_with(&[(0, 16, 1)]);
        let mut w = worker(Rc::clone(&store));
        w.enqueue(WorkerMsg::Request(RequestId(0)));
        let mut events = Vec::new();
        for step in 0..20u64 {
            w.tick(Time::from_ms(step as f64));
            events.extend(w.drain_events());
        }
        assert_eq!(
            events,
            vec![WorkerEvent::RequestComplete { req: RequestId(0) }]
        );
    }
}
