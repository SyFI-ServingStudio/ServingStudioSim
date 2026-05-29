//! `BareboneWorker` — the minimal viable iter-wise unified worker (L5 design.md
//! §3.4). One request group, `Strict` admission, whole-prefill-in-one-iter, an
//! iter-wise three-state `BatchFsmState` cursor.
//!
//! Reconciliations vs the design example (see plan):
//!   - event-driven surface (`enqueue`/`tick`/`status`) so the L6 `simple_dp`
//!     pool can drive it; `complete_iter` pushes a self-tagged
//!     `WorkerEventCommon::RequestComplete` into the caller's event sink.
//!   - the request slab is the shared `RequestStore`, injected at construction as
//!     `SharedRequests` and borrowed transiently inside each method (no per-tick
//!     `&mut RequestStore` parameter). Request/session logging stays in L7; the
//!     optional worker-local `CostLogger` records per-iteration cost rows.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::{ArchGroupInput, IterwiseUnifiedModel, UnifiedArchInput};
use crate::common::{PoolId, RequestId, SharedRequests, Time, WorkerId};
use crate::worker::admission_helpers::Batch;
use crate::worker::cost_buffers::CostBuffers;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::types::{
    BatchFsmState, IterCursor, WorkerConfig, WorkerEventCommon, WorkerFsmState, WorkerMsgCommon,
    WorkerStatus,
};

// ── Runtime ────────────────────────────────────────────────────────────────────

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
    /// Eval scratch buffers + cost-log writer. Hides what would otherwise be
    /// five separate fields and a ~50-line block at end of `start_iter`.
    cost: CostBuffers,
}

impl<M: IterwiseUnifiedModel> BareboneWorker<M> {
    /// Barebone has no transfers, so it registers its GPU block in the shared
    /// cluster (for the run-meta report) and drops the handle — no field needed.
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
        cluster
            .borrow_mut()
            .allocate(pool.0, id.0, model.gpus_per_replica(), gpu_name);
        // KvPool capacity in tokens. `attn_kv_bytes` is per-GPU; one attn shard
        // spans `num_attn_shards()` GPUs and stores the full KV (each GPU holds
        // a per-rank slice that sums to the model-level total). So group memory
        // = `num_attn_shards × attn_kv_bytes`; dividing by `total_kv_bytes_per_token`
        // (the wire size — full model, all ranks summed) yields the per-group
        // token capacity. L5 owns the division; arch owns the footprint.
        let group_kv_bytes =
            config.attn_kv_bytes.saturating_mul(model.num_attn_shards().max(1) as u64);
        let kv_capacity = (group_kv_bytes / model.total_kv_bytes_per_token().max(1)).max(1);
        let cost = CostBuffers::new(cost_log_dir, pool_tag, id, model.as_ref());
        Self {
            id,
            model,
            requests,
            config,
            runtime: WorkerRuntime::new(),
            batches: vec![Batch::new(0, kv_capacity)],
            cost,
        }
    }

    // ── L6-facing surface ────────────────────────────────────────────────────

    pub fn enqueue(&mut self, msg: WorkerMsgCommon) {
        let WorkerMsgCommon::Request(rid) = msg;
        self.runtime.pending_prefills.push_back(rid);
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

    pub fn tick(&mut self, now: Time, events: &mut Vec<WorkerEventCommon>) -> Option<Time> {
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
                        self.complete_iter(now, events);
                        self.runtime.worker_fsm_state = Idle; // Active → Idle
                        continue; // forward into the Idle arm to form the next batch
                    }
                },
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
        let cost_time = self.cost.run_iter(
            self.model.as_ref(),
            &arch_input,
            self.id,
            self.runtime.iter_counter as u64,
            now,
        );
        self.runtime.iter_compute_start = now;
        now + cost_time
    }

    // ── Stage 3: complete_iter (token bookkeeping + KV transitions) ────────────

    fn complete_iter(&mut self, now: Time, events: &mut Vec<WorkerEventCommon>) {
        let mut completed: Vec<RequestId> = Vec::new();

        // (a) live decodes produced one token. Iterate the decode set directly
        // (it is not mutated here — `advance_decodes` in (b) does that next) so no
        // intermediate id Vec is materialized; `store` is a separate field, so the
        // shared borrow of `batches` and the `RefCell` borrow_mut don't conflict.
        {
            let log_tokens = self.config.log_output_token_times;
            let mut store = self.requests.borrow_mut();
            for (rid, _) in self.batches[0].iter_decoding() {
                let r = &mut store[rid];
                r.record_token(now, log_tokens);
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
            // Iterate `prefill_admits` in place (cleared below after the deferred
            // finalize) instead of cloning it.
            for &rid in &self.batches[0].prefill_admits {
                let r = &mut store[rid];
                r.prefill_processed = r.prompt_len;
                r.record_first_token(now, self.config.log_output_token_times);
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
            events.push(WorkerEventCommon::RequestComplete {
                worker: self.id,
                req: rid,
            });
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
        // Admission point: this request leaves the pending queue and starts
        // prefill. Advance the store's admitted-prefix watermark so dense
        // `request_state` snapshots log it (and skip the still-pending tail).
        store.mark_admitted(rid);
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
        if let Some(pos) = self.runtime.pending_prefills.iter().position(|&x| x == rid) {
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
    use crate::test_helpers::{shared_with, test_cluster, FakeModel};
    use std::rc::Rc;

    /// Drive ticks at coarse 1ms steps until no work remains or a step cap hits.
    fn run_to_quiescence<M: IterwiseUnifiedModel>(
        w: &mut BareboneWorker<M>,
        max_steps: u64,
    ) -> Vec<WorkerEventCommon> {
        let mut all = Vec::new();
        for step in 0..max_steps {
            w.tick(Time::from_ms(step as f64), &mut all);
        }
        all
    }

    #[test]
    fn single_request_prefill_then_decode_completes() {
        let store = shared_with(&[(0, 16, 3)]); // prompt 16, 3 decode tokens
        let model = Arc::new(FakeModel::for_ms(1.0));
        let mut w = BareboneWorker::new(
            WorkerId(0),
            "main",
            model,
            Rc::clone(&store),
            WorkerConfig::default(),
            None,
            crate::common::PoolId(0),
            "test-gpu",
            test_cluster(),
        );
        w.enqueue(WorkerMsgCommon::Request(RequestId(0)));

        let events = run_to_quiescence(&mut w, 50);
        assert_eq!(
            events,
            vec![WorkerEventCommon::RequestComplete {
                worker: WorkerId(0),
                req: RequestId(0)
            }]
        );
        let s = store.borrow();
        let r = &s[RequestId(0)];
        assert!(r.completed);
        assert_eq!(r.tokens_emitted, 3); // first token + 2 decode tokens
        assert!(r.first_token_time.is_some());
    }

    #[test]
    fn three_requests_all_complete() {
        let store = shared_with(&[(0, 8, 2), (1, 8, 2), (2, 8, 2)]);
        let model = Arc::new(FakeModel::for_ms(1.0));
        let mut w = BareboneWorker::new(
            WorkerId(0),
            "main",
            model,
            Rc::clone(&store),
            WorkerConfig::default(),
            None,
            crate::common::PoolId(0),
            "test-gpu",
            test_cluster(),
        );
        for id in [0, 1, 2] {
            w.enqueue(WorkerMsgCommon::Request(RequestId(id)));
        }
        let events = run_to_quiescence(&mut w, 200);
        assert_eq!(events.len(), 3);
        let s = store.borrow();
        for id in [0, 1, 2] {
            assert!(s[RequestId(id)].completed, "req {id} should complete");
        }
    }
}
