//! `HpUnifiedWorker` — the multi-group iter-wise unified worker (L5 design.md
//! §3.5 "+HP groups"). Same FSM as the barebone worker, but maintains **N
//! independent `Batch` containers** (one per attention DP shard, `N =
//! model.num_attn_dp_groups()`) instead of one. A fresh prefill is round-robined
//! across the groups; the whole iteration is still costed jointly (one shared
//! `BatchFsmState` cursor, one `eval_iter` per iter — the arch's `Max` fan-out
//! over the per-group attention slots supplies the DP wallclock).
//!
//! Kept as its own file (not folded into the barebone template): the barebone
//! worker stays the minimal single-group reference, and this is the multi-group
//! specialization. The shared admission primitives (`Batch` / `KvAdmission` /
//! `LoadBalance` / `KvPool`) and the public worker vocabulary (`WorkerConfig` /
//! `WorkerMsgCommon` / `WorkerEventCommon` / `WorkerStatus` / FSM enums) are reused, not
//! duplicated.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::{ArchGroupInput, IterwiseUnifiedModel, UnifiedArchInput};
use crate::common::{PoolId, RequestId, SharedRequests, Time, WorkerId};
use crate::worker::admission_helpers::{Batch, LoadBalance};
use crate::worker::cost_buffers::CostBuffers;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::iter_worker::IterWorker;
use crate::worker::types::{
    BatchFsmState, IterCursor, WorkerConfig, WorkerEventCommon, WorkerFsmState, WorkerMsgCommon,
    WorkerStatus,
};

/// Worker-local runtime state (mirrors the barebone worker's, plus the live
/// `LoadBalance` cursor used to route fresh prefills across the N groups).
struct HpRuntime {
    pending_prefills: VecDeque<RequestId>,
    promised: HashMap<RequestId, (u16, u64)>,
    request_to_group: HashMap<RequestId, u16>,
    iter_counter: u32,
    iter_compute_start: Time,
    worker_fsm_state: WorkerFsmState,
    batch_fsm_state: BatchFsmState,
    balance: LoadBalance,
}

impl HpRuntime {
    fn new(balance: LoadBalance) -> Self {
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
            balance,
        }
    }
}

pub struct HpUnifiedWorker<M: IterwiseUnifiedModel> {
    pub id: WorkerId,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    runtime: HpRuntime,
    /// One container per attention DP shard; length = `model.num_attn_dp_groups()`.
    batches: Vec<Batch>,
    /// Eval scratch buffers + cost-log writer.
    cost: CostBuffers,
}

impl<M: IterwiseUnifiedModel> HpUnifiedWorker<M> {
    /// HP/DP has no transfers, so it registers its GPU block in the shared
    /// cluster (for the run-meta report) and drops the handle.
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
        let num_groups = model.num_attn_dp_groups().max(1) as usize;
        // Each DP shard is an independent attn TP group with its own KV cache.
        // KvPool capacity in tokens — see `unified::new` for the derivation.
        // `attn_kv_bytes` is per-GPU; one attn shard spans `num_attn_shards()`
        // GPUs; the model's `total_kv_bytes_per_token` is summed across them.
        let group_kv_bytes =
            config.attn_kv_bytes.saturating_mul(model.num_attn_shards().max(1) as u64);
        let kv_capacity = (group_kv_bytes / model.total_kv_bytes_per_token().max(1)).max(1);
        let batches: Vec<Batch> = (0..num_groups)
            .map(|g| Batch::new(g as u16, kv_capacity))
            .collect();
        // Round-robin fresh prefills across the groups (config.balance is a hint;
        // a single-group degenerate still works since choose(1) == 0).
        let balance = match config.balance {
            LoadBalance::Single if num_groups > 1 => LoadBalance::RoundRobin { next: 0 },
            other => other,
        };
        let cost = CostBuffers::new(cost_log_dir, pool_tag, id, model.as_ref());
        Self {
            id,
            model,
            requests,
            config,
            runtime: HpRuntime::new(balance),
            batches,
            cost,
        }
    }

    // ── tick: state-forwarding loop (§3.2.1; identical shape to barebone) ──────

    fn tick_inner(&mut self, now: Time, events: &mut Vec<WorkerEventCommon>) -> Option<Time> {
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
                        self.complete_iter(now, events);
                        self.runtime.worker_fsm_state = Idle;
                        continue;
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

    // ── Stage 1: form_batch — admit one fresh prefill into a balance-chosen group ─

    fn form_batch(&mut self) -> bool {
        let had_decode = self
            .batches
            .iter()
            .any(|b| b.iter_decoding().next().is_some());

        // Phase A: route one fresh prefill to a group (RoundRobin over N).
        if let Some(&rid) = self.runtime.pending_prefills.front() {
            let (p, d) = {
                let store = self.requests.borrow();
                let r = &store[rid];
                (r.prompt_len, r.decode_len)
            };
            let gid = self.runtime.balance.choose(self.batches.len()) as u16;
            let group_promised = self.group_promised_kv(gid);
            if self
                .config
                .admission
                .try_admit(&self.batches[gid as usize], group_promised, p, d)
            {
                self.runtime.pending_prefills.pop_front();
                self.promise(gid, rid, p, d, 0);
            }
        }

        // Phase B: drain ready promises into their group's prefill_admits.
        self.drain_promises_into_admits();
        let had_prefill = self.batches.iter().any(|b| !b.prefill_admits.is_empty());

        if !had_decode && !had_prefill {
            return false;
        }
        self.runtime.iter_counter += 1;
        true
    }

    // ── Stage 2: start_iter (build N-group ArchInput, cost query) → compute_end ──

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

    // ── Stage 3: complete_iter — same bookkeeping as barebone, per group ────────

    fn complete_iter(&mut self, now: Time, events: &mut Vec<WorkerEventCommon>) {
        let log_tokens = self.config.log_output_token_times;
        for gid in 0..self.batches.len() {
            let mut completed: Vec<RequestId> = Vec::new();

            // (a) live decodes produced one token. Iterate the decode set in place
            // (not mutated here — `advance_decodes` in (b) does that next), so no
            // intermediate id Vec; `store` is a separate field, so the shared borrow
            // of `batches` and the `RefCell` borrow_mut don't conflict. `log_tokens`
            // gates the per-token array (the hot-path cost — see RequestRecord).
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

            // (c) prefills resolved → first token; complete now or enter decode set.
            // Iterate `prefill_admits` in place (cleared below after the deferred
            // finalize) instead of cloning it.
            let mut to_finalize: Vec<(RequestId, u64, u32)> = Vec::new();
            {
                let mut store = self.requests.borrow_mut();
                for &rid in &self.batches[gid].prefill_admits {
                    let r = &mut store[rid];
                    r.prefill_processed = r.prompt_len;
                    r.record_first_token(now, log_tokens);
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
                self.batches[gid].finalize_to_decode(rid, kv, remaining);
            }
            self.batches[gid].prefill_admits.clear();

            // (d) emit + release completed requests.
            for rid in completed {
                let current_kv = self.batches[gid]
                    .decodes
                    .iter()
                    .find(|(r, _)| *r == rid)
                    .map(|(_, s)| s.current_kv)
                    .unwrap_or(0);
                self.batches[gid].release(rid, current_kv);
                self.runtime.request_to_group.remove(&rid);
                events.push(WorkerEventCommon::RequestComplete {
                    worker: self.id,
                    req: rid,
                });
            }
        }
    }

    // ── build_arch_input (§3.10) — one ArchGroupInput per DP shard ──────────────

    fn build_arch_input(&self) -> UnifiedArchInput {
        let store = self.requests.borrow();
        let mut groups = Vec::with_capacity(self.batches.len());
        for b in &self.batches {
            let mut prefill_chunk_pairs = Vec::new();
            let mut prefill_tokens = 0u32;
            for &rid in &b.prefill_admits {
                let r = &store[rid];
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

            groups.push(ArchGroupInput {
                batch_tokens: prefill_tokens + decode_tokens,
                prefill_tokens,
                decode_tokens,
                prefill_chunk_pairs,
                decode_kv_lens,
                total_kv_len: total_kv as u32,
            });
        }
        UnifiedArchInput {
            groups,
            tokens_per_source_rank: Vec::new(),
        }
    }

    // ── lifecycle helpers (gid-aware; same as barebone) ─────────────────────────

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

impl<M: IterwiseUnifiedModel> IterWorker for HpUnifiedWorker<M> {
    type Msg = WorkerMsgCommon;
    type Event = WorkerEventCommon;

    fn id(&self) -> WorkerId {
        self.id
    }

    fn enqueue(&mut self, msg: Self::Msg) {
        let WorkerMsgCommon::Request(rid) = msg;
        self.runtime.pending_prefills.push_back(rid);
    }

    fn tick(&mut self, now: Time, events: &mut Vec<Self::Event>) -> Option<Time> {
        self.tick_inner(now, events)
    }

    fn status(&self) -> WorkerStatus {
        let live_decodes: u32 = self
            .batches
            .iter()
            .map(|b| b.iter_decoding().count() as u32)
            .sum();
        let prefill_admits: u32 = self
            .batches
            .iter()
            .map(|b| b.prefill_admits.len() as u32)
            .sum();
        let active = live_decodes + self.runtime.promised.len() as u32 + prefill_admits;
        WorkerStatus {
            queued_requests: self.runtime.pending_prefills.len() as u32,
            active_requests: active,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::PoolId;
    use crate::test_helpers::{shared_with, test_cluster, FakeModel};
    use std::rc::Rc;

    fn worker(store: SharedRequests, dp_groups: u16) -> HpUnifiedWorker<FakeModel> {
        HpUnifiedWorker::new(
            WorkerId(0),
            "main",
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
    fn builds_one_batch_per_dp_group() {
        let w = worker(shared_with(&[]), 2);
        assert_eq!(w.batches.len(), 2);
    }

    #[test]
    fn round_robin_spreads_fresh_prefills_across_groups() {
        // Two long-decode requests, two groups: RR routes them to distinct groups.
        // One prefill is admitted per iter (1ms each), so by ~step 2 both are live
        // in decode; the long decode budget keeps them alive past the probe.
        let store = shared_with(&[(0, 4, 50), (1, 4, 50)]);
        let mut w = worker(store, 2);
        w.enqueue(WorkerMsgCommon::Request(RequestId(0)));
        w.enqueue(WorkerMsgCommon::Request(RequestId(1)));
        let mut events = Vec::new();
        for step in 0..20u64 {
            w.tick(Time::from_ms(step as f64), &mut events);
        }
        let g0 = w.batches[0].iter_decoding().count();
        let g1 = w.batches[1].iter_decoding().count();
        assert_eq!(
            (g0, g1),
            (1, 1),
            "RoundRobin must place one request in each group"
        );
    }

    #[test]
    fn build_arch_input_yields_one_group_per_shard() {
        let w = worker(shared_with(&[]), 3);
        let ai = w.build_arch_input();
        assert_eq!(ai.groups.len(), 3);
    }

    #[test]
    fn all_arrivals_complete() {
        let store = shared_with(&[(0, 4, 2), (1, 4, 2), (2, 4, 2)]);
        let mut w = worker(store, 2);
        for id in 0..3u32 {
            w.enqueue(WorkerMsgCommon::Request(RequestId(id)));
        }
        let mut completed = Vec::new();
        let mut events = Vec::new();
        for step in 0..500u64 {
            w.tick(Time::from_ms(step as f64), &mut events);
            for e in events.drain(..) {
                let WorkerEventCommon::RequestComplete { req, .. } = e;
                completed.push(req);
            }
        }
        completed.sort_by_key(|r| r.0);
        assert_eq!(completed, (0..3).map(RequestId).collect::<Vec<_>>());
    }
}
