//! `PdPrefillWorker` — the prefill half of a PD (prefill/decode disaggregation)
//! deployment. Iter-wise like the barebone worker, and admits + costs prefills the
//! same way, but at iter end it does **not** keep the request to decode: it emits
//! `PdPrefillEvent::PrefillDone` so L6 hands the request (its prompt KV already
//! computed) off to a decode pool, and releases its own transient state.
//!
//! v1 simplification: the prefill→decode KV transfer is not modeled (the handoff
//! is a control-plane event only; the shared `RequestStore` carries the request's
//! prefill state across pools). The local KvPool therefore stays empty — a prefill
//! worker reserves KV only transiently during admission, never finalizes a decode.
//!
//! Copied from `BareboneWorker` intentionally for now: the PD lifecycle differs
//! enough that a shared iter-wise core should be introduced only with a concrete
//! maintenance win, not just because the shapes rhyme.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::{ArchGroupInput, IterwiseUnifiedModel, UnifiedArchInput};
use crate::common::{PdStage, PoolId, RequestId, SharedRequests, Time, WorkerId};
use crate::log::{KvSampler, KvSubmit};
use crate::worker::cost_buffers::CostBuffers;
use crate::worker::admission_helpers::Batch;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::iter_worker::IterWorker;
use crate::worker::types::{
    BatchFsmState, IterCursor, PdPrefillEvent, PdPrefillMsg, WorkerConfig, WorkerFsmState,
    WorkerStatus,
};

struct PrefillRuntime {
    pending_prefills: VecDeque<RequestId>,
    promised: HashMap<RequestId, (u16, u64)>,
    request_to_group: HashMap<RequestId, u16>,
    /// Requests whose prefill is done and whose KV is still resident here
    /// pending the decode side's pull. Each entry's KV-token count counts
    /// against the worker's admission budget so the prefill side doesn't
    /// over-commit beyond what its physical KV can actually hold. Drained
    /// when the decode worker acks via `PdPrefillMsg::ReleaseKv`.
    held: HashMap<RequestId, u64>,
    /// Running sum of `held` values — kept incrementally so `try_admit`
    /// stays O(1). Always equals `held.values().sum()`.
    held_kv_tokens: u64,
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
            held: HashMap::new(),
            held_kv_tokens: 0,
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
    /// This worker's pool; paired with `id` to globally identify the worker in
    /// `record_stage` (`id` alone is only unique within a pool).
    pool: PoolId,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    runtime: PrefillRuntime,
    batches: Vec<Batch>, // length 1 (prefill is single-group in v1)
    /// Eval scratch buffers + cost-log writer.
    cost: CostBuffers,
    /// Per-worker KV occupancy sampler (single group); `None` when no log dir.
    kv: Option<KvSampler>,
    /// This worker's send-side comm group id, registered with the shared cluster
    /// at construction (covers the `model.num_attn_shards()` GPUs that hold KV).
    /// Stamped into every emitted `PrefillDone`; the cluster knows the underlying
    /// link count and free-time, the worker keeps only this opaque id.
    send_gid: u16,
}

impl<M: IterwiseUnifiedModel> PdPrefillWorker<M> {
    /// PD prefill self-registers its GPU block in the cluster and stores the
    /// returned base for emit-time `PrefillDone` stamping. It does not keep the
    /// cluster handle — only the decode side calls `submit_transfer`.
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
        let send_gid = {
            let mut c = cluster.borrow_mut();
            let gpu_base = c.allocate(pool.0, id.0, model.gpus_per_replica(), gpu_name, pool_tag);
            // Arch invariant: `num_attn_shards() ≤ gpus_per_replica`, so the
            // attn-shard prefix is the comm group covering KV storage.
            c.register_comm_group(gpu_base, model.num_attn_shards().max(1), pool_tag, id.0)
        };
        // KvPool capacity in tokens. See `unified::new` for the derivation:
        // group memory = `num_attn_shards × attn_kv_bytes`, divided by the
        // model-level `total_kv_bytes_per_token` (all ranks summed).
        let group_kv_bytes =
            config.attn_kv_bytes.saturating_mul(model.num_attn_shards().max(1) as u64);
        let kv_capacity = (group_kv_bytes / model.total_kv_bytes_per_token().max(1)).max(1);
        // Report the pool's static capacity to the run-meta registry (single group),
        // and open the sampler (borrow `cost_log_dir` before `CostBuffers` moves it).
        cluster
            .borrow_mut()
            .register_kv_capacity(pool_tag, pool.0, id.0, 0, kv_capacity);
        let kv = KvSampler::open_opt(cost_log_dir.as_deref(), pool_tag, id, 1, config.kv_log_stride);
        let cost =
            CostBuffers::new_iter(cost_log_dir, pool_tag, id, model.as_ref(), config.gpu_time_multiplier);
        // Arch invariant: `num_attn_shards() ≤ gpus_per_replica == gpus_per_worker`,
        // so no clamp against the worker's GPU range is needed — the model is
        // authoritative for shard count.
        Self {
            id,
            pool,
            model,
            requests,
            config,
            runtime: PrefillRuntime::new(),
            batches: vec![Batch::new(0, kv_capacity)],
            cost,
            kv,
            send_gid,
        }
    }

    // ── tick: state-forwarding loop (identical to barebone) ────────────────────

    fn tick_inner(&mut self, now: Time, events: &mut Vec<PdPrefillEvent>) -> Option<Time> {
        use IterCursor::*;
        use WorkerFsmState::*;
        loop {
            match self.runtime.worker_fsm_state {
                Idle => {
                    if !self.form_batch(now) {
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

    // ── Stage 1: form_batch — admit one fresh prefill by prompt KV only ─────────

    fn form_batch(&mut self, now: Time) -> bool {
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
            // Reservation includes both freshly-promised admits and any KV
            // still held pending decode acks (post-PrefillDone, pre-ReleaseKv).
            // Held KV physically occupies the worker's KV cache, so it must
            // gate new admissions just like a promised one.
            let group_promised = self.group_promised_kv(0) + self.runtime.held_kv_tokens;
            if self
                .config
                .admission
                .try_admit(&self.batches[0], group_promised, p, 0)
            {
                self.runtime.pending_prefills.pop_front();
                self.promise(now, 0, rid, p, 0, 0);
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
        let cost_time = self.cost.run_iter(
            self.model.as_ref(),
            &arch_input,
            self.runtime.iter_counter as u64,
            now,
        );
        self.runtime.iter_compute_start = now;
        now + cost_time
    }

    // ── Stage 3: complete_iter — emit first token, then hand off to decode ──────

    fn complete_iter(&mut self, now: Time, events: &mut Vec<PdPrefillEvent>) {
        let log_tokens = self.config.log_output_token_times;
        // Sender side is fully determined by this worker's pre-registered comm
        // group; emit stamps the gid and the cluster resolves link count /
        // free-time at `submit_transfer` time.
        let send_gid = self.send_gid;
        let worker_id = self.id;
        let pool = self.pool;
        let log_stage = self.config.log_stage_transitions;
        // The prefill produced the request's first output token (TTFT). Then,
        // instead of entering a local decode set, the request is handed off to a
        // decode pool: emit RequestComplete if it needed only that one token,
        // else PrefillDone (L6 routes it to the decode pool). One pass over the
        // admits — `self.requests` (RefCell), `self.runtime`, `self.batches[0]`
        // are disjoint fields, so split-borrows let us read+write per-record
        // and remove `request_to_group` in the same loop without a scratch Vec.
        let mut store = self.requests.borrow_mut();
        let n = self.batches[0].prefill_admits.len();
        for i in 0..n {
            let rid = self.batches[0].prefill_admits[i];
            let r = &mut store[rid];
            r.prefill_processed = r.prompt_len;
            r.record_first_token(now, log_tokens);
            // Hand off in tokens — KvPool's native unit. The decode worker
            // multiplies by its arch's `total_kv_bytes_per_token` only when
            // calling `cluster.submit_transfer`; bytes are a wire-level concern.
            let kv_tokens = (r.prompt_len + r.prefix_kv) as u64;
            let complete = r.is_complete();
            // Location (while `r` still borrowed): a single-token request finishes
            // here (Done); otherwise its KV is held on this worker awaiting the
            // decode-side pull (`PrefillDoneAwaitPull`) — the decode worker
            // records the rest.
            r.record_stage(
                now,
                if complete {
                    PdStage::Done
                } else {
                    PdStage::PrefillDoneAwaitPull
                } as u16,
                pool,
                worker_id,
                log_stage,
            );
            // r is unused below — NLL releases the borrow on `store`.
            self.runtime.request_to_group.remove(&rid);
            if complete {
                events.push(PdPrefillEvent::RequestComplete {
                    worker: worker_id,
                    req: rid,
                });
            } else {
                // KV stays resident on this worker until the decode side acks
                // the pull via `PdPrefillMsg::ReleaseKv`. Tracking it in `held`
                // keeps it gating future admissions.
                self.runtime.held.insert(rid, kv_tokens);
                self.runtime.held_kv_tokens += kv_tokens;
                events.push(PdPrefillEvent::PrefillDone {
                    worker: worker_id,
                    req: rid,
                    send_gid,
                    kv_tokens,
                });
            }
        }
        drop(store);
        self.batches[0].prefill_admits.clear();

        // Sample this iteration's KV occupancy. A prefill worker's local KvPool
        // stays empty (it never finalizes a decode); its real resident KV is the
        // `held` set (prefilled, awaiting the decode side's pull), and it has no
        // decode drain, so `projected_peak` equals the held total. `promised_kv` is
        // the admitted-but-not-yet-prefilled reservation.
        if self.kv.is_some() {
            let held = self.runtime.held_kv_tokens;
            let submit = KvSubmit {
                active_kv: held,
                projected_peak: held,
                promised_kv: self.group_promised_kv(0),
            };
            self.kv.as_mut().unwrap().submit(0, submit, now);
        }
    }

    /// Release this request's held KV reservation. Called from `enqueue` on
    /// receipt of `PdPrefillMsg::ReleaseKv` (decode side finished its pull). A
    /// missing entry is silently ignored — possible if the request was
    /// already cancelled via `release_request` before the ack arrived.
    fn drop_held(&mut self, rid: RequestId) {
        if let Some(tokens) = self.runtime.held.remove(&rid) {
            self.runtime.held_kv_tokens = self.runtime.held_kv_tokens.saturating_sub(tokens);
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

    // ── lifecycle helpers (PD prefill: no local decode KV finalization) ─────────

    fn promise(&mut self, now: Time, gid: u16, rid: RequestId, p: u32, d: u32, prefix: u32) {
        self.runtime
            .promised
            .insert(rid, (gid, (p + prefix + d) as u64));
        self.runtime.request_to_group.insert(rid, gid);
        let mut store = self.requests.borrow_mut();
        store.mark_admitted(rid);
        let r = &mut store[rid];
        r.active_chunk_len = p;
        r.prefix_kv = prefix;
        // Location: pending → prefilling on this prefill worker.
        r.record_stage(
            now,
            PdStage::Prefill as u16,
            self.pool,
            self.id,
            self.config.log_stage_transitions,
        );
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
        // External cancellation while KV was held pending decode ack.
        self.drop_held(rid);
        None
    }
}

impl<M: IterwiseUnifiedModel> IterWorker for PdPrefillWorker<M> {
    type Msg = PdPrefillMsg;
    type Event = PdPrefillEvent;

    fn id(&self) -> WorkerId {
        self.id
    }

    fn enqueue(&mut self, msg: Self::Msg) {
        match msg {
            PdPrefillMsg::Request(rid) => {
                self.runtime.pending_prefills.push_back(rid);
                // Location: request now queued on this prefill worker. Stamped at
                // its arrival time (enqueue carries no clock; arrival is on record).
                let mut store = self.requests.borrow_mut();
                let arrival = store[rid].arrival_time;
                store[rid].record_stage(
                    arrival,
                    PdStage::PendingPrefill as u16,
                    self.pool,
                    self.id,
                    self.config.log_stage_transitions,
                );
            }
            // Decode side has finished pulling — drop the held reservation.
            PdPrefillMsg::ReleaseKv { req } => self.drop_held(req),
        }
    }

    fn tick(&mut self, now: Time, events: &mut Vec<Self::Event>) -> Option<Time> {
        self.tick_inner(now, events)
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
    use crate::common::PoolId;
    use crate::test_helpers::{shared_with, test_cluster, FakeModel};
    use std::rc::Rc;

    fn worker(store: SharedRequests) -> PdPrefillWorker<FakeModel> {
        PdPrefillWorker::new(
            WorkerId(0),
            "prefill",
            Arc::new(FakeModel::for_ms(1.0)),
            store,
            WorkerConfig::default(),
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        )
    }

    #[test]
    fn prefill_emits_prefilldone_handoff() {
        // A multi-token request: prefill emits the first token then hands off.
        let store = shared_with(&[(0, 16, 3)]);
        let mut w = worker(Rc::clone(&store));
        w.enqueue(PdPrefillMsg::Request(RequestId(0)));
        let mut events = Vec::new();
        for step in 0..20u64 {
            w.tick(Time::from_ms(step as f64), &mut events);
        }
        // PrefillDone is token-based at the worker boundary: kv_tokens =
        // prompt_len (16) + prefix_kv (0). The worker registers one comm
        // group at construction → send_gid=0.
        assert_eq!(
            events,
            vec![PdPrefillEvent::PrefillDone {
                worker: WorkerId(0),
                req: RequestId(0),
                send_gid: 0,
                kv_tokens: 16,
            }]
        );
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
        w.enqueue(PdPrefillMsg::Request(RequestId(0)));
        let mut events = Vec::new();
        for step in 0..20u64 {
            w.tick(Time::from_ms(step as f64), &mut events);
        }
        assert_eq!(
            events,
            vec![PdPrefillEvent::RequestComplete {
                worker: WorkerId(0),
                req: RequestId(0)
            }]
        );
        assert!(store.borrow()[RequestId(0)].completed);
    }
}
