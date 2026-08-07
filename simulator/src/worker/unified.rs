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
use crate::log::{KvSampler, KvSubmit};
use crate::worker::admission_helpers::{prefill_fits_budget, Batch};
use crate::worker::cost_buffers::CostBuffers;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::prefix_cache::PrefixCache;
use crate::worker::types::{
    BatchFsmState, IterCursor, WorkerConfig, WorkerEventCommon, WorkerFsmState, WorkerMsgCommon,
    WorkerStatus,
};

// ── Runtime ────────────────────────────────────────────────────────────────────

struct WorkerRuntime {
    pending_prefills: VecDeque<RequestId>,
    promised: HashMap<RequestId, (u16, u64)>,
    request_to_group: HashMap<RequestId, u16>,
    /// KV-offloaded (preempted) decodes, FIFO: `(rid, kv_tokens)`. Swap-in has
    /// priority over fresh admissions. Empty unless `kv_offload` is configured.
    swapped: VecDeque<(RequestId, u64)>,
    /// Host-pool bytes currently holding swapped KV.
    host_used_bytes: u64,
    /// Swap transfer time accrued since the last iteration start; drained into
    /// the next iteration's compute window (the swap stalls this worker).
    pending_transfer: Time,
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
            swapped: VecDeque::new(),
            host_used_bytes: 0,
            pending_transfer: Time::ZERO,
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
    /// Per-worker KV occupancy sampler; `None` when no log dir. The worker only
    /// `submit`s per iteration — throttle + running-max live in `KvSampler`.
    kv: Option<KvSampler>,
    /// Session-scoped prefix-cache model; `None` = legacy always-hit replay.
    prefix_cache: Option<PrefixCache>,
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
        let group_kv_bytes = config
            .attn_kv_bytes
            .saturating_mul(model.num_attn_shards().max(1) as u64);
        let kv_capacity = (group_kv_bytes / model.total_kv_bytes_per_token().max(1)).max(1);
        // Report the pool's static token capacity to the run-meta registry, and
        // open the per-worker KV occupancy sampler (borrowing `cost_log_dir` before
        // it is moved into `CostBuffers`). Barebone is single-group (group 0).
        cluster
            .borrow_mut()
            .register_kv_capacity(pool_tag, pool.0, id.0, 0, kv_capacity);
        let kv = KvSampler::open_opt(
            cost_log_dir.as_deref(),
            pool_tag,
            id,
            1,
            config.kv_log_stride,
        );
        let cost = CostBuffers::new_iter(
            cost_log_dir,
            pool_tag,
            id,
            model.as_ref(),
            config.gpu_time_multiplier,
        );
        let prefix_cache = config.prefix_cache_bytes.map(|b| {
            PrefixCache::new(
                b / model.total_kv_bytes_per_token().max(1),
                config.prefix_cache_policy,
            )
        });
        Self {
            id,
            model,
            requests,
            config,
            runtime: WorkerRuntime::new(),
            batches: vec![Batch::new(0, kv_capacity)],
            cost,
            kv,
            prefix_cache,
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
            + self.batches[0].prefill_admits.len() as u32
            + self.runtime.swapped.len() as u32;
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

    // ── KV offload (preemption swap; see WorkerConfig::kv_offload) ────────────

    /// One offload step, run at batch formation: (1) swap FIFO-parked requests
    /// back in while their KV fits (they outrank fresh admissions); (2) if the
    /// pending head is KV-blocked but would fit an empty pool, preempt the
    /// NEWEST decodes to the host pool until the head fits or the host is full.
    /// The would-fit-empty guard stops a monster request from thrashing the
    /// pool it can never fit. Transfer time accrues into `pending_transfer`
    /// (drained into the next iteration's compute window by `start_iter`).
    fn drive_kv_offload(&mut self) {
        let Some(off) = self.config.kv_offload else {
            return;
        };
        let bytes_per_token = self.model.total_kv_bytes_per_token().max(1);
        let transfer_time =
            |bytes: u64| -> Time { Time::from_ms(bytes as f64 / (off.host_bw_gbps * 1e9) * 1e3) };

        // (1) swap-in, FIFO.
        while let Some(&(rid, kv_tokens)) = self.runtime.swapped.front() {
            let remaining = {
                let store = self.requests.borrow();
                let r = &store[rid];
                r.decode_len.saturating_sub(r.tokens_emitted)
            };
            let group_promised = self.group_promised_kv(0);
            if !self.config.admission.try_admit(
                &self.batches[0],
                group_promised,
                kv_tokens as u32,
                remaining,
            ) {
                break;
            }
            self.runtime.swapped.pop_front();
            let bytes = kv_tokens.saturating_mul(bytes_per_token);
            self.runtime.host_used_bytes = self.runtime.host_used_bytes.saturating_sub(bytes);
            self.runtime.pending_transfer = self.runtime.pending_transfer + transfer_time(bytes);
            self.batches[0].finalize_to_decode(rid, kv_tokens, remaining);
            self.runtime.request_to_group.insert(rid, 0);
        }

        // (2) swap-out for a blocked-but-feasible pending head.
        let Some(&head) = self.runtime.pending_prefills.front() else {
            return;
        };
        let (p, d, prefix) = {
            let store = self.requests.borrow();
            let r = &store[head];
            (r.prompt_len, r.decode_len, r.prefix_kv)
        };
        let demand = u64::from(p) + u64::from(prefix) + u64::from(d);
        if demand > self.batches[0].kv.kv_capacity {
            return; // would never fit even empty — not a preemption case
        }
        loop {
            let group_promised = self.group_promised_kv(0);
            if self
                .config
                .admission
                .try_admit(&self.batches[0], group_promised, p + prefix, d)
            {
                break; // head fits now
            }
            let Some(&(victim, ref state)) = self.batches[0].decodes.last() else {
                break; // nothing left to preempt
            };
            let kv = state.current_kv;
            let bytes = kv.saturating_mul(bytes_per_token);
            if self.runtime.host_used_bytes + bytes > off.host_capacity_bytes {
                break; // host pool full
            }
            self.batches[0].release(victim, kv);
            self.runtime.swapped.push_back((victim, kv));
            self.runtime.host_used_bytes += bytes;
            self.runtime.pending_transfer = self.runtime.pending_transfer + transfer_time(bytes);
        }
    }

    fn form_batch(&mut self) -> bool {
        self.drive_kv_offload();
        if let Some(cap) = self.config.chunk_prefill_tokens {
            return self.form_batch_chunked(cap);
        }
        let had_decode = self.batches[0].iter_decoding().next().is_some();

        // Phase A: admit fresh prefill(s) from the queue. Without a token budget
        // barebone admits one per iter; with `max_batch_tokens` it reserves the
        // budget for the live decodes (1 tok/req) then fills the remainder with
        // whole prefills (see `prefill_fits_budget`). The KV gate always applies.
        match self.config.max_batch_tokens {
            None => {
                if let Some(&rid) = self.runtime.pending_prefills.front() {
                    let (p, d, prefix) = {
                        let store = self.requests.borrow();
                        let r = &store[rid];
                        (r.prompt_len, r.decode_len, r.prefix_kv)
                    };
                    let group_promised = self.group_promised_kv(0);
                    // The trace-declared cached prefix occupies KV alongside the
                    // prefilled prompt, so the gate demands `p + prefix`.
                    if self.config.admission.try_admit(
                        &self.batches[0],
                        group_promised,
                        p + prefix,
                        d,
                    ) {
                        self.runtime.pending_prefills.pop_front();
                        self.promise(0, rid, p, d, prefix);
                    }
                }
            }
            Some(budget) => {
                let decode_tokens = self.batches[0].iter_decoding().count() as u32;
                let mut admitted = 0u32;
                while let Some(&rid) = self.runtime.pending_prefills.front() {
                    let (p, d, prefix) = {
                        let store = self.requests.borrow();
                        let r = &store[rid];
                        (r.prompt_len, r.decode_len, r.prefix_kv)
                    };
                    // Token budget covers computed (uncached) tokens only; the
                    // cached prefix costs no prefill compute.
                    if !prefill_fits_budget(budget, decode_tokens, admitted, p) {
                        break; // token budget exhausted for this iter
                    }
                    let group_promised = self.group_promised_kv(0);
                    if !self.config.admission.try_admit(
                        &self.batches[0],
                        group_promised,
                        p + prefix,
                        d,
                    ) {
                        break; // KV gate blocks the FIFO head; retry next iter
                    }
                    self.runtime.pending_prefills.pop_front();
                    self.promise(0, rid, p, d, prefix);
                    admitted += p;
                }
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

    /// Chunked-prefill batch formation: `cap` is a HARD per-iteration token
    /// budget. Live decodes reserve 1 token each; the remainder is dealt FIFO —
    /// first to in-flight partial prefills (already in `prefill_admits`), then
    /// to fresh admissions from the queue. A request's iteration chunk is
    /// `min(prefill_target − prefill_processed, budget_left)`; zero-chunk
    /// requests stay admitted and resume next iteration. When the batch is
    /// otherwise empty, one chunk is force-dealt even if decodes alone consumed
    /// the cap (a decode-saturated batch must not starve prefill forever).
    fn form_batch_chunked(&mut self, cap: u32) -> bool {
        let had_decode = self.batches[0].iter_decoding().next().is_some();
        let decode_tokens = self.batches[0].iter_decoding().count() as u32;
        let mut left = cap.saturating_sub(decode_tokens);

        // Admit fresh requests while chunk budget remains beyond the in-flight
        // partials' demand. The KV gate reserves the FULL context up front
        // (prompt + prefix + decode), exactly as whole-prefill admission does.
        let inflight_demand: u32 = {
            let store = self.requests.borrow();
            self.batches[0]
                .prefill_admits
                .iter()
                .map(|&rid| {
                    let r = &store[rid];
                    r.prefill_target.saturating_sub(r.prefill_processed)
                })
                .sum()
        };
        while left > inflight_demand.min(left) {
            let Some(&rid) = self.runtime.pending_prefills.front() else {
                break;
            };
            let (p, d, prefix) = {
                let store = self.requests.borrow();
                let r = &store[rid];
                (r.prompt_len, r.decode_len, r.prefix_kv)
            };
            let group_promised = self.group_promised_kv(0);
            if !self
                .config
                .admission
                .try_admit(&self.batches[0], group_promised, p + prefix, d)
            {
                break; // KV gate blocks the FIFO head; retry next iter
            }
            self.runtime.pending_prefills.pop_front();
            self.promise(0, rid, p, d, prefix);
            // Only the first chunk of this request lands this iter; whether more
            // fresh requests admit depends on the budget left after it.
            left = left.saturating_sub(p.min(left));
        }
        self.drain_promises_into_admits();

        // Deal chunks FIFO over everything admitted (partials first — they were
        // pushed in admission order and retained across iterations).
        let mut budget = cap.saturating_sub(decode_tokens);
        let mut dealt_any = false;
        {
            let mut store = self.requests.borrow_mut();
            for &rid in &self.batches[0].prefill_admits {
                let r = &mut store[rid];
                let remaining = r.prefill_target.saturating_sub(r.prefill_processed);
                let chunk = if budget == 0 && !dealt_any && !had_decode {
                    // Force-deal one chunk when nothing else runs this iter.
                    remaining.min(cap)
                } else {
                    remaining.min(budget)
                };
                r.active_chunk_len = chunk;
                budget = budget.saturating_sub(chunk);
                dealt_any |= chunk > 0;
            }
        }

        if !had_decode && !dealt_any {
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
            self.runtime.iter_counter as u64,
            now,
        );
        self.runtime.iter_compute_start = now;
        // Swap traffic (KV offload) stalls the worker: the accrued transfer
        // time extends this iteration's compute window, then resets.
        let transfer = self.runtime.pending_transfer;
        self.runtime.pending_transfer = Time::ZERO;
        now + cost_time + transfer
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

        // (c) prefills advance by this iter's chunk; a request whose target is
        // reached resolves to first token (complete now or enter the decode
        // set), an unfinished chunked partial stays admitted for the next iter.
        // Whole-prefill mode deals the full prompt as one chunk, so "advance +
        // check target" covers both modes.
        let mut to_finalize: Vec<(RequestId, u64, u32)> = Vec::new();
        let mut still_prefilling: Vec<RequestId> = Vec::new();
        {
            let mut store = self.requests.borrow_mut();
            for &rid in &self.batches[0].prefill_admits {
                let r = &mut store[rid];
                r.prefill_processed += r.active_chunk_len;
                r.active_chunk_len = 0;
                if r.is_prefill() {
                    still_prefilling.push(rid);
                    continue;
                }
                r.record_first_token(now, self.config.log_output_token_times);
                if r.is_complete() {
                    completed.push(rid);
                } else {
                    // Recomputed miss tokens live inside prefill_target, so
                    // this is the full context regardless of hit rate.
                    let kv = (r.prefill_target + r.prefix_kv) as u64;
                    let remaining = r.decode_len.saturating_sub(r.tokens_emitted);
                    to_finalize.push((rid, kv, remaining));
                }
            }
        }
        for (rid, kv, remaining) in to_finalize {
            self.batches[0].finalize_to_decode(rid, kv, remaining);
        }
        self.batches[0].prefill_admits = still_prefilling;

        // (d) emit + release completed requests; the freed context is retained
        // in the session prefix cache (the model behind future `prefix_kv` hits).
        for rid in completed {
            if let Some(cache) = &mut self.prefix_cache {
                let store = self.requests.borrow();
                let r = &store[rid];
                if let Some(session) = r.session {
                    let ctx = u64::from(r.prefill_target)
                        + u64::from(r.prefix_kv)
                        + u64::from(r.tokens_emitted);
                    cache.insert(session, ctx);
                }
            }
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

        // Sample this iteration's settled KV occupancy (post advance / finalize /
        // release). Values are read first so the immutable `&self` borrows
        // (`group_promised_kv`, `batches`) end before `self.kv` is borrowed mut.
        if self.kv.is_some() {
            let submit = KvSubmit {
                active_kv: self.batches[0].kv.active_kv,
                projected_peak: self.batches[0].projected_peak_kv(),
                promised_kv: self.group_promised_kv(0),
            };
            self.kv.as_mut().unwrap().submit(0, submit, now);
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
            if r.active_chunk_len == 0 {
                continue; // chunked partial with no budget this iter
            }
            // (prefix_len, append_len) — matches FlashInfer attention input.
            // Already-processed chunks are cached context, so they join the
            // trace-declared prefix on the attention side.
            prefill_chunk_pairs.push((r.prefix_kv + r.prefill_processed, r.active_chunk_len));
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
        // Prefix-cache model: the declared prefix only hits up to what this
        // request's session left resident here; the shortfall is recomputed
        // (prefill_target grows by the miss). Sessionless requests are cold.
        // Legacy (no cache configured) keeps the always-hit replay.
        let hit = match (&mut self.prefix_cache, r.session) {
            (Some(cache), Some(session)) => cache.lookup_touch(session, prefix),
            (Some(_), None) => 0,
            (None, _) => prefix,
        };
        r.prefix_kv = hit;
        r.prefill_target = p + (prefix - hit);
        // Whole-prefill mode runs the full target as one chunk; chunked mode
        // overwrites this in the dealing pass.
        r.active_chunk_len = r.prefill_target;
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

    // ── per-iter prefill token budget (max_batch_tokens) ─────────────────────

    fn cfg_budget(n: u32) -> WorkerConfig {
        WorkerConfig {
            max_batch_tokens: Some(n),
            ..WorkerConfig::default()
        }
    }

    fn worker_with(store: SharedRequests, config: WorkerConfig) -> BareboneWorker<FakeModel> {
        BareboneWorker::new(
            WorkerId(0),
            "main",
            Arc::new(FakeModel::for_ms(1.0)),
            store,
            config,
            None,
            crate::common::PoolId(0),
            "test-gpu",
            test_cluster(),
        )
    }

    #[test]
    fn budget_admits_multiple_prefills_in_one_iter() {
        // budget 20, four 8-token prefills: one form_batch admits 8+8 = 16 (≤20),
        // then rejects the third (24 > 20). Two admitted, two still queued.
        let store = shared_with(&[(0, 8, 0), (1, 8, 0), (2, 8, 0), (3, 8, 0)]);
        let mut w = worker_with(store, cfg_budget(20));
        for id in [0, 1, 2, 3] {
            w.enqueue(WorkerMsgCommon::Request(RequestId(id)));
        }
        w.form_batch();
        assert_eq!(w.batches[0].prefill_admits.len(), 2);
        assert_eq!(w.runtime.pending_prefills.len(), 2);
    }

    #[test]
    fn none_budget_admits_one_prefill_per_iter() {
        // Default config (no budget) keeps the legacy one-prefill/iter behavior.
        let store = shared_with(&[(0, 8, 0), (1, 8, 0), (2, 8, 0)]);
        let mut w = worker_with(store, WorkerConfig::default());
        for id in [0, 1, 2] {
            w.enqueue(WorkerMsgCommon::Request(RequestId(id)));
        }
        w.form_batch();
        assert_eq!(w.batches[0].prefill_admits.len(), 1);
        assert_eq!(w.runtime.pending_prefills.len(), 2);
    }

    // ── chunked prefill (chunk_prefill_tokens) ───────────────────────────────

    fn cfg_chunk(n: u32) -> WorkerConfig {
        WorkerConfig {
            chunk_prefill_tokens: Some(n),
            ..WorkerConfig::default()
        }
    }

    #[test]
    fn chunked_prefill_splits_prompt_across_iters() {
        // cap 4, prompt 10: chunks 4+4+2 → first token lands after 3 iterations
        // (3ms with the 1ms FakeModel), then 2 decode iters complete the request.
        let store = shared_with(&[(0, 10, 3)]);
        let mut w = worker_with(Rc::clone(&store), cfg_chunk(4));
        w.enqueue(WorkerMsgCommon::Request(RequestId(0)));

        // Two iterations in: still prefilling, nothing emitted.
        for step in 0..2 {
            let mut ev = Vec::new();
            w.tick(Time::from_ms(step as f64), &mut ev);
        }
        {
            let s = store.borrow();
            let r = &s[RequestId(0)];
            assert!(r.first_token_time.is_none(), "prefill must span >2 iters");
            assert!(r.prefill_processed > 0 && r.prefill_processed < 10);
        }

        let events = run_to_quiescence(&mut w, 50);
        assert_eq!(events.len(), 1);
        let s = store.borrow();
        let r = &s[RequestId(0)];
        assert!(r.completed);
        assert_eq!(r.prefill_processed, 10);
        // 3 prefill iters at 1ms each; whole-prefill mode would emit at 1ms.
        assert!(r.first_token_time.unwrap() >= Time::from_ms(3.0));
    }

    #[test]
    fn chunked_prefill_decodes_reserve_budget_first() {
        // Req 0 (prompt 4, decode 8) finishes prefill first and decodes; req 1
        // (prompt 12) then chunks through cap 4 with 1 token/iter reserved for
        // the live decode → 3-token chunks. Both must complete, and req 1's
        // prefill must span ≥ 4 iterations (12 / 3).
        let store = shared_with(&[(0, 4, 8), (1, 12, 2)]);
        let mut w = worker_with(Rc::clone(&store), cfg_chunk(4));
        w.enqueue(WorkerMsgCommon::Request(RequestId(0)));
        w.enqueue(WorkerMsgCommon::Request(RequestId(1)));
        let events = run_to_quiescence(&mut w, 100);
        assert_eq!(events.len(), 2);
        let s = store.borrow();
        assert!(s[RequestId(0)].completed);
        assert!(s[RequestId(1)].completed);
    }

    #[test]
    fn chunked_prefill_matches_whole_prefill_totals() {
        // Same three requests, chunked vs whole: identical token totals; only
        // the timing differs. Guards the bookkeeping (prefill_processed /
        // finalize) against drift between the two paths.
        for cfg in [WorkerConfig::default(), cfg_chunk(6)] {
            let store = shared_with(&[(0, 9, 2), (1, 5, 3), (2, 14, 1)]);
            let mut w = worker_with(Rc::clone(&store), cfg);
            for id in [0, 1, 2] {
                w.enqueue(WorkerMsgCommon::Request(RequestId(id)));
            }
            let events = run_to_quiescence(&mut w, 200);
            let s = store.borrow();
            for (id, p, d) in [(0u32, 9u32, 2u32), (1, 5, 3), (2, 14, 1)] {
                let r = &s[RequestId(id)];
                assert!(
                    r.completed,
                    "req {id} incomplete: processed={} target={} emitted={} events={}",
                    r.prefill_processed,
                    r.prefill_target,
                    r.tokens_emitted,
                    events.len()
                );
                assert_eq!(r.prefill_processed, p);
                assert_eq!(r.tokens_emitted, d);
            }
        }
    }

    // ── prefix cache (prefix_cache_bytes + session) ──────────────────────────

    #[test]
    fn prefix_cache_miss_recomputes_then_hits() {
        use crate::common::Request;
        // FakeModel: 1 KV byte per token, so prefix_cache_bytes == tokens.
        let cfg = WorkerConfig {
            prefix_cache_bytes: Some(1000),
            ..WorkerConfig::default()
        };
        let store: SharedRequests = shared_with(&[]);
        // Two sequential calls of session 7, both declaring a 100-token cached
        // prefix (as an agent trace would after its first turn).
        for (id, p, d) in [(0u32, 20u32, 2u32), (1, 30, 2)] {
            store.borrow_mut().insert(
                &Request::new(RequestId(id), p, d, Time::ZERO)
                    .with_prefix_kv(100)
                    .with_session(Some(7)),
            );
        }
        let mut w = worker_with(Rc::clone(&store), cfg);

        // Call 0: cold session → the declared prefix MISSES and is recomputed.
        w.enqueue(WorkerMsgCommon::Request(RequestId(0)));
        run_to_quiescence(&mut w, 50);
        {
            let s = store.borrow();
            let r = &s[RequestId(0)];
            assert!(r.completed);
            assert_eq!(r.prefix_kv, 0, "cold session must not hit");
            assert_eq!(r.prefill_target, 120, "missed prefix is recomputed");
        }

        // Call 1: session 7 now resident (20+100+2 = 122 tokens) → full hit.
        w.enqueue(WorkerMsgCommon::Request(RequestId(1)));
        run_to_quiescence(&mut w, 50);
        let s = store.borrow();
        let r = &s[RequestId(1)];
        assert!(r.completed);
        assert_eq!(r.prefix_kv, 100, "warm session hits the declared prefix");
        assert_eq!(r.prefill_target, 30, "no recompute on a hit");
    }

    #[test]
    fn prefix_cache_disabled_keeps_always_hit_replay() {
        use crate::common::Request;
        let store: SharedRequests = shared_with(&[]);
        store.borrow_mut().insert(
            &Request::new(RequestId(0), 20, 2, Time::ZERO)
                .with_prefix_kv(100)
                .with_session(Some(7)),
        );
        let mut w = worker_with(Rc::clone(&store), WorkerConfig::default());
        w.enqueue(WorkerMsgCommon::Request(RequestId(0)));
        run_to_quiescence(&mut w, 50);
        let s = store.borrow();
        let r = &s[RequestId(0)];
        assert!(r.completed);
        assert_eq!(r.prefix_kv, 100);
        assert_eq!(r.prefill_target, 20);
    }

    // ── KV offload (preemption swap) ─────────────────────────────────────────

    #[test]
    fn kv_offload_preempts_and_resumes_blocked_head() {
        use crate::worker::types::KvOffloadConfig;
        // Tiny KV pool via a tiny attn budget: FakeModel is 1 byte/token, so
        // attn_kv_bytes 100 = 100-token pool. Req 0 (60+40=100 KV) fills it;
        // req 1 (60+40) blocks. Without offload req 1 waits for req 0 to fully
        // drain 40 decode tokens; with offload req 0 is preempted, req 1 runs,
        // then req 0 swaps back in — BOTH must complete either way, and the
        // sequencing difference is visible in first_token_time.
        let cfg = WorkerConfig {
            attn_kv_bytes: 100,
            kv_offload: Some(KvOffloadConfig {
                host_capacity_bytes: 1_000,
                host_bw_gbps: 1e-6, // slow link → swap time visibly nonzero
            }),
            ..WorkerConfig::default()
        };
        let store = shared_with(&[(0, 60, 40), (1, 60, 40)]);
        let mut w = worker_with(Rc::clone(&store), cfg);
        w.enqueue(WorkerMsgCommon::Request(RequestId(0)));
        w.enqueue(WorkerMsgCommon::Request(RequestId(1)));
        let events = run_to_quiescence(&mut w, 400);
        assert_eq!(events.len(), 2, "both requests must complete under swap");
        let s = store.borrow();
        assert!(s[RequestId(0)].completed);
        assert!(s[RequestId(1)].completed);
        // The preempted request resumed with full token accounting.
        assert_eq!(s[RequestId(0)].tokens_emitted, 40);
        assert_eq!(s[RequestId(1)].tokens_emitted, 40);
    }

    #[test]
    fn kv_offload_never_thrashes_on_infeasible_monster() {
        use crate::worker::types::KvOffloadConfig;
        // Req 1's context (150+50) exceeds the 100-token pool outright: the
        // would-fit-empty guard must keep it from evicting req 0, which
        // completes normally; the monster stays pending (stuck by capacity,
        // not by thrash).
        let cfg = WorkerConfig {
            attn_kv_bytes: 100,
            kv_offload: Some(KvOffloadConfig {
                host_capacity_bytes: 1_000,
                host_bw_gbps: 55.0,
            }),
            ..WorkerConfig::default()
        };
        let store = shared_with(&[(0, 40, 20), (1, 150, 50)]);
        let mut w = worker_with(Rc::clone(&store), cfg);
        w.enqueue(WorkerMsgCommon::Request(RequestId(0)));
        w.enqueue(WorkerMsgCommon::Request(RequestId(1)));
        let events = run_to_quiescence(&mut w, 200);
        assert_eq!(events.len(), 1);
        let s = store.borrow();
        assert!(
            s[RequestId(0)].completed,
            "victim candidate must not starve"
        );
        assert!(!s[RequestId(1)].completed);
    }

    #[test]
    fn budget_force_admits_single_overlong_prefill() {
        // budget 4 but the head prompt is 10 (> budget) and nothing admitted yet:
        // force-admit exactly one; the second over-long prefill waits its turn.
        let store = shared_with(&[(0, 10, 0), (1, 10, 0)]);
        let mut w = worker_with(store, cfg_budget(4));
        for id in [0, 1] {
            w.enqueue(WorkerMsgCommon::Request(RequestId(id)));
        }
        w.form_batch();
        assert_eq!(w.batches[0].prefill_admits.len(), 1);
        assert_eq!(w.runtime.pending_prefills.len(), 1);
    }
}
