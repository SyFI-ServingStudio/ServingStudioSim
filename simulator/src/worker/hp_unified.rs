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
use crate::log::{KvSampler, KvSubmit};
use crate::worker::admission_helpers::{prefill_fits_budget, Batch, LoadBalance};
use crate::worker::cost_buffers::CostBuffers;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::iter_worker::IterWorker;
use crate::worker::prefix_cache::PrefixCache;
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
    /// KV-offloaded (preempted) decodes per group, FIFO `(rid, kv_tokens)`.
    swapped: Vec<VecDeque<(RequestId, u64)>>,
    /// Host-pool bytes holding swapped KV (shared across groups).
    host_used_bytes: u64,
    /// Swap transfer time accrued since last iteration start (see barebone).
    pending_transfer: Time,
    /// Promised requests whose host-tier prefix load has not finished: rid →
    /// load-completion time. Gated out of `drain_promises_into_admits` until
    /// then (their KV reservation already counts via `promised`).
    load_ready: HashMap<RequestId, Time>,
    iter_counter: u32,
    iter_compute_start: Time,
    worker_fsm_state: WorkerFsmState,
    batch_fsm_state: BatchFsmState,
    balance: LoadBalance,
}

impl HpRuntime {
    fn new(balance: LoadBalance, num_groups: usize) -> Self {
        Self {
            pending_prefills: VecDeque::new(),
            promised: HashMap::new(),
            request_to_group: HashMap::new(),
            swapped: (0..num_groups).map(|_| VecDeque::new()).collect(),
            host_used_bytes: 0,
            pending_transfer: Time::ZERO,
            load_ready: HashMap::new(),
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
    /// Per-worker KV occupancy sampler (one stream, per-group rows); `None` when no
    /// log dir. The worker only `submit`s — throttle + running-max live in `KvSampler`.
    kv: Option<KvSampler>,
    /// Session-scoped prefix-cache model, one per DP group (each shard retains
    /// its own sessions' KV); empty when disabled = legacy always-hit replay.
    prefix_caches: Vec<PrefixCache>,
    /// Host (CPU-DRAM) prefix-cache tier, one per DP group; empty = disabled.
    /// Sessions write through on completion; a GPU-tier shortfall served here
    /// is LOADED over the group's host link instead of recomputed.
    host_tiers: Vec<PrefixCache>,
    /// Per-group host-link busy-until clock: loads serialize FIFO per group.
    host_link_free: Vec<Time>,
    /// Session → pinned DP group (`session_sticky_groups`): first admission
    /// routes via the balancer, later rounds reuse the pin so they land where
    /// the session's prefix-cache entries live.
    session_groups: HashMap<u32, u16>,
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
        let group_kv_bytes = config
            .attn_kv_bytes
            .saturating_mul(model.num_attn_shards().max(1) as u64);
        let kv_capacity = (group_kv_bytes / model.total_kv_bytes_per_token().max(1)).max(1);
        let batches: Vec<Batch> = (0..num_groups)
            .map(|g| Batch::new(g as u16, kv_capacity))
            .collect();
        // Report each group's static token capacity to the run-meta registry, and
        // open the per-worker sampler across all groups (borrow `cost_log_dir`
        // before `CostBuffers` moves it).
        {
            let mut cl = cluster.borrow_mut();
            for g in 0..num_groups {
                cl.register_kv_capacity(pool_tag, pool.0, id.0, g as u16, kv_capacity);
            }
        }
        let kv = KvSampler::open_opt(
            cost_log_dir.as_deref(),
            pool_tag,
            id,
            num_groups,
            config.kv_log_stride,
        );
        // Round-robin fresh prefills across the groups (config.balance is a hint;
        // a single-group degenerate still works since choose(1) == 0).
        let balance = match config.balance {
            LoadBalance::Single if num_groups > 1 => LoadBalance::RoundRobin { next: 0 },
            other => other,
        };
        let cost = CostBuffers::new_iter(
            cost_log_dir,
            pool_tag,
            id,
            model.as_ref(),
            config.gpu_time_multiplier,
        );
        let prefix_caches = match config.prefix_cache_bytes {
            Some(b) => {
                let tokens = b / model.total_kv_bytes_per_token().max(1);
                (0..num_groups)
                    .map(|_| PrefixCache::new(tokens, config.prefix_cache_policy))
                    .collect()
            }
            None => Vec::new(),
        };
        let host_tiers = match config.prefix_cache_host_bytes {
            Some(b) => {
                let tokens = b / model.total_kv_bytes_per_token().max(1);
                (0..num_groups)
                    .map(|_| PrefixCache::new(tokens, config.prefix_cache_policy))
                    .collect()
            }
            None => Vec::new(),
        };
        Self {
            id,
            model,
            requests,
            config,
            runtime: HpRuntime::new(balance, num_groups),
            batches,
            cost,
            kv,
            prefix_caches,
            host_tiers,
            host_link_free: vec![Time::ZERO; num_groups],
            session_groups: HashMap::new(),
        }
    }

    // ── tick: state-forwarding loop (§3.2.1; identical shape to barebone) ──────

    fn tick_inner(&mut self, now: Time, events: &mut Vec<WorkerEventCommon>) -> Option<Time> {
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

    // ── Stage 1: form_batch — admit one fresh prefill into a balance-chosen group ─

    /// Per-group KV-offload step (see the barebone `drive_kv_offload` for the
    /// policy; the host pool budget is shared across groups).
    fn drive_kv_offload(&mut self) {
        let Some(off) = self.config.kv_offload else {
            return;
        };
        let bytes_per_token = self.model.total_kv_bytes_per_token().max(1);
        let transfer_time =
            |bytes: u64| -> Time { Time::from_ms(bytes as f64 / (off.host_bw_gbps * 1e9) * 1e3) };
        for g in 0..self.batches.len() {
            // (1) swap-in, FIFO.
            while let Some(&(rid, kv_tokens)) = self.runtime.swapped[g].front() {
                let remaining = {
                    let store = self.requests.borrow();
                    let r = &store[rid];
                    r.decode_len.saturating_sub(r.tokens_emitted)
                };
                let group_promised = self.group_promised_kv(g as u16);
                if !self.config.admission.try_admit(
                    &self.batches[g],
                    group_promised,
                    kv_tokens as u32,
                    remaining,
                ) {
                    break;
                }
                self.runtime.swapped[g].pop_front();
                let bytes = kv_tokens.saturating_mul(bytes_per_token);
                self.runtime.host_used_bytes = self.runtime.host_used_bytes.saturating_sub(bytes);
                self.runtime.pending_transfer =
                    self.runtime.pending_transfer + transfer_time(bytes);
                self.batches[g].finalize_to_decode(rid, kv_tokens, remaining);
                self.runtime.request_to_group.insert(rid, g as u16);
            }
        }
        // (2) swap-out for a blocked-but-feasible pending head, on the group RR
        // would route it to (peek without advancing the cursor).
        let Some(&head) = self.runtime.pending_prefills.front() else {
            return;
        };
        let (p, d, prefix) = {
            let store = self.requests.borrow();
            let r = &store[head];
            (r.prompt_len, r.decode_len, r.prefix_kv)
        };
        let g = self.runtime.balance.peek(self.batches.len());
        let demand = u64::from(p) + u64::from(prefix) + u64::from(d);
        if demand > self.batches[g].kv.kv_capacity {
            return;
        }
        loop {
            let group_promised = self.group_promised_kv(g as u16);
            if self
                .config
                .admission
                .try_admit(&self.batches[g], group_promised, p + prefix, d)
            {
                break;
            }
            let Some(&(victim, ref state)) = self.batches[g].decodes.last() else {
                break;
            };
            let kv = state.current_kv;
            let bytes = kv.saturating_mul(bytes_per_token);
            if self.runtime.host_used_bytes + bytes > off.host_capacity_bytes {
                break;
            }
            self.batches[g].release(victim, kv);
            self.runtime.swapped[g].push_back((victim, kv));
            self.runtime.host_used_bytes += bytes;
            self.runtime.pending_transfer = self.runtime.pending_transfer + transfer_time(bytes);
        }
    }

    fn form_batch(&mut self, now: Time) -> bool {
        self.drive_kv_offload();
        if let Some(cap) = self.config.chunk_prefill_tokens {
            return self.form_batch_chunked(cap, now);
        }
        let had_decode = self
            .batches
            .iter()
            .any(|b| b.iter_decoding().next().is_some());

        // Phase A: route fresh prefill(s) to a group (RoundRobin over N). Without
        // a token budget one prefill is admitted per iter; with `max_batch_tokens`
        // the budget applies PER DP group — each group reserves its own live
        // decodes (1 tok/req) then fills its own remainder with whole prefills.
        match self.config.max_batch_tokens {
            None => {
                if let Some(&rid) = self.runtime.pending_prefills.front() {
                    let (p, d, prefix) = {
                        let store = self.requests.borrow();
                        let r = &store[rid];
                        (r.prompt_len, r.decode_len, r.prefix_kv)
                    };
                    let gid = self.route_group(rid);
                    let group_promised = self.group_promised_kv(gid);
                    // The trace-declared cached prefix occupies KV alongside the
                    // prefilled prompt, so the gate demands `p + prefix`.
                    if self.config.admission.try_admit(
                        &self.batches[gid as usize],
                        group_promised,
                        p + prefix,
                        d,
                    ) {
                        self.runtime.pending_prefills.pop_front();
                        self.promise(gid, rid, p, d, prefix, now);
                    }
                }
            }
            Some(budget) => {
                let n = self.batches.len();
                // Per-group decode reserve (fixed for this iter) + admitted tally.
                let decode_tokens: Vec<u32> = self
                    .batches
                    .iter()
                    .map(|b| b.iter_decoding().count() as u32)
                    .collect();
                let mut admitted = vec![0u32; n];
                while let Some(&rid) = self.runtime.pending_prefills.front() {
                    let (p, d, prefix) = {
                        let store = self.requests.borrow();
                        let r = &store[rid];
                        (r.prompt_len, r.decode_len, r.prefix_kv)
                    };
                    // RR picks the head's target group; if that group can't take it
                    // (its own budget or the KV gate), stop — the head waits for a
                    // later iter (the cursor has advanced, so RR tries the next
                    // group then). Naive: no cross-group re-routing within an iter.
                    let gid = self.route_group(rid);
                    let g = gid as usize;
                    if !prefill_fits_budget(budget, decode_tokens[g], admitted[g], p) {
                        break;
                    }
                    let group_promised = self.group_promised_kv(gid);
                    if !self.config.admission.try_admit(
                        &self.batches[g],
                        group_promised,
                        p + prefix,
                        d,
                    ) {
                        break;
                    }
                    self.runtime.pending_prefills.pop_front();
                    self.promise(gid, rid, p, d, prefix, now);
                    admitted[g] += p;
                }
            }
        }

        // Phase B: drain ready promises into their group's prefill_admits.
        self.drain_promises_into_admits(now);
        let had_prefill = self.batches.iter().any(|b| !b.prefill_admits.is_empty());

        if !had_decode && !had_prefill {
            return false;
        }
        self.runtime.iter_counter += 1;
        true
    }

    /// Chunked-prefill batch formation, applied PER DP group: each group has a
    /// HARD `cap`-token iteration budget (its live decodes reserve 1 token
    /// each), dealt FIFO to in-flight partial prefills then fresh admissions
    /// (RR-routed to a group, KV-gated on the full context). See the barebone
    /// `form_batch_chunked` for the single-group semantics this mirrors.
    fn form_batch_chunked(&mut self, cap: u32, now: Time) -> bool {
        let had_decode = self
            .batches
            .iter()
            .any(|b| b.iter_decoding().next().is_some());
        let n = self.batches.len();
        let decode_tokens: Vec<u32> = self
            .batches
            .iter()
            .map(|b| b.iter_decoding().count() as u32)
            .collect();

        // Fresh admissions: RR picks the head's target group; admit while that
        // group still has chunk budget beyond its in-flight partials' demand.
        // Naive like the legacy path: a blocked head stops admission this iter.
        let mut left: Vec<u32> = (0..n)
            .map(|g| {
                let inflight: u32 = {
                    let store = self.requests.borrow();
                    self.batches[g]
                        .prefill_admits
                        .iter()
                        .map(|&rid| {
                            let r = &store[rid];
                            r.prefill_target.saturating_sub(r.prefill_processed)
                        })
                        .sum()
                };
                cap.saturating_sub(decode_tokens[g])
                    .saturating_sub(inflight.min(cap))
            })
            .collect();
        // Under sticky groups a blocked request's target group is FIXED, so
        // stopping at the first blocked head would head-of-line block requests
        // bound for other (possibly idle) groups. Scan the whole pending queue
        // (bounded by the closed-loop cap) and admit whatever fits its own
        // group; without sticky groups the scan degenerates to the legacy
        // front-first behavior after the first failure per group would anyway
        // (balancer state advances identically per admitted request).
        let mut idx = 0;
        let mut blocked_groups = vec![false; n];
        while idx < self.runtime.pending_prefills.len() {
            let rid = self.runtime.pending_prefills[idx];
            let (p, d, prefix) = {
                let store = self.requests.borrow();
                let r = &store[rid];
                (r.prompt_len, r.decode_len, r.prefix_kv)
            };
            let gid = self.route_group(rid);
            let g = gid as usize;
            // Per-group FIFO: once a group rejects a request, don't admit a
            // LATER request into that same group past it this iteration.
            if blocked_groups[g] {
                idx += 1;
                continue;
            }
            let group_promised = self.group_promised_kv(gid);
            if left[g] == 0
                || !self
                    .config
                    .admission
                    .try_admit(&self.batches[g], group_promised, p + prefix, d)
            {
                blocked_groups[g] = true;
                if !self.config.session_sticky_groups {
                    break; // legacy front-first semantics
                }
                idx += 1;
                continue;
            }
            self.runtime.pending_prefills.remove(idx);
            self.promise(gid, rid, p, d, prefix, now);
            left[g] = left[g].saturating_sub(p.min(left[g]));
        }
        self.drain_promises_into_admits(now);

        // Deal chunks FIFO per group (partials first — retained in admission
        // order across iterations). Force-deal one chunk on an otherwise-empty
        // worker so a decode-free, budget-starved iter can't stall forever.
        let mut dealt_any = false;
        for g in 0..n {
            let mut budget = cap.saturating_sub(decode_tokens[g]);
            let mut store = self.requests.borrow_mut();
            for &rid in &self.batches[g].prefill_admits {
                let r = &mut store[rid];
                let remaining = r.prefill_target.saturating_sub(r.prefill_processed);
                let chunk = if budget == 0 && !dealt_any && !had_decode {
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
            self.runtime.iter_counter as u64,
            now,
        );
        self.runtime.iter_compute_start = now;
        // Swap traffic (KV offload) stalls the worker (see barebone start_iter).
        let transfer = self.runtime.pending_transfer;
        self.runtime.pending_transfer = Time::ZERO;
        now + cost_time + transfer
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

            // (c) prefills advance by this iter's chunk; a request whose target
            // is reached resolves to first token, an unfinished chunked partial
            // stays admitted. Whole-prefill mode deals the full target as one
            // chunk, so "advance + check target" covers both modes.
            let mut to_finalize: Vec<(RequestId, u64, u32)> = Vec::new();
            let mut still_prefilling: Vec<RequestId> = Vec::new();
            {
                let mut store = self.requests.borrow_mut();
                for &rid in &self.batches[gid].prefill_admits {
                    let r = &mut store[rid];
                    r.prefill_processed += r.active_chunk_len;
                    r.active_chunk_len = 0;
                    if r.is_prefill() {
                        still_prefilling.push(rid);
                        continue;
                    }
                    r.record_first_token(now, log_tokens);
                    if r.is_complete() {
                        completed.push(rid);
                    } else {
                        // Full context: recomputed miss tokens live inside
                        // prefill_target.
                        let kv = (r.prefill_target + r.prefix_kv) as u64;
                        let remaining = r.decode_len.saturating_sub(r.tokens_emitted);
                        to_finalize.push((rid, kv, remaining));
                    }
                }
            }
            for (rid, kv, remaining) in to_finalize {
                self.batches[gid].finalize_to_decode(rid, kv, remaining);
            }
            self.batches[gid].prefill_admits = still_prefilling;

            // (d) emit + release completed requests; retain the freed context in
            // the group's session prefix cache.
            for rid in completed {
                {
                    let store = self.requests.borrow();
                    let r = &store[rid];
                    if let Some(session) = r.session {
                        let ctx = u64::from(r.prefill_target)
                            + u64::from(r.prefix_kv)
                            + u64::from(r.tokens_emitted);
                        if let Some(cache) = self.prefix_caches.get_mut(gid) {
                            cache.insert(session, ctx);
                        }
                        // Write-through: the host tier retains the session too,
                        // so a later GPU-tier eviction downgrades the next
                        // round to a load instead of a recompute.
                        if let Some(shared) = &self.config.shared_host_tier {
                            shared.insert(session, ctx, self.model.total_kv_bytes_per_token());
                        } else if let Some(tier) = self.host_tiers.get_mut(gid) {
                            tier.insert(session, ctx);
                        }
                    }
                }
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

            // Sample this group's settled KV occupancy for this iteration (values
            // read before `self.kv` is borrowed mut, so the `&self` reads end first).
            if self.kv.is_some() {
                let submit = KvSubmit {
                    active_kv: self.batches[gid].kv.active_kv,
                    projected_peak: self.batches[gid].projected_peak_kv(),
                    promised_kv: self.group_promised_kv(gid as u16),
                };
                self.kv.as_mut().unwrap().submit(gid as u16, submit, now);
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
                if r.active_chunk_len == 0 {
                    continue; // chunked partial with no budget this iter
                }
                // Already-processed chunks are cached context: they join the
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

    fn promise(&mut self, gid: u16, rid: RequestId, p: u32, d: u32, prefix: u32, now: Time) {
        self.runtime
            .promised
            .insert(rid, (gid, (p + prefix + d) as u64));
        self.runtime.request_to_group.insert(rid, gid);
        let mut store = self.requests.borrow_mut();
        store.mark_admitted(rid);
        let r = &mut store[rid];
        // Prefix-cache model (per DP group): the declared prefix only hits up
        // to what this session left resident on the serving group; the
        // shortfall is recomputed. See the barebone worker's `promise`.
        let hit = match (self.prefix_caches.get_mut(gid as usize), r.session) {
            (Some(cache), Some(session)) => cache.lookup_touch(session, prefix),
            (Some(_), None) => 0,
            (None, _) => prefix,
        };
        // Host tier (two-level cache): the GPU-tier shortfall is served from
        // the host tier when resident there — those tokens are LOADED over the
        // group's host link (FIFO-serialized) instead of recomputed. The
        // request is held out of the prefill admits until the load completes.
        let mut hit = hit;
        if hit < prefix {
            let kv_per_tok = self.model.total_kv_bytes_per_token();
            let host_hit = match (&self.config.shared_host_tier, r.session) {
                (Some(shared), Some(session)) => shared.lookup_touch(session, prefix, kv_per_tok),
                _ => match (self.host_tiers.get_mut(gid as usize), r.session) {
                    (Some(tier), Some(session)) => tier.lookup_touch(session, prefix),
                    _ => 0,
                },
            };
            let loaded = host_hit.saturating_sub(hit);
            if loaded > 0 {
                let bytes = u64::from(loaded).saturating_mul(self.model.total_kv_bytes_per_token());
                let dur_ms = bytes as f64 / (self.config.prefix_cache_host_bw_gbps * 1e9) * 1e3;
                let g = gid as usize;
                let start = self.host_link_free[g].max(now);
                let ready = start + Time::from_ms(dur_ms);
                self.host_link_free[g] = ready;
                self.runtime.load_ready.insert(rid, ready);
                hit += loaded;
            }
        }
        r.prefix_kv = hit;
        r.prefill_target = p + (prefix - hit);
        // Whole-prefill mode runs the full target as one chunk; chunked mode
        // overwrites this in the dealing pass.
        r.active_chunk_len = r.prefill_target;
    }

    /// Group for a fresh admission: the session's pinned group when
    /// `session_sticky_groups` is on (pinning it via the balancer on first
    /// sight), else the balancer's per-round choice.
    fn route_group(&mut self, rid: RequestId) -> u16 {
        let n = self.batches.len();
        if self.config.session_sticky_groups {
            let session = self.requests.borrow()[rid].session;
            if let Some(sess) = session {
                if let Some(&g) = self.session_groups.get(&sess) {
                    return g;
                }
                let g = self.runtime.balance.choose(n) as u16;
                self.session_groups.insert(sess, g);
                return g;
            }
        }
        self.runtime.balance.choose(n) as u16
    }

    fn drain_promises_into_admits(&mut self, now: Time) {
        // A promised request whose host-tier prefix load is still in flight
        // stays promised (KV reserved) until the load completes.
        let drained: Vec<(u16, RequestId)> = self
            .runtime
            .promised
            .iter()
            .filter(|(rid, _)| {
                self.runtime
                    .load_ready
                    .get(rid)
                    .is_none_or(|ready| now >= *ready)
            })
            .map(|(&rid, &(gid, _))| (gid, rid))
            .collect();
        for (gid, rid) in drained {
            self.runtime.promised.remove(&rid);
            self.runtime.load_ready.remove(&rid);
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

    fn worker_cfg(
        store: SharedRequests,
        dp_groups: u16,
        config: WorkerConfig,
    ) -> HpUnifiedWorker<FakeModel> {
        HpUnifiedWorker::new(
            WorkerId(0),
            "main",
            Arc::new(FakeModel { ms: 1.0, dp_groups }),
            store,
            config,
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        )
    }

    #[test]
    fn host_tier_serves_gpu_miss_as_gated_load() {
        // One group; a session resident in the HOST tier only. The request's
        // declared 100-token prefix must (a) hit via the host tier (no
        // recompute: prefill_target stays at the prompt), and (b) hold the
        // request out of prefill_admits until the load completes on the
        // modeled host link, then admit it.
        let store = shared_with(&[(0, 8, 2)]);
        {
            let mut s = store.borrow_mut();
            let r = &mut s[RequestId(0)];
            r.prefix_kv = 100;
            r.prefill_target = 8;
            // session 7, declared prefix 100
        }
        store.borrow_mut()[RequestId(0)].session = Some(7);
        let cfg = WorkerConfig {
            prefix_cache_bytes: Some(0), // GPU tier present but empty-capacity
            prefix_cache_host_bytes: Some(1_000_000),
            prefix_cache_host_bw_gbps: 55.0,
            ..WorkerConfig::default()
        };
        let mut w = worker_cfg(store, 1, cfg);
        // FakeModel's total_kv_bytes_per_token sizes the tier; make session 7
        // resident host-side with 100 retained tokens.
        w.host_tiers[0].insert(7, 100);
        w.enqueue(WorkerMsgCommon::Request(RequestId(0)));
        w.form_batch(Time::ZERO);
        // Load in flight at t=0: promised, not yet admitted.
        assert_eq!(w.batches[0].prefill_admits.len(), 0);
        assert_eq!(w.runtime.promised.len(), 1);
        let ready = *w
            .runtime
            .load_ready
            .get(&RequestId(0))
            .expect("load scheduled");
        assert!(ready > Time::ZERO);
        // Host hit covered the whole prefix: nothing recomputed.
        {
            let s = w.requests.borrow();
            assert_eq!(s[RequestId(0)].prefix_kv, 100);
            assert_eq!(s[RequestId(0)].prefill_target, 8);
        }
        // After the load completes, the next formation admits it.
        w.form_batch(ready);
        assert_eq!(w.batches[0].prefill_admits.len(), 1);
        assert!(w.runtime.load_ready.is_empty());
    }

    #[test]
    fn session_sticky_groups_pin_rounds_to_one_group() {
        // Two groups, four sessionful one-round requests: sessions 5,5,9,9.
        // With sticky groups, both rounds of a session land on the group the
        // session was pinned to at first admission, regardless of the RR
        // balancer's per-round rotation.
        let store = shared_with(&[(0, 8, 0), (1, 8, 0), (2, 8, 0), (3, 8, 0)]);
        for (id, sess) in [(0u32, 5u32), (1, 9), (2, 5), (3, 9)] {
            store.borrow_mut()[RequestId(id)].session = Some(sess);
        }
        let cfg = WorkerConfig {
            max_batch_tokens: Some(64),
            session_sticky_groups: true,
            ..WorkerConfig::default()
        };
        let mut w = worker_cfg(store, 2, cfg);
        for id in 0..4u32 {
            w.enqueue(WorkerMsgCommon::Request(RequestId(id)));
        }
        w.form_batch(Time::ZERO);
        let group_of = |w: &HpUnifiedWorker<FakeModel>, id: u32| {
            *w.runtime.request_to_group.get(&RequestId(id)).unwrap()
        };
        assert_eq!(
            group_of(&w, 0),
            group_of(&w, 2),
            "session 5 split across groups"
        );
        assert_eq!(
            group_of(&w, 1),
            group_of(&w, 3),
            "session 9 split across groups"
        );
        assert_ne!(group_of(&w, 0), group_of(&w, 1), "sessions should spread");
    }

    #[test]
    fn per_dp_budget_admits_independently_per_group() {
        // Two groups, budget 16 PER group, six 8-token prefills. RoundRobin
        // spreads them, so one form_batch fills each group to 16 (two prefills)
        // → 4 admitted (2 per group), 2 still queued. The budget is applied per
        // DP node, not globally (a global 16 would admit only two total).
        let store = shared_with(&[
            (0, 8, 0),
            (1, 8, 0),
            (2, 8, 0),
            (3, 8, 0),
            (4, 8, 0),
            (5, 8, 0),
        ]);
        let cfg = WorkerConfig {
            max_batch_tokens: Some(16),
            ..WorkerConfig::default()
        };
        let mut w = worker_cfg(store, 2, cfg);
        for id in 0..6u32 {
            w.enqueue(WorkerMsgCommon::Request(RequestId(id)));
        }
        w.form_batch(Time::ZERO);
        assert_eq!(w.batches[0].prefill_admits.len(), 2);
        assert_eq!(w.batches[1].prefill_admits.len(), 2);
        assert_eq!(w.runtime.pending_prefills.len(), 2);
    }
}
