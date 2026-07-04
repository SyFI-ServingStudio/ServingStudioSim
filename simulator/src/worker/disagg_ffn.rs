//! `DisaggFfnWorker` — the ffn half of an AFD (attention-FFN disaggregation)
//! deployment. **No KV, no batch admission**: it is a pure compute/transfer FSM
//! that the ffn pool (L6) feeds pre-composed [`FfnTask`]s. Each task is one
//! micro-batch's section for one layer step (`Bootstrap` / `Bridge` / `Terminal`,
//! see [`FfnTaskKind`]); the worker pulls the section's input from the attn side
//! (attn→ffn), runs the matching arch cost method(s), and emits a
//! [`FfnWorkerEvent`] — [`SectionReady`](FfnWorkerEvent::SectionReady) for a
//! Bootstrap/Bridge QKV handoff, [`IterComplete`](FfnWorkerEvent::IterComplete)
//! for a Terminal token.
//!
//! Double-buffering (port of moesim `ffn_worker.rs`): at most one task `computing`
//! (`current`) and one `pulling` (`next`) at a time, so the next section's input
//! transfer overlaps the current section's compute. Extra tasks queue in
//! `incoming`. The two live timestamps — `current.compute_end` and
//! `next.pull_end` — are the worker's only wakeups (same shape as
//! `pd_decode::next_wakeup`).
//!
//! Cost dispatch by kind (L4 §4.1 Bridge/Bootstrap/Terminal convention):
//!   - `Bootstrap`        → `prologue_cost` (embed) + `pre_attn_cost(0)` (qkv layer 0); input local (no pull)
//!   - `Bridge{upstream}` → `post_attn_cost(upstream)` (o_proj+router+MoE + fused pre(upstream+1))
//!   - `Terminal`         → `post_attn_cost(last)` + `epilogue_cost` (final_norm + lm_head); emits the token
//!
//! v1 / flow-integration notes: `FfnTask` carries the flat total workload
//! (`reqs: Vec<RequestId>`); the worker — the only L5↔L4 bridge — owns the DP-shard
//! partition. Since the ffn side has no attention (no per-request KV locality) it
//! just splits the total token count evenly across `num_dp_groups` and builds the
//! L4 `FfnArchInput` (L6 never groups, counts, or touches arch vocabulary).
//! cost_log goes through [`CostBuffers`]: one row per section per layer, tagged
//! `section` + `layer` + `batch_id` = the pipeline slot (so a slot's whole forward
//! shares an `iter_id`). Attn→ffn input pulls preserve one source chunk per
//! attention worker, so the source comm-group ownership stays with the producer.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::{FfnArchInput, FfnLayerwiseModel};
use crate::common::SharedRequests;
use crate::common::{PoolId, RequestId, Time, WorkerId};
use crate::worker::cost_buffers::CostBuffers;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::iter_worker::IterWorker;
use crate::worker::types::{
    FfnTask, FfnTaskKind, FfnWorkerEvent, FfnWorkerMsg, WorkerConfig, WorkerStatus,
};

/// `min` of two `Option<Time>`, treating `None` as "no constraint".
fn min_opt(a: Option<Time>, b: Time) -> Option<Time> {
    Some(a.map_or(b, |c| c.min(b)))
}

/// A task whose input is transferring (attn→ffn); promoted to `current` once
/// `now >= pull_end`.
struct PullingTask {
    task: FfnTask,
    pull_end: Time,
}

/// A task whose section is computing; finished (event emitted) once
/// `now >= compute_end`.
struct RunningTask {
    task: FfnTask,
    compute_end: Time,
}

pub struct DisaggFfnWorker<M: FfnLayerwiseModel> {
    pub id: WorkerId,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cluster: SharedGpuCluster,
    /// This worker's comm group — both the recv endpoint for the attn→ffn input
    /// pull AND the send endpoint the next attn layer pulls QKV from.
    gid: u16,
    /// Tasks handed in but not yet pulling (double-buffer holds ≤1 pull + ≤1 compute).
    incoming: VecDeque<FfnTask>,
    next_task: Option<PullingTask>,
    current_task: Option<RunningTask>,
    /// Eval scratch + the per-section `cost_log` writer (one row per building block:
    /// `prologue` / `pre_attn` / `post_attn` / `post_attn_last` / `epilogue`). Honest
    /// per-layer rows — the homogeneous layers compress hard under the writer's ZSTD.
    cost: CostBuffers,
    /// Per-slot forward-pass counter (the row `iter_id`), bumped on each `Bootstrap`
    /// so every section of one slot's forward shares an id. Indexed by slot.
    iter_seq: Vec<u64>,
}

impl<M: FfnLayerwiseModel> DisaggFfnWorker<M> {
    /// Self-registers its GPU block + a comm group in the shared cluster (used as
    /// both pull-recv and QKV-send endpoint) and keeps the cluster handle for
    /// runtime `submit_transfer`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: WorkerId,
        model: Arc<M>,
        requests: SharedRequests,
        config: WorkerConfig,
        pool: PoolId,
        gpu_name: &str,
        cluster: SharedGpuCluster,
        cost_log_dir: Option<PathBuf>,
        pool_tag: &'static str,
    ) -> Self {
        let gid = {
            let mut c = cluster.borrow_mut();
            let base = c.allocate(pool.0, id.0, model.gpus_per_replica(), gpu_name);
            // One comm group spanning the whole ffn replica's residing fabric
            // (`gpus_per_replica` = ep_size links). A `FfnTask` carries the producer
            // sources reported by the attn workers; the worker submits one transfer per
            // source into this recv endpoint. Keeping the ffn recv group at ep_size
            // prevents the destination leg from degenerating into an artificial
            // single-link bottleneck.
            c.register_comm_group(base, model.gpus_per_replica(), pool_tag, id.0)
        };
        let cost = CostBuffers::new(cost_log_dir, pool_tag, id, &model.cost_log_manifest());
        Self {
            id,
            model,
            requests,
            config,
            cluster,
            gid,
            incoming: VecDeque::new(),
            next_task: None,
            current_task: None,
            cost,
            iter_seq: Vec::new(),
        }
    }

    fn tick_inner(&mut self, now: Time, events: &mut Vec<FfnWorkerEvent>) -> Option<Time> {
        loop {
            let mut progressed = false;

            // 1. finish the computing task if its compute window closed.
            if let Some(rt) = &self.current_task {
                if now >= rt.compute_end {
                    let rt = self.current_task.take().unwrap();
                    self.finish_task(rt.task, now, events);
                    progressed = true;
                }
            }
            // 2. promote the pulling task to computing if its input landed and the
            //    compute slot is free.
            if self.current_task.is_none() {
                if let Some(pt) = &self.next_task {
                    if now >= pt.pull_end {
                        let pt = self.next_task.take().unwrap();
                        self.current_task = Some(self.start_compute(pt.task, now));
                        progressed = true;
                    }
                }
            }
            // 3. start pulling the next queued task if the pull slot is free.
            if self.next_task.is_none() {
                if let Some(task) = self.incoming.pop_front() {
                    self.next_task = Some(self.start_pull(task, now));
                    progressed = true;
                }
            }

            if !progressed {
                break;
            }
        }
        self.next_wakeup(now)
    }

    fn next_wakeup(&self, now: Time) -> Option<Time> {
        let mut wake: Option<Time> = None;
        // After the tick loop, a live `current` always has a future compute_end and
        // a live `next` may still be pulling. A pull that already landed but is
        // stuck behind a busy compute slot is covered by the compute_end wakeup, so
        // only count `pull_end` while it is still in the future.
        if let Some(rt) = &self.current_task {
            if rt.compute_end > now {
                wake = min_opt(wake, rt.compute_end);
            }
        }
        if let Some(pt) = &self.next_task {
            if pt.pull_end > now {
                wake = min_opt(wake, pt.pull_end);
            }
        }
        wake
    }

    fn start_pull(&mut self, task: FfnTask, now: Time) -> PullingTask {
        // Bootstrap input is local (the embedded tokens) → no transfer. Otherwise
        // the section's input arrives from potentially many attn workers at once:
        // one CONCURRENT gather (latency paid once, senders parallel, receiver
        // drains the aggregate) — not N serialized per-source transfers.
        let sources: Vec<(u16, u64)> = task
            .pull_sources
            .iter()
            .filter(|source| source.bytes > 0)
            .map(|source| (source.send_gid, source.bytes))
            .collect();
        let pull_end = if sources.is_empty() {
            now
        } else {
            self.cluster
                .borrow_mut()
                .submit_gather(now, &sources, self.gid, "afd_ffn_pull", "")
        };
        PullingTask { task, pull_end }
    }

    fn start_compute(&mut self, mut task: FfnTask, now: Time) -> RunningTask {
        // A Bootstrap opens a new forward pass for this slot → bump its row `iter_id`,
        // which the rest of the pass (Bridges + Terminal) reuses.
        let slot = task.slot as usize;
        if matches!(task.kind, FfnTaskKind::Bootstrap) {
            if slot >= self.iter_seq.len() {
                self.iter_seq.resize(slot + 1, 0);
            }
            self.iter_seq[slot] += 1;
            // The Bootstrap runs before any attn layer-output has reported, so the pool
            // could not pre-sum its token count — derive it once here from the store and
            // stamp the task, so this iteration's build + handoff reuse it (Bridge and
            // Terminal already arrive with `tokens` summed pool-side).
            task.tokens = self.workload_tokens(&task.reqs);
        }
        let iter_id = self.iter_seq.get(slot).copied().unwrap_or(0);
        // Empty batches are valid control-plane no-ops in the layer -1 / all-worker
        // barrier path. They still emit the normal downstream event, but must not hit
        // model cost/profile lookup with zero-token shapes.
        let dt = if task.reqs.is_empty() {
            Time::ZERO
        } else {
            let input = self.build_arch_input(task.tokens);
            self.compute_time(task.kind, task.slot, iter_id, &input, now)
        };
        RunningTask {
            compute_end: now + dt,
            task,
        }
    }

    /// Partition the workload across the model's DP shards and build the L4 input.
    /// The worker owns the partition. The ffn side has NO attention (no KV, no
    /// per-request locality), so a token's originating request/shard is irrelevant
    /// internally — only the per-shard *count* matters (the replicated post_norm +
    /// router + MoE home-reduce, `Max`-fanned over shards). So we simply split the
    /// total token count evenly across `num_dp_groups` (remainder to the first
    /// groups). L6 never groups, counts, or constructs arch vocabulary; the worker
    /// is the only L5↔L4 bridge. The ffn arch reads only the per-shard token count
    /// (qkv / o_proj / router / MoE are token-count driven), so [`FfnArchInput`] is
    /// just `tokens_per_group` — no attention-shaped vocabulary.
    fn build_arch_input(&self, total: u64) -> FfnArchInput {
        let num_groups = self.model.num_dp_groups().max(1) as u64;
        let base = (total / num_groups) as u32;
        let rem = total % num_groups;
        FfnArchInput {
            tokens_per_group: (0..num_groups)
                .map(|g| base + if g < rem { 1 } else { 0 })
                .collect(),
        }
    }

    /// Total query tokens this task contributes (prefill = its prompt length,
    /// decode = 1 per request) — sizes the FFN batch and the ffn→attn handoff bytes.
    /// Derived from the store's `is_prefill()` discriminator, mirroring the attn
    /// worker's `batch_query_tokens` so both sides of one layer agree on the token
    /// count. (v1 is non-chunked, so prefill is the whole prompt in one pass; a
    /// chunked prefill would switch this to `active_chunk_len`.)
    fn workload_tokens(&self, reqs: &[RequestId]) -> u64 {
        let store = self.requests.borrow();
        reqs.iter()
            .map(|&rid| {
                let r = &store[rid];
                if r.is_prefill() {
                    r.prompt_len as u64
                } else {
                    1
                }
            })
            .sum()
    }

    /// Sum the wall time of the section(s) this task kind covers, and (if logging is
    /// enabled) write one `cost_log` row per section via [`CostBuffers::run_section`]. The
    /// eval is identical to the non-logging path — `run_section` returns the same
    /// aggregate the timing uses — so the sim clock is unchanged. `start` is the
    /// task's compute-window opening; sections are laid out back-to-back from it
    /// (each row's `wall_start_ms` = the running cursor) so the trace reads in order.
    fn compute_time(
        &mut self,
        kind: FfnTaskKind,
        slot: u8,
        iter_id: u64,
        input: &FfnArchInput,
        start: Time,
    ) -> Time {
        let model = Arc::clone(&self.model);
        let last = model.num_layers().saturating_sub(1) as usize;
        let bid = slot as u64;
        let mut ms = 0.0f64;
        let mut cursor = start;
        match kind {
            FfnTaskKind::Bootstrap => {
                let agg = self.cost.run_section(
                    "prologue",
                    -1,
                    iter_id,
                    bid,
                    &input.tokens_per_group,
                    Some(input.tokens_per_group.as_slice()),
                    cursor,
                    |s, sc, inp| match inp {
                        Some(i) => model.prologue_cost_with_inputs(input, s, sc, i),
                        None => model.prologue_cost(input, s, sc),
                    },
                );
                let t = agg.m.time_ms as f64;
                ms += t;
                cursor += Time::from_ms(t);
                let agg = self.cost.run_section(
                    "pre_attn",
                    0,
                    iter_id,
                    bid,
                    &input.tokens_per_group,
                    Some(input.tokens_per_group.as_slice()),
                    cursor,
                    |s, sc, inp| match inp {
                        Some(i) => model.pre_attn_cost_with_inputs(0, input, s, sc, i),
                        None => model.pre_attn_cost(0, input, s, sc),
                    },
                );
                let t = agg.m.time_ms as f64;
                ms += t;
                cursor += Time::from_ms(t);
            }
            FfnTaskKind::Bridge { upstream } => {
                let agg = self.cost.run_section(
                    "post_attn",
                    upstream as i16,
                    iter_id,
                    bid,
                    &input.tokens_per_group,
                    Some(input.tokens_per_group.as_slice()),
                    cursor,
                    |s, sc, inp| match inp {
                        Some(i) => {
                            model.post_attn_cost_with_inputs(upstream as usize, input, s, sc, i)
                        }
                        None => model.post_attn_cost(upstream as usize, input, s, sc),
                    },
                );
                ms += agg.m.time_ms as f64;
            }
            FfnTaskKind::Terminal => {
                let agg = self.cost.run_section(
                    "post_attn_last",
                    last as i16,
                    iter_id,
                    bid,
                    &input.tokens_per_group,
                    Some(input.tokens_per_group.as_slice()),
                    cursor,
                    |s, sc, inp| match inp {
                        Some(i) => model.post_attn_cost_with_inputs(last, input, s, sc, i),
                        None => model.post_attn_cost(last, input, s, sc),
                    },
                );
                let t = agg.m.time_ms as f64;
                ms += t;
                cursor += Time::from_ms(t);
                let agg = self.cost.run_section(
                    "epilogue",
                    -1,
                    iter_id,
                    bid,
                    &input.tokens_per_group,
                    Some(input.tokens_per_group.as_slice()),
                    cursor,
                    |s, sc, inp| match inp {
                        Some(i) => model.epilogue_cost_with_inputs(input, s, sc, i),
                        None => model.epilogue_cost(input, s, sc),
                    },
                );
                ms += agg.m.time_ms as f64;
            }
        }
        Time::from_ms(ms)
    }

    fn finish_task(&mut self, task: FfnTask, now: Time, events: &mut Vec<FfnWorkerEvent>) {
        match task.kind {
            FfnTaskKind::Terminal => {
                // Terminal emits the iteration's output token for every request in
                // the batch; bookkeeping (and the per-request completion split) is
                // the worker's (plan §3).
                let log_tokens = self.config.log_output_token_times;
                let mut completed = Vec::new();
                {
                    let mut store = self.requests.borrow_mut();
                    for &rid in &task.reqs {
                        let r = &mut store[rid];
                        if r.tokens_emitted == 0 {
                            // Prefill resolves on this Terminal: advance the store's
                            // prefill marker so `is_prefill()` flips to decode. Every
                            // other worker does this; in AFD the attn shard reads it as
                            // its authoritative prefill/decode discriminator, so the ffn
                            // (which owns token emission) must maintain it here.
                            r.prefill_processed = r.prompt_len;
                            r.record_first_token(now, log_tokens);
                        } else {
                            r.record_token(now, log_tokens);
                        }
                        // `record_token` sets `completed` itself, but `record_first_token`
                        // does not — so a `decode_len == 1` request completing on its very
                        // first Terminal needs the flag set here. Authoritative completion
                        // is the worker's (the sim census only reads the flag).
                        if r.is_complete() {
                            r.completed = true;
                            completed.push(rid);
                        }
                    }
                }
                events.push(FfnWorkerEvent::IterComplete {
                    worker: self.id,
                    slot: task.slot,
                    reqs: task.reqs,
                    completed,
                });
            }
            FfnTaskKind::Bootstrap | FfnTaskKind::Bridge { .. } => {
                // Produces this section's QKV for the downstream attn layer; the
                // attn side pulls `ffn_to_attn_bytes_per_token × tokens`. L6 maps
                // `kind` to that downstream layer. `task.tokens` was set once (pool-side
                // for Bridge, worker-side for Bootstrap) — no per-layer store re-scan.
                let out_bytes = self.model.ffn_to_attn_bytes_per_token() * task.tokens;
                events.push(FfnWorkerEvent::SectionReady {
                    worker: self.id,
                    slot: task.slot,
                    kind: task.kind,
                    reqs: task.reqs,
                    out_send_gid: self.gid,
                    out_bytes,
                });
            }
        }
    }
}

impl<M: FfnLayerwiseModel> IterWorker for DisaggFfnWorker<M> {
    type Msg = FfnWorkerMsg;
    type Event = FfnWorkerEvent;

    fn id(&self) -> WorkerId {
        self.id
    }

    fn enqueue(&mut self, msg: Self::Msg) {
        match msg {
            FfnWorkerMsg::Task(task) => self.incoming.push_back(task),
        }
    }

    fn tick(&mut self, now: Time, events: &mut Vec<Self::Event>) -> Option<Time> {
        self.tick_inner(now, events)
    }

    fn status(&self) -> WorkerStatus {
        let queued = self.incoming.len() + usize::from(self.next_task.is_some());
        WorkerStatus {
            queued_requests: queued as u32,
            active_requests: u32::from(self.current_task.is_some()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::RequestId;
    use crate::test_helpers::{shared_with, test_cluster};
    use crate::timing::cache::interp::{CoverageFlags, Metrics4};
    use crate::timing::LeafMetrics;
    use crate::worker::gpu_cluster::SharedGpuCluster;
    use crate::worker::FfnPullSource;
    use std::rc::Rc;

    struct FakeFfn {
        ms: f64,
        layers: u32,
    }
    fn lm(ms: f64) -> LeafMetrics {
        LeafMetrics {
            m: Metrics4 {
                time_ms: ms as f32,
                flops: 0.0,
                bytes: 0.0,
                energy_j: 0.0,
            },
            coverage: CoverageFlags::EMPTY,
        }
    }
    impl FfnLayerwiseModel for FakeFfn {
        fn num_layers(&self) -> u32 {
            self.layers
        }
        fn gpus_per_replica(&self) -> u16 {
            1
        }
        fn num_dp_groups(&self) -> u16 {
            1
        }
        fn ffn_to_attn_bytes_per_token(&self) -> u64 {
            2
        }
        fn pre_attn_cost(
            &self,
            _l: usize,
            _b: &FfnArchInput,
            s: &mut Vec<LeafMetrics>,
            _sc: &mut Vec<LeafMetrics>,
        ) -> LeafMetrics {
            s.clear();
            lm(self.ms)
        }
        fn post_attn_cost(
            &self,
            _l: usize,
            _b: &FfnArchInput,
            s: &mut Vec<LeafMetrics>,
            _sc: &mut Vec<LeafMetrics>,
        ) -> LeafMetrics {
            s.clear();
            lm(self.ms)
        }
        fn prologue_cost(
            &self,
            _b: &FfnArchInput,
            s: &mut Vec<LeafMetrics>,
            _sc: &mut Vec<LeafMetrics>,
        ) -> LeafMetrics {
            s.clear();
            lm(self.ms)
        }
        fn epilogue_cost(
            &self,
            _b: &FfnArchInput,
            s: &mut Vec<LeafMetrics>,
            _sc: &mut Vec<LeafMetrics>,
        ) -> LeafMetrics {
            s.clear();
            lm(self.ms)
        }
    }

    fn worker(store: SharedRequests, cluster: SharedGpuCluster) -> DisaggFfnWorker<FakeFfn> {
        DisaggFfnWorker::new(
            WorkerId(0),
            Arc::new(FakeFfn { ms: 1.0, layers: 2 }),
            store,
            WorkerConfig::default(),
            PoolId(0),
            "test-gpu",
            cluster,
            None,
            "ffn",
        )
    }

    /// Register a 1-link sender comm group (the attn side) so a Bridge/Terminal
    /// pull has a real `send_gid`. Allocates a dummy GPU first so `base` lines up.
    fn register_test_sender(cluster: &SharedGpuCluster) -> u16 {
        let mut c = cluster.borrow_mut();
        c.allocate(99, 99, 1, "attn-gpu");
        c.register_comm_group(0, 1, "attn", 99)
    }

    #[test]
    fn bootstrap_no_pull_computes_and_emits() {
        // A fresh (un-prefilled) request: `is_prefill()` ⇒ workload = prompt_len = 8.
        let store = shared_with(&[(0, 8, 4)]);
        let mut w = worker(Rc::clone(&store), test_cluster());
        w.enqueue(FfnWorkerMsg::Task(FfnTask {
            kind: FfnTaskKind::Bootstrap,
            slot: 0,
            reqs: vec![RequestId(0)],
            pull_sources: Vec::new(),
            tokens: 0,
        }));
        let mut events = Vec::new();
        for step in 0..10u64 {
            w.tick(Time::from_ms(step as f64), &mut events);
        }
        // cost = prologue + pre_attn = 2 ms; out_bytes = ffn_to_attn(2) × 8 = 16.
        assert_eq!(
            events,
            vec![FfnWorkerEvent::SectionReady {
                worker: WorkerId(0),
                slot: 0,
                kind: FfnTaskKind::Bootstrap,
                reqs: vec![RequestId(0)],
                out_send_gid: w.gid,
                out_bytes: 16,
            }]
        );
    }

    #[test]
    fn empty_bootstrap_is_zero_time_control_noop() {
        let store = shared_with(&[]);
        let mut w = worker(Rc::clone(&store), test_cluster());
        w.enqueue(FfnWorkerMsg::Task(FfnTask {
            kind: FfnTaskKind::Bootstrap,
            slot: 0,
            reqs: Vec::new(),
            pull_sources: Vec::new(),
            tokens: 0,
        }));
        let mut events = Vec::new();
        let wake = w.tick(Time::ZERO, &mut events);

        assert_eq!(wake, None, "empty Bootstrap finishes in the current tick");
        assert_eq!(
            events,
            vec![FfnWorkerEvent::SectionReady {
                worker: WorkerId(0),
                slot: 0,
                kind: FfnTaskKind::Bootstrap,
                reqs: Vec::new(),
                out_send_gid: w.gid,
                out_bytes: 0,
            }]
        );
    }

    #[test]
    fn terminal_records_token_and_completes() {
        // decode_len 1 → the single token completes the request.
        let store = shared_with(&[(0, 8, 1)]);
        let mut w = worker(Rc::clone(&store), test_cluster());
        w.enqueue(FfnWorkerMsg::Task(FfnTask {
            kind: FfnTaskKind::Terminal,
            slot: 0,
            reqs: vec![RequestId(0)],
            pull_sources: Vec::new(),
            tokens: 0,
        }));
        let mut events = Vec::new();
        for step in 0..10u64 {
            w.tick(Time::from_ms(step as f64), &mut events);
        }
        assert_eq!(
            events,
            vec![FfnWorkerEvent::IterComplete {
                worker: WorkerId(0),
                slot: 0,
                reqs: vec![RequestId(0)],
                completed: vec![RequestId(0)],
            }]
        );
        assert!(store.borrow()[RequestId(0)].completed);
        assert_eq!(store.borrow()[RequestId(0)].tokens_emitted, 1);
    }

    #[test]
    fn empty_terminal_is_zero_time_control_noop() {
        let store = shared_with(&[]);
        let mut w = worker(Rc::clone(&store), test_cluster());
        w.enqueue(FfnWorkerMsg::Task(FfnTask {
            kind: FfnTaskKind::Terminal,
            slot: 0,
            reqs: Vec::new(),
            pull_sources: Vec::new(),
            tokens: 0,
        }));
        let mut events = Vec::new();
        let wake = w.tick(Time::ZERO, &mut events);

        assert_eq!(wake, None, "empty Terminal finishes in the current tick");
        assert_eq!(
            events,
            vec![FfnWorkerEvent::IterComplete {
                worker: WorkerId(0),
                slot: 0,
                reqs: Vec::new(),
                completed: Vec::new(),
            }]
        );
    }

    #[test]
    fn double_buffer_pulls_bridge_while_bootstrap_computes() {
        let store = shared_with(&[(0, 8, 4), (1, 8, 4)]);
        let cluster = test_cluster();
        let sender = register_test_sender(&cluster);
        let mut w = worker(Rc::clone(&store), Rc::clone(&cluster));
        // Bootstrap (no pull) computes while the Bridge (with a pull) fetches.
        w.enqueue(FfnWorkerMsg::Task(FfnTask {
            kind: FfnTaskKind::Bootstrap,
            slot: 0,
            reqs: vec![RequestId(0)],
            pull_sources: Vec::new(),
            tokens: 0,
        }));
        w.enqueue(FfnWorkerMsg::Task(FfnTask {
            kind: FfnTaskKind::Bridge { upstream: 0 },
            slot: 1,
            reqs: vec![RequestId(1)],
            pull_sources: vec![FfnPullSource {
                send_gid: sender,
                bytes: 4096,
            }],
            tokens: 0,
        }));
        let mut events = Vec::new();
        // At t=0 the Bootstrap is already computing AND the Bridge is already
        // pulling (double-buffer) — the worker reports a future wakeup, not idle.
        let wake = w.tick(Time::ZERO, &mut events);
        assert!(
            wake.is_some(),
            "worker has live compute + pull, must wake later"
        );
        assert!(w.current_task.is_some(), "Bootstrap should be computing");
        assert!(
            w.next_task.is_some(),
            "Bridge should be pulling concurrently"
        );
        for step in 1..200u64 {
            w.tick(Time::from_ms(step as f64), &mut events);
        }
        // Both sections finish, Bootstrap first. Both are Bridge/Bootstrap → SectionReady.
        assert_eq!(events.len(), 2, "both tasks complete");
        assert!(matches!(
            events[0],
            FfnWorkerEvent::SectionReady {
                kind: FfnTaskKind::Bootstrap,
                ..
            }
        ));
        assert!(matches!(
            events[1],
            FfnWorkerEvent::SectionReady {
                kind: FfnTaskKind::Bridge { upstream: 0 },
                ..
            }
        ));
    }
}
