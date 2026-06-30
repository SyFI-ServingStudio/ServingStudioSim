//! `afd_attn_pool` — the attn half **and the cross-attn aggregator** of the AFD
//! flow (L6a, ref `AttentionScheduler`). It owns the [`DisaggAttnWorker`](crate::worker::DisaggAttnWorker)s
//! and does the three jobs that make AFD aggregation work. It does NOT reuse
//! `SimpleDpPoolController` (per the AFD plan): aggregation + the per-layer flush
//! barrier are the whole point, and that controller has neither.
//!
//!   1. **Placement** ([`admit`](AfdAttnPoolController::admit)): a fresh request is
//!      pinned to the least-loaded attn worker (KV locality, sticky for its whole
//!      life) and admitted there with a pure `Admit`.
//!   2. **Aggregate + barrier** ([`aggregate`](AfdAttnPoolController::aggregate)):
//!      per slot it gathers the workers' `IterStart`s into one ffn prolog
//!      (`Bootstrap`), then holds a per-layer flush barrier over the iteration's
//!      participants — only once **every** participant has reported layer `L` does
//!      it emit ONE aggregated [`FfnTask`] (the big fused batch) for that layer. A
//!      late joiner (its `IterStart` lands while the slot is in-flight) is buffered
//!      and folded into the **next** iteration — never a second concurrent one.
//!   3. **Scatter + complete** ([`apply_ffn_events`](AfdAttnPoolController::apply_ffn_events)):
//!      an ffn `SectionReady` fans the next layer's QKV back to the iteration's
//!      participants as slot-addressed `ReadyNotification`s; an ffn `IterComplete`
//!      releases finished requests' KV and surfaces their completion.
//!
//! **How cross-worker lockstep falls out** (diverges from ref's all-workers
//! `deferred_empty_advance`): a participant cannot advance its slot to layer `L+1`
//! until it receives the layer-`(L+1)` `ReadyNotification`, which the barrier only
//! produces once ALL participants finished layer `L`. Empty slots never participate
//! (they stay dormant in the worker), so there is no empty-slot churn and no
//! `SlotFlushed`.
//!
//! **Why it cannot deadlock**: every barrier waits on exactly the workers recorded
//! as `members` at prolog fire, and `members` is built from buffered `IterStart`s
//! *after* `finish_iteration` has stripped just-completed requests out of them — so
//! a worker whose whole batch finished (its slot now empty, never to report again)
//! is never carried into the next iteration's barrier. All transitions are forward;
//! a blocked slot simply waits for the ffn it already dispatched.

use std::collections::HashMap;
use std::sync::Arc;

use crate::arch::contract::AttnLayerwiseModel;
use crate::common::{PoolId, RequestId, SharedRequests, Time, WorkerId};
use crate::worker::{
    AttnWorkerEvent, AttnWorkerMsg, DisaggAttnWorker, FfnTask, FfnTaskKind, FfnWorkerEvent,
    IterWorker, SharedGpuCluster, WorkerConfig,
};

/// Pipeline depth — matches the worker's `NUM_SLOTS`. Cross-worker alignment is by
/// this slot index + layer; never by request.
const NUM_SLOTS: usize = 3;

/// Sentinel for a quiescent worker (mirrors `simple_dp`).
const NO_WAKEUP_TIME: Time = Time::from_ns(u64::MAX);

/// The flush barrier for one in-flight layer of one slot: which participants have
/// reported, plus the aggregate they contribute to the next [`FfnTask`].
#[derive(Default)]
struct LayerBarrier {
    /// The layer being collected (set from the first report; lockstep ⇒ all match).
    layer: u16,
    /// Participants that have reported this layer (barrier completes at `members`).
    reported: Vec<WorkerId>,
    /// Union of reported request sets — the fused ffn batch for this layer.
    reqs: Vec<RequestId>,
    /// Sum of reported attn→ffn handoff bytes (the ffn's single fused pull size).
    pull_bytes: u64,
    /// Representative source endpoint (first reporter with non-zero bytes).
    send_gid: u16,
}

/// Per-slot aggregation state.
struct SlotSched {
    /// An iteration is mid-flight: a prolog fired and its `Terminal` has not yet
    /// completed. While set, fresh `IterStart`s buffer in `pending`.
    in_flight: bool,
    /// Participants of the in-flight iteration (the workers whose `IterStart` formed
    /// it). The barrier waits on exactly this set; the scatter fans to exactly it.
    members: Vec<WorkerId>,
    /// The current layer's flush barrier.
    barrier: LayerBarrier,
    /// `IterStart`s that arrived while in-flight (or before the next prolog fired) —
    /// folded into the next iteration. `(worker, reqs)`, one entry per worker.
    pending: Vec<(WorkerId, Vec<RequestId>)>,
}

impl SlotSched {
    fn new() -> Self {
        Self {
            in_flight: false,
            members: Vec::new(),
            barrier: LayerBarrier::default(),
            pending: Vec::new(),
        }
    }
}

pub struct AfdAttnPoolController<M: AttnLayerwiseModel> {
    workers: Vec<DisaggAttnWorker<M>>,
    /// Hot wakeup filter, parallel to `workers` (same scheme as `simple_dp`).
    worker_wakeup_times: Vec<Time>,
    /// Layer count — the barrier flushes `Bridge{L}` for `L < num_layers - 1` and
    /// `Terminal` for the last layer.
    num_layers: u16,
    slots: [SlotSched; NUM_SLOTS],
    /// req → its pinned worker (sticky, KV locality): routes `Release` on completion.
    req_worker: HashMap<RequestId, WorkerId>,
}

impl<M: AttnLayerwiseModel> AfdAttnPoolController<M> {
    // ── Construction ──────────────────────────────────────────────────────────
    /// Build the attn pool's `num_workers` workers (one DP shard each), each handed
    /// the shared `cluster` so it self-registers its GPU block + comm group. The
    /// disagg attn worker's `new` is 7-arg, so this builds them directly rather than
    /// through `UnifiedWorkerFactory`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_workers: u16,
        model: Arc<M>,
        requests: SharedRequests,
        config: WorkerConfig,
        pool: PoolId,
        gpu_name: &str,
        cluster: &SharedGpuCluster,
        cost_log_dir: Option<std::path::PathBuf>,
    ) -> Self {
        assert!(num_workers > 0, "afd attn pool needs at least one worker");
        let num_layers = model.num_layers() as u16;
        let workers: Vec<DisaggAttnWorker<M>> = (0..num_workers)
            .map(|i| {
                DisaggAttnWorker::new(
                    WorkerId(i),
                    Arc::clone(&model),
                    std::rc::Rc::clone(&requests),
                    config,
                    pool,
                    gpu_name,
                    std::rc::Rc::clone(cluster),
                    cost_log_dir.clone(),
                    "attn",
                )
            })
            .collect();
        let n = workers.len();
        Self {
            worker_wakeup_times: vec![NO_WAKEUP_TIME; n],
            workers,
            num_layers,
            slots: std::array::from_fn(|_| SlotSched::new()),
            req_worker: HashMap::new(),
        }
    }

    // ── Placement (called by the flow on arrival) ─────────────────────────────

    /// Pin a fresh request to the least-KV-loaded attn worker (KV locality, sticky for
    /// its whole life) and admit it there with a pure `Admit`. The worker reserves
    /// the KV, picks a local slot, and announces the iteration itself via `IterStart`.
    pub fn admit(&mut self, req: RequestId) {
        let idx = self.least_kv_loaded_worker();
        self.req_worker.insert(req, WorkerId(idx as u16));
        self.workers[idx].enqueue(AttnWorkerMsg::Admit { req });
        self.wake(idx);
    }

    /// The shard with the least estimated peak KV. KV is the binding resource on the
    /// attn side, so placement balances each worker's reported projected-peak KV (its
    /// resident decode trajectory + reserved-not-resident prefills) rather than a
    /// request count — the worker owns its KV accounting, the pool just reads it.
    fn least_kv_loaded_worker(&self) -> usize {
        (0..self.workers.len())
            .min_by_key(|&i| (self.workers[i].estimated_peak_kv(), i))
            .unwrap_or(0)
    }

    // ── Tick driving (called by the flow) ─────────────────────────────────────

    /// Sweep wakeup times and tick only due workers, each pushing its self-tagged
    /// events into the caller's sink (same one-sweep shape as `simple_dp`).
    pub fn tick_collect(&mut self, now: Time, events: &mut Vec<AttnWorkerEvent>) {
        for (wakeup, worker) in self
            .worker_wakeup_times
            .iter_mut()
            .zip(self.workers.iter_mut())
        {
            if *wakeup <= now {
                *wakeup = worker.tick(now, events).unwrap_or(NO_WAKEUP_TIME);
            }
        }
    }

    // ── (3) Apply ffn events: scatter QKV + release/complete ──────────────────

    /// Consume this tick's ffn events: a `SectionReady` scatters the produced QKV to
    /// the slot's participants; an `IterComplete` releases finished requests and
    /// records their completion in `completed` (the flow surfaces them to L7). Runs
    /// before the attn workers tick, so the scattered notifications / releases are in
    /// hand when they next advance.
    pub fn apply_ffn_events(&mut self, events: &[FfnWorkerEvent], completed: &mut Vec<RequestId>) {
        for ev in events {
            match ev {
                FfnWorkerEvent::SectionReady {
                    slot,
                    kind,
                    out_send_gid,
                    out_bytes,
                    ..
                } => self.scatter(*slot as usize, *kind, *out_send_gid, *out_bytes),
                FfnWorkerEvent::IterComplete {
                    slot,
                    completed: done,
                    ..
                } => self.finish_iteration(*slot as usize, done, completed),
            }
        }
    }

    /// Fan a finished section's QKV back to the slot's participants as slot-addressed
    /// `ReadyNotification`s for the downstream attn layer (`Bootstrap → 0`,
    /// `Bridge{u} → u+1`). v1 splits `out_bytes` evenly across participants — the
    /// balanced-load approximation (the attn→ffn pull is summed exactly; only this
    /// ffn→attn scatter is even-split). `Terminal` produces no `SectionReady`.
    fn scatter(&mut self, slot: usize, kind: FfnTaskKind, send_gid: u16, out_bytes: u64) {
        let layer = match kind {
            FfnTaskKind::Bootstrap => 0,
            FfnTaskKind::Bridge { upstream } => upstream + 1,
            FfnTaskKind::Terminal => {
                debug_assert!(false, "Terminal yields IterComplete, not SectionReady");
                return;
            }
        };
        let members = self.slots[slot].members.clone();
        if members.is_empty() {
            return;
        }
        let per = out_bytes / members.len() as u64;
        for w in members {
            let idx = w.0 as usize;
            self.workers[idx].enqueue(AttnWorkerMsg::ReadyNotification {
                slot: slot as u8,
                layer,
                send_gid,
                bytes: per,
            });
            self.wake(idx);
        }
    }

    /// An iteration's `Terminal` completed: release each finished request's KV on its
    /// pinned worker, record the completion, and free the slot. Crucially it also
    /// strips the just-completed requests out of any buffered next-iteration
    /// `IterStart`s — a worker whose whole batch finished (its slot now empty, never
    /// to report) must NOT be carried into the next iteration's barrier, or that
    /// barrier would wait forever (deadlock). Continuing requests loop back via a
    /// fresh `IterStart`, so a new prolog fires on the next `aggregate`.
    fn finish_iteration(&mut self, slot: usize, done: &[RequestId], completed: &mut Vec<RequestId>) {
        for &req in done {
            if let Some(w) = self.req_worker.remove(&req) {
                let idx = w.0 as usize;
                self.workers[idx].enqueue(AttnWorkerMsg::Release { req });
                self.wake(idx);
                // No controller-side load counter to decrement: the worker drops the
                // request's KV on `Release`, and the pool reads `estimated_peak_kv`
                // straight from the worker at the next placement.
            }
            completed.push(req);
        }
        let s = &mut self.slots[slot];
        s.in_flight = false;
        s.members.clear();
        s.barrier = LayerBarrier::default();
        for entry in &mut s.pending {
            entry.1.retain(|r| !done.contains(r));
        }
        s.pending.retain(|(_, reqs)| !reqs.is_empty());
    }

    // ── (2) Aggregate attn events: buffer starts, barrier layers, fire prologs ─

    /// Fold this tick's attn events into ffn work: buffer each `IterStart`, advance
    /// the per-layer flush barrier on each `AttnLayerOutputsReady`, then fire a
    /// prolog for any idle slot that has buffered starts. Returns the aggregated
    /// `FfnTask`s (in `tasks`) for the flow to submit to the ffn pool.
    pub fn aggregate(&mut self, events: &[AttnWorkerEvent], tasks: &mut Vec<FfnTask>) {
        for ev in events {
            match ev {
                AttnWorkerEvent::IterStart { worker, slot, reqs } => {
                    self.slots[*slot as usize]
                        .pending
                        .push((*worker, reqs.clone()));
                }
                AttnWorkerEvent::AttnLayerOutputsReady {
                    worker,
                    slot,
                    reqs,
                    layer,
                    send_gid,
                    bytes,
                } => self.on_layer_output(
                    *slot as usize,
                    *worker,
                    reqs,
                    *layer,
                    *send_gid,
                    *bytes,
                    tasks,
                ),
            }
        }
        // Fire a prolog for each idle slot that has buffered starts (aggregating this
        // tick's same-slot `IterStart`s into one fused `Bootstrap`).
        for slot in 0..NUM_SLOTS {
            self.try_fire_prolog(slot, tasks);
        }
    }

    /// Advance a slot's flush barrier with one participant's layer output. When the
    /// last participant reports, flush ONE aggregated `FfnTask` for that layer
    /// (`Bridge{L}` for a mid layer, `Terminal` for the last) and reset the barrier
    /// for the next layer (the participant set is unchanged within an iteration).
    #[allow(clippy::too_many_arguments)]
    fn on_layer_output(
        &mut self,
        slot: usize,
        worker: WorkerId,
        reqs: &[RequestId],
        layer: u16,
        send_gid: u16,
        bytes: u64,
        tasks: &mut Vec<FfnTask>,
    ) {
        let num_layers = self.num_layers;
        let s = &mut self.slots[slot];
        debug_assert!(s.in_flight, "layer-{layer} output for an idle slot {slot}");
        if s.barrier.reported.is_empty() {
            s.barrier.layer = layer;
        }
        debug_assert_eq!(
            s.barrier.layer, layer,
            "slot {slot} barrier mixes layers {} and {layer} — lockstep broken",
            s.barrier.layer
        );
        if s.barrier.reported.contains(&worker) {
            debug_assert!(false, "worker {worker:?} reported slot {slot} twice in one layer");
            return;
        }
        s.barrier.reported.push(worker);
        s.barrier.reqs.extend_from_slice(reqs);
        s.barrier.pull_bytes += bytes;
        if bytes > 0 && s.barrier.send_gid == 0 {
            s.barrier.send_gid = send_gid;
        }
        if s.barrier.reported.len() != s.members.len() {
            return;
        }
        let kind = if layer + 1 < num_layers {
            FfnTaskKind::Bridge { upstream: layer }
        } else {
            FfnTaskKind::Terminal
        };
        let b = std::mem::take(&mut s.barrier);
        tasks.push(FfnTask {
            kind,
            slot: slot as u8,
            reqs: b.reqs,
            send_gid: b.send_gid,
            pull_bytes: b.pull_bytes,
        });
    }

    /// Fire a prolog for an idle slot with buffered starts: aggregate the buffered
    /// `IterStart`s into one fused `Bootstrap` (the participants of the new
    /// iteration), and record the participant set for the barrier + scatter.
    fn try_fire_prolog(&mut self, slot: usize, tasks: &mut Vec<FfnTask>) {
        let s = &mut self.slots[slot];
        if s.in_flight || s.pending.is_empty() {
            return;
        }
        let mut members = Vec::with_capacity(s.pending.len());
        let mut reqs = Vec::new();
        for (w, rs) in s.pending.drain(..) {
            debug_assert!(!members.contains(&w), "worker {w:?} double-announced slot {slot}");
            members.push(w);
            reqs.extend(rs);
        }
        s.members = members;
        s.in_flight = true;
        s.barrier = LayerBarrier::default();
        tasks.push(FfnTask {
            kind: FfnTaskKind::Bootstrap,
            slot: slot as u8,
            reqs,
            send_gid: 0,
            pull_bytes: 0,
        });
    }

    fn wake(&mut self, idx: usize) {
        if self.worker_wakeup_times[idx] == NO_WAKEUP_TIME {
            self.worker_wakeup_times[idx] = Time::ZERO;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{PoolId, RequestId};
    use crate::test_helpers::{prefilled_store, test_cluster, FakeAttn};
    use std::rc::Rc;

    fn pool(num_workers: u16, store: SharedRequests) -> AfdAttnPoolController<FakeAttn> {
        AfdAttnPoolController::new(
            num_workers,
            Arc::new(FakeAttn { ms: 1.0, layers: 2 }),
            store,
            WorkerConfig::default(),
            PoolId(0),
            "test-gpu",
            &test_cluster(),
            None,
        )
    }

    /// One drive step: tick the workers, then aggregate their events into ffn tasks.
    fn step(p: &mut AfdAttnPoolController<FakeAttn>, now: Time, tasks: &mut Vec<FfnTask>) {
        let mut events = Vec::new();
        p.tick_collect(now, &mut events);
        p.aggregate(&events, tasks);
    }

    /// `admit` pins a request and the worker's `IterStart` aggregates into a single
    /// `Bootstrap` prolog (one participant, one request).
    #[test]
    fn admit_aggregates_iter_start_into_bootstrap() {
        let store = prefilled_store(&[(0, 8, 4)]);
        let mut p = pool(1, Rc::clone(&store));
        p.admit(RequestId(0));
        let mut tasks = Vec::new();
        step(&mut p, Time::ZERO, &mut tasks);
        assert_eq!(tasks.len(), 1, "one fused prolog");
        let t = &tasks[0];
        assert!(matches!(t.kind, FfnTaskKind::Bootstrap));
        assert_eq!(t.reqs, vec![RequestId(0)]);
        assert_eq!(t.pull_bytes, 0, "bootstrap input is local");
        assert!(p.slots[0].in_flight);
        assert_eq!(p.slots[0].members, vec![WorkerId(0)]);
    }

    /// Two workers, same slot, same tick → ONE `Bootstrap` carrying both requests
    /// (the defining AFD aggregation), with both workers as participants.
    #[test]
    fn two_workers_aggregate_into_one_big_prolog() {
        let store = prefilled_store(&[(0, 8, 4), (1, 8, 4)]);
        let mut p = pool(2, Rc::clone(&store));
        p.admit(RequestId(0)); // → worker 0 (least loaded)
        p.admit(RequestId(1)); // → worker 1
        assert_eq!(p.req_worker[&RequestId(0)], WorkerId(0));
        assert_eq!(p.req_worker[&RequestId(1)], WorkerId(1));
        let mut tasks = Vec::new();
        step(&mut p, Time::ZERO, &mut tasks);
        assert_eq!(tasks.len(), 1, "two workers fold into ONE prolog");
        let mut got = tasks[0].reqs.clone();
        got.sort_by_key(|r| r.0);
        assert_eq!(got, vec![RequestId(0), RequestId(1)]);
        assert_eq!(p.slots[0].members.len(), 2, "both workers participate");
    }

    /// The per-layer flush barrier: with a fast worker and a slow worker, the fast
    /// worker's layer-0 output does NOT flush a `Bridge` until the slow worker also
    /// reports layer 0 — then ONE aggregated `Bridge` carries both.
    #[test]
    fn flush_barrier_waits_for_all_participants() {
        let store = prefilled_store(&[(0, 8, 4), (1, 8, 4)]);
        let mut p = pool(2, Rc::clone(&store));
        p.admit(RequestId(0));
        p.admit(RequestId(1));
        let mut tasks = Vec::new();
        step(&mut p, Time::ZERO, &mut tasks); // prolog (Bootstrap)
        tasks.clear();

        // Scatter layer-0 QKV to both participants (as the ffn SectionReady would).
        p.scatter(0, FfnTaskKind::Bootstrap, 7, 4);
        // Only worker 0 gets ticked far enough to finish layer 0 — manually feed just
        // its output to the barrier; the barrier must NOT flush yet.
        p.on_layer_output(0, WorkerId(0), &[RequestId(0)], 0, 11, 2, &mut tasks);
        assert!(tasks.is_empty(), "barrier holds until every participant reports");
        // Worker 1 reports layer 0 → the barrier flushes ONE aggregated Bridge.
        p.on_layer_output(0, WorkerId(1), &[RequestId(1)], 0, 22, 2, &mut tasks);
        assert_eq!(tasks.len(), 1);
        assert!(matches!(tasks[0].kind, FfnTaskKind::Bridge { upstream: 0 }));
        assert_eq!(tasks[0].pull_bytes, 4, "summed attn→ffn bytes (2 + 2)");
        let mut got = tasks[0].reqs.clone();
        got.sort_by_key(|r| r.0);
        assert_eq!(got, vec![RequestId(0), RequestId(1)]);
    }

    /// The last layer flushes a `Terminal` (not a `Bridge`); `IterComplete` then
    /// releases the finished request, frees the slot, and surfaces the completion.
    #[test]
    fn last_layer_flushes_terminal_then_complete_frees_slot() {
        let store = prefilled_store(&[(0, 8, 4)]);
        let mut p = pool(1, Rc::clone(&store));
        p.admit(RequestId(0));
        let mut tasks = Vec::new();
        step(&mut p, Time::ZERO, &mut tasks); // prolog
        tasks.clear();

        // Last layer (num_layers=2 ⇒ layer 1) → Terminal.
        p.on_layer_output(0, WorkerId(0), &[RequestId(0)], 1, 9, 2, &mut tasks);
        assert_eq!(tasks.len(), 1);
        assert!(matches!(tasks[0].kind, FfnTaskKind::Terminal));

        // The ffn Terminal completes the request → release + free the slot.
        let mut completed = Vec::new();
        let ev = vec![FfnWorkerEvent::IterComplete {
            worker: WorkerId(0),
            slot: 0,
            reqs: vec![RequestId(0)],
            completed: vec![RequestId(0)],
        }];
        p.apply_ffn_events(&ev, &mut completed);
        assert_eq!(completed, vec![RequestId(0)]);
        assert!(!p.slots[0].in_flight, "slot freed after completion");
        assert!(!p.req_worker.contains_key(&RequestId(0)), "KV released");
    }

    /// Deadlock guard: a worker whose whole batch completes is stripped from the
    /// buffered next-iteration start, so the next prolog's barrier does not wait on a
    /// now-empty worker. Worker 1 keeps a continuing request and forms the next
    /// iteration alone.
    #[test]
    fn completed_worker_dropped_from_next_iteration() {
        let store = prefilled_store(&[(0, 8, 4), (1, 8, 4)]);
        let mut p = pool(2, Rc::clone(&store));
        p.admit(RequestId(0)); // worker 0
        p.admit(RequestId(1)); // worker 1
        let mut tasks = Vec::new();
        step(&mut p, Time::ZERO, &mut tasks); // prolog with both
        tasks.clear();
        assert_eq!(p.slots[0].members.len(), 2);

        // Both workers re-announce the next iteration (decode loop-back) while the
        // slot is still in-flight → buffered in `pending`.
        p.aggregate(
            &[
                AttnWorkerEvent::IterStart {
                    worker: WorkerId(0),
                    slot: 0,
                    reqs: vec![RequestId(0)],
                },
                AttnWorkerEvent::IterStart {
                    worker: WorkerId(1),
                    slot: 0,
                    reqs: vec![RequestId(1)],
                },
            ],
            &mut tasks,
        );
        assert!(tasks.is_empty(), "no prolog while in-flight");

        // The Terminal completes ONLY request 0 (worker 0's whole batch) → worker 0
        // must drop out of the next iteration; worker 1 continues.
        let mut completed = Vec::new();
        p.apply_ffn_events(
            &[FfnWorkerEvent::IterComplete {
                worker: WorkerId(0),
                slot: 0,
                reqs: vec![RequestId(0), RequestId(1)],
                completed: vec![RequestId(0)],
            }],
            &mut completed,
        );
        // Next aggregate fires the prolog from the cleaned pending: worker 1 only.
        p.aggregate(&[], &mut tasks);
        assert_eq!(tasks.len(), 1, "next iteration fires");
        assert!(matches!(tasks[0].kind, FfnTaskKind::Bootstrap));
        assert_eq!(tasks[0].reqs, vec![RequestId(1)]);
        assert_eq!(p.slots[0].members, vec![WorkerId(1)], "completed worker 0 dropped");
    }
}
