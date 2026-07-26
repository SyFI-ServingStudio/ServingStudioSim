//! `afd_attn_pool` — the attn half **and the cross-attn aggregator** of the AFD
//! flow (L6a, ref `AttentionScheduler`). It owns the [`DisaggAttnWorker`](crate::worker::DisaggAttnWorker)s
//! and does the three jobs that make AFD aggregation work. It does NOT reuse
//! `SimpleDpPoolController` (per the AFD plan): aggregation + the per-layer flush
//! barrier are the whole point, and that controller has neither.
//!
//!   1. **Placement** ([`admit`](AfdAttnPoolController::admit)): a fresh request is
//!      pinned to the least-KV-loaded attn worker (KV locality, sticky for its whole
//!      life) and admitted there with a pure `Admit`.
//!   2. **Aggregate + barrier** ([`aggregate`](AfdAttnPoolController::aggregate)):
//!      per slot it treats `IterStart` as the layer-(-1) barrier and emits one ffn
//!      prolog (`Bootstrap`) only after **every worker in the pool** has reported that
//!      slot's start boundary. `reqs` may be empty: an all-empty slot is a legal no-op
//!      iteration. It then holds the same all-worker barrier for each attention layer.
//!   3. **Scatter + complete** ([`apply_ffn_events`](AfdAttnPoolController::apply_ffn_events)):
//!      an ffn `SectionReady` fans the next layer's QKV back to **all** workers as
//!      slot-addressed `ReadyNotification`s; an ffn `IterComplete` releases finished
//!      requests' KV, surfaces their completion, and wraps every worker's slot to the
//!      next iteration.
//!
//! **All workers march every active slot in lockstep** (L6 spec; matches ref's
//! all-workers `deferred_empty_advance`, but worker-self-contained). Once any worker's
//! `IterStart` activates a slot (`in_flight`), the controller scatters that slot's
//! per-layer `ReadyNotification` to ALL workers and releases the barrier with a
//! `SlotFlushed` to ALL workers. So a worker whose slot is EMPTY for an active batch
//! still runs each layer as a 0-token no-op and reports it — keeping it a barrier
//! member. The empty slot does not burst ahead because the worker parks it in
//! `AwaitFlush` after each layer (see [`DisaggAttnWorker`]); it advances exactly one
//! layer per `SlotFlushed`. There is therefore NO per-iteration membership set: the
//! barrier divisor is the constant `workers.len()`.
//!
//! **Mid-layer vs last-layer flush.** A mid layer's barrier flushes a `Bridge{L}`
//! ffn task AND immediately sends `SlotFlushed{L}` (so workers wrap to `L+1` and pull
//! the Bridge's QKV). The LAST layer flushes a `Terminal` but does NOT send
//! `SlotFlushed`: the next iteration's `Bootstrap` consumes the `Terminal`'s output
//! (the new token), so the wrap-to-layer-0 is deferred to `finish_iteration` (after
//! the `Terminal` completes), where the slot is idle again.
//!
//! **Membership lock.** Each worker locks its local slot membership when it emits
//! `IterStart` for the layer-(-1) barrier, even if its local request set is empty.
//! Bootstrap only fires after every worker has reported, so a request admitted after
//! that report waits worker-side (`pending_insert`) for the slot's next layer-0 wrap.
//! There is no separate pool-forced membership lock.
//!
//! **Why it cannot deadlock**: every barrier waits on the constant `workers.len()`,
//! and every worker reports every active `(slot, layer)` whether its slot is empty or
//! not (it keeps lockstep-reporting 0-token layers until the batch slot goes idle
//! pool-wide). So the count is never short and no completed worker has to be stripped
//! out. All transitions are forward; a blocked slot simply waits for the ffn it
//! already dispatched.

use std::sync::Arc;

use crate::arch::contract::AttnLayerwiseModel;
use crate::common::{IdMap, PoolId, RequestId, SharedRequests, Time, WorkerId};
use crate::worker::{
    AfdAttnWorker, AttnWorkerEvent, AttnWorkerMsg, DisaggAttnWorker, FfnPullSource, FfnTask,
    FfnTaskKind, FfnWorkerEvent, SharedGpuCluster, WorkerConfig,
};

/// Pipeline depth — matches the worker's `NUM_SLOTS`. Cross-worker alignment is by
/// this slot index + layer; never by request.
const NUM_SLOTS: usize = 3;

/// Sentinel for a quiescent worker (mirrors `simple_dp`).
const NO_WAKEUP_TIME: Time = Time::from_ns(u64::MAX);

/// Split `total` ffn→attn bytes across `n` workers by their token share `weights`
/// (indexed by worker; a shorter/empty slice reads missing entries as 0). Worker `i`
/// gets `total * weights[i] / sum` — the fraction of the batch's query tokens it owns,
/// which (QKV bytes being uniform per token) is exactly its share of the transfer; an
/// empty shard gets 0. Integer-truncated per shard, so the parts can sum to `< total`
/// by at most `n - 1` bytes — negligible against a multi-KB transfer and below cache
/// resolution. `sum == 0` (only before a slot's first barrier has produced weights)
/// falls back to an even split.
fn split_by_token_share(total: u64, weights: &[u64], n: usize) -> Vec<u64> {
    let sum: u64 = weights.iter().sum();
    (0..n)
        .map(|i| {
            if sum == 0 {
                total / n as u64
            } else {
                total * weights.get(i).copied().unwrap_or(0) / sum
            }
        })
        .collect()
}

/// The flush barrier for one in-flight layer of one slot: which participants have
/// reported, plus the aggregate they contribute to the next [`FfnTask`].
#[derive(Default)]
struct LayerBarrier {
    /// The layer being collected (set from the first report; lockstep ⇒ all match).
    layer: u16,
    /// Workers that reported this layer, each with its query-token count (barrier
    /// completes at `workers.len()`). The token counts become the slot's next
    /// `scatter_weights` — the split key for the ffn→attn QKV fan-out.
    reported: Vec<(WorkerId, u64)>,
    /// Union of reported request sets — the fused ffn batch for this layer.
    reqs: Vec<RequestId>,
    /// Per-worker attn→ffn handoff sources. The source worker owns `send_gid`; the
    /// barrier preserves the producer split instead of collapsing multiple producers.
    pull_sources: Vec<FfnPullSource>,
}

/// The pre-attention barrier for one slot. It has the same fan-in shape as a layer
/// barrier, but carries only the union of requests that joined this iteration; an
/// empty union is a legal no-op iteration.
#[derive(Default)]
struct StartBarrier {
    reported: Vec<WorkerId>,
    reqs: Vec<RequestId>,
}

/// Per-slot aggregation state. An iteration runs across ALL workers in lockstep
/// (every worker reports every active `(slot, layer)`), so there is no per-iteration
/// membership set and the barrier divisor is the constant `workers.len()`.
struct SlotSched {
    /// An iteration is mid-flight: a `Bootstrap` fired and its `Terminal` has not yet
    /// completed (`finish_iteration` clears it). While set, no new `Bootstrap` fires —
    /// one iteration per slot at a time. A worker cannot re-announce until
    /// `finish_iteration` wraps it back to layer 0, so a fresh `IterStart` never lands
    /// on an in-flight slot.
    in_flight: bool,
    /// The layer-(-1) start barrier. `Bootstrap` fires only when every worker has
    /// reported this slot's start boundary, even if some or all reports carry no reqs.
    start: StartBarrier,
    /// The current layer's flush barrier (completes once ALL workers reported it).
    barrier: LayerBarrier,
    /// Per-worker query-token count (indexed by worker), refreshed from each completed
    /// barrier's reports. It is the split key for the ffn→attn `scatter`: worker `w` owns
    /// `weights[w]` of the batch's query tokens, so it receives that fraction of each
    /// layer's QKV bytes (an empty worker gets 0). Token share is fixed across a forward
    /// pass, so the value from layer `L-1`'s barrier splits layer `L`'s scatter exactly;
    /// it is carried across the `Bootstrap` (not reset) so even layer 0's scatter — which
    /// precedes this iteration's first barrier — splits by the last known share (exact
    /// for a steady decode loop). Empty only before the very first barrier ⇒ even split.
    scatter_weights: Vec<u64>,
}

impl SlotSched {
    fn new() -> Self {
        Self {
            in_flight: false,
            start: StartBarrier::default(),
            barrier: LayerBarrier::default(),
            scatter_weights: Vec::new(),
        }
    }
}

pub struct AfdAttnPoolController<W>
where
    W: AfdAttnWorker,
    W::Msg: From<AttnWorkerMsg>,
{
    workers: Vec<W>,
    /// Hot wakeup filter, parallel to `workers` (same scheme as `simple_dp`).
    worker_wakeup_times: Vec<Time>,
    /// Layer count — the barrier flushes `Bridge{L}` for `L < num_layers - 1` and
    /// `Terminal` for the last layer.
    num_layers: u16,
    slots: [SlotSched; NUM_SLOTS],
    /// req → its pinned worker (sticky, KV locality): routes `Release` on completion.
    req_worker: IdMap<RequestId, WorkerId>,
}

impl<M: AttnLayerwiseModel> AfdAttnPoolController<DisaggAttnWorker<M>> {
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
        Self::from_workers(num_layers, workers)
    }
}

impl<W> AfdAttnPoolController<W>
where
    W: AfdAttnWorker,
    W::Msg: From<AttnWorkerMsg>,
{
    /// Assemble the AFD aggregator around already-built workers. This is the
    /// construction seam for protocol variants whose worker needs additional
    /// runtime state (for example, an initial PD KV-pull FSM).
    pub fn from_workers(num_layers: u16, workers: Vec<W>) -> Self {
        assert!(
            !workers.is_empty(),
            "afd attn pool needs at least one worker"
        );
        assert!(num_layers > 0, "afd attn pool needs at least one layer");
        let n = workers.len();
        Self {
            // Start due once so every worker can publish its layer-(-1) boundary for
            // every slot. After that, wakeups are driven by admits, ffn scatters, and
            // slot flushes as usual.
            worker_wakeup_times: vec![Time::ZERO; n],
            workers,
            num_layers,
            slots: std::array::from_fn(|_| SlotSched::new()),
            req_worker: IdMap::default(),
        }
    }

    // ── Placement (called by the flow on arrival) ─────────────────────────────

    /// Pin a fresh request to the least-KV-loaded attn worker (KV locality, sticky for
    /// its whole life) and admit it there with a pure `Admit`. The worker reserves
    /// the KV, picks a local slot, and announces the iteration itself via `IterStart`.
    pub fn admit(&mut self, req: RequestId) {
        self.admit_msg(req, AttnWorkerMsg::Admit { req }.into());
    }

    /// Pin a request using the normal KV-local placement, but deliver a
    /// variant-specific message. PD-for-AFD uses this to send a handoff instead
    /// of the fresh-request `Admit` message.
    pub fn admit_msg(&mut self, req: RequestId, msg: W::Msg) {
        let idx = self.least_kv_loaded_worker();
        self.req_worker.insert(req, WorkerId(idx as u16));
        self.workers[idx].enqueue(msg);
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

    /// Fan a finished section's QKV back to **all** workers as slot-addressed
    /// `ReadyNotification`s for the downstream attn layer (`Bootstrap → 0`,
    /// `Bridge{u} → u+1`). Every worker is notified (even one whose slot is empty this
    /// iteration) so it runs the layer as a 0-token no-op and stays a barrier member —
    /// the lockstep the all-workers barrier needs. `out_bytes` is split by each worker's
    /// **token share** (`scatter_weights`, fixed for the iteration): worker `w` owns
    /// `weights[w]` of the batch's query tokens, and the QKV bytes are uniform per token,
    /// so it receives exactly that fraction — an empty shard gets 0 bytes (a 0-byte pull,
    /// but still the notification, so it advances its no-op layer). This is the exact
    /// dual of the attn→ffn pull, which is summed exactly. `total == 0` is the normal
    /// all-empty no-op control path. `Terminal` produces no `SectionReady`.
    fn scatter(&mut self, slot: usize, kind: FfnTaskKind, send_gid: u16, out_bytes: u64) {
        let layer = match kind {
            FfnTaskKind::Bootstrap => 0,
            FfnTaskKind::Bridge { upstream } => upstream + 1,
            FfnTaskKind::Terminal => {
                debug_assert!(false, "Terminal yields IterComplete, not SectionReady");
                return;
            }
        };
        let n = self.workers.len();
        let per_worker = split_by_token_share(out_bytes, &self.slots[slot].scatter_weights, n);
        for idx in 0..n {
            self.workers[idx].enqueue(
                AttnWorkerMsg::ReadyNotification {
                    slot: slot as u8,
                    layer,
                    send_gid,
                    bytes: per_worker[idx],
                }
                .into(),
            );
            self.wake(idx);
        }
    }

    /// An iteration's `Terminal` completed: release each finished request's KV on its
    /// pinned worker, record the completion, free the slot, and wrap EVERY worker's
    /// slot back to layer 0 (the deferred last-layer flush). The last layer holds its
    /// `SlotFlushed` until here — unlike a mid-layer `Bridge`, the next iteration's
    /// `Bootstrap` consumes this `Terminal`'s output, so the wrap must wait for the
    /// slot to be idle. Empty workers (parked in `AwaitFlush` at the last layer for
    /// lockstep) wrap too; continuing decoders then re-announce `IterStart` from layer
    /// 0 against an idle slot, so the next `Bootstrap` fires cleanly with no buffer.
    fn finish_iteration(
        &mut self,
        slot: usize,
        done: &[RequestId],
        completed: &mut Vec<RequestId>,
    ) {
        for &req in done {
            if let Some(w) = self.req_worker.remove(&req) {
                let idx = w.0 as usize;
                self.workers[idx].enqueue(AttnWorkerMsg::Release { req }.into());
                self.wake(idx);
                // No controller-side load counter to decrement: the worker drops the
                // request's KV on `Release`, and the pool reads `estimated_peak_kv`
                // straight from the worker at the next placement.
            }
            completed.push(req);
        }
        self.slots[slot].in_flight = false;
        self.slots[slot].barrier = LayerBarrier::default();
        // Release the last layer's barrier now that the iteration is done: wrap every
        // worker's slot (including the just-released / empty ones) from `AwaitFlush` at
        // the last layer back to layer 0. `Release` above already ran for the completed
        // requests, so a worker emptied by it wraps to a dormant layer-0 slot.
        let last = self.num_layers - 1;
        for idx in 0..self.workers.len() {
            self.workers[idx].enqueue(
                AttnWorkerMsg::SlotFlushed {
                    slot: slot as u8,
                    layer: last,
                }
                .into(),
            );
            self.wake(idx);
        }
    }

    // ── (2) Aggregate attn events: buffer starts, barrier layers, fire prologs ─

    /// Fold this tick's attn events into ffn work: `IterStart` feeds the persistent
    /// layer-(-1) start barrier for its slot, and `AttnLayerOutputsReady` feeds the
    /// per-layer flush barrier. Returns the aggregated `FfnTask`s (in `tasks`) for the
    /// flow to submit to the ffn pool.
    pub fn aggregate(&mut self, events: &[AttnWorkerEvent], tasks: &mut Vec<FfnTask>) {
        for ev in events {
            match ev {
                // PD-for-AFD flow consumes this side-band ack before calling
                // `aggregate`; it has no effect on the AFD layer barriers.
                AttnWorkerEvent::KvPullComplete { .. } => {}
                AttnWorkerEvent::IterStart { worker, slot, reqs } => {
                    self.on_iter_start(*slot as usize, *worker, reqs, tasks);
                }
                AttnWorkerEvent::AttnLayerOutputsReady {
                    worker,
                    slot,
                    reqs,
                    tokens,
                    layer,
                    send_gid,
                    bytes,
                } => {
                    let flushed = self.on_layer_output(
                        *slot as usize,
                        *worker,
                        reqs,
                        *tokens,
                        *layer,
                        *send_gid,
                        *bytes,
                        tasks,
                    );
                    // A MID layer's barrier release: advance every worker to `L+1` so it
                    // pulls the Bridge's QKV. The LAST layer holds its flush for
                    // `finish_iteration` (the next Bootstrap consumes the Terminal's output).
                    if flushed && *layer + 1 < self.num_layers {
                        self.flush_slot_layer(*slot as usize, *layer);
                    }
                }
            }
        }
    }

    /// Advance the layer-(-1) start barrier for one slot. Bootstrap fires only when
    /// every worker has reached the slot's start boundary; `reqs` may be empty, so an
    /// all-empty slot still runs the normal no-op control path.
    fn on_iter_start(
        &mut self,
        slot: usize,
        worker: WorkerId,
        reqs: &[RequestId],
        tasks: &mut Vec<FfnTask>,
    ) -> bool {
        let n = self.workers.len();
        let s = &mut self.slots[slot];
        debug_assert!(
            !s.in_flight,
            "IterStart for in-flight slot {slot} — a worker re-announced before the \
             iteration finished"
        );
        if s.in_flight {
            return false;
        }
        // Duplicate-report guard (debug-only): re-announcing a slot's start is a
        // lockstep violation that cannot happen in a valid run. Assert in debug; skip
        // the O(workers) scan in release (O(workers²) per iteration per slot at scale).
        debug_assert!(
            !s.start.reported.contains(&worker),
            "worker {worker:?} reported slot {slot}'s start twice"
        );
        s.start.reported.push(worker);
        s.start.reqs.extend_from_slice(reqs);
        if s.start.reported.len() != n {
            return false;
        }

        let reqs = std::mem::take(&mut s.start.reqs);
        s.start.reported.clear();
        self.fire_bootstrap(slot, reqs, tasks);
        true
    }

    /// Advance a slot's flush barrier with one worker's layer output; pure barrier
    /// accounting with NO worker side-effects. Returns `true` when this report COMPLETED
    /// the barrier (every worker has now reported — the constant `workers.len()`), having
    /// pushed ONE aggregated `FfnTask` for that layer (`Bridge{L}` mid, `Terminal` last).
    /// The caller releases a mid-layer barrier to the workers via [`flush_slot_layer`];
    /// the last layer's release is deferred to `finish_iteration` (the next `Bootstrap`
    /// consumes the `Terminal`'s output, so the wrap-to-layer-0 must wait until idle).
    #[allow(clippy::too_many_arguments)]
    fn on_layer_output(
        &mut self,
        slot: usize,
        worker: WorkerId,
        reqs: &[RequestId],
        tokens: u64,
        layer: u16,
        send_gid: u16,
        bytes: u64,
        tasks: &mut Vec<FfnTask>,
    ) -> bool {
        let num_layers = self.num_layers;
        let n = self.workers.len();
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
        // Duplicate-report guard (debug-only): a worker reporting the same (slot, layer)
        // twice breaks lockstep and cannot happen in a valid run. The O(workers) scan
        // per report is O(workers²) per completed layer-barrier — small today but scales
        // with the attn pool — so run it only under `debug_assert` and push in release.
        debug_assert!(
            !s.barrier.reported.iter().any(|&(w, _)| w == worker),
            "worker {worker:?} reported slot {slot} twice in one layer"
        );
        s.barrier.reported.push((worker, tokens));
        s.barrier.reqs.extend_from_slice(reqs);
        s.barrier
            .pull_sources
            .push(FfnPullSource { send_gid, bytes });
        if s.barrier.reported.len() != n {
            return false;
        }
        // Barrier complete: ALL workers reported this layer → flush ONE aggregated task.
        let kind = if layer + 1 >= num_layers {
            FfnTaskKind::Terminal
        } else {
            FfnTaskKind::Bridge { upstream: layer }
        };
        let b = std::mem::take(&mut s.barrier);
        // Refresh the ffn→attn scatter split from this layer's per-worker token counts,
        // reusing the slot's `scatter_weights` allocation (fully overwritten each
        // completed barrier) instead of allocating a fresh `vec![0; n]` per layer.
        s.scatter_weights.clear();
        s.scatter_weights.resize(n, 0);
        for &(w, t) in &b.reported {
            s.scatter_weights[w.0 as usize] = t;
        }
        // Sum the per-shard query-token counts the workers already reported (the same
        // numbers feeding `scatter_weights`). The shards partition `reqs`, so this
        // equals a fresh `workload_tokens(reqs)` scan on the ffn side — hand it over
        // so the ffn worker never re-walks the store per layer.
        let total_tokens: u64 = b.reported.iter().map(|&(_, t)| t).sum();
        tasks.push(FfnTask {
            kind,
            slot: slot as u8,
            reqs: b.reqs,
            pull_sources: b.pull_sources,
            tokens: total_tokens,
        });
        true
    }

    /// Release a completed MID-layer barrier: advance every worker's slot from
    /// `AwaitFlush` at `layer` to `Wait` at `layer + 1` (where it pulls the Bridge's
    /// QKV). All workers are parked in `AwaitFlush` at `layer` by the time the barrier
    /// completes, so this fans `SlotFlushed` to all of them.
    fn flush_slot_layer(&mut self, slot: usize, layer: u16) {
        for idx in 0..self.workers.len() {
            self.workers[idx].enqueue(
                AttnWorkerMsg::SlotFlushed {
                    slot: slot as u8,
                    layer,
                }
                .into(),
            );
            self.wake(idx);
        }
    }

    /// Fire the ffn prolog for an idle slot: one fused `Bootstrap` carrying the union of
    /// all workers' `IterStart` request sets, possibly empty. Marks the slot in-flight
    /// and resets the layer barrier. Membership was already locked worker-side when each
    /// worker emitted its `IterStart`; the prolog's layer-0 QKV returns via `scatter` to
    /// ALL workers.
    fn fire_bootstrap(&mut self, slot: usize, reqs: Vec<RequestId>, tasks: &mut Vec<FfnTask>) {
        debug_assert!(
            !self.slots[slot].in_flight,
            "double Bootstrap for slot {slot}"
        );
        self.slots[slot].in_flight = true;
        self.slots[slot].start = StartBarrier::default();
        self.slots[slot].barrier = LayerBarrier::default();
        // Note: `scatter_weights` is NOT reset here — it carries over from the previous
        // iteration's last barrier, so layer 0's scatter (which precedes this iteration's
        // first barrier) still splits by a token share, exact for a steady decode loop.
        tasks.push(FfnTask {
            kind: FfnTaskKind::Bootstrap,
            slot: slot as u8,
            reqs,
            pull_sources: Vec::new(),
            // A Bootstrap precedes any attn layer-output, so no per-shard count exists
            // yet; the worker fills `tokens` from the store once at `start_compute`.
            tokens: 0,
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

    fn pool(
        num_workers: u16,
        store: SharedRequests,
    ) -> AfdAttnPoolController<DisaggAttnWorker<FakeAttn>> {
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
    fn step(
        p: &mut AfdAttnPoolController<DisaggAttnWorker<FakeAttn>>,
        now: Time,
        tasks: &mut Vec<FfnTask>,
    ) {
        let mut events = Vec::new();
        p.tick_collect(now, &mut events);
        p.aggregate(&events, tasks);
    }

    fn task_for_slot(tasks: &[FfnTask], slot: u8) -> &FfnTask {
        tasks
            .iter()
            .find(|t| t.slot == slot)
            .unwrap_or_else(|| panic!("missing task for slot {slot}"))
    }

    /// Drive every worker through ONE attn layer end-to-end: scatter that layer's input
    /// QKV (simulating the ffn `SectionReady`; 0 bytes ⇒ no real transfer / sender needed),
    /// then tick + aggregate until the barrier flushes its `FfnTask` and (mid layer)
    /// releases the workers. Mirrors a real flow step, so the workers legitimately park
    /// in — and are released from — `AwaitFlush`. `upstream` is the ffn task whose output
    /// feeds this layer (`Bootstrap → 0`, `Bridge{u} → u+1`). Returns the flushed tasks.
    fn run_layer(
        p: &mut AfdAttnPoolController<DisaggAttnWorker<FakeAttn>>,
        slot: usize,
        upstream: FfnTaskKind,
        t0: u64,
    ) -> Vec<FfnTask> {
        p.scatter(slot, upstream, 0, 0);
        let mut tasks = Vec::new();
        for s in 0..40u64 {
            let mut events = Vec::new();
            p.tick_collect(Time::from_ms((t0 + s) as f64), &mut events);
            p.aggregate(&events, &mut tasks);
        }
        tasks
    }

    /// `admit` pins a request and the worker's `IterStart` aggregates into a single
    /// `Bootstrap` prolog (one request), marking the slot in-flight.
    #[test]
    fn admit_aggregates_iter_start_into_bootstrap() {
        let store = prefilled_store(&[(0, 8, 4)]);
        let mut p = pool(1, Rc::clone(&store));
        p.admit(RequestId(0));
        let mut tasks = Vec::new();
        step(&mut p, Time::ZERO, &mut tasks);
        assert_eq!(tasks.len(), NUM_SLOTS, "one prolog per slot boundary");
        let t = task_for_slot(&tasks, 0);
        assert!(matches!(t.kind, FfnTaskKind::Bootstrap));
        assert_eq!(t.reqs, vec![RequestId(0)]);
        assert!(t.pull_sources.is_empty(), "bootstrap input is local");
        assert!(p.slots[0].in_flight);
        for slot in 1..NUM_SLOTS {
            let t = task_for_slot(&tasks, slot as u8);
            assert!(matches!(t.kind, FfnTaskKind::Bootstrap));
            assert!(t.reqs.is_empty(), "empty slot {slot} is a no-op Bootstrap");
        }
    }

    /// Two workers, same slot, same tick → ONE `Bootstrap` carrying both requests
    /// (the defining AFD aggregation).
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
        assert_eq!(tasks.len(), NUM_SLOTS, "one prolog per slot boundary");
        let mut got = task_for_slot(&tasks, 0).reqs.clone();
        got.sort_by_key(|r| r.0);
        assert_eq!(got, vec![RequestId(0), RequestId(1)]);
        assert!(p.slots[0].in_flight);
        for slot in 1..NUM_SLOTS {
            assert!(
                task_for_slot(&tasks, slot as u8).reqs.is_empty(),
                "slot {slot} has only empty start reports"
            );
        }
    }

    /// Bootstrap is the layer-(-1) barrier: a non-empty worker's `IterStart` is not
    /// enough until every worker, including empty shards, has reported the same slot.
    #[test]
    fn bootstrap_start_barrier_waits_for_empty_worker() {
        let store = prefilled_store(&[(0, 8, 4)]);
        let mut p = pool(2, Rc::clone(&store));
        let mut tasks = Vec::new();

        assert!(!p.on_iter_start(0, WorkerId(0), &[RequestId(0)], &mut tasks));
        assert!(tasks.is_empty(), "start barrier waits for worker 1");
        assert!(p.on_iter_start(0, WorkerId(1), &[], &mut tasks));

        assert_eq!(tasks.len(), 1);
        assert!(matches!(tasks[0].kind, FfnTaskKind::Bootstrap));
        assert_eq!(tasks[0].slot, 0);
        assert_eq!(tasks[0].reqs, vec![RequestId(0)]);
        assert!(p.slots[0].in_flight);
    }

    /// All-empty start barriers are still legal no-op iterations. The FFN worker
    /// handles the resulting empty task without model lookup.
    #[test]
    fn all_empty_start_barrier_still_fires_noop_bootstrap() {
        let store = prefilled_store(&[]);
        let mut p = pool(2, Rc::clone(&store));
        let mut tasks = Vec::new();

        assert!(!p.on_iter_start(0, WorkerId(0), &[], &mut tasks));
        assert!(p.on_iter_start(0, WorkerId(1), &[], &mut tasks));

        assert_eq!(tasks.len(), 1);
        assert!(matches!(tasks[0].kind, FfnTaskKind::Bootstrap));
        assert_eq!(tasks[0].slot, 0);
        assert!(tasks[0].reqs.is_empty());
        assert!(p.slots[0].in_flight);
    }

    /// The per-layer flush barrier waits for EVERY worker (`workers.len()`): the first
    /// worker's layer-0 output does NOT flush a `Bridge` until the second also reports
    /// layer 0 — then ONE aggregated `Bridge` carries both. (The mid-layer `SlotFlushed`
    /// fan-out to all workers is exercised end-to-end by the afd.rs flow tests.)
    #[test]
    fn flush_barrier_waits_for_all_workers() {
        let store = prefilled_store(&[(0, 8, 4), (1, 8, 4)]);
        let mut p = pool(2, Rc::clone(&store));
        p.admit(RequestId(0));
        p.admit(RequestId(1));
        let mut tasks = Vec::new();
        step(&mut p, Time::ZERO, &mut tasks); // prolog (Bootstrap)
        tasks.clear();

        // Feed only worker 0's layer-0 output to the barrier; it must NOT flush yet
        // (workers.len() == 2, only 1 reported). `on_layer_output` is pure barrier
        // accounting — the mid-layer SlotFlushed fan-out is the caller's job (aggregate),
        // exercised end-to-end by the afd.rs flow tests. Args: (slot, worker, reqs,
        // tokens, layer, send_gid, bytes).
        assert!(!p.on_layer_output(0, WorkerId(0), &[RequestId(0)], 1, 0, 11, 2, &mut tasks));
        assert!(tasks.is_empty(), "barrier holds until every worker reports");
        // Worker 1 reports layer 0 → the barrier completes and flushes ONE Bridge.
        assert!(p.on_layer_output(0, WorkerId(1), &[RequestId(1)], 1, 0, 22, 2, &mut tasks));
        assert_eq!(tasks.len(), 1);
        assert!(matches!(tasks[0].kind, FfnTaskKind::Bridge { upstream: 0 }));
        assert_eq!(
            tasks[0].pull_sources,
            vec![
                FfnPullSource {
                    send_gid: 11,
                    bytes: 2,
                },
                FfnPullSource {
                    send_gid: 22,
                    bytes: 2,
                },
            ],
            "attn→ffn sources stay split by producer worker"
        );
        let mut got = tasks[0].reqs.clone();
        got.sort_by_key(|r| r.0);
        assert_eq!(got, vec![RequestId(0), RequestId(1)]);
    }

    /// The last layer flushes a `Terminal` (not a `Bridge`) but holds its `SlotFlushed`;
    /// the ffn `IterComplete` then releases the finished request, frees the slot, and —
    /// being the deferred last-layer flush — wraps the worker and surfaces the completion.
    /// The worker is driven through both layers for real so it legitimately parks in
    /// `AwaitFlush` at the last layer when `IterComplete`'s `SlotFlushed` arrives.
    #[test]
    fn last_layer_flushes_terminal_then_complete_frees_slot() {
        let store = prefilled_store(&[(0, 8, 4)]);
        let mut p = pool(1, Rc::clone(&store));
        p.admit(RequestId(0));
        let mut tasks = Vec::new();
        step(&mut p, Time::ZERO, &mut tasks); // prolog (Bootstrap)
        assert!(matches!(tasks[0].kind, FfnTaskKind::Bootstrap));

        // Layer 0 flushes a Bridge; layer 1 (the last, num_layers=2) flushes a Terminal.
        let l0 = run_layer(&mut p, 0, FfnTaskKind::Bootstrap, 1);
        assert_eq!(l0.len(), 1);
        assert!(matches!(l0[0].kind, FfnTaskKind::Bridge { upstream: 0 }));
        let l1 = run_layer(&mut p, 0, FfnTaskKind::Bridge { upstream: 0 }, 50);
        assert_eq!(l1.len(), 1);
        assert!(matches!(l1[0].kind, FfnTaskKind::Terminal));
        assert!(
            p.slots[0].in_flight,
            "Terminal does not free the slot — IterComplete does"
        );

        // The ffn Terminal completes the request → release + free the slot + wrap worker.
        let mut completed = Vec::new();
        p.apply_ffn_events(
            &[FfnWorkerEvent::IterComplete {
                worker: WorkerId(0),
                slot: 0,
                reqs: vec![RequestId(0)],
                completed: vec![RequestId(0)],
            }],
            &mut completed,
        );
        assert_eq!(completed, vec![RequestId(0)]);
        assert!(!p.slots[0].in_flight, "slot freed after completion");
        assert!(!p.req_worker.contains_key(&RequestId(0)), "KV released");
    }

    /// All-workers barrier (the L6 spec): a worker with NO request in a slot is STILL a
    /// required member — it lockstep-runs a 0-token layer — so the barrier divisor is
    /// the constant `workers.len()`, never an opportunistic subset (and a completed
    /// worker that goes empty is therefore never "dropped"). With the lone request on
    /// worker 0, the slot's barrier does not flush until the empty worker 1 also reports.
    #[test]
    fn empty_worker_is_a_required_barrier_member() {
        let store = prefilled_store(&[(0, 8, 4)]);
        let mut p = pool(2, Rc::clone(&store));
        p.admit(RequestId(0)); // → worker 0; worker 1 has nothing in slot 0
        let mut tasks = Vec::new();
        step(&mut p, Time::ZERO, &mut tasks); // Bootstrap [r0], slot 0 in-flight
        tasks.clear();

        // Worker 0 (the only one with a request) reports layer 0 — NOT enough.
        // Args: (slot, worker, reqs, tokens, layer, send_gid, bytes).
        assert!(!p.on_layer_output(0, WorkerId(0), &[RequestId(0)], 1, 0, 11, 2, &mut tasks));
        assert!(tasks.is_empty(), "1 of 2 workers reported — barrier holds");
        // The empty worker 1 reports its 0-token layer → barrier completes (workers.len()).
        assert!(p.on_layer_output(0, WorkerId(1), &[], 0, 0, 22, 0, &mut tasks));
        assert_eq!(
            tasks.len(),
            1,
            "barrier completes once EVERY worker reports"
        );
        assert!(matches!(tasks[0].kind, FfnTaskKind::Bridge { upstream: 0 }));
        assert_eq!(
            tasks[0].reqs,
            vec![RequestId(0)],
            "empty worker adds no tokens"
        );
        assert_eq!(
            tasks[0].pull_sources,
            vec![
                FfnPullSource {
                    send_gid: 11,
                    bytes: 2,
                },
                FfnPullSource {
                    send_gid: 22,
                    bytes: 0,
                },
            ],
            "source metadata preserves the empty worker's zero-byte report"
        );
    }

    /// The ffn→attn scatter splits by each worker's token share, NOT evenly: a shard
    /// owning more query tokens pulls proportionally more of the QKV; an empty shard
    /// pulls 0. Before any barrier (no weights yet) it falls back to an even split.
    #[test]
    fn scatter_splits_bytes_by_token_share() {
        // 3 shards holding 8 / 2 / 0 tokens → 100 bytes split 80 / 20 / 0.
        assert_eq!(split_by_token_share(100, &[8, 2, 0], 3), vec![80, 20, 0]);
        // Single busy shard owns all the bytes (the old members-split result).
        assert_eq!(split_by_token_share(100, &[5, 0], 2), vec![100, 0]);
        // No weights yet (before the first barrier) ⇒ even split.
        assert_eq!(split_by_token_share(9, &[], 3), vec![3, 3, 3]);
        // Integer truncation loses at most n-1 bytes (7/3 each = 2, sum 6 ≤ 7).
        assert_eq!(split_by_token_share(7, &[1, 1, 1], 3), vec![2, 2, 2]);
    }
}
