//! `simple_dp` — a reusable DP pool controller plus the single-pool unified flow.
//! L6a (`SimpleDpPoolController`) + L6b (`SimpleDpFlow`) live in one file because
//! the deployment is tiny, but stay two structs (L6 design.md §「simple DP」).

use std::path::PathBuf;

use crate::common::{PoolId, Request, RequestId, SharedRequests, Time, WorkerId};
use crate::worker::{
    CostSource, GpuCluster, IterWorker, MigratableWorker, SharedGpuCluster, WorkerEventCommon,
    WorkerMsgCommon,
};

use super::super::migration::{MigrationOrder, MigrationPolicy, MigrationTrigger, WorkerLoad};
use super::super::training::{TrainingConfig, TrainingPool};
use super::super::{BackgroundWork, Flow, OrchAction, WorkerFactory};

/// Sentinel for a quiescent worker. `Option<Time>` would add a tag; the simulator
/// clock is a `u64` newtype, so this keeps the hot wakeup array dense while still
/// carrying a typed timestamp.
const NO_WAKEUP_TIME: Time = Time::from_ns(u64::MAX);

// ── Configs & policy (simple_dp-specific; L6 design.md §「simple DP」) ──────────

/// Worker placement within a DP pool.
///
/// `TraceDirected` is not a load heuristic: each arrival names its own worker
/// through the trace's `placement` tag, and the pool obeys it. That makes a
/// placement sequence reproducible, which a load metric cannot be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DpPlacementPolicy {
    LeastQueued,
    RoundRobin,
    TraceDirected,
}

/// Config for one DP pool — pure orchestration (which workers, how to place).
/// The GPU facts each worker spans live behind its [`WorkerFactory`] and on the
/// worker's paired model/arch contract, not here.
pub struct SimpleDpPoolConfig {
    pub pool: PoolId,
    pub num_workers: u16,
    pub placement: DpPlacementPolicy,
}

/// Config for the whole simple_dp deployment.
pub struct SimpleDpConfig {
    pub dp_pool: SimpleDpPoolConfig,
    /// The pool's migration hook, or `None` for the default: never migrate.
    ///
    /// `None` rather than a do-nothing policy so a pool that does not opt in
    /// pays nothing — no per-tick virtual call and no load snapshot — and every
    /// existing preset keeps today's tick path instruction for instruction.
    pub migration: Option<MigrationTrigger>,
    /// The RL training side, or `None` for the default: the run ends when
    /// generation does. Same reasoning as `migration` — off means the objects
    /// are never built, not that they sit idle.
    pub training: Option<TrainingConfig>,
    /// Where the training blocks write their `cost_log`. Inference workers get
    /// theirs through the factory; the training blocks are built here, so this
    /// is their path in.
    pub log_dir: Option<PathBuf>,
}

// ── L6a: pool-local orchestration ─────────────────────────────────────────────

pub struct SimpleDpPoolController<W: IterWorker> {
    pool: PoolId,
    workers: Vec<W>,
    /// Hot wakeup filter: `NO_WAKEUP_TIME` means quiescent; any real timestamp is
    /// the earliest sim time at which the worker may advance. The L7 driver still
    /// calls the flow at its configured tick cadence, so this wakes on the first
    /// outer tick whose `now >= wakeup_time`.
    worker_wakeup_times: Vec<Time>,
    placement: DpPlacementPolicy,
    rr_next: usize,
    /// Current host of each worker index: `redirect[i] == WorkerId(i)` while
    /// worker `i` is live, and the worker it was drained onto once it retires.
    /// A trace-declared target is resolved through this, so a request pinned to
    /// a retired worker follows the work that already left it rather than
    /// refilling a machine the pool has decided to empty.
    ///
    /// Only ever written by `migrate`, which retires the source and requires a
    /// live destination — so the chain is at most one hop and cannot cycle.
    redirect: Vec<WorkerId>,
    /// Prompt-group bookkeeping, present only while a migration policy is
    /// installed. Nothing else in the pool needs it, and a pool that never
    /// migrates should not pay to maintain it.
    groups: Option<GroupLedger>,
}

/// Which worker holds each prompt group, and how much of each group is left.
///
/// A *prompt group* is a block of `size` consecutive request ids. The group is
/// in flight until its last member completes — the reading an RL rollout needs,
/// because a training step cannot consume a prompt's samples until the slowest
/// one lands, so a nearly-finished group still pins a worker.
///
/// Derived from the id rather than declared per request: the trace formats that
/// carry grouped work number their samples in contiguous blocks, and a column
/// would have to be validated against that anyway. A mismatch is caught, not
/// tolerated — see the split-group assertion in `arrived`.
#[derive(Debug)]
struct GroupLedger {
    size: u32,
    /// Worker currently holding each group, by group id.
    host: Vec<WorkerId>,
    /// Members of each group not yet complete, by group id.
    unfinished: Vec<u32>,
    /// Groups with at least one unfinished member, by worker index.
    per_worker: Vec<u32>,
    /// Requests completed on each worker. A policy that fires on a drop in
    /// load needs to tell "this block has drained" from "this block has not
    /// started yet", and the two look identical in `per_worker` alone.
    completed_per_worker: Vec<u32>,
}

impl GroupLedger {
    fn new(size: u32, num_workers: usize) -> Self {
        assert!(size > 0, "a prompt group needs at least one request");
        Self {
            size,
            host: Vec::new(),
            unfinished: Vec::new(),
            per_worker: vec![0; num_workers],
            completed_per_worker: vec![0; num_workers],
        }
    }

    fn group_of(&self, request: RequestId) -> usize {
        request.0 as usize / self.size as usize
    }

    fn arrived(&mut self, request: RequestId, worker: WorkerId) {
        let group = self.group_of(request);
        if group >= self.unfinished.len() {
            self.unfinished.resize(group + 1, 0);
            self.host.resize(group + 1, worker);
        }
        if self.unfinished[group] == 0 {
            self.host[group] = worker;
            self.per_worker[worker.0 as usize] += 1;
        } else {
            assert_eq!(
                self.host[group], worker,
                "prompt group {group} is split across workers {:?} and {worker:?}; \
                 a group-counting migration policy needs each group on one worker",
                self.host[group],
            );
        }
        self.unfinished[group] += 1;
    }

    /// Retire one member, returning the group if that was its last one —
    /// the moment the group becomes useful to a trainer, and the only moment
    /// anything downstream cares about.
    fn completed(&mut self, request: RequestId) -> Option<usize> {
        let group = self.group_of(request);
        let host = self.host[group].0 as usize;
        let left = &mut self.unfinished[group];
        assert!(*left > 0, "prompt group {group} completed more than it holds");
        *left -= 1;
        let emptied = *left == 0;
        if emptied {
            self.per_worker[host] -= 1;
        }
        self.completed_per_worker[host] += 1;
        emptied.then_some(group)
    }

    /// Groups still in flight on `worker`, lowest id first — the order a
    /// `Scatter` deals destinations out in.
    fn resident(&self, worker: WorkerId, out: &mut Vec<usize>) {
        out.clear();
        out.extend(
            (0..self.unfinished.len())
                .filter(|group| self.unfinished[*group] > 0 && self.host[*group] == worker),
        );
    }

    /// The lowest-id group still in flight on `worker` — the one a `MoveGroup`
    /// hands over. Same ordering as [`Self::resident`], so a release that moves
    /// its groups one at a time visits them in the order a `Scatter` would have
    /// dealt them out.
    fn first_resident(&self, worker: WorkerId) -> Option<usize> {
        (0..self.unfinished.len())
            .find(|group| self.unfinished[*group] > 0 && self.host[*group] == worker)
    }

    /// Every request id belonging to `group`, finished or not. Ids are numbered
    /// in consecutive blocks, so this is arithmetic rather than a lookup — and
    /// the caller hands the whole block to the worker, which knows which of them
    /// it is actually still holding.
    fn members(&self, group: usize) -> Vec<RequestId> {
        let first = group as u32 * self.size;
        (first..first + self.size).map(RequestId).collect()
    }

    fn moved(&mut self, group: usize, to: WorkerId) {
        let from = self.host[group];
        if from == to {
            return;
        }
        self.per_worker[from.0 as usize] -= 1;
        self.per_worker[to.0 as usize] += 1;
        self.host[group] = to;
    }
}

impl<W: IterWorker> SimpleDpPoolController<W> {
    // ── Construction ──────────────────────────────────────────────────────────
    /// Build the pool's workers, each handed the shared `cluster` so it can
    /// self-register its GPU block. Sharing one cluster across pools keeps GPU
    /// ids globally unique — a multi-pool deployment threads the same cluster
    /// into every pool's `new`, so ids continue (`allocate` appends from the
    /// current length) instead of every pool restarting at 0.
    pub fn new<F>(cfg: &SimpleDpPoolConfig, factory: &F, cluster: &SharedGpuCluster) -> Self
    where
        F: WorkerFactory<W>,
    {
        assert!(cfg.num_workers > 0, "simple_dp needs at least one worker");
        let workers: Vec<W> = (0..cfg.num_workers)
            .map(|i| factory.build(i, cfg.pool, cluster))
            .collect();
        Self {
            pool: cfg.pool,
            worker_wakeup_times: vec![NO_WAKEUP_TIME; workers.len()],
            redirect: (0..workers.len() as u16).map(WorkerId).collect(),
            workers,
            placement: cfg.placement,
            rr_next: 0,
            groups: None,
        }
    }

    /// Start counting prompt groups of `size` requests. Called once, when a
    /// migration policy is installed; a pool without one never tracks them.
    pub fn track_groups(&mut self, size: u32) {
        self.groups = Some(GroupLedger::new(size, self.workers.len()));
    }

    /// Record that `request` has been placed on `worker`. Separate from the
    /// enqueue itself because a resume is not an arrival: the ledger already
    /// counts a migrated request, and re-counting it would double its group.
    pub fn note_arrival(&mut self, request: RequestId, worker: WorkerId) {
        if let Some(groups) = self.groups.as_mut() {
            groups.arrived(request, worker);
        }
    }

    /// Record a completion, returning the prompt group it just finished off (if
    /// any). A pool that tracks no groups returns `None` for every request.
    pub fn note_completion(&mut self, request: RequestId) -> Option<usize> {
        self.groups
            .as_mut()
            .and_then(|groups| groups.completed(request))
    }

    /// Every request id of `group`. Panics if the pool tracks no groups — a
    /// caller asking about one has already been told a group exists.
    pub fn group_members(&self, group: usize) -> Vec<RequestId> {
        self.groups
            .as_ref()
            .expect("group membership needs the ledger a group-aware policy installs")
            .members(group)
    }

    // ── Outward API (called by L6b) ───────────────────────────────────────────

    /// Pick a worker via the pool's placement policy and enqueue `msg` into it.
    /// Deployment-agnostic primitive over `W::Msg`: the unified flow admits a
    /// plain `Request` (via [`Self::admit`] convenience); a PD flow uses this
    /// to enqueue a `Handoff` whose content does not depend on the chosen
    /// worker — the receiver fills in its own destination block.
    pub fn admit_msg(&mut self, msg: W::Msg) {
        let idx = self.choose_worker_idx();
        self.workers[idx].enqueue(msg);
        // An idle worker has no scheduled wakeup; new work must make it due
        // immediately. A computing worker already has a `compute_end` wakeup, so
        // keep that instead of forcing an early poll.
        if self.worker_wakeup_times[idx] == NO_WAKEUP_TIME {
            self.worker_wakeup_times[idx] = Time::ZERO;
        }
    }

    /// Route `msg` to a *specific* worker (bypasses placement). Used for
    /// targeted acks like PD's `ReleaseKv`, where the message must reach the
    /// exact worker that holds the addressed request's KV. Panics if
    /// `worker_id` is outside this pool — silently dropping the ack would
    /// leak the held reservation forever, so a bad id is treated as a bug.
    pub fn route_msg_to(&mut self, worker_id: WorkerId, msg: W::Msg) {
        self.enqueue_at(worker_id.0 as usize, msg);
    }

    fn enqueue_at(&mut self, idx: usize, msg: W::Msg) {
        self.workers[idx].enqueue(msg);
        if self.worker_wakeup_times[idx] == NO_WAKEUP_TIME {
            self.worker_wakeup_times[idx] = Time::ZERO;
        }
    }

    fn choose_worker_idx(&mut self) -> usize {
        match self.placement {
            DpPlacementPolicy::LeastQueued => self.choose_least_queued_idx(),
            DpPlacementPolicy::RoundRobin => self.choose_round_robin_idx(),
            // A trace-directed pool never asks a policy which worker to use —
            // the flow reads the request's own target and calls `route_msg_to`.
            DpPlacementPolicy::TraceDirected => {
                unreachable!("a trace-directed pool routes through route_msg_to, not placement")
            }
        }
    }

    fn choose_least_queued_idx(&self) -> usize {
        self.workers
            .iter()
            .enumerate()
            .min_by_key(|(idx, w)| {
                let s = w.status();
                (s.queued_requests, s.active_requests, *idx)
            })
            .map(|(idx, _)| idx)
            .expect("simple_dp has at least one worker")
    }

    fn choose_round_robin_idx(&mut self) -> usize {
        let idx = self.rr_next;
        self.rr_next = (self.rr_next + 1) % self.workers.len();
        idx
    }

    pub fn pool(&self) -> PoolId {
        self.pool
    }

    pub fn placement(&self) -> DpPlacementPolicy {
        self.placement
    }

    pub fn num_workers(&self) -> usize {
        self.workers.len()
    }

    /// Resolve a trace-declared worker to the one currently hosting its work.
    ///
    /// Identity until something migrates. Panics on an out-of-pool id for the
    /// same reason `route_msg_to` does: the alternative is a request that
    /// silently never runs.
    pub fn resolve(&self, worker: WorkerId) -> WorkerId {
        let mut current = worker;
        for _ in 0..=self.redirect.len() {
            let next = self.redirect[current.0 as usize];
            if next == current {
                return current;
            }
            current = next;
        }
        unreachable!("redirect chain cycled; migrate never retires a destination")
    }

    /// Read every worker's load for a migration policy. Retired workers stay in
    /// the snapshot so a policy sees the whole pool, not just its live part.
    pub fn snapshot_loads(&self, out: &mut Vec<WorkerLoad>) {
        out.clear();
        out.extend(self.workers.iter().enumerate().map(|(idx, worker)| {
            let status = worker.status();
            WorkerLoad {
                worker: WorkerId(idx as u16),
                queued_requests: status.queued_requests,
                active_requests: status.active_requests,
                in_flight_groups: self
                    .groups
                    .as_ref()
                    .map_or(0, |groups| groups.per_worker[idx]),
                completed_requests: self
                    .groups
                    .as_ref()
                    .map_or(0, |groups| groups.completed_per_worker[idx]),
                retired: self.redirect[idx] != WorkerId(idx as u16),
            }
        }));
    }

    // ── Tick driving ──────────────────────────────────────────────────────────
    /// Sweep worker wakeup times and tick only due workers, each pushing its
    /// self-tagged events into the caller's sink. One sweep — no separate drain
    /// pass; the worker already stamps its id so the pool needs no per-worker
    /// attribution loop. The sink type is `Vec<W::Event>` so each pool gets its
    /// own role-specific event stream (a barebone pool's sink takes
    /// `WorkerEventCommon`, a PD prefill pool's sink takes `PdPrefillEvent`,
    /// etc.).
    pub fn tick_collect(&mut self, now: Time, events: &mut Vec<W::Event>) {
        debug_assert_eq!(self.worker_wakeup_times.len(), self.workers.len());
        for (wakeup_time, worker) in self
            .worker_wakeup_times
            .iter_mut()
            .zip(self.workers.iter_mut())
        {
            if *wakeup_time <= now {
                *wakeup_time = worker.tick(now, events).unwrap_or(NO_WAKEUP_TIME);
            }
        }
    }
}

// Convenience: every Msg enum impls `From<RequestId>` so the universal
// "admit a request" entry point can stay one method. Decoupled into its own
// `impl` block so workers whose Msg does not (or cannot) carry Request — none
// today, but the bound keeps the trait surface minimal — still compile.
impl<W: IterWorker> SimpleDpPoolController<W>
where
    W::Msg: From<RequestId>,
{
    /// Admit a fresh request (the universal entry) — wraps in the chosen
    /// worker's `W::Msg::from(rid)`. PD-side admits via `admit_msg` directly.
    pub fn admit(&mut self, rid: RequestId) {
        let idx = self.choose_worker_idx();
        self.note_arrival(rid, WorkerId(idx as u16));
        self.enqueue_at(idx, W::Msg::from(rid));
    }

    /// Admit a fresh request to a *named* worker — the trace-directed entry.
    /// Unlike `route_msg_to` this is an arrival, so it reaches the ledger.
    pub fn admit_to(&mut self, worker: WorkerId, rid: RequestId) {
        self.note_arrival(rid, worker);
        self.enqueue_at(worker.0 as usize, W::Msg::from(rid));
    }
}

// Migration is a capability, not part of the pool contract: the extra bounds
// live on their own impl block so PD's pools — which reuse this controller with
// a non-migratable worker — are unaffected.
impl<W: MigratableWorker> SimpleDpPoolController<W>
where
    W::Msg: From<WorkerMsgCommon>,
{
    /// Execute one order: drain `src`, retire it onto `dst`, and re-admit every
    /// drained request there. Returns how many requests moved.
    ///
    /// L6 never learns what happens to those requests. It asks the source to
    /// give them up and hands the ids to the destination; whether the
    /// destination recomputes them, and how much work that is, belongs to L5.
    ///
    /// Panics on `src == dst`, an out-of-pool id, or a destination that has
    /// itself been retired. All three are policy bugs, and quietly skipping any
    /// of them would strand requests on a worker nothing routes to again.
    pub fn migrate(
        &mut self,
        now: Time,
        order: &MigrationOrder,
        scratch: &mut Vec<RequestId>,
    ) -> usize {
        let src = order.src().0 as usize;
        assert!(
            src < self.workers.len(),
            "migration {order:?} names a worker outside a pool of {}",
            self.workers.len()
        );
        match order {
            MigrationOrder::Consolidate { src: _, dst } => {
                self.check_destination(order, *dst);
                assert_ne!(order.src(), *dst, "a migration must name two workers");
                scratch.clear();
                self.workers[src].drain_resident(now, scratch);
                self.retire(src, *dst);
                for request in std::mem::take(scratch) {
                    self.resume_on(now, *dst, request);
                    scratch.push(request);
                }
                scratch.len()
            }
            MigrationOrder::Scatter {
                src: _,
                dsts,
                retire_to,
            } => {
                // Where each group is must be read before the drain: draining
                // is what makes the source's residency empty, and the ledger is
                // the only record of which request belongs to which group.
                let mut resident = Vec::new();
                if let Some(groups) = self.groups.as_ref() {
                    groups.resident(order.src(), &mut resident);
                }
                assert_eq!(
                    resident.len(),
                    dsts.len(),
                    "migration {order:?} names {} destination(s) for {} in-flight group(s); \
                     the policy and the pool disagree about what the source holds",
                    dsts.len(),
                    resident.len(),
                );
                self.check_destination(order, *retire_to);
                assert_ne!(
                    order.src(),
                    *retire_to,
                    "a released worker must redirect somewhere other than itself"
                );
                for dst in dsts {
                    self.check_destination(order, *dst);
                    assert_ne!(order.src(), *dst, "a migration must name two workers");
                }

                scratch.clear();
                self.workers[src].drain_resident(now, scratch);
                self.retire(src, *retire_to);
                let drained = std::mem::take(scratch);
                for (group, dst) in resident.iter().zip(dsts) {
                    if let Some(groups) = self.groups.as_mut() {
                        groups.moved(*group, *dst);
                    }
                }
                for request in &drained {
                    let dst = self
                        .groups
                        .as_ref()
                        .map(|groups| groups.host[groups.group_of(*request)])
                        .expect("a scatter needs the group ledger a migration policy installs");
                    self.resume_on(now, dst, *request);
                }
                *scratch = drained;
                scratch.len()
            }
            MigrationOrder::MoveGroup { src: _, dst } => {
                self.check_destination(order, *dst);
                assert_ne!(order.src(), *dst, "a migration must name two workers");
                let groups = self
                    .groups
                    .as_ref()
                    .expect("a group move needs the group ledger a migration policy installs");
                let group = groups.first_resident(order.src()).unwrap_or_else(|| {
                    panic!("migration {order:?} moves a group off a worker holding none")
                });
                let members = groups.members(group);
                scratch.clear();
                // The source keeps running: only this group's requests leave,
                // and the worker is neither retired nor redirected. That is the
                // whole point — a block being handed back one group at a time is
                // still a working engine until its last group is gone.
                self.workers[src].drain_requests(now, &members, scratch);
                self.groups
                    .as_mut()
                    .expect("checked above")
                    .moved(group, *dst);
                for request in std::mem::take(scratch) {
                    self.resume_on(now, *dst, request);
                    scratch.push(request);
                }
                scratch.len()
            }
        }
    }

    fn check_destination(&self, order: &MigrationOrder, dst: WorkerId) {
        let idx = dst.0 as usize;
        assert!(
            idx < self.workers.len(),
            "migration {order:?} names a worker outside a pool of {}",
            self.workers.len()
        );
        assert_eq!(
            self.redirect[idx], dst,
            "migration {order:?} targets a worker that was itself retired"
        );
    }

    /// Take `src` out of the pool. Later trace arrivals pinned to it follow the
    /// work that already left rather than refilling a machine being emptied.
    fn retire(&mut self, src: usize, to: WorkerId) {
        self.redirect[src] = to;
        // The source keeps whatever wakeup it had: an in-flight forward pass is
        // still running on its GPU even though its requests have left.
    }

    fn resume_on(&mut self, now: Time, dst: WorkerId, request: RequestId) {
        self.enqueue_at(
            dst.0 as usize,
            W::Msg::from(WorkerMsgCommon::Resume { req: request, at: now }),
        );
    }
}

// ── L6b: deployment flow (the object L7 calls) ────────────────────────────────

pub struct SimpleDpFlow<W: IterWorker<Event = WorkerEventCommon>> {
    requests: SharedRequests,
    dp_pool: SimpleDpPoolController<W>,
    /// Shared run-level GPU cluster (registry + transfer oracle), built here and
    /// threaded into the pool's construction so workers self-register and (PD
    /// only) keep a handle for runtime transfers. simple_dp has one pool today,
    /// but the ownership shape generalizes to multi-pool (allocate into the
    /// same cluster, ids continue).
    cluster: SharedGpuCluster,
    /// Reused per-tick event sink — workers push `WorkerEventCommon`s (a unified
    /// deployment's workers only emit `RequestComplete`) into it during
    /// `tick_collect`, then it is drained here and cleared for the next tick.
    events: Vec<WorkerEventCommon>,
    /// The pool's migration hook, or `None` when the pool never migrates — the
    /// default, and the reason the untouched tick path stays free of both the
    /// virtual call and the snapshot below.
    ///
    /// Boxed as a trait object so a study can plug its own rule in a test
    /// without the flow gaining a type parameter that every deployment, preset
    /// and factory would then have to spell out.
    migration: Option<Box<dyn MigrationPolicy>>,
    /// The RL training side, or `None` when this deployment only generates.
    /// Holds its own blocks, so a run with it on keeps going after the last
    /// request lands — see [`Flow::background`].
    training: Option<TrainingPool>,
    /// Reused per-tick migration scratch, allocated only on the first tick that
    /// actually runs a policy.
    loads: Vec<WorkerLoad>,
    orders: Vec<MigrationOrder>,
    migrated: Vec<RequestId>,
}

impl<W> SimpleDpFlow<W>
where
    W: IterWorker<Event = WorkerEventCommon>,
{
    pub fn new<F>(cfg: SimpleDpConfig, factory: F) -> Self
    where
        F: WorkerFactory<W>,
    {
        let requests = std::rc::Rc::clone(factory.requests());
        // simple_dp has no inter-worker transfers, but the cluster is still the
        // GPU registry — wire a sentinel `CostSource` whose `submit_transfer`
        // would return ~zero if ever called (it isn't: only PD decode workers
        // call it, and there are none here).
        let cluster: SharedGpuCluster = std::rc::Rc::new(std::cell::RefCell::new(GpuCluster::new(
            CostSource::analytic(f64::INFINITY),
        )));
        let mut dp_pool = SimpleDpPoolController::new(&cfg.dp_pool, &factory, &cluster);
        // Both sides count the same prompt groups, so they share one ledger and
        // must agree on what a group is.
        if let (Some(trigger), Some(training)) = (cfg.migration.as_ref(), cfg.training.as_ref()) {
            assert_eq!(
                trigger.group_size(),
                training.group_size,
                "migration and training disagree about the prompt group size",
            );
        }
        let group_size = cfg
            .migration
            .as_ref()
            .map(MigrationTrigger::group_size)
            .or(cfg.training.as_ref().map(|training| training.group_size));
        if let Some(size) = group_size {
            dp_pool.track_groups(size);
        }
        let training = cfg
            .training
            .map(|training| TrainingPool::new(training, dp_pool.num_workers(), cfg.log_dir));
        Self {
            requests,
            dp_pool,
            cluster,
            events: Vec::new(),
            migration: cfg
                .migration
                .map(|trigger| Box::new(trigger) as Box<dyn MigrationPolicy>),
            training,
            loads: Vec::new(),
            orders: Vec::new(),
            migrated: Vec::new(),
        }
    }

    /// Install a migration hook this deployment's presets cannot name — the
    /// "function hook" half of the policy surface, used by studies and tests
    /// that implement [`MigrationPolicy`] themselves.
    pub fn set_migration_policy(
        &mut self,
        policy: Option<Box<dyn MigrationPolicy>>,
        group_size: u32,
    ) {
        if policy.is_some() {
            self.dp_pool.track_groups(group_size);
        }
        self.migration = policy;
    }
}

impl<W> Flow for SimpleDpFlow<W>
where
    W: MigratableWorker<Event = WorkerEventCommon>,
    W::Msg: From<RequestId> + From<WorkerMsgCommon>,
{
    fn on_arrival(&mut self, req: Request) {
        let rid = req.core.id;
        let declared = req.core.placement.worker;
        self.requests.borrow_mut().insert(req);
        match (self.dp_pool.placement(), declared) {
            (DpPlacementPolicy::TraceDirected, Some(target)) => {
                assert!(
                    (target.0 as usize) < self.dp_pool.num_workers(),
                    "{rid:?} names worker {} but the pool has {} of them",
                    target.0,
                    self.dp_pool.num_workers(),
                );
                let host = self.dp_pool.resolve(target);
                self.dp_pool.admit_to(host, rid);
            }
            // Both mismatches are configuration errors, and both are refused
            // rather than papered over: falling back to a load policy would
            // silently produce a run that is not the placement the trace
            // describes, which is the one thing this mode exists to guarantee.
            (DpPlacementPolicy::TraceDirected, None) => panic!(
                "a trace-directed pool needs a target_worker on every request, \
                 but {rid:?} declared none"
            ),
            (other, Some(_)) => panic!(
                "{rid:?} declares a target_worker, but this pool places by {other:?}; \
                 set the pool's placement to trace-directed or drop the placement tag"
            ),
            (_, None) => self.dp_pool.admit(rid),
        }
    }

    fn tick(&mut self, now: Time) -> Vec<OrchAction> {
        // Migration runs before the workers tick, so a destination can start
        // prefilling what it just took over in this same tick, and so every
        // decision reads the state at the tick boundary rather than a
        // half-advanced pool.
        if let Some(policy) = self.migration.as_mut() {
            self.dp_pool.snapshot_loads(&mut self.loads);
            self.orders.clear();
            policy.decide(now, &self.loads, &mut self.orders);
            // Moved out and back so `migrate` can borrow the pool mutably while
            // the order list keeps its allocation across ticks.
            let mut orders = std::mem::take(&mut self.orders);
            for order in orders.drain(..) {
                self.dp_pool.migrate(now, &order, &mut self.migrated);
            }
            self.orders = orders;
        }

        let mut events = std::mem::take(&mut self.events);
        events.clear();
        self.dp_pool.tick_collect(now, &mut events);
        let mut actions = Vec::new();
        for ev in events.drain(..) {
            let WorkerEventCommon::RequestComplete { req, .. } = ev;
            let finished_group = self.dp_pool.note_completion(req);
            // A group reaches the trainer only when its slowest sample lands,
            // carrying what that step will run over: every member's prompt plus
            // everything it generated.
            if let (Some(group), Some(training)) = (finished_group, self.training.as_mut()) {
                let store = self.requests.borrow();
                training.admit(
                    self.dp_pool
                        .group_members(group)
                        .into_iter()
                        .map(|member| {
                            let record = &store[member];
                            record.request.definition.prompt_tokens
                                + record.progress.output_tokens_emitted
                        })
                        .collect(),
                );
            }
            actions.push(OrchAction::Complete { req });
        }
        self.events = events;

        // The training side reads the pool *after* this tick's completions and
        // migrations, so a block that empties now is the trainer's now.
        if self.training.is_some() {
            self.dp_pool.snapshot_loads(&mut self.loads);
            let training = self.training.as_mut().expect("checked");
            training.observe(&self.loads);
            training.tick(now);
        }
        actions
    }

    fn cluster(&self) -> &SharedGpuCluster {
        &self.cluster
    }

    fn background(&self) -> BackgroundWork {
        match self.training.as_ref() {
            Some(training) => BackgroundWork {
                outstanding: training.outstanding(),
                completed: training.chunks_completed(),
            },
            None => BackgroundWork::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::RequestStore;
    use crate::common::UnifiedStage;
    use crate::orchestrator::training::TrainingConfig;
    use crate::orchestrator::UnifiedWorkerFactory;
    use crate::test_helpers::{text_request, text_request_on, FakeModel};
    use crate::worker::{
        build_barebone_worker, build_chunked_prefill_worker, BareboneWorker, ChunkedPrefillWorker,
        WorkerConfig,
    };
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::Arc;

    fn build_flow(
        num_workers: u16,
        placement: DpPlacementPolicy,
    ) -> (SimpleDpFlow<BareboneWorker<FakeModel>>, SharedRequests) {
        build_flow_with(num_workers, placement, None)
    }

    fn build_flow_with(
        num_workers: u16,
        placement: DpPlacementPolicy,
        migration: Option<MigrationTrigger>,
    ) -> (SimpleDpFlow<BareboneWorker<FakeModel>>, SharedRequests) {
        build_flow_training(num_workers, placement, migration, None)
    }

    fn build_flow_training(
        num_workers: u16,
        placement: DpPlacementPolicy,
        migration: Option<MigrationTrigger>,
        training: Option<TrainingConfig>,
    ) -> (SimpleDpFlow<BareboneWorker<FakeModel>>, SharedRequests) {
        let store: SharedRequests = Rc::new(RefCell::new(RequestStore::new()));
        let factory = UnifiedWorkerFactory::new(
            Arc::new(FakeModel::for_ms(1.0)),
            Rc::clone(&store),
            WorkerConfig::default(),
            None,
            "test-gpu".to_string(),
            "main",
            build_barebone_worker::<FakeModel>,
        );
        let cfg = SimpleDpConfig {
            dp_pool: SimpleDpPoolConfig {
                pool: PoolId(0),
                num_workers,
                placement,
            },
            migration,
            training,
            log_dir: None,
        };
        (SimpleDpFlow::new(cfg, factory), store)
    }

    /// Run until nothing is left, returning completions in id order.
    fn drain_to_completion(flow: &mut SimpleDpFlow<BareboneWorker<FakeModel>>) -> Vec<RequestId> {
        let mut completed = Vec::new();
        for step in 0..500u64 {
            for a in flow.tick(Time::from_ms(step as f64)) {
                let OrchAction::Complete { req } = a;
                completed.push(req);
            }
        }
        completed.sort_by_key(|r| r.0);
        completed
    }

    /// Every worker this request was ever observed on, in visit order.
    fn workers_visited(store: &SharedRequests, req: RequestId) -> Vec<WorkerId> {
        store.borrow()[req]
            .lifecycle
            .stage_log
            .iter()
            .map(|event| event.worker)
            .collect()
    }

    #[test]
    fn all_arrivals_complete() {
        let (mut flow, _store) = build_flow(2, DpPlacementPolicy::RoundRobin);
        for id in 0..5u32 {
            flow.on_arrival(text_request(RequestId(id), 8, 2, Time::ZERO));
        }
        let mut completed = Vec::new();
        for step in 0..500u64 {
            for a in flow.tick(Time::from_ms(step as f64)) {
                let OrchAction::Complete { req } = a;
                completed.push(req);
            }
        }
        completed.sort_by_key(|r| r.0);
        assert_eq!(
            completed,
            (0..5).map(RequestId).collect::<Vec<_>>(),
            "every arrival should complete exactly once"
        );
    }

    #[test]
    fn round_robin_spreads_across_workers() {
        // 4 arrivals over 2 workers, RoundRobin → 2 each. We can't see worker
        // internals directly, but all must still complete (smoke for placement).
        let (mut flow, _store) = build_flow(2, DpPlacementPolicy::RoundRobin);
        for id in 0..4u32 {
            flow.on_arrival(text_request(RequestId(id), 4, 1, Time::ZERO));
        }
        let mut n = 0;
        for step in 0..200u64 {
            n += flow.tick(Time::from_ms(step as f64)).len();
        }
        assert_eq!(n, 4);
    }

    #[test]
    fn a_trace_directed_pool_puts_every_request_where_the_trace_said() {
        // Worker 2 is named by every row even though it is the busiest place to
        // put them — the whole point is that load does not get a vote.
        let (mut flow, store) = build_flow(3, DpPlacementPolicy::TraceDirected);
        for id in 0..4u32 {
            flow.on_arrival(text_request_on(
                RequestId(id),
                8,
                2,
                Time::ZERO,
                WorkerId(2),
            ));
        }

        let completed = drain_to_completion(&mut flow);

        assert_eq!(completed, (0..4).map(RequestId).collect::<Vec<_>>());
        for id in 0..4u32 {
            let visited = workers_visited(&store, RequestId(id));
            assert!(!visited.is_empty(), "request {id} recorded no stage");
            assert!(
                visited.iter().all(|worker| *worker == WorkerId(2)),
                "request {id} ran on {visited:?}, not only on the declared worker 2"
            );
        }
    }

    #[test]
    fn a_trace_directed_pool_spreads_exactly_as_the_trace_spells_it() {
        let (mut flow, store) = build_flow(2, DpPlacementPolicy::TraceDirected);
        // Deliberately lopsided: three on worker 0, one on worker 1. A load
        // policy would never produce this, which is what makes it evidence.
        let declared = [WorkerId(0), WorkerId(0), WorkerId(0), WorkerId(1)];
        for (id, worker) in declared.iter().enumerate() {
            flow.on_arrival(text_request_on(
                RequestId(id as u32),
                8,
                2,
                Time::ZERO,
                *worker,
            ));
        }

        drain_to_completion(&mut flow);

        for (id, worker) in declared.iter().enumerate() {
            let visited = workers_visited(&store, RequestId(id as u32));
            assert!(
                visited.iter().all(|seen| seen == worker),
                "request {id} was declared on {worker:?} but ran on {visited:?}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "needs a target_worker on every request")]
    fn a_trace_directed_pool_refuses_a_request_that_names_no_worker() {
        let (mut flow, _store) = build_flow(2, DpPlacementPolicy::TraceDirected);
        flow.on_arrival(text_request(RequestId(0), 8, 2, Time::ZERO));
    }

    #[test]
    #[should_panic(expected = "places by LeastQueued")]
    fn a_load_placed_pool_refuses_a_request_that_names_a_worker() {
        // Silently ignoring the column would produce a run that is not the
        // placement the trace describes, with nothing in the output saying so.
        let (mut flow, _store) = build_flow(2, DpPlacementPolicy::LeastQueued);
        flow.on_arrival(text_request_on(RequestId(0), 8, 2, Time::ZERO, WorkerId(1)));
    }

    #[test]
    #[should_panic(expected = "but the pool has 2 of them")]
    fn a_target_outside_the_pool_is_refused() {
        let (mut flow, _store) = build_flow(2, DpPlacementPolicy::TraceDirected);
        flow.on_arrival(text_request_on(RequestId(0), 8, 2, Time::ZERO, WorkerId(5)));
    }

    // ── Migration ─────────────────────────────────────────────────────────────

    /// A chunked-prefill pool: the family that can take a resumed request over,
    /// because it already re-prefills a request whose KV it dropped.
    fn build_chunked_flow(
        num_workers: u16,
        placement: DpPlacementPolicy,
        migration: Option<MigrationTrigger>,
    ) -> (
        SimpleDpFlow<ChunkedPrefillWorker<FakeModel>>,
        SharedRequests,
    ) {
        let store: SharedRequests = Rc::new(RefCell::new(RequestStore::new()));
        let factory = UnifiedWorkerFactory::new(
            Arc::new(FakeModel::for_ms(1.0)),
            Rc::clone(&store),
            WorkerConfig {
                max_batch_tokens: Some(64),
                ..WorkerConfig::default()
            },
            None,
            "test-gpu".to_string(),
            "main",
            build_chunked_prefill_worker::<FakeModel>,
        );
        let cfg = SimpleDpConfig {
            dp_pool: SimpleDpPoolConfig {
                pool: PoolId(0),
                num_workers,
                placement,
            },
            migration,
            training: None,
            log_dir: None,
        };
        (SimpleDpFlow::new(cfg, factory), store)
    }

    /// The "function hook" a study writes: one order at a chosen time, so the
    /// test pins migration mechanics rather than a trigger's heuristics.
    struct MigrateOnce {
        at: Time,
        order: MigrationOrder,
        fired: bool,
    }

    impl MigrationPolicy for MigrateOnce {
        fn decide(&mut self, now: Time, _loads: &[WorkerLoad], out: &mut Vec<MigrationOrder>) {
            if !self.fired && now >= self.at {
                self.fired = true;
                out.push(self.order.clone());
            }
        }
    }

    fn stage_codes(store: &SharedRequests, req: RequestId) -> Vec<u16> {
        store.borrow()[req]
            .lifecycle
            .stage_log
            .iter()
            .map(|event| event.code)
            .collect()
    }

    #[test]
    fn a_pool_never_migrates_unless_it_was_asked_to() {
        // The default has to be free as well as inert: `None`, not a boxed
        // policy that decides to do nothing every tick.
        let (mut flow, store) = build_flow(2, DpPlacementPolicy::RoundRobin);
        assert!(flow.migration.is_none(), "migration is opt-in");

        for id in 0..5u32 {
            flow.on_arrival(text_request(RequestId(id), 8, 3, Time::ZERO));
        }
        let completed = drain_to_completion(&mut flow);

        assert_eq!(completed, (0..5).map(RequestId).collect::<Vec<_>>());
        for id in 0..5u32 {
            let request = RequestId(id);
            assert_eq!(store.borrow()[request].telemetry.retraction_count, 0);
            assert!(
                !stage_codes(&store, request).contains(&(UnifiedStage::Suspended as u16)),
                "request {id} was suspended by a pool that has no migration policy"
            );
            let visited = workers_visited(&store, request);
            assert!(
                visited.iter().all(|worker| *worker == visited[0]),
                "request {id} moved workers without a migration policy: {visited:?}"
            );
        }
    }

    #[test]
    fn releasing_a_train_group_moves_each_prompt_group_whole() {
        // Four workers in two blocks of two, prompt groups of two requests.
        // Block 0 holds a short group on worker 0 and a long one on worker 1;
        // block 1 holds six groups and stays busy throughout. Once the short
        // group lands, block 0 is down to one group — 2 samples, under the
        // 6-sample threshold — and the block is handed back.
        //
        // The assertion that matters is cohesion: both members of the long
        // group must land on the SAME destination. The policy counts load in
        // whole groups, so a group split across two workers would be counted
        // twice for the rest of the run.
        let (mut flow, store) = build_chunked_flow(
            4,
            DpPlacementPolicy::TraceDirected,
            Some(MigrationTrigger::train_group_samples_below(6, 2, 2, Time::ZERO)),
        );
        let declared = |id: u32| match id {
            0..=1 => WorkerId(0),
            2..=3 => WorkerId(1),
            4..=9 => WorkerId(2),
            _ => WorkerId(3),
        };
        for id in 0..16u32 {
            let output = if id < 2 { 2 } else { 30 };
            flow.on_arrival(text_request_on(
                RequestId(id),
                8,
                output,
                Time::ZERO,
                declared(id),
            ));
        }
        const LATE: RequestId = RequestId(16);

        let mut completed = Vec::new();
        for step in 0..900u64 {
            let now = Time::from_ms(step as f64);
            if step == 200 {
                flow.on_arrival(text_request_on(LATE, 8, 2, now, WorkerId(0)));
            }
            for action in flow.tick(now) {
                let OrchAction::Complete { req } = action;
                completed.push(req);
            }
        }

        assert_eq!(completed.len(), 17, "a release must not lose or duplicate work");
        let landed = |id: u32| {
            *workers_visited(&store, RequestId(id))
                .last()
                .expect("a completed request has stages")
        };
        assert_eq!(
            landed(2),
            landed(3),
            "the long prompt group was split across {:?} and {:?}",
            landed(2),
            landed(3)
        );
        assert!(
            landed(2).0 >= 2,
            "the long group stayed on the released block: {:?}",
            landed(2)
        );
        for id in 2..4u32 {
            let request = RequestId(id);
            assert_eq!(
                store.borrow()[request].telemetry.retraction_count,
                1,
                "request {id} moved, so it must have re-prefilled exactly once"
            );
            assert!(
                stage_codes(&store, request).contains(&(UnifiedStage::Suspended as u16)),
                "request {id} moved without recording a suspension"
            );
        }
        for id in 0..2u32 {
            assert_eq!(
                store.borrow()[RequestId(id)].telemetry.retraction_count,
                0,
                "request {id} finished before the release and must not have moved"
            );
        }
        // Worker 0 held nothing by the time the block fired, and is retired all
        // the same: a block is only useful to training once all of it is free.
        assert!(
            landed(16).0 >= 2,
            "a row pinned to a released worker must follow the work that left: {:?}",
            landed(16)
        );
    }

    #[test]
    fn migration_drains_the_source_and_redirects_what_the_trace_sends_it_later() {
        let (mut flow, store) = build_chunked_flow(2, DpPlacementPolicy::TraceDirected, None);
        flow.set_migration_policy(
            Some(Box::new(MigrateOnce {
                at: Time::from_ms(3.0),
                order: MigrationOrder::Consolidate {
                    src: WorkerId(0),
                    dst: WorkerId(1),
                },
                fired: false,
            })),
            1,
        );
        // Every row names worker 0, including the one that arrives after the
        // pool has already emptied it.
        for id in 0..3u32 {
            flow.on_arrival(text_request_on(RequestId(id), 8, 6, Time::ZERO, WorkerId(0)));
        }
        const LATE: RequestId = RequestId(3);

        let mut completed = Vec::new();
        for step in 0..500u64 {
            let now = Time::from_ms(step as f64);
            if step == 10 {
                flow.on_arrival(text_request_on(LATE, 8, 2, now, WorkerId(0)));
            }
            for action in flow.tick(now) {
                let OrchAction::Complete { req } = action;
                completed.push(req);
            }
        }

        completed.sort_by_key(|request| request.0);
        assert_eq!(
            completed,
            vec![RequestId(0), RequestId(1), RequestId(2), LATE],
            "a migration must not lose or duplicate a request"
        );
        for id in 0..3u32 {
            let request = RequestId(id);
            let codes = stage_codes(&store, request);
            let suspended = codes
                .iter()
                .position(|code| *code == UnifiedStage::Suspended as u16)
                .unwrap_or_else(|| panic!("request {id} never left worker 0: {codes:?}"));
            assert_eq!(codes[suspended + 1], UnifiedStage::Pending as u16);
            let visited = workers_visited(&store, request);
            assert_eq!(visited[suspended], WorkerId(0));
            assert_eq!(visited[suspended + 1], WorkerId(1));
            let record = &store.borrow()[request];
            assert_eq!(record.telemetry.retraction_count, 1);
            assert!(
                record.telemetry.first_output_time.unwrap()
                    <= record.lifecycle.stage_log[suspended].time,
                "a migrated request keeps the TTFT it already earned"
            );
        }
        assert_eq!(
            workers_visited(&store, LATE).first(),
            Some(&WorkerId(1)),
            "a row pinned to a retired worker must follow the work that left it"
        );
    }

    #[test]
    fn a_group_move_takes_one_group_and_leaves_the_source_running() {
        // Two prompt groups of two on worker 0. Moving one must take exactly
        // that group's requests, leave the other group where it is, and leave
        // worker 0 a working member of the pool — unretired, so a later arrival
        // pinned to it still lands on it. That last part is what separates this
        // from `Scatter`: a block being handed back one group at a time is not
        // gone until its last group is.
        let (mut flow, store) = build_chunked_flow(2, DpPlacementPolicy::TraceDirected, None);
        flow.set_migration_policy(
            Some(Box::new(MigrateOnce {
                at: Time::from_ms(3.0),
                order: MigrationOrder::MoveGroup {
                    src: WorkerId(0),
                    dst: WorkerId(1),
                },
                fired: false,
            })),
            2,
        );
        for id in 0..4u32 {
            flow.on_arrival(text_request_on(RequestId(id), 8, 6, Time::ZERO, WorkerId(0)));
        }
        // Ids are dense, so the late row opens a third group of its own.
        const LATE: RequestId = RequestId(4);

        let mut completed = Vec::new();
        for step in 0..500u64 {
            let now = Time::from_ms(step as f64);
            if step == 10 {
                flow.on_arrival(text_request_on(LATE, 8, 2, now, WorkerId(0)));
            }
            for action in flow.tick(now) {
                let OrchAction::Complete { req } = action;
                completed.push(req);
            }
        }

        completed.sort_by_key(|request| request.0);
        assert_eq!(
            completed,
            vec![RequestId(0), RequestId(1), RequestId(2), RequestId(3), LATE],
        );
        // Group 0 is ids 0 and 1 — the lowest-id group resident on the source.
        for id in 0..2u32 {
            let visited = workers_visited(&store, RequestId(id));
            assert_eq!(
                visited.last(),
                Some(&WorkerId(1)),
                "request {id} belongs to the moved group"
            );
            assert_eq!(store.borrow()[RequestId(id)].telemetry.retraction_count, 1);
        }
        for id in 2..4u32 {
            assert!(
                workers_visited(&store, RequestId(id))
                    .iter()
                    .all(|worker| *worker == WorkerId(0)),
                "request {id} was not in the moved group and must not have moved"
            );
            assert_eq!(store.borrow()[RequestId(id)].telemetry.retraction_count, 0);
        }
        assert_eq!(
            workers_visited(&store, LATE).first(),
            Some(&WorkerId(0)),
            "a group move does not retire its source, so the trace still reaches it"
        );
    }

    fn training_cfg(group_size: u32, workers_per_block: u16) -> TrainingConfig {
        TrainingConfig {
            workers_per_block,
            group_size,
            cost: crate::worker::TrainChunkCost {
                tokens_per_s: 1_000.0,
                overhead: Time::ZERO,
            },
            bulk_grab: 1,
            tail_threshold: 0,
            expected_groups: 0,
        }
    }

    /// The default is no training at all: the flow builds no blocks and reports
    /// nothing outstanding, so a preset that never asked for it sees the run it
    /// has always seen.
    #[test]
    fn training_off_is_the_default_and_changes_nothing() {
        let (mut flow, _store) = build_flow(2, DpPlacementPolicy::RoundRobin);
        for id in 0..4u32 {
            flow.on_arrival(text_request(RequestId(id), 8, 2, Time::ZERO));
        }
        let completed = drain_to_completion(&mut flow);
        assert_eq!(completed.len(), 4);
        assert!(flow.training.is_none());
        assert_eq!(flow.background(), BackgroundWork::default());
    }

    /// A prompt group is worth nothing to the trainer until its slowest sample
    /// lands — and what it is then worth is every member's prompt plus
    /// everything that member generated.
    #[test]
    fn a_group_reaches_training_only_when_its_slowest_sample_lands() {
        let (mut flow, _store) = build_flow_training(
            2,
            DpPlacementPolicy::TraceDirected,
            None,
            Some(training_cfg(2, 2)),
        );
        // One group of two on one engine (a group never spans engines), with
        // one sample far slower than the other.
        flow.on_arrival(text_request_on(RequestId(0), 8, 1, Time::ZERO, WorkerId(0)));
        flow.on_arrival(text_request_on(RequestId(1), 8, 20, Time::ZERO, WorkerId(0)));

        let mut first_done = None;
        for step in 0..500u64 {
            let now = Time::from_ms(step as f64);
            for action in flow.tick(now) {
                let OrchAction::Complete { req } = action;
                if req == RequestId(0) {
                    first_done = Some(step);
                }
            }
            if first_done == Some(step) {
                assert_eq!(
                    flow.training.as_ref().expect("training on").queued(),
                    (0, 0),
                    "seven-eighths of a group is still nothing to train on"
                );
            }
        }
        assert!(first_done.is_some(), "the fast sample should have completed");
        let training = flow.training.as_ref().expect("training on");
        assert_eq!(training.queued(), (0, 0), "the group was picked up");
        assert_eq!(training.groups_per_block(), [1]);
        // The chunk's token count shows up as its duration: at 1,000 tok/s with
        // no overhead, (8 + 1) + (8 + 20) tokens is 37 ms and nothing else is.
        let (start, end) = training.window().expect("one chunk ran");
        assert_eq!(end.as_ms() - start.as_ms(), 37.0);
    }
}
