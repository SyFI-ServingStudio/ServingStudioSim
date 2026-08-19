//! `GpuCluster` — the unified GPU registry **and** inter-worker transfer timing
//! oracle (L5 / `doc/detailed_design/L5.md`).
//!
//! Two roles folded into one object:
//!   1. **Registry** — flat `gpus: Vec<GpuInfo>` indexed by run-level GPU id
//!      (0..N). Workers self-register their owned block at construction via
//!      [`GpuCluster::allocate`]; the resulting list is what L7 serializes into
//!      `raw/run_meta.json` (the reporting subset is the `gpus` field; `groups`
//!      / `cost` are `#[serde(skip)]`).
//!   2. **Timing oracle** — per-**comm-group** send/recv free-times, fed by a
//!      [`CostSource`] (profiled `p2p_inter` kernel or analytic bandwidth).
//!      [`submit_transfer`](GpuCluster::submit_transfer) is called by PD decode
//!      workers each handoff; compute is NOT modeled here (it lives in each
//!      worker's iteration FSM via `start_iter → compute_end`).
//!
//! Why per-group state (not per-GPU): the members of a comm group always sync —
//! every transfer using a group writes the same `send_free` (or `recv_free`) to
//! every GPU in the range. Per-GPU storage duplicated that one value across the
//! range and forced a O(count) loop on every submit; per-group state is O(1).
//! Workers register their group once at construction ([`register_comm_group`])
//! and receive a `u16` id that flows through `PrefillDone` / `Handoff` /
//! `TransferPlan` — no `(base, count)` plumbing anymore.
//!
//! Why merge registry + timing: the two grow together as workers come online.
//! Keeping them apart forced a two-phase "build inventory first, then build
//! cluster from its size" dance; folding them lets the flow build one
//! `Rc<RefCell<GpuCluster>>` up front, thread it through controller → factory →
//! worker, and have every worker `allocate` (+ register its group) itself.
//!
//! A PD prefill→decode KV handoff is a transfer from a sender comm group to a
//! receiver comm group. Each side's aggregated bandwidth scales with its own
//! link count, so a higher attention-TP / head-parallel degree on a side makes
//! that side faster, and the slower (fewer-link) side bounds the transfer.
//!
//! Per-link cost comes from a [`CostSource`]: production holds the `p2p_inter` L1
//! kernel and reads its profiled inter-node send/recv-vs-message-size curve via
//! `eval`; tests hold an analytic per-link bandwidth. The enum is the test seam —
//! the kernel can't be built without a perf-api bridge + `profile.db` rows, so the
//! cluster's contention-logic unit tests use the analytic variant instead.
//!
//! Send/recv coupling: the two legs share a single `start` time =
//! `max(now, all src.send_free, all dst.recv_free)`. A receiver cannot start
//! pulling before the sender has a free send link to push from; a sender cannot
//! start pushing before the receiver has a free recv link to drain into.
//! Modeling them as independent legs (the previous shape) under-counted
//! contention: a `dst.recv_free` could be written ahead of the actual byte
//! arrival when the sender was the bottleneck, letting a subsequent transfer to
//! the same dst start too early. Within the shared `start`, send/recv durations
//! still differ when link counts differ (each side's `bytes/count` × link cost),
//! and each side's free-time advances by its own duration — so the slower side
//! still bounds `pull_end` = `max(send_end, recv_end)`. The whole object is
//! shared (`Rc<RefCell<>>`, like `SharedRequests`) across the PD pools because a
//! transfer touches GPUs owned by two different workers/pools.

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

use serde::Serialize;

use crate::common::Time;
use crate::log::{GpuClusterEntry, NetworkLogger};
use crate::timing::kernels::{P2pInterKernel, P2pInterKernelInput};

/// One physical GPU and who owns it. The cluster's `gpus` field is a flat list
/// of these (the reporting / `run_meta.json` subset); `pool` + `worker_id` make
/// the worker→gpu and pool→gpu groupings derivable without a second table.
///
/// `pool_tag` is the worker pool's role literal (e.g. `attn` / `ffn` / `prefill`),
/// stamped on every GPU at [`allocate`](GpuCluster::allocate) regardless of whether
/// the owning worker holds a KV pool or a comm group. It is the authoritative source
/// `run_meta.json` uses for each worker row's tag, so downstream consumers never have
/// to reverse-recover a non-KV worker's role through `comm_groups`.
#[derive(Clone, Debug, Serialize)]
pub struct GpuInfo {
    pub id: u16,
    pub name: String,
    pub pool: u16,
    pub worker_id: u16,
    pub pool_tag: String,
}

/// Where a transfer's per-link time comes from. `Kernel` is the profiled
/// inter-node p2p curve (production); `Analytic` is a constant per-link bandwidth
/// (`bytes_per_ms`) used as the unit-test seam / a bridge-free fallback.
pub enum CostSource {
    Kernel(P2pInterKernel),
    Analytic { bytes_per_ms: f64 },
}

impl CostSource {
    /// Analytic source from a per-link bandwidth in GB/s (`1 GB/s = 1e6 bytes/ms`).
    pub fn analytic(gbps: f64) -> Self {
        CostSource::Analytic {
            bytes_per_ms: (gbps * 1e6).max(f64::MIN_POSITIVE),
        }
    }

    /// Time for one link to carry `message_size_bytes`.
    fn link_time(&self, message_size_bytes: u64) -> Time {
        match self {
            CostSource::Kernel(k) => {
                let leaf = k.eval(&P2pInterKernelInput { message_size_bytes });
                Time::from_ms(leaf.m.time_ms.max(0.0) as f64)
            }
            CostSource::Analytic { bytes_per_ms } => {
                Time::from_ms(message_size_bytes as f64 / bytes_per_ms)
            }
        }
    }

    /// One-time per-link **latency** (α): the p2p curve's y-intercept — the cost
    /// of an ~empty transfer, all setup/propagation, no bytes. Extrapolated
    /// linearly to 0 bytes from the two smallest profiled sweep points (both deep
    /// in the latency-bound floor, so the extrapolation is ≈ the floor value):
    /// `α = 2·t(b0) − t(2·b0)`. `Analytic` is a pure bandwidth line with no floor
    /// (α = 0). This is the cost a gather pays **once**, not per source.
    fn latency(&self) -> Time {
        match self {
            CostSource::Analytic { .. } => Time::ZERO,
            CostSource::Kernel(_) => {
                // 2^10 = the p2p_inter sweep floor (see p2p_inter.rs sweep_grid).
                let b0 = 1024u64;
                let t0 = self.link_time(b0).as_ms();
                let t1 = self.link_time(2 * b0).as_ms();
                Time::from_ms((2.0 * t0 - t1).max(0.0))
            }
        }
    }
}

/// One NCCL-style communication group: the contiguous block of GPU ids whose
/// streams move data as a unit (an attention-shard set for a PD worker, an
/// all-to-all group elsewhere). Each group carries both directional free-times
/// — `send_free` for when it's acting as a transfer's src side, `recv_free` for
/// when it's acting as the dst side — so one register-once group covers both
/// directions a worker may operate in. Members of a group always sync; per-GPU
/// stream state would duplicate the same value across the range.
#[derive(Clone, Copy, Debug)]
struct CommGroup {
    /// First GPU id in this group; members are the contiguous block
    /// `[base, base+count)`. Exported (with `count`) via
    /// [`comm_groups`](GpuCluster::comm_groups) into `run_meta.json`.
    base: u16,
    /// Number of GPUs = link count for aggregate-bandwidth math.
    count: u16,
    /// Earliest time this group's send streams are idle.
    send_free: Time,
    /// Earliest time this group's recv streams are idle.
    recv_free: Time,
    /// The worker that owns this group, supplied at registration. Lets
    /// `submit_transfer` resolve *both* endpoints of a transfer (sender = the
    /// `send_gid` group's owner, receiver = the `recv_gid` group's) into the
    /// `gpu_cluster` log without any caller-side identity plumbing. `pool_tag` is
    /// a worker pool literal (`&'static str`); the pair keys the same `(pool_tag,
    /// worker_id)` space `cost_log` uses, so the trace overlays a transfer onto
    /// the receiving worker's row.
    owner_pool_tag: &'static str,
    owner_worker_id: u16,
}

#[derive(Serialize)]
pub struct GpuCluster {
    /// Flat per-GPU registry, indexed by global GPU id (dense `0..num_gpus`).
    /// This is the **reporting** subset — the only field L7 serializes into
    /// `raw/run_meta.json` (the other two are runtime state).
    pub gpus: Vec<GpuInfo>,
    /// Comm groups registered by workers via [`register_comm_group`]. Indexed
    /// by the `u16` id returned at registration time; each `submit_transfer`
    /// touches exactly one src group + one dst group, no per-GPU loop.
    #[serde(skip)]
    groups: Vec<CommGroup>,
    /// Per-link transfer cost — the profiled `p2p_inter` kernel or an analytic
    /// bandwidth (see [`CostSource`]).
    #[serde(skip)]
    cost: CostSource,
    /// The `gpu_cluster` stream writer, attached once a log dir is available
    /// ([`attach_logger`](GpuCluster::attach_logger)). `None` for tests and
    /// no-log runs — `submit_transfer` then records nothing. One writer for the
    /// whole run (the cluster is the single shared transfer oracle).
    #[serde(skip)]
    logger: Option<NetworkLogger>,
    /// Per-worker × group KV-pool token capacity, as
    /// `(pool_tag, pool, worker_id, group_id, capacity_tokens)`. A static per-run
    /// fact each worker reports at construction via
    /// [`register_kv_capacity`](GpuCluster::register_kv_capacity) (right after
    /// [`allocate`](GpuCluster::allocate)); L7 folds it into `run_meta.json`'s
    /// per-worker `kv_pools` so the `kv_snapshot` occupancy series need not repeat
    /// the constant. `pool_tag` is carried so `run_meta` stamps the same
    /// `(pool_tag, worker_id, group_id)` key the `kv_snapshot` rows use — an exact
    /// analyzer join even when `worker_id` collides across pools (PD prefill#0 vs
    /// decode#0). `#[serde(skip)]` like the other runtime tables — exported via
    /// [`kv_capacities`](GpuCluster::kv_capacities), not the derived `Serialize`.
    #[serde(skip)]
    kv_caps: Vec<(&'static str, u16, u16, u16, u64)>,
}

impl GpuCluster {
    /// Build an empty cluster carrying `cost`. Workers append their own GPU
    /// blocks at construction via [`allocate`](GpuCluster::allocate) and their
    /// comm groups via [`register_comm_group`](GpuCluster::register_comm_group);
    /// the cluster starts with zero GPUs and zero groups.
    pub fn new(cost: CostSource) -> Self {
        Self {
            gpus: Vec::new(),
            groups: Vec::new(),
            cost,
            logger: None,
            kv_caps: Vec::new(),
        }
    }

    /// Attach the `gpu_cluster` log writer for this run, opening
    /// `<log_dir>/raw/gpu_cluster.parquet`. Called by the flow after building the
    /// cluster (PD / AFD) when `cfg.io.log_dir` is set; every subsequent
    /// [`submit_transfer`](GpuCluster::submit_transfer) then records one row. A
    /// failed open is logged and left as `None` (transfer timing still runs; only
    /// the log is absent) so a logging problem never aborts the sim.
    pub fn attach_logger(&mut self, log_dir: &Path) {
        match NetworkLogger::open(log_dir) {
            Ok(l) => self.logger = Some(l),
            Err(e) => tracing::warn!("failed to open gpu_cluster log: {e:#}"),
        }
    }

    /// Register `n` contiguous-id GPUs owned by `(pool, worker_id)`, all sharing
    /// `name` and the worker's `pool_tag`, and return the base id of the new block.
    /// Ids continue from the current length, so calling once per worker yields a
    /// dense `0..total` id space across all pools. Every worker calls this exactly
    /// once at construction, so stamping `pool_tag` here is what makes each worker's
    /// role tag reach `run_meta.json` independent of KV / comm-group registration.
    pub fn allocate(
        &mut self,
        pool: u16,
        worker_id: u16,
        n: u16,
        name: &str,
        pool_tag: &str,
    ) -> u16 {
        let base = self.gpus.len() as u16;
        for offset in 0..n {
            self.gpus.push(GpuInfo {
                id: base + offset,
                name: name.to_string(),
                pool,
                worker_id,
                pool_tag: pool_tag.to_string(),
            });
        }
        base
    }

    /// Record the KV-pool token `capacity` of `(pool_tag, pool, worker_id,
    /// group_id)` — a static per-run fact the worker reports at construction, right
    /// after [`allocate`](GpuCluster::allocate). L7 folds these into
    /// `run_meta.json`'s per-worker `kv_pools`, keeping the `kv_snapshot` occupancy
    /// series capacity-free. `pool_tag` is the worker's cost-log pool literal, so
    /// `run_meta` can key capacity by the same `(pool_tag, worker_id, group_id)` the
    /// snapshot rows use. Multi-group workers (HP / PD decode) call once per group.
    pub fn register_kv_capacity(
        &mut self,
        pool_tag: &'static str,
        pool: u16,
        worker_id: u16,
        group_id: u16,
        capacity: u64,
    ) {
        self.kv_caps
            .push((pool_tag, pool, worker_id, group_id, capacity));
    }

    /// Register one comm group covering `[base, base+count)`, owned by worker
    /// `(pool_tag, worker_id)`, and return its id. Workers register their
    /// attn-shard endpoint set once at construction; the returned id is what flows
    /// through `PrefillDone` / `Handoff` / `TransferPlan`. The owner identity is
    /// carried so `submit_transfer` can label both ends of a transfer in the
    /// `gpu_cluster` log — `pool_tag` is the worker's cost-log pool literal.
    pub fn register_comm_group(
        &mut self,
        base: u16,
        count: u16,
        pool_tag: &'static str,
        worker_id: u16,
    ) -> u16 {
        let gid = self.groups.len() as u16;
        self.groups.push(CommGroup {
            base,
            count,
            send_free: Time::ZERO,
            recv_free: Time::ZERO,
            owner_pool_tag: pool_tag,
            owner_worker_id: worker_id,
        });
        gid
    }

    pub fn num_gpus(&self) -> usize {
        self.gpus.len()
    }

    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }

    /// Registered comm groups as `(gid, base, count, owner_pool_tag,
    /// owner_worker_id)`, `gid` = registration index. Members are the contiguous
    /// GPU block `[base, base+count)`. The one read path for L7 to export the
    /// gid→gpu-set mapping into `run_meta.json` — the internal `groups` table is
    /// `#[serde(skip)]`, so this accessor is how the otherwise-private group
    /// identity leaves the cluster.
    pub fn comm_groups(&self) -> impl Iterator<Item = (u16, u16, u16, &'static str, u16)> + '_ {
        self.groups.iter().enumerate().map(|(gid, g)| {
            (
                gid as u16,
                g.base,
                g.count,
                g.owner_pool_tag,
                g.owner_worker_id,
            )
        })
    }

    /// Registered KV-pool capacities as `(pool_tag, pool, worker_id, group_id,
    /// capacity_tokens)`, in registration order (group-ordered within a worker).
    /// The one read path for L7 to export per-worker `kv_pools` into
    /// `run_meta.json` — the internal `kv_caps` table is `#[serde(skip)]`.
    pub fn kv_capacities(&self) -> impl Iterator<Item = (&'static str, u16, u16, u16, u64)> + '_ {
        self.kv_caps.iter().copied()
    }

    /// Submit `bytes` from `send_gid` (acting as src) to `recv_gid` (acting as
    /// dst), starting no earlier than `now`. Modeled as one **synchronized**
    /// collective: the transfer begins only once both the src group's send
    /// streams and the dst group's recv streams are idle, and both sides remain
    /// busy until the **bottleneck** side finishes. So the duration is
    /// `max(send_dur, recv_dur)` where each leg's hypothetical-at-full-bw time
    /// is `bytes / side_count / per_link_bw`; the faster side is throttled
    /// (buffers fill / drain at the slower side's pace) and its link is busy
    /// for the full collective duration, not just its own. Returns the time
    /// the KV is resident at the destination = `start + transfer_dur`.
    /// `send_gid` and `recv_gid` may refer to the same group (a self-loop is a
    /// no-op in practice but the API accepts it).
    ///
    /// `kind` (a stable category literal, e.g. `pd_kv_pull`) and `tag` (a
    /// free-form per-deployment identifier, often `""`) are recorded verbatim into
    /// the `gpu_cluster` log alongside the resolved `start`/`end` window and both
    /// endpoints' worker identity — but only when a [`NetworkLogger`] is attached;
    /// otherwise they are ignored and `tag` is never cloned.
    pub fn submit_transfer(
        &mut self,
        now: Time,
        send_gid: u16,
        recv_gid: u16,
        bytes: u64,
        kind: &'static str,
        tag: &str,
    ) -> Time {
        let s = self.groups[send_gid as usize];
        let r = self.groups[recv_gid as usize];
        if s.count == 0 || r.count == 0 {
            return now;
        }
        let send_per_link = (bytes as f64 / s.count as f64).round() as u64;
        let recv_per_link = (bytes as f64 / r.count as f64).round() as u64;
        let send_dur = self.cost.link_time(send_per_link);
        let recv_dur = self.cost.link_time(recv_per_link);
        // Slower side bounds the collective; faster side is held throttled at
        // the bottleneck pace, so both links stay busy for the same window.
        let transfer_dur = send_dur.max(recv_dur);
        let start = now.max(s.send_free).max(r.recv_free);
        let end = start + transfer_dur;
        self.groups[send_gid as usize].send_free = end;
        self.groups[recv_gid as usize].recv_free = end;
        // Record the on-wire window (incl. queueing) with both endpoints resolved
        // from the two groups' owners. A logging failure warns but never aborts
        // the transfer — the timing above is already committed.
        if let Some(logger) = self.logger.as_mut() {
            if let Err(e) = logger.record(GpuClusterEntry {
                net_start_ms: start.as_ms(),
                net_end_ms: end.as_ms(),
                src_pool_tag: s.owner_pool_tag,
                src_worker_id: s.owner_worker_id,
                dst_pool_tag: r.owner_pool_tag,
                dst_worker_id: r.owner_worker_id,
                send_gid,
                recv_gid,
                send_count: s.count,
                recv_count: r.count,
                bytes,
                kind,
                tag: tag.to_string(),
                // Coupled transfer: the sender is held for the whole collective,
                // so its link frees at `end` (no early release, unlike a gather).
                send_end_ms: end.as_ms(),
            }) {
                tracing::warn!("gpu_cluster log record failed: {e:#}");
            }
        }
        end
    }

    /// Submit a **fan-in gather**: every `(send_gid, bytes)` in `sources` delivers
    /// to `recv_gid` as ONE concurrent collective (e.g. the AFD ffn side pulling
    /// each layer's input from all attn workers at once). Unlike N separate
    /// [`submit_transfer`](Self::submit_transfer) calls — which serialize on the
    /// shared `recv_gid` and pay the wire latency once per source — a gather:
    ///   - pays the per-link latency **once** (`α`, overlapped across all sources);
    ///   - runs the senders **in parallel**, each holding its own send links only
    ///     for its transmission slice (`transfer_time`, latency-stripped), so a
    ///     sender is free for its next push as soon as its bytes are on the wire;
    ///   - drains the **aggregate** byte total across the receiver's links.
    ///     Duration = `α + max(slowest sender transmission, receiver aggregate drain)`;
    ///     returns the time all bytes are resident at `recv_gid`. A single-source
    ///     gather has the same arrival as `submit_transfer` (α + max ≡ link_time), but
    ///     frees the sender after its own slice rather than the coupled collective.
    ///
    /// `kind`/`tag` are logged per source over the shared `[start, arrival]` window
    /// (only when a [`NetworkLogger`] is attached; `tag` is cloned per source then).
    pub fn submit_gather(
        &mut self,
        now: Time,
        sources: &[(u16, u64)],
        recv_gid: u16,
        kind: &'static str,
        tag: &str,
    ) -> Time {
        let r = self.groups[recv_gid as usize];
        if r.count == 0 {
            return now;
        }
        // Live sources only: non-empty bytes over a non-empty sender group. The
        // collective can't start until every involved send link AND the recv link
        // is idle. `live` carries (send_gid, bytes, send_count).
        // α (the one-time wire latency) is invariant per link — a pure function of
        // the fixed cost curve — so evaluate it ONCE here rather than re-deriving it
        // (2 constant-input curve evals) inside every per-source cost lookup.
        let alpha = self.cost.latency();
        let alpha_ms = alpha.as_ms();
        // `xfer` = each sender's bandwidth-only transmission slice `link_time(bytes)−α`:
        // the time it actively pushes bits, latency stripped (the collective pays α
        // once, not per sender). It is reused three times below — the send-side
        // bound, `send_free`, and the log row — so evaluate it ONCE per source and
        // carry it in `live` instead of re-running the curve lookup three times.
        let mut live: Vec<(u16, u64, u16, Time)> = Vec::with_capacity(sources.len());
        let mut total_bytes: u64 = 0;
        let mut start = now.max(r.recv_free);
        for &(sg, bytes) in sources {
            let s = self.groups[sg as usize];
            if bytes == 0 || s.count == 0 {
                continue;
            }
            start = start.max(s.send_free);
            total_bytes += bytes;
            let per_link = (bytes as f64 / s.count as f64).round() as u64;
            let xfer = Time::from_ms((self.cost.link_time(per_link).as_ms() - alpha_ms).max(0.0));
            live.push((sg, bytes, s.count, xfer));
        }
        if live.is_empty() {
            return now;
        }
        // Senders push in parallel: the send-side bound is the slowest single
        // sender's transmission slice (each over its own links), NOT the sum.
        let mut send_xfer_max = Time::ZERO;
        for &(_, _, _, xfer) in &live {
            send_xfer_max = send_xfer_max.max(xfer);
        }
        // The receiver drains the aggregate byte total across its links.
        let recv_per_link = (total_bytes as f64 / r.count as f64).round() as u64;
        let recv_xfer =
            Time::from_ms((self.cost.link_time(recv_per_link).as_ms() - alpha_ms).max(0.0));
        // Latency paid ONCE for the whole gather (overlapped across sources).
        let arrival = start + alpha + send_xfer_max.max(recv_xfer);
        // Bookkeeping: each sender frees after its own transmission slice (no
        // latency, no coupling to the collective); the receiver is busy until
        // every byte lands (preserves the "recv_free tracks resident time" rule).
        for &(sg, _, _, xfer) in &live {
            self.groups[sg as usize].send_free = start + xfer;
        }
        self.groups[recv_gid as usize].recv_free = arrival;
        // One log row per source over the shared window (resolved endpoints).
        if self.logger.is_some() {
            for &(sg, bytes, count, xfer) in &live {
                let s = self.groups[sg as usize];
                // Sender frees after its own transmission slice (latency-stripped) —
                // strictly before `arrival`, so the send slice ends inside the
                // collective's window (the overlap `analyze trace` visualizes).
                let send_end = start + xfer;
                let entry = GpuClusterEntry {
                    net_start_ms: start.as_ms(),
                    net_end_ms: arrival.as_ms(),
                    src_pool_tag: s.owner_pool_tag,
                    src_worker_id: s.owner_worker_id,
                    dst_pool_tag: r.owner_pool_tag,
                    dst_worker_id: r.owner_worker_id,
                    send_gid: sg,
                    recv_gid,
                    send_count: count,
                    recv_count: r.count,
                    bytes,
                    kind,
                    tag: tag.to_string(),
                    send_end_ms: send_end.as_ms(),
                };
                if let Some(logger) = self.logger.as_mut() {
                    if let Err(e) = logger.record(entry) {
                        tracing::warn!("gpu_cluster gather log record failed: {e:#}");
                    }
                }
            }
        }
        arrival
    }
}

/// Shared run-level handle: one cluster threaded into the PD pools (see module docs).
pub type SharedGpuCluster = Rc<RefCell<GpuCluster>>;

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a fresh cluster at 1 GB/s analytic per-link cost. The reporting
    /// `gpus` list is irrelevant to timing tests, so we skip `allocate`.
    fn cluster() -> GpuCluster {
        GpuCluster::new(CostSource::analytic(1.0))
    }

    /// Register a group with a throwaway owner identity — these tests probe
    /// timing/contention only, and the cluster has no logger, so the owner is
    /// never read.
    fn reg(c: &mut GpuCluster, base: u16, count: u16) -> u16 {
        c.register_comm_group(base, count, "test", 0)
    }

    /// Submit a transfer with a throwaway `kind`/`tag` (no logger attached).
    fn xfer(c: &mut GpuCluster, now: Time, s: u16, d: u16, bytes: u64) -> Time {
        c.submit_transfer(now, s, d, bytes, "test", "")
    }

    #[test]
    fn recv_leg_dominates_when_fewer_dst_links() {
        // 8 send links vs 2 recv links: recv carries bytes/2 per link → slower.
        let mut c = cluster();
        let s = reg(&mut c, 0, 8);
        let d = reg(&mut c, 8, 2);
        let bytes = 8_000_000u64; // 8 MB
        let end = xfer(&mut c, Time::ZERO, s, d, bytes);
        // send: (8e6/8)/1e6 = 1ms; recv: (8e6/2)/1e6 = 4ms → resident at 4ms.
        assert!((end.as_ms() - 4.0).abs() < 1e-6, "got {}", end.as_ms());
    }

    #[test]
    fn more_links_is_faster_aggregated_bandwidth() {
        let bytes = 8_000_000u64;
        let mut c4 = cluster();
        let s4 = reg(&mut c4, 0, 4);
        let d4 = reg(&mut c4, 4, 4);
        let e4 = xfer(&mut c4, Time::ZERO, s4, d4, bytes);
        let mut c8 = cluster();
        let s8 = reg(&mut c8, 0, 8);
        let d8 = reg(&mut c8, 8, 8);
        let e8 = xfer(&mut c8, Time::ZERO, s8, d8, bytes);
        assert!(
            e8.as_ms() < e4.as_ms(),
            "8 links should beat 4: {} vs {}",
            e8.as_ms(),
            e4.as_ms()
        );
    }

    #[test]
    fn same_group_serializes() {
        // Two back-to-back transfers re-using the same src+dst groups queue up.
        let mut c = cluster();
        let s = reg(&mut c, 0, 1);
        let d = reg(&mut c, 1, 1);
        let bytes = 1_000_000u64; // 1 MB → 1ms on a single link
        let e1 = xfer(&mut c, Time::ZERO, s, d, bytes);
        let e2 = xfer(&mut c, Time::ZERO, s, d, bytes);
        assert!((e1.as_ms() - 1.0).abs() < 1e-6, "first {}", e1.as_ms());
        assert!((e2.as_ms() - 2.0).abs() < 1e-6, "second {}", e2.as_ms());
    }

    #[test]
    fn disjoint_groups_do_not_serialize() {
        let mut c = cluster();
        let s1 = reg(&mut c, 0, 1);
        let d1 = reg(&mut c, 1, 1);
        let s2 = reg(&mut c, 2, 1);
        let d2 = reg(&mut c, 3, 1);
        let bytes = 1_000_000u64;
        let e1 = xfer(&mut c, Time::ZERO, s1, d1, bytes);
        let e2 = xfer(&mut c, Time::ZERO, s2, d2, bytes);
        assert!((e1.as_ms() - 1.0).abs() < 1e-6);
        assert!(
            (e2.as_ms() - 1.0).abs() < 1e-6,
            "disjoint groups should not queue: {}",
            e2.as_ms()
        );
    }

    /// Regression for "fast side held by bottleneck": when the recv side is
    /// slower (fewer links), the sender is throttled to the recv pace and its
    /// `send_free` must reflect the full `max(send_dur, recv_dur)` collective
    /// duration, not just `send_dur`. A subsequent transfer reusing that sender
    /// to a fast receiver must wait for the full collective to finish.
    #[test]
    fn recv_bottleneck_holds_sender_too() {
        let mut c = cluster();
        let s = reg(&mut c, 0, 8); // 8-link sender
        let r_slow = reg(&mut c, 8, 2); // 2-link receiver — bottleneck
        let r_fast = reg(&mut c, 10, 8); // 8-link receiver — fast
                                         // 8MB: send_dur=1ms (8 links), recv_dur=4ms (2 links). collective=4ms.
        let e1 = xfer(&mut c, Time::ZERO, s, r_slow, 8_000_000);
        assert!((e1.as_ms() - 4.0).abs() < 1e-6, "got {}", e1.as_ms());
        // 1MB, symmetric 8-link both sides: send_dur=recv_dur=0.125ms. But the
        // sender was held for the prior collective until t=4. start=4, end=4.125.
        // The previously-decoupled bookkeeping would have set s.send_free to 1
        // (just send_dur of T1), letting T2 start at t=1 — wrong.
        let e2 = xfer(&mut c, Time::ZERO, s, r_fast, 1_000_000);
        assert!(
            (e2.as_ms() - 4.125).abs() < 1e-6,
            "sender should be held by recv bottleneck of T1, got {}",
            e2.as_ms()
        );
    }

    /// Regression for the previously-decoupled legs bug: a sender that's busy
    /// for the next 10 ms must push back the recv side's start (and thus its
    /// bookkeeping), even when the receiver's recv link was free at `now`.
    #[test]
    fn send_busy_blocks_recv_start_and_bookkeeping() {
        let mut c = cluster();
        // Three sender groups (each 1 link) + one receiver group (1 link); the
        // sender groups are disjoint, the receiver is shared so we can probe
        // its recv_free bookkeeping.
        let s_busy = reg(&mut c, 0, 1);
        let d_far = reg(&mut c, 5, 1);
        let d_probe = reg(&mut c, 4, 1);
        let s_disjoint = reg(&mut c, 2, 1);
        // Pre-occupy the busy sender out to t=10ms via a 10MB transfer.
        let _pre = xfer(&mut c, Time::ZERO, s_busy, d_far, 10_000_000);
        // 1MB transfer from the busy sender to d_probe. Receiver was idle but
        // sender's send_free=10 → start=10, end=11.
        let end = xfer(&mut c, Time::ZERO, s_busy, d_probe, 1_000_000);
        assert!((end.as_ms() - 11.0).abs() < 1e-6, "got {}", end.as_ms());
        // d_probe.recv_free should now be 11. A subsequent transfer from a
        // disjoint sender into d_probe must therefore wait until t=11, not t=0.
        let end2 = xfer(&mut c, Time::ZERO, s_disjoint, d_probe, 1_000_000);
        assert!(
            (end2.as_ms() - 12.0).abs() < 1e-6,
            "recv group's recv_free should track the coupled end, got {}",
            end2.as_ms()
        );
    }

    /// Multi-source gather with a throwaway kind/tag (no logger).
    fn gather(c: &mut GpuCluster, now: Time, sources: &[(u16, u64)], d: u16) -> Time {
        c.submit_gather(now, sources, d, "test", "")
    }

    #[test]
    fn gather_runs_senders_in_parallel_not_serialized() {
        // A=2MB, B=6MB, each over a 1-link sender, into a 2-link receiver.
        // Senders run in parallel → send bound = slowest = 6ms; receiver drains
        // the 8MB aggregate over 2 links = 4ms → arrival = max(6,4) = 6ms.
        // Analytic α=0. (N serialized per-source transfers would cost 8ms.)
        let mut c = cluster();
        let sa = reg(&mut c, 0, 1);
        let sb = reg(&mut c, 1, 1);
        let d = reg(&mut c, 2, 2);
        let end = gather(&mut c, Time::ZERO, &[(sa, 2_000_000), (sb, 6_000_000)], d);
        assert!((end.as_ms() - 6.0).abs() < 1e-6, "got {}", end.as_ms());
    }

    #[test]
    fn gather_receiver_drains_the_aggregate() {
        // 4 sources × 1MB into a 2-link receiver. Each 1-link sender pushes
        // 1MB=1ms in parallel; the receiver drains 4MB/2=2MB per link=2ms →
        // arrival = max(1,2) = 2ms. Four serialized transfers would cost 4ms.
        let mut c = cluster();
        let d = reg(&mut c, 0, 2);
        let srcs: Vec<(u16, u64)> = (0..4)
            .map(|i| (reg(&mut c, 10 + i, 1), 1_000_000))
            .collect();
        let end = gather(&mut c, Time::ZERO, &srcs, d);
        assert!((end.as_ms() - 2.0).abs() < 1e-6, "got {}", end.as_ms());
    }

    #[test]
    fn gather_frees_each_sender_after_its_own_slice() {
        // A=2MB, B=6MB, 1-link each, into a 2-link receiver (collective ends at
        // 6ms). A's send link must free after its OWN 2ms transmission, not the
        // 6ms collective: a follow-up 1MB transfer from A (1-link → 1ms) to a
        // fast idle receiver therefore starts at t=2 and ends at t=3.
        let mut c = cluster();
        let sa = reg(&mut c, 0, 1);
        let sb = reg(&mut c, 1, 1);
        let d = reg(&mut c, 2, 2);
        let fast = reg(&mut c, 4, 1);
        let _ = gather(&mut c, Time::ZERO, &[(sa, 2_000_000), (sb, 6_000_000)], d);
        let end = xfer(&mut c, Time::ZERO, sa, fast, 1_000_000);
        assert!(
            (end.as_ms() - 3.0).abs() < 1e-6,
            "A should free at 2ms, got end {}",
            end.as_ms()
        );
    }

    #[test]
    fn gather_single_source_matches_transfer_arrival() {
        // One source, 4-link sender → 2-link receiver, 8MB. send 8MB/4=2ms, recv
        // 8MB/2=4ms → arrival 4ms — identical to the equivalent submit_transfer
        // (α + max(a−α, b−α) ≡ max(a, b); α=0 here anyway).
        let mut c = cluster();
        let s = reg(&mut c, 0, 4);
        let d = reg(&mut c, 4, 2);
        let g = gather(&mut c, Time::ZERO, &[(s, 8_000_000)], d);
        assert!((g.as_ms() - 4.0).abs() < 1e-6, "got {}", g.as_ms());
    }
}
