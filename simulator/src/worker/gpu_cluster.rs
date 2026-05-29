//! `GpuCluster` — the unified GPU registry **and** inter-worker transfer timing
//! oracle (L5 / `docs/layers.md` §「GpuCluster」).
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
//! and receive a `u16` id that flows through `SendSpec` / `Handoff` /
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
use std::rc::Rc;

use serde::Serialize;

use crate::common::Time;
use crate::timing::kernels::{P2pInterKernel, P2pInterKernelInput};

/// One physical GPU and who owns it. The cluster's `gpus` field is a flat list
/// of these (the reporting / `run_meta.json` subset); `pool` + `worker_id` make
/// the worker→gpu and pool→gpu groupings derivable without a second table.
#[derive(Clone, Debug, Serialize)]
pub struct GpuInfo {
    pub id: u16,
    pub name: String,
    pub pool: u16,
    pub worker_id: u16,
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
        CostSource::Analytic { bytes_per_ms: (gbps * 1e6).max(f64::MIN_POSITIVE) }
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
    /// First GPU id in this group (informational, debugging).
    #[allow(dead_code)]
    base: u16,
    /// Number of GPUs = link count for aggregate-bandwidth math.
    count: u16,
    /// Earliest time this group's send streams are idle.
    send_free: Time,
    /// Earliest time this group's recv streams are idle.
    recv_free: Time,
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
        }
    }

    /// Register `n` contiguous-id GPUs owned by `(pool, worker_id)`, all sharing
    /// `name`, and return the base id of the new block. Ids continue from the
    /// current length, so calling once per worker yields a dense `0..total` id
    /// space across all pools.
    pub fn allocate(&mut self, pool: u16, worker_id: u16, n: u16, name: &str) -> u16 {
        let base = self.gpus.len() as u16;
        for offset in 0..n {
            self.gpus.push(GpuInfo {
                id: base + offset,
                name: name.to_string(),
                pool,
                worker_id,
            });
        }
        base
    }

    /// Register one comm group covering `[base, base+count)` and return its id.
    /// Workers register their attn-shard endpoint set once at construction; the
    /// returned id is what flows through `SendSpec` / `Handoff` / `TransferPlan`.
    pub fn register_comm_group(&mut self, base: u16, count: u16) -> u16 {
        let gid = self.groups.len() as u16;
        self.groups.push(CommGroup {
            base,
            count,
            send_free: Time::ZERO,
            recv_free: Time::ZERO,
        });
        gid
    }

    pub fn num_gpus(&self) -> usize {
        self.gpus.len()
    }

    pub fn num_groups(&self) -> usize {
        self.groups.len()
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
    pub fn submit_transfer(
        &mut self,
        now: Time,
        send_gid: u16,
        recv_gid: u16,
        bytes: u64,
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
        end
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

    #[test]
    fn recv_leg_dominates_when_fewer_dst_links() {
        // 8 send links vs 2 recv links: recv carries bytes/2 per link → slower.
        let mut c = cluster();
        let s = c.register_comm_group(0, 8);
        let d = c.register_comm_group(8, 2);
        let bytes = 8_000_000u64; // 8 MB
        let end = c.submit_transfer(Time::ZERO, s, d, bytes);
        // send: (8e6/8)/1e6 = 1ms; recv: (8e6/2)/1e6 = 4ms → resident at 4ms.
        assert!((end.as_ms() - 4.0).abs() < 1e-6, "got {}", end.as_ms());
    }

    #[test]
    fn more_links_is_faster_aggregated_bandwidth() {
        let bytes = 8_000_000u64;
        let mut c4 = cluster();
        let s4 = c4.register_comm_group(0, 4);
        let d4 = c4.register_comm_group(4, 4);
        let e4 = c4.submit_transfer(Time::ZERO, s4, d4, bytes);
        let mut c8 = cluster();
        let s8 = c8.register_comm_group(0, 8);
        let d8 = c8.register_comm_group(8, 8);
        let e8 = c8.submit_transfer(Time::ZERO, s8, d8, bytes);
        assert!(e8.as_ms() < e4.as_ms(), "8 links should beat 4: {} vs {}", e8.as_ms(), e4.as_ms());
    }

    #[test]
    fn same_group_serializes() {
        // Two back-to-back transfers re-using the same src+dst groups queue up.
        let mut c = cluster();
        let s = c.register_comm_group(0, 1);
        let d = c.register_comm_group(1, 1);
        let bytes = 1_000_000u64; // 1 MB → 1ms on a single link
        let e1 = c.submit_transfer(Time::ZERO, s, d, bytes);
        let e2 = c.submit_transfer(Time::ZERO, s, d, bytes);
        assert!((e1.as_ms() - 1.0).abs() < 1e-6, "first {}", e1.as_ms());
        assert!((e2.as_ms() - 2.0).abs() < 1e-6, "second {}", e2.as_ms());
    }

    #[test]
    fn disjoint_groups_do_not_serialize() {
        let mut c = cluster();
        let s1 = c.register_comm_group(0, 1);
        let d1 = c.register_comm_group(1, 1);
        let s2 = c.register_comm_group(2, 1);
        let d2 = c.register_comm_group(3, 1);
        let bytes = 1_000_000u64;
        let e1 = c.submit_transfer(Time::ZERO, s1, d1, bytes);
        let e2 = c.submit_transfer(Time::ZERO, s2, d2, bytes);
        assert!((e1.as_ms() - 1.0).abs() < 1e-6);
        assert!((e2.as_ms() - 1.0).abs() < 1e-6, "disjoint groups should not queue: {}", e2.as_ms());
    }

    /// Regression for "fast side held by bottleneck": when the recv side is
    /// slower (fewer links), the sender is throttled to the recv pace and its
    /// `send_free` must reflect the full `max(send_dur, recv_dur)` collective
    /// duration, not just `send_dur`. A subsequent transfer reusing that sender
    /// to a fast receiver must wait for the full collective to finish.
    #[test]
    fn recv_bottleneck_holds_sender_too() {
        let mut c = cluster();
        let s = c.register_comm_group(0, 8);     // 8-link sender
        let r_slow = c.register_comm_group(8, 2); // 2-link receiver — bottleneck
        let r_fast = c.register_comm_group(10, 8); // 8-link receiver — fast
        // 8MB: send_dur=1ms (8 links), recv_dur=4ms (2 links). collective=4ms.
        let e1 = c.submit_transfer(Time::ZERO, s, r_slow, 8_000_000);
        assert!((e1.as_ms() - 4.0).abs() < 1e-6, "got {}", e1.as_ms());
        // 1MB, symmetric 8-link both sides: send_dur=recv_dur=0.125ms. But the
        // sender was held for the prior collective until t=4. start=4, end=4.125.
        // The previously-decoupled bookkeeping would have set s.send_free to 1
        // (just send_dur of T1), letting T2 start at t=1 — wrong.
        let e2 = c.submit_transfer(Time::ZERO, s, r_fast, 1_000_000);
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
        let s_busy = c.register_comm_group(0, 1);
        let d_far = c.register_comm_group(5, 1);
        let d_probe = c.register_comm_group(4, 1);
        let s_disjoint = c.register_comm_group(2, 1);
        // Pre-occupy the busy sender out to t=10ms via a 10MB transfer.
        let _pre = c.submit_transfer(Time::ZERO, s_busy, d_far, 10_000_000);
        // 1MB transfer from the busy sender to d_probe. Receiver was idle but
        // sender's send_free=10 → start=10, end=11.
        let end = c.submit_transfer(Time::ZERO, s_busy, d_probe, 1_000_000);
        assert!((end.as_ms() - 11.0).abs() < 1e-6, "got {}", end.as_ms());
        // d_probe.recv_free should now be 11. A subsequent transfer from a
        // disjoint sender into d_probe must therefore wait until t=11, not t=0.
        let end2 = c.submit_transfer(Time::ZERO, s_disjoint, d_probe, 1_000_000);
        assert!(
            (end2.as_ms() - 12.0).abs() < 1e-6,
            "recv group's recv_free should track the coupled end, got {}",
            end2.as_ms()
        );
    }
}
