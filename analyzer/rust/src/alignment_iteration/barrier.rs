//! The measured critical path of one iteration, under the barrier model.
//!
//! A synchronizing collective is a barrier: every rank has to arrive before any
//! rank leaves. An iteration is therefore not one race between the ranks but a
//! chain of short races separated by barriers, and a different rank can win each
//! one.
//!
//! ```text
//! T_begin                                                              T_end
//!   |---- segment 0 ----[ barrier 0 ]---- segment 1 ----[ barrier 1 ]----|
//!        rank 2 slowest                    rank 0 slowest
//! ```
//!
//! * A **barrier** is a maximal run of consecutive synchronizing positions in
//!   program order. It is a run, not a single kernel, because a fused reduction
//!   can emit two positions of one operation (the MNNVL all-reduce and its
//!   Lamport norm), and splitting them would insert an empty segment. Its cost is
//!   `net = max_r(last end) - max_r(first start)`: from the last rank to arrive
//!   to the last rank to leave. The time an early rank sits at the barrier is
//!   `skew`. Skew is reported, but it is not on the path, because it coincides
//!   with the slow rank's busy time.
//! * A **segment** is the window between one barrier's exit and the next one's
//!   entry. Within it each rank's busy time is the union of its non-collective
//!   launches across its streams, clipped to the window. The rank with the
//!   largest union wins the segment. The window is the same for every rank, so
//!   this is also the rank with the least idle time.
//!
//! The windows and the barriers tile `[T_begin, T_end]` over every rank, so
//!
//! ```text
//! wall          = critical_path + idle
//! critical_path = critical_busy + collective
//! critical_busy = on_path_kernel_sum - hidden_same_stream - hidden_cross_stream
//! ```
//!
//! hold exactly, in integer nanoseconds.
//!
//! Which position a winner's busy time belongs to is decided by one sweep
//! ([`attribute`]): sort by start, let the wider launch win a tie, and give each
//! launch only the part no earlier launch already covered. The covered part is
//! charged to the launch holding the running maximum end. That launch started no
//! later, so it covers the whole stretch on its own and the charge is exact. The
//! charge is split by whether the two launches share a stream (PDL) or not.
//!
//! [`launch_overlaps`] is a separate, diagnostic view. For every launch it
//! reports how much of it overlaps other launches on the same device and which
//! ones they are, whether or not those launches are on the path.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{bail, Result};

use super::{IterationMeasurement, KernelLaunch};

/// The operation key of a position with no mapped operation. It only orders the
/// split of a multi-operation barrier. Callers read the per-position shares.
const UNMAPPED: &str = "<unmapped>";

/// One run of consecutive synchronizing positions and its cross-rank extent.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Barrier {
    pub positions: Vec<usize>,
    /// The last rank's arrival: max over ranks of that rank's first start.
    pub enter_ns: u64,
    /// The last rank's departure: max over ranks of that rank's last end.
    pub exit_ns: u64,
    /// The barrier's length on the path. It is `exit - enter`, except that it
    /// starts no earlier than the previous barrier's exit, so two barriers can
    /// never both charge one stretch.
    pub net_ns: u64,
    /// How long the earliest arrival waited for the latest one. Off the path.
    pub skew_ns: u64,
    /// The rank that left last, which is where the collective's own time is read.
    pub last_exit_device: i64,
}

/// One window between two barriers, or between an iteration edge and a barrier.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Segment {
    pub start_ns: u64,
    pub end_ns: u64,
    /// The rank with the largest busy union. The lowest device wins a tie.
    pub winner: Option<i64>,
    pub busy_ns_by_device: BTreeMap<i64, u64>,
    /// The winner's gaps between its first and last busy instant.
    pub idle_internal_ns: u64,
    /// The winner's time before its first and after its last busy instant.
    pub idle_boundary_ns: u64,
    /// The rank that gates the segment's end: the last to arrive at the closing
    /// barrier, or for the final segment the rank that finished last.
    pub gating_device_id: Option<i64>,
    /// The gating rank's own non-busy time in the window.
    pub gating_gap_ns: u64,
}

impl Segment {
    pub fn window_ns(&self) -> u64 {
        self.end_ns - self.start_ns
    }
}

/// The barrier-model reduction of one iteration.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct BarrierPath {
    pub wall_ns: u64,
    pub critical_busy_ns: u64,
    pub collective_ns: u64,
    pub collective_skew_ns: u64,
    pub idle_internal_ns: u64,
    pub idle_boundary_ns: u64,
    /// The winners' own launch durations, clipped to their segments.
    pub on_path_kernel_sum_ns: u64,
    /// Every launch on every rank, collectives included.
    pub all_rank_kernel_sum_ns: u64,
    pub hidden_same_stream_ns: u64,
    pub hidden_cross_stream_ns: u64,
    /// Non-collective launch time inside barrier windows, on the last rank to
    /// leave. It is already inside `collective_ns`, so it is not subtracted again.
    pub hidden_under_collective_same_stream_ns: u64,
    pub hidden_under_collective_cross_stream_ns: u64,
    pub critical_rank_switches: usize,
    pub critical_busy_ns_by_device: BTreeMap<i64, u64>,
    /// Per position, its share of the critical path. Sums to `critical_path_ns`.
    pub kernel_critical_ns: Vec<u64>,
    /// Per position, its winners' clipped durations. Sums to
    /// `on_path_kernel_sum_ns`.
    pub kernel_on_path_ns: Vec<u64>,
    pub segments: Vec<Segment>,
    pub barriers: Vec<Barrier>,
}

impl BarrierPath {
    pub fn critical_path_ns(&self) -> u64 {
        self.critical_busy_ns + self.collective_ns
    }

    pub fn idle_ns(&self) -> u64 {
        self.idle_internal_ns + self.idle_boundary_ns
    }
}

/// What a device's launches say about one of them.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct LaunchOverlap {
    /// Intersection with the union of every other launch on this device.
    pub overlap_ns: u64,
    /// Intersection with the union of the other launches on this launch's stream.
    pub same_stream_ns: u64,
    /// Intersection with the union of the launches on other streams.
    pub cross_stream_ns: u64,
    /// `(position, ns)` per overlapping position, in position order. A position
    /// launched more than once on this device is folded into one entry.
    pub partners: Vec<(usize, u64)>,
}

/// Streams are compared within one device. A capture without stream ids falls
/// back to the track, which the parser derives one-to-one from the stream.
fn stream_key(launch: &KernelLaunch) -> (bool, u64) {
    match launch.stream_id {
        Some(stream_id) => (true, stream_id),
        None => (false, launch.track_index as u64),
    }
}

/// One non-collective launch in sweep order.
#[derive(Clone, Copy)]
struct RankLaunch {
    start_ns: u64,
    end_ns: u64,
    track_index: usize,
    position: usize,
    stream: (bool, u64),
}

/// One rank's non-collective launches, sorted once for the whole iteration.
///
/// `reach[i]` is the largest end among launches `0..=i`, so it never decreases.
/// A prefix whose reach is at or before `lo` cannot overlap a window starting at
/// `lo`, which makes [`RankLaunches::window`] two binary searches.
struct RankLaunches {
    launches: Vec<RankLaunch>,
    reach: Vec<u64>,
}

impl RankLaunches {
    fn new(mut launches: Vec<RankLaunch>) -> Self {
        launches.sort_unstable_by_key(sweep_key);
        let mut furthest = 0;
        let reach = launches
            .iter()
            .map(|launch| {
                furthest = furthest.max(launch.end_ns);
                furthest
            })
            .collect();
        Self { launches, reach }
    }

    /// The launches that can overlap `[lo, hi)`, still in sweep order.
    fn window(&self, lo: u64, hi: u64) -> &[RankLaunch] {
        let first = self.reach.partition_point(|&reach| reach <= lo);
        let last = self
            .launches
            .partition_point(|launch| launch.start_ns <= hi);
        &self.launches[first..last.max(first)]
    }
}

/// Start first, the wider launch on a tie, then track and position so the order
/// never depends on how the inventory happened to insert its rows.
fn sweep_key(launch: &RankLaunch) -> (u64, std::cmp::Reverse<u64>, usize, usize) {
    (
        launch.start_ns,
        std::cmp::Reverse(launch.end_ns),
        launch.track_index,
        launch.position,
    )
}

/// One rank's covered time in a window, split across the launches that cover it.
#[derive(Default)]
struct Attribution {
    union_ns: u64,
    /// `(position, ns)` in sweep order.
    slices: Vec<(usize, u64)>,
    /// `(position, clipped ns)` per launch.
    clipped: Vec<(usize, u64)>,
    hidden_same_stream_ns: u64,
    hidden_cross_stream_ns: u64,
    first_ns: Option<u64>,
    last_ns: Option<u64>,
}

/// Sweep one rank's launches clipped to `[lo, hi)`.
///
/// The sort is redone per window because it is keyed on the clipped interval:
/// clipping a launch that straddles `lo` can reorder it against one that starts
/// inside the window.
fn attribute(launches: &[RankLaunch], lo: u64, hi: u64) -> Attribution {
    let mut ordered: Vec<RankLaunch> = launches
        .iter()
        .filter_map(|launch| {
            let start_ns = launch.start_ns.max(lo);
            let end_ns = launch.end_ns.min(hi);
            (end_ns > start_ns).then_some(RankLaunch {
                start_ns,
                end_ns,
                ..*launch
            })
        })
        .collect();
    ordered.sort_unstable_by_key(sweep_key);

    let mut out = Attribution::default();
    // The launch holding the running maximum end covers everything already
    // covered from its own start onward, and it started no later than the
    // launch being swept.
    let mut cover: Option<RankLaunch> = None;
    for launch in ordered {
        let duration_ns = launch.end_ns - launch.start_ns;
        out.clipped.push((launch.position, duration_ns));
        out.first_ns.get_or_insert(launch.start_ns);
        out.last_ns = Some(
            out.last_ns
                .map_or(launch.end_ns, |last| last.max(launch.end_ns)),
        );
        let exclusive_ns = match cover {
            None => duration_ns,
            Some(cover) => launch
                .end_ns
                .saturating_sub(cover.end_ns.max(launch.start_ns)),
        };
        if exclusive_ns > 0 {
            out.slices.push((launch.position, exclusive_ns));
            out.union_ns += exclusive_ns;
        }
        if let Some(cover) = cover {
            let hidden_ns = duration_ns - exclusive_ns;
            if cover.stream == launch.stream {
                out.hidden_same_stream_ns += hidden_ns;
            } else {
                out.hidden_cross_stream_ns += hidden_ns;
            }
        }
        if cover.is_none_or(|cover| launch.end_ns > cover.end_ns) {
            cover = Some(launch);
        }
    }
    out
}

/// Divide `total` across `(key, weight)` pairs without losing a nanosecond.
///
/// The heaviest key comes first, and the last key in that order takes the
/// remainder. With no weight at all the split is even, and the remainder goes
/// to the first keys.
fn split_by_weight<K: Clone + Ord>(
    total: u64,
    keys: &[K],
    weights: &BTreeMap<K, u64>,
) -> Vec<(K, u64)> {
    let total_weight: u64 = weights.values().sum();
    if total_weight == 0 {
        let count = keys.len() as u64;
        let (share, extra) = (total / count, total % count);
        return keys
            .iter()
            .enumerate()
            .map(|(index, key)| (key.clone(), share + u64::from((index as u64) < extra)))
            .collect();
    }
    let mut ordered: Vec<_> = weights.iter().collect();
    ordered.sort_by(|(left, lw), (right, rw)| rw.cmp(lw).then_with(|| left.cmp(right)));
    let mut out = Vec::with_capacity(ordered.len());
    let mut assigned = 0;
    for (key, weight) in &ordered[..ordered.len() - 1] {
        let part = (u128::from(total) * u128::from(**weight) / u128::from(total_weight)) as u64;
        out.push(((*key).clone(), part));
        assigned += part;
    }
    out.push((ordered[ordered.len() - 1].0.clone(), total - assigned));
    out
}

/// The lowest device holding the largest value.
fn first_max(values: impl IntoIterator<Item = (i64, u64)>) -> Option<i64> {
    let mut best: Option<(i64, u64)> = None;
    for (device, value) in values {
        if best.is_none_or(|(_, current)| value > current) {
            best = Some((device, value));
        }
    }
    best.map(|(device, _)| device)
}

pub(super) fn barrier_path(measurement: &IterationMeasurement) -> Result<BarrierPath> {
    let kernels = &measurement.kernels;
    let mut path = BarrierPath {
        kernel_critical_ns: vec![0; kernels.len()],
        kernel_on_path_ns: vec![0; kernels.len()],
        ..Default::default()
    };
    let all = kernels.iter().flat_map(|(_, item)| &item.launches);
    let (Some(t_begin), Some(t_end)) = (
        all.clone().map(|launch| launch.start_ns).min(),
        all.clone().map(|launch| launch.end_ns).max(),
    ) else {
        return Ok(path);
    };
    path.wall_ns = t_end - t_begin;
    path.all_rank_kernel_sum_ns = all.map(|launch| launch.end_ns - launch.start_ns).sum();

    let operation_key =
        |position: usize| -> &str { kernels[position].1.operation.as_deref().unwrap_or(UNMAPPED) };

    // Barriers: maximal runs of consecutive synchronizing positions.
    struct Run {
        positions: Vec<usize>,
        first_start: BTreeMap<i64, u64>,
        last_end: BTreeMap<i64, u64>,
        operation_weights: BTreeMap<String, u64>,
        position_weights: Vec<u64>,
    }
    let mut runs: Vec<Run> = Vec::new();
    let mut index = 0;
    while index < kernels.len() {
        if !kernels[index].1.synchronizing {
            index += 1;
            continue;
        }
        let mut positions = vec![index];
        while index + 1 < kernels.len() && kernels[index + 1].1.synchronizing {
            index += 1;
            positions.push(index);
        }
        index += 1;

        let mut first_start: BTreeMap<i64, u64> = BTreeMap::new();
        let mut last_end: BTreeMap<i64, u64> = BTreeMap::new();
        let mut operation_weights: BTreeMap<String, u64> = BTreeMap::new();
        let mut position_weights = Vec::with_capacity(positions.len());
        for &position in &positions {
            // This position's own last-arrival-to-last-exit extent.
            let mut starts: BTreeMap<i64, u64> = BTreeMap::new();
            let mut ends: BTreeMap<i64, u64> = BTreeMap::new();
            for launch in &kernels[position].1.launches {
                let start = starts.entry(launch.device_id).or_insert(launch.start_ns);
                *start = (*start).min(launch.start_ns);
                let end = ends.entry(launch.device_id).or_insert(launch.end_ns);
                *end = (*end).max(launch.end_ns);
            }
            for (device, start) in &starts {
                let first = first_start.entry(*device).or_insert(*start);
                *first = (*first).min(*start);
                let last = last_end.entry(*device).or_insert(ends[device]);
                *last = (*last).max(ends[device]);
            }
            let extent = match (starts.values().max(), ends.values().max()) {
                (Some(enter), Some(exit)) => exit.saturating_sub(*enter),
                _ => 0,
            };
            if !starts.is_empty() {
                *operation_weights
                    .entry(operation_key(position).to_owned())
                    .or_default() += extent;
            }
            position_weights.push(extent);
        }
        if first_start.is_empty() {
            continue;
        }
        let devices: BTreeSet<i64> = first_start.keys().copied().collect();
        if devices != measurement.device_ids {
            bail!(
                "synchronizing positions {:?} ran on devices {:?}, not on every device of the \
                 iteration {:?}; a barrier over part of the ranks (a collective inside one \
                 data-parallel group) is not supported",
                positions,
                devices,
                measurement.device_ids
            );
        }
        runs.push(Run {
            positions,
            first_start,
            last_end,
            operation_weights,
            position_weights,
        });
    }

    // Non-collective launches per rank, for the segment races.
    let mut by_device: BTreeMap<i64, Vec<RankLaunch>> = BTreeMap::new();
    for (position, (_, item)) in kernels.iter().enumerate() {
        if item.synchronizing {
            continue;
        }
        for launch in &item.launches {
            by_device
                .entry(launch.device_id)
                .or_default()
                .push(RankLaunch {
                    start_ns: launch.start_ns,
                    end_ns: launch.end_ns,
                    track_index: launch.track_index,
                    position,
                    stream: stream_key(launch),
                });
        }
    }
    let ranks: BTreeMap<i64, RankLaunches> = by_device
        .into_iter()
        .map(|(device, launches)| (device, RankLaunches::new(launches)))
        .collect();
    let last_finisher = first_max(kernels.iter().flat_map(|(_, item)| &item.launches).fold(
        BTreeMap::<i64, u64>::new(),
        |mut ends, launch| {
            let end = ends.entry(launch.device_id).or_insert(launch.end_ns);
            *end = (*end).max(launch.end_ns);
            ends
        },
    ));

    // Tile [T_begin, T_end]: a segment before every barrier, then the tail.
    let mut cursor = t_begin;
    for run_index in 0..=runs.len() {
        let run = runs.get(run_index);
        let lo = cursor;
        let hi = lo.max(run.map_or(t_end, |run| *run.first_start.values().max().unwrap()));

        let attributed: BTreeMap<i64, Attribution> = ranks
            .iter()
            .map(|(device, launches)| (*device, attribute(launches.window(lo, hi), lo, hi)))
            .collect();
        let busy_ns_by_device: BTreeMap<i64, u64> = attributed
            .iter()
            .map(|(device, attribution)| (*device, attribution.union_ns))
            .collect();
        let winner = first_max(busy_ns_by_device.iter().map(|(d, v)| (*d, *v)));
        let window_ns = hi - lo;
        let (mut idle_internal_ns, mut idle_boundary_ns) = (0, window_ns);
        if let Some(device) = winner {
            let won = &attributed[&device];
            for &(position, ns) in &won.slices {
                path.kernel_critical_ns[position] += ns;
            }
            for &(position, ns) in &won.clipped {
                path.kernel_on_path_ns[position] += ns;
                path.on_path_kernel_sum_ns += ns;
            }
            path.hidden_same_stream_ns += won.hidden_same_stream_ns;
            path.hidden_cross_stream_ns += won.hidden_cross_stream_ns;
            path.critical_busy_ns += won.union_ns;
            *path.critical_busy_ns_by_device.entry(device).or_default() += won.union_ns;
            if let (Some(first), Some(last)) = (won.first_ns, won.last_ns) {
                idle_boundary_ns = (first - lo) + (hi - last);
                idle_internal_ns = (last - first) - won.union_ns;
            }
        }
        path.idle_internal_ns += idle_internal_ns;
        path.idle_boundary_ns += idle_boundary_ns;
        let gating_device_id = match run {
            Some(run) => first_max(run.first_start.iter().map(|(d, v)| (*d, *v))),
            None => last_finisher,
        };
        let gating_gap_ns = gating_device_id.map_or(window_ns, |device| {
            window_ns - busy_ns_by_device.get(&device).copied().unwrap_or(0)
        });
        path.segments.push(Segment {
            start_ns: lo,
            end_ns: hi,
            winner,
            busy_ns_by_device,
            idle_internal_ns,
            idle_boundary_ns,
            gating_device_id,
            gating_gap_ns,
        });

        let Some(run) = run else { break };
        let enter_ns = *run.first_start.values().max().unwrap();
        let exit_ns = *run.last_end.values().max().unwrap();
        let net_ns = exit_ns.saturating_sub(hi);
        let skew_ns = enter_ns - *run.first_start.values().min().unwrap();
        let last_exit_device = first_max(run.last_end.iter().map(|(d, v)| (*d, *v))).unwrap();
        path.collective_ns += net_ns;
        path.collective_skew_ns += skew_ns;
        cursor = hi.max(exit_ns);

        // Charge the net to operations, then each operation's share to its
        // positions, both by extent.
        let operations: Vec<String> = run
            .positions
            .iter()
            .map(|&position| operation_key(position).to_owned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let operation_shares = if operations.len() == 1 {
            vec![(operations[0].clone(), net_ns)]
        } else {
            split_by_weight(net_ns, &operations, &run.operation_weights)
        };
        for (operation, share) in operation_shares {
            let members: Vec<usize> = run
                .positions
                .iter()
                .copied()
                .filter(|&position| operation_key(position) == operation)
                .collect();
            let weights: BTreeMap<usize, u64> = run
                .positions
                .iter()
                .zip(&run.position_weights)
                .filter(|(position, weight)| **weight > 0 && members.contains(position))
                .map(|(position, weight)| (*position, *weight))
                .collect();
            for (position, ns) in split_by_weight(share, &members, &weights) {
                path.kernel_critical_ns[position] += ns;
            }
        }

        // Non-collective work that ran inside the barrier window on the rank
        // that left last. Same stream means the stream the collective ran on.
        if net_ns > 0 {
            let collective_streams: BTreeSet<(bool, u64)> = run
                .positions
                .iter()
                .flat_map(|&position| &kernels[position].1.launches)
                .filter(|launch| launch.device_id == last_exit_device)
                .map(stream_key)
                .collect();
            if let Some(launches) = ranks.get(&last_exit_device) {
                for launch in launches.window(hi, exit_ns) {
                    let clipped = launch
                        .end_ns
                        .min(exit_ns)
                        .saturating_sub(launch.start_ns.max(hi));
                    if collective_streams.contains(&launch.stream) {
                        path.hidden_under_collective_same_stream_ns += clipped;
                    } else {
                        path.hidden_under_collective_cross_stream_ns += clipped;
                    }
                }
            }
        }
        path.barriers.push(Barrier {
            positions: run.positions.clone(),
            enter_ns,
            exit_ns,
            net_ns,
            skew_ns,
            last_exit_device,
        });
    }

    let owners: Vec<Option<i64>> = path
        .segments
        .iter()
        .filter(|segment| segment.window_ns() > 0)
        .map(|segment| segment.winner)
        .collect();
    path.critical_rank_switches = owners.windows(2).filter(|pair| pair[0] != pair[1]).count();

    let accounted = path.critical_path_ns() + path.idle_ns();
    if accounted != path.wall_ns {
        bail!(
            "barrier path does not tile the iteration: critical path {} + idle {} != wall {} ns",
            path.critical_path_ns(),
            path.idle_ns(),
            path.wall_ns
        );
    }
    Ok(path)
}

/// Per position, one [`LaunchOverlap`] per launch, in the position's launch order.
pub(super) fn launch_overlaps(measurement: &IterationMeasurement) -> Vec<Vec<LaunchOverlap>> {
    let kernels = &measurement.kernels;
    let mut out: Vec<Vec<LaunchOverlap>> = kernels
        .iter()
        .map(|(_, item)| vec![LaunchOverlap::default(); item.launches.len()])
        .collect();

    // `(start, end, position, launch index, stream)` per device.
    type DeviceLaunch = (u64, u64, usize, usize, (bool, u64));
    let mut by_device: BTreeMap<i64, Vec<DeviceLaunch>> = BTreeMap::new();
    for (position, (_, item)) in kernels.iter().enumerate() {
        for (launch_index, launch) in item.launches.iter().enumerate() {
            if launch.end_ns > launch.start_ns {
                by_device.entry(launch.device_id).or_default().push((
                    launch.start_ns,
                    launch.end_ns,
                    position,
                    launch_index,
                    stream_key(launch),
                ));
            }
        }
    }

    for launches in by_device.values_mut() {
        launches.sort_unstable();
        // Every overlapping pair, found once from its earlier-starting side.
        let mut pairs: Vec<Vec<usize>> = vec![Vec::new(); launches.len()];
        for left in 0..launches.len() {
            let end = launches[left].1;
            for right in left + 1..launches.len() {
                if launches[right].0 >= end {
                    break;
                }
                pairs[left].push(right);
                pairs[right].push(left);
            }
        }
        for (me, others) in pairs.iter().enumerate() {
            if others.is_empty() {
                continue;
            }
            let (start, end, position, launch_index, stream) = launches[me];
            let clip = |other: usize| {
                let (other_start, other_end, ..) = launches[other];
                (other_start.max(start), other_end.min(end))
            };
            let covered = |filter: &dyn Fn(usize) -> bool| -> u64 {
                let intervals: Vec<(u64, u64)> = others
                    .iter()
                    .copied()
                    .filter(|&other| filter(other))
                    .map(clip)
                    .collect();
                super::interval_union_ns(&intervals)
            };
            let mut partners: BTreeMap<usize, Vec<(u64, u64)>> = BTreeMap::new();
            for &other in others {
                partners
                    .entry(launches[other].2)
                    .or_default()
                    .push(clip(other));
            }
            out[position][launch_index] = LaunchOverlap {
                overlap_ns: covered(&|_| true),
                same_stream_ns: covered(&|other| launches[other].4 == stream),
                cross_stream_ns: covered(&|other| launches[other].4 != stream),
                partners: partners
                    .into_iter()
                    .map(|(partner, intervals)| (partner, super::interval_union_ns(&intervals)))
                    .collect(),
            };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::IterationKernelAggregate;
    use super::*;

    /// `(device, start, end, track)`
    type Launch = (i64, u64, u64, usize);

    fn position(
        operation: &str,
        synchronizing: bool,
        launches: &[Launch],
    ) -> (String, IterationKernelAggregate) {
        let launches: Vec<KernelLaunch> = launches
            .iter()
            .map(|&(device_id, start_ns, end_ns, track_index)| KernelLaunch {
                device_id,
                start_ns,
                end_ns,
                correlation_id: None,
                track_index,
                stream_id: Some(100 + track_index as u64),
            })
            .collect();
        (
            operation.to_owned(),
            IterationKernelAggregate {
                operation: (!operation.is_empty()).then(|| operation.to_owned()),
                synchronizing,
                device_ids: launches.iter().map(|launch| launch.device_id).collect(),
                launches,
                ..Default::default()
            },
        )
    }

    fn measurement(kernels: Vec<(String, IterationKernelAggregate)>) -> IterationMeasurement {
        let device_ids = kernels
            .iter()
            .flat_map(|(_, item)| item.device_ids.iter().copied())
            .collect();
        IterationMeasurement {
            kernels,
            inventory_kernels: Vec::new(),
            phase_summaries: Vec::new(),
            device_ids,
            busy_union_ms: 0.0,
            track_intervals: BTreeMap::new(),
        }
    }

    fn assert_identities(path: &BarrierPath) {
        assert_eq!(path.critical_path_ns() + path.idle_ns(), path.wall_ns);
        assert_eq!(
            path.critical_busy_ns,
            path.on_path_kernel_sum_ns - path.hidden_same_stream_ns - path.hidden_cross_stream_ns
        );
        assert_eq!(
            path.kernel_critical_ns.iter().sum::<u64>(),
            path.critical_path_ns()
        );
        assert_eq!(
            path.kernel_on_path_ns.iter().sum::<u64>(),
            path.on_path_kernel_sum_ns
        );
    }

    #[test]
    fn a_single_rank_without_collectives_is_its_busy_union() {
        let path = barrier_path(&measurement(vec![
            position("a", false, &[(0, 0, 10, 0)]),
            position("b", false, &[(0, 12, 20, 0)]),
        ]))
        .unwrap();
        assert_eq!(path.critical_busy_ns, 18);
        assert_eq!(path.collective_ns, 0);
        assert_eq!(path.idle_internal_ns, 2);
        assert_eq!(path.idle_boundary_ns, 0);
        assert_eq!(path.kernel_critical_ns, vec![10, 8]);
        assert_identities(&path);
    }

    #[test]
    fn each_segment_is_won_by_its_own_slowest_rank() {
        // Rank 1 is slow before the barrier and rank 0 after it. A single-device
        // path would charge one rank's whole timeline; the barrier path takes
        // the slow side of each race.
        let path = barrier_path(&measurement(vec![
            position("attn", false, &[(0, 0, 4, 0), (1, 0, 10, 0)]),
            position("ar", true, &[(0, 4, 13, 0), (1, 10, 13, 0)]),
            position("moe", false, &[(0, 13, 25, 0), (1, 13, 16, 0)]),
        ]))
        .unwrap();
        assert_eq!(path.segments[0].winner, Some(1));
        assert_eq!(path.segments[1].winner, Some(0));
        assert_eq!(path.critical_rank_switches, 1);
        assert_eq!(path.collective_ns, 3);
        assert_eq!(path.collective_skew_ns, 6);
        assert_eq!(path.critical_busy_ns, 22);
        assert_eq!(path.kernel_critical_ns, vec![10, 3, 12]);
        assert_eq!(path.segments[0].gating_device_id, Some(1));
        assert_eq!(
            path.critical_busy_ns_by_device,
            BTreeMap::from([(0, 12), (1, 10)])
        );
        assert_identities(&path);
    }

    #[test]
    fn the_segment_winner_is_the_busiest_rank_even_when_another_arrives_last() {
        // Rank 0 is busy for 8 and arrives at 8. Rank 1 is busy for 6 with a gap
        // and arrives at 10. The winner is rank 0; rank 1 is the gating rank and
        // its gap is reported beside it.
        let path = barrier_path(&measurement(vec![
            position("a", false, &[(0, 0, 8, 0), (1, 0, 3, 0)]),
            position("b", false, &[(1, 7, 10, 0)]),
            position("ar", true, &[(0, 8, 12, 0), (1, 10, 12, 0)]),
        ]))
        .unwrap();
        let segment = &path.segments[0];
        assert_eq!(segment.winner, Some(0));
        assert_eq!(segment.idle_boundary_ns, 2);
        assert_eq!(segment.gating_device_id, Some(1));
        assert_eq!(segment.gating_gap_ns, 4);
        assert_identities(&path);
    }

    #[test]
    fn a_multi_operation_barrier_splits_its_net_by_extent_without_rounding_loss() {
        let path = barrier_path(&measurement(vec![
            position("x", true, &[(0, 0, 7, 0), (1, 0, 7, 0)]),
            position("y", true, &[(0, 7, 10, 0), (1, 7, 10, 0)]),
            position("y", true, &[(0, 10, 11, 0), (1, 10, 11, 0)]),
        ]))
        .unwrap();
        assert_eq!(path.barriers.len(), 1);
        assert_eq!(path.collective_ns, 11);
        // x weighs 7 and y weighs 3 + 1: 11 * 7 / 11 = 7 to x, the rest to y,
        // and y's 4 goes 3 : 1 to its two positions.
        assert_eq!(path.kernel_critical_ns, vec![7, 3, 1]);
        assert_identities(&path);
    }

    #[test]
    fn same_and_cross_stream_overlap_are_removed_and_told_apart() {
        // One rank: a PDL successor on track 0 starts 2 before its predecessor
        // ends, and a side-stream kernel on track 1 runs under both.
        let path = barrier_path(&measurement(vec![
            position("a", false, &[(0, 0, 10, 0)]),
            position("b", false, &[(0, 8, 20, 0)]),
            position("side", false, &[(0, 12, 15, 1)]),
        ]))
        .unwrap();
        assert_eq!(path.on_path_kernel_sum_ns, 25);
        assert_eq!(path.hidden_same_stream_ns, 2);
        assert_eq!(path.hidden_cross_stream_ns, 3);
        assert_eq!(path.critical_busy_ns, 20);
        assert_identities(&path);
    }

    #[test]
    fn three_way_overlap_is_charged_once() {
        let path = barrier_path(&measurement(vec![
            position("a", false, &[(0, 0, 10, 0)]),
            position("b", false, &[(0, 2, 8, 1)]),
            position("c", false, &[(0, 4, 12, 0)]),
        ]))
        .unwrap();
        assert_eq!(path.critical_busy_ns, 12);
        assert_eq!(path.hidden_cross_stream_ns, 6);
        assert_eq!(path.hidden_same_stream_ns, 6);
        assert_identities(&path);
    }

    #[test]
    fn work_under_a_collective_is_reported_but_not_charged_twice() {
        let path = barrier_path(&measurement(vec![
            position("ar", true, &[(0, 0, 10, 0), (1, 0, 10, 0)]),
            position("next", false, &[(0, 8, 14, 0), (1, 10, 14, 0)]),
            position("side", false, &[(0, 2, 5, 1)]),
        ]))
        .unwrap();
        assert_eq!(path.collective_ns, 10);
        assert_eq!(path.hidden_under_collective_same_stream_ns, 2);
        assert_eq!(path.hidden_under_collective_cross_stream_ns, 3);
        assert_eq!(path.critical_path_ns(), 14);
        assert_identities(&path);
    }

    #[test]
    fn a_barrier_that_starts_inside_the_previous_one_is_not_charged_twice() {
        let path = barrier_path(&measurement(vec![
            position("ar1", true, &[(0, 0, 10, 0), (1, 0, 10, 0)]),
            position("gap", false, &[(0, 10, 11, 0)]),
            position("ar2", true, &[(0, 6, 15, 0), (1, 6, 15, 0)]),
        ]))
        .unwrap();
        assert_eq!(path.collective_ns, 15);
        assert_identities(&path);
    }

    #[test]
    fn a_barrier_over_part_of_the_ranks_is_an_error() {
        let error = barrier_path(&measurement(vec![
            position("a", false, &[(0, 0, 4, 0), (1, 0, 4, 0), (2, 0, 4, 0)]),
            position("ar", true, &[(0, 4, 6, 0), (1, 4, 6, 0)]),
        ]))
        .unwrap_err();
        assert!(error.to_string().contains("not on every device"), "{error}");
    }

    #[test]
    fn overlap_names_partners_and_streams_including_a_collective() {
        let overlaps = launch_overlaps(&measurement(vec![
            position("ar", true, &[(0, 0, 10, 0)]),
            position("pdl", false, &[(0, 7, 20, 0)]),
            position("side", false, &[(0, 12, 18, 1)]),
        ]));
        assert_eq!(
            overlaps[1][0],
            LaunchOverlap {
                overlap_ns: 9,
                same_stream_ns: 3,
                cross_stream_ns: 6,
                partners: vec![(0, 3), (2, 6)],
            }
        );
        assert_eq!(overlaps[0][0].partners, vec![(1, 3)]);
        assert_eq!(overlaps[2][0].cross_stream_ns, 6);
    }
}
