//! Per-node clock alignment of a multi-node capture.
//!
//! A multi-node capture is one Nsight report per node, folded onto the head
//! node's timeline by the difference of the sessions' UTC start stamps
//! (`alignment/profiler/multinode.py::merge_nsys_sqlite`). Those stamps come
//! from each node's wall clock, so every peer node keeps a constant residual
//! offset, typically a few hundred microseconds. That is longer than one layer
//! of a decode step. The barrier model reads raw timestamps across ranks, so an
//! uncorrected offset puts different layers of different ranks into the same
//! segment, and the winning rank's busy time is charged to the wrong
//! operations.
//!
//! The capture itself measures the offset. A synchronizing collective cannot
//! finish on any rank before every rank has contributed, and its last step is a
//! short exchange, so all ranks leave it within a few microseconds of each
//! other in true time. A node's offset is therefore the median, over the
//! iteration's synchronizing positions that span nodes, of that node's exit
//! minus the reference node's exit. A node's exit is the median exit of its
//! own devices. The reference is the node holding the lowest device id, which
//! is the head, so corrected times stay on the head's clock and keep matching
//! the head-clock workload metrics.
//!
//! Each iteration is aligned on its own collectives, so a slow drift between
//! node clocks over a long capture is followed. A node that an iteration
//! cannot align on its own (fewer than [`MIN_SAMPLES`] spanning positions, or
//! an implausible estimate) takes the run-wide offset instead: the median of
//! that node's per-iteration estimates over the whole capture
//! ([`ClockDomains::with_run_offsets`]). Merge offsets are constant per node,
//! so the run median is the best estimate an iteration without its own
//! evidence has. Only when no iteration of the run aligns a node is it left
//! unshifted. [`OffsetSource`] records which estimate each node used.
//!
//! Devices map to nodes by the merge's numbering: node `k` owns devices
//! `k * L .. (k + 1) * L`, where `L` is the per-node device count. Without a
//! known `L`, or on a single node, there is one clock and nothing is shifted.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

use super::IterationKernelAggregate;

/// Fewest node-spanning synchronizing positions an iteration needs before its
/// median exit difference is trusted as an offset.
pub(super) const MIN_SAMPLES: usize = 3;

/// An estimate beyond this is not a clock offset (the merge's stamps are good
/// to well under a millisecond); it means the exits did not line up, and the
/// iteration is left unshifted rather than moved by a wrong amount.
pub(super) const MAX_OFFSET_NS: i64 = 5_000_000;

/// Which clock each device's timestamps were taken on.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct ClockDomains {
    /// Devices on one node, or `None` when the capture has one clock.
    devices_per_node: Option<i64>,
    /// Run-wide fallback: the reference node the medians are relative to, and
    /// `node -> median per-iteration offset`. `None` until calibrated.
    run_offsets: Option<(i64, BTreeMap<i64, i64>)>,
}

/// One iteration's own clock evidence.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct IterationEstimate {
    /// Node holding the iteration's lowest device id; offsets are relative to it.
    pub reference: Option<i64>,
    /// Every other node with launches in the iteration.
    pub peers: std::collections::BTreeSet<i64>,
    /// `node -> offset` for the peers this iteration aligned on its own.
    pub offsets: BTreeMap<i64, i64>,
}

/// Which estimate put a peer node on the reference clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum OffsetSource {
    /// The iteration's own node-spanning collectives.
    Iteration,
    /// The run-wide median, because the iteration had too little evidence.
    RunMedian,
    /// Neither: the node is left unshifted.
    None,
}

#[derive(Deserialize)]
struct ProfileResult {
    #[serde(default)]
    cuda_visible_devices: Option<String>,
    #[serde(default)]
    parsed_device_ids: Option<Vec<i64>>,
}

impl ClockDomains {
    pub fn single() -> Self {
        Self::default()
    }

    pub fn per_node(devices_per_node: i64) -> Self {
        Self {
            devices_per_node: (devices_per_node > 0).then_some(devices_per_node),
            run_offsets: None,
        }
    }

    /// Whether timestamps come from more than one clock.
    pub fn is_multi_node(&self) -> bool {
        self.devices_per_node.is_some()
    }

    /// Calibrate the run-wide fallback from every iteration's own estimate.
    /// The medians are taken relative to the reference node most iterations
    /// share (the head), over the iterations that share it, and a node needs
    /// [`MIN_SAMPLES`] aligned iterations before its median is trusted.
    pub fn with_run_offsets<'a>(
        mut self,
        estimates: impl IntoIterator<Item = &'a IterationEstimate>,
    ) -> Self {
        let mut by_reference: BTreeMap<i64, BTreeMap<i64, Vec<i64>>> = BTreeMap::new();
        let mut iterations_by_reference: BTreeMap<i64, usize> = BTreeMap::new();
        for estimate in estimates {
            let Some(reference) = estimate.reference else { continue };
            *iterations_by_reference.entry(reference).or_default() += 1;
            let samples = by_reference.entry(reference).or_default();
            for (node, offset) in &estimate.offsets {
                samples.entry(*node).or_default().push(*offset);
            }
        }
        let Some(reference) = iterations_by_reference
            .iter()
            .max_by_key(|(node, count)| (**count, std::cmp::Reverse(**node)))
            .map(|(node, _)| *node)
        else {
            self.run_offsets = None;
            return self;
        };
        let medians: BTreeMap<i64, i64> = by_reference
            .remove(&reference)
            .unwrap_or_default()
            .into_iter()
            .filter(|(_, values)| values.len() >= MIN_SAMPLES)
            .map(|(node, mut values)| (node, median_i64(&mut values)))
            .collect();
        self.run_offsets = (!medians.is_empty()).then_some((reference, medians));
        self
    }

    /// Offset and source for every peer node of one iteration: its own
    /// estimate where it has one, else the run median when that is relative
    /// to the same reference node, else none (offset 0).
    pub fn resolve(&self, estimate: &IterationEstimate) -> BTreeMap<i64, (i64, OffsetSource)> {
        estimate
            .peers
            .iter()
            .map(|node| {
                let resolved = if let Some(offset) = estimate.offsets.get(node) {
                    (*offset, OffsetSource::Iteration)
                } else if let Some(offset) = self
                    .run_offsets
                    .as_ref()
                    .filter(|(reference, _)| Some(*reference) == estimate.reference)
                    .and_then(|(_, medians)| medians.get(node))
                {
                    (*offset, OffsetSource::RunMedian)
                } else {
                    (0, OffsetSource::None)
                };
                (*node, resolved)
            })
            .collect()
    }

    /// Read the per-node device count from the capture's `profile_result.json`
    /// (`cuda_visible_devices` is one node's list). A capture without it, or
    /// whose devices fit on one node, has one clock.
    pub fn from_profile_dir(profile_log_dir: &Path) -> Result<Self> {
        let path = profile_log_dir.join("profile_result.json");
        if !path.exists() {
            return Ok(Self::single());
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("read {}", path.display()))?;
        let result: ProfileResult =
            serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
        let per_node = result
            .cuda_visible_devices
            .as_deref()
            .map(|list| list.split(',').filter(|item| !item.trim().is_empty()).count() as i64)
            .unwrap_or(0);
        let total = result.parsed_device_ids.as_ref().map_or(0, |ids| ids.len() as i64);
        if per_node > 0 && total > per_node {
            Ok(Self::per_node(per_node))
        } else {
            Ok(Self::single())
        }
    }

    pub fn node_of(&self, device_id: i64) -> i64 {
        self.devices_per_node.map_or(0, |size| device_id.div_euclid(size))
    }
}

fn median_i64(values: &mut [i64]) -> i64 {
    values.sort_unstable();
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        // Midpoint without overflow; rounds toward negative infinity.
        values[middle - 1] + (values[middle] - values[middle - 1]).div_euclid(2)
    } else {
        values[middle]
    }
}

/// Per-node offset to subtract from that node's timestamps, keyed by node,
/// from this iteration's own collectives only. The reference node is absent
/// (its offset is zero), and so is every node that could not be aligned.
#[cfg(test)]
pub(super) fn node_offsets(
    kernels: &BTreeMap<String, IterationKernelAggregate>,
    domains: &ClockDomains,
) -> BTreeMap<i64, i64> {
    let mut offsets = estimate_iteration(kernels, domains).offsets;
    offsets.retain(|_, offset| *offset != 0);
    offsets
}

/// This iteration's reference node, peer nodes, and the peers it aligns on
/// its own collectives.
pub(super) fn estimate_iteration(
    kernels: &BTreeMap<String, IterationKernelAggregate>,
    domains: &ClockDomains,
) -> IterationEstimate {
    if domains.devices_per_node.is_none() {
        return IterationEstimate::default();
    }
    let nodes: std::collections::BTreeSet<i64> = kernels
        .values()
        .flat_map(|item| item.launches.iter().map(|launch| domains.node_of(launch.device_id)))
        .collect();
    let Some(reference) = kernels
        .values()
        .flat_map(|item| item.launches.iter().map(|launch| launch.device_id))
        .min()
        .map(|device| domains.node_of(device))
    else {
        return IterationEstimate::default();
    };
    let peers = nodes.into_iter().filter(|node| *node != reference).collect();
    let mut samples: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
    for item in kernels.values().filter(|item| item.synchronizing) {
        let mut exit_by_device: BTreeMap<i64, u64> = BTreeMap::new();
        for launch in &item.launches {
            let exit = exit_by_device.entry(launch.device_id).or_insert(launch.end_ns);
            *exit = (*exit).max(launch.end_ns);
        }
        let mut exits_by_node: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
        for (device, exit) in exit_by_device {
            exits_by_node
                .entry(domains.node_of(device))
                .or_default()
                .push(exit as i64);
        }
        if exits_by_node.len() < 2 {
            continue;
        }
        let Some(reference_exits) = exits_by_node.get(&reference) else {
            continue;
        };
        let reference_exit = median_i64(&mut reference_exits.clone());
        for (node, mut exits) in exits_by_node {
            if node != reference {
                samples
                    .entry(node)
                    .or_default()
                    .push(median_i64(&mut exits) - reference_exit);
            }
        }
    }
    let offsets = samples
        .into_iter()
        .filter(|(_, values)| values.len() >= MIN_SAMPLES)
        .map(|(node, mut values)| (node, median_i64(&mut values)))
        .filter(|(_, offset)| offset.abs() <= MAX_OFFSET_NS)
        .collect();
    IterationEstimate {
        reference: Some(reference),
        peers,
        offsets,
    }
}

/// Move `ns`, taken on a node whose clock reads `offset` ahead of the
/// reference, onto the reference clock.
pub(super) fn shift(ns: u64, offset: i64) -> u64 {
    (ns as i64 - offset) as u64
}

#[cfg(test)]
mod tests {
    use super::super::KernelLaunch;
    use super::*;

    fn launch(device_id: i64, start_ns: u64, end_ns: u64) -> KernelLaunch {
        KernelLaunch {
            device_id,
            start_ns,
            end_ns,
            correlation_id: None,
            track_index: 0,
            stream_id: Some(7),
        }
    }

    /// Two nodes of two devices each. Node 1's clock reads `skew` ahead. Each
    /// collective `k` really ends at `1000 * (k + 1)` on every rank, give or
    /// take a few nanoseconds of genuine exit spread.
    fn kernels(skew: i64, collectives: usize) -> BTreeMap<String, IterationKernelAggregate> {
        let mut out = BTreeMap::new();
        for k in 0..collectives {
            let end = 1_000_000 * (k as u64 + 1);
            let launches = (0..4)
                .map(|device| {
                    let jitter = (device as u64 * 3) % 5;
                    let local = if device >= 2 { skew } else { 0 };
                    let end = (end + jitter) as i64 + local;
                    launch(device, (end - 20_000) as u64, end as u64)
                })
                .collect();
            out.insert(
                format!("forward/ar{k}"),
                IterationKernelAggregate {
                    synchronizing: true,
                    launches,
                    ..Default::default()
                },
            );
        }
        out.insert(
            "forward/gemm".to_owned(),
            IterationKernelAggregate {
                launches: vec![launch(3, 10, 5_000_000)],
                ..Default::default()
            },
        );
        out
    }

    #[test]
    fn a_peer_node_offset_is_read_from_collective_exits() {
        let offsets = node_offsets(&kernels(257_000, 8), &ClockDomains::per_node(2));
        let node1 = offsets[&1];
        assert!((node1 - 257_000).abs() <= 5, "node 1 offset {node1}");
        assert!(!offsets.contains_key(&0));
        assert_eq!(shift(1_257_000, node1), 1_257_000 - node1 as u64);
    }

    #[test]
    fn a_node_running_behind_gets_a_negative_offset() {
        let offsets = node_offsets(&kernels(-120_000, 8), &ClockDomains::per_node(2));
        assert!((offsets[&1] + 120_000).abs() <= 5);
    }

    #[test]
    fn one_clock_or_too_few_collectives_shifts_nothing() {
        assert!(node_offsets(&kernels(257_000, 8), &ClockDomains::single()).is_empty());
        // Every device on one node: no second clock to compare.
        assert!(node_offsets(&kernels(257_000, 8), &ClockDomains::per_node(4)).is_empty());
        assert!(node_offsets(&kernels(257_000, MIN_SAMPLES - 1), &ClockDomains::per_node(2))
            .is_empty());
    }

    #[test]
    fn an_implausible_estimate_is_rejected() {
        let offsets = node_offsets(&kernels(MAX_OFFSET_NS + 1, 8), &ClockDomains::per_node(2));
        assert!(offsets.is_empty());
    }

    #[test]
    fn an_iteration_without_evidence_takes_the_run_median() {
        let domains = ClockDomains::per_node(2);
        let aligned: Vec<_> = [256_990, 257_000, 257_020]
            .iter()
            .map(|skew| estimate_iteration(&kernels(*skew, 8), &domains))
            .collect();
        let domains = domains.with_run_offsets(&aligned);
        // Too few collectives of its own: the run median (257_000 +- jitter).
        let sparse = estimate_iteration(&kernels(257_000, MIN_SAMPLES - 1), &domains);
        assert!(sparse.offsets.is_empty());
        let (offset, source) = domains.resolve(&sparse)[&1];
        assert_eq!(source, OffsetSource::RunMedian);
        assert!((offset - 257_000).abs() <= 25, "run median {offset}");
        // Enough evidence of its own: the iteration's estimate wins.
        let own = estimate_iteration(&kernels(300_000, 8), &domains);
        let (offset, source) = domains.resolve(&own)[&1];
        assert_eq!(source, OffsetSource::Iteration);
        assert!((offset - 300_000).abs() <= 5);
    }

    #[test]
    fn without_a_calibrated_run_a_sparse_iteration_is_left_unshifted() {
        let domains = ClockDomains::per_node(2);
        let sparse = estimate_iteration(&kernels(257_000, MIN_SAMPLES - 1), &domains);
        assert_eq!(domains.resolve(&sparse)[&1], (0, OffsetSource::None));
        // Fewer than MIN_SAMPLES aligned iterations do not calibrate a median.
        let one = [estimate_iteration(&kernels(257_000, 8), &domains)];
        let domains = domains.with_run_offsets(&one);
        assert_eq!(domains.resolve(&sparse)[&1], (0, OffsetSource::None));
        // A single clock has no peers to resolve.
        assert!(estimate_iteration(&kernels(257_000, 8), &ClockDomains::single()).peers.is_empty());
    }

    #[test]
    fn the_run_median_is_relative_to_the_shared_reference_node() {
        let domains = ClockDomains::per_node(2);
        let aligned: Vec<_> =
            (0..3).map(|_| estimate_iteration(&kernels(257_000, 8), &domains)).collect();
        let domains = domains.with_run_offsets(&aligned);
        // An iteration whose reference is another node must not borrow node 0's
        // medians.
        let foreign = IterationEstimate {
            reference: Some(1),
            peers: [0].into_iter().collect(),
            offsets: BTreeMap::new(),
        };
        assert_eq!(domains.resolve(&foreign)[&0], (0, OffsetSource::None));
    }

    #[test]
    fn devices_map_to_nodes_by_the_merge_numbering() {
        let domains = ClockDomains::per_node(4);
        assert_eq!(
            (0..16).map(|d| domains.node_of(d)).collect::<Vec<_>>(),
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        );
        assert_eq!(ClockDomains::single().node_of(13), 0);
    }

    #[test]
    fn the_profile_result_names_the_per_node_device_count() {
        let dir = std::env::temp_dir().join(format!("clock-domains-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("profile_result.json"),
            r#"{"cuda_visible_devices": "0,1,2,3", "parsed_device_ids": [0,1,2,3,4,5,6,7]}"#,
        )
        .unwrap();
        assert_eq!(ClockDomains::from_profile_dir(&dir).unwrap(), ClockDomains::per_node(4));
        std::fs::write(
            dir.join("profile_result.json"),
            r#"{"cuda_visible_devices": "0,1,2,3", "parsed_device_ids": [0,1,2,3]}"#,
        )
        .unwrap();
        assert_eq!(ClockDomains::from_profile_dir(&dir).unwrap(), ClockDomains::single());
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(ClockDomains::from_profile_dir(&dir).unwrap(), ClockDomains::single());
    }
}
