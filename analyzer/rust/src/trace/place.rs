//! Placement walk: lay one iteration's cost tree onto a Perfetto track as nested
//! slices. Mirrors `CostTree::aggregate` (sim side) but emits BEGIN/END slices
//! instead of summing metrics — `Sum` lays children sequentially, `Scale{n}`
//! repeats its child subtree `n` times (one wrapper each, named from the arch's
//! repeat-unit label if present), `Max`
//! spreads children across parallel child-tracks. Durations accumulate in
//! integer ns from one rounded base so nesting stays exact (a parent's END is
//! the accumulated end of its children).

use crate::perfetto::{Annotation, TraceWriter};
use crate::trace::manifest::{node_time, FlatCostNode, Manifest};

/// Per-iteration placement context. `slot_ns[slot]` is this iter's leaf duration
/// (pre-rounded to ns); `slot_input[slot]` is the captured kernel input JSON
/// (empty if the run lacked `--cost-log-slot-input`).
///
/// `expanded` selects how `Max` (parallel branches) renders. Default (`false`) =
/// critical-path collapse: only the bottleneck (largest `node_time`) branch is
/// laid inline on the same track, so a whole compute iteration stays one lane.
/// `true` = the legacy fan-out, each branch on its own `child_track` lane.
pub struct Placer<'a> {
    manifest: &'a Manifest,
    slot_ns: &'a [i64],
    slot_input: &'a [String],
    /// Per-slot achieved FLOPs / bytes (slot-aligned to `slot_ns`), from the
    /// `slot_flops` / `slot_bytes` cost-log columns. Used to annotate each leaf
    /// slice with its achieved TFLOP/s and GB/s. Empty when the run predates the
    /// columns; leaf annotation is then simply skipped.
    slot_flops: &'a [f64],
    slot_bytes: &'a [f64],
    expanded: bool,
}

impl<'a> Placer<'a> {
    pub fn new(
        manifest: &'a Manifest,
        slot_ns: &'a [i64],
        slot_input: &'a [String],
        slot_flops: &'a [f64],
        slot_bytes: &'a [f64],
        expanded: bool,
    ) -> Self {
        Self {
            manifest,
            slot_ns,
            slot_input,
            slot_flops,
            slot_bytes,
            expanded,
        }
    }

    /// Place the whole tree (root = node 0) on `track` starting at `t0` ns.
    /// Returns the laid-out duration in ns (≈ `total_time_ms`, the drift guard).
    pub fn place_root(&self, w: &mut TraceWriter, track: u64, t0: i64) -> i64 {
        self.place(w, track, 0, t0, 1.0)
    }

    /// The composite label the arch attached to flat node `idx` (worklet name,
    /// repeat-unit noun, …), or `None` if it left this node unlabeled.
    fn node_label(&self, idx: usize) -> Option<&str> {
        self.manifest
            .node_labels
            .get(idx)
            .and_then(|o| o.as_deref())
    }

    fn label_or(&self, idx: usize, default: &str) -> String {
        self.node_label(idx).unwrap_or(default).to_string()
    }

    /// Place node `idx` at `t0`, returning its **displayed** duration in ns. Every
    /// natural leaf duration is multiplied by `scale` before emission; `scale` is
    /// `1.0` everywhere except inside a critical-path `Max` with `overlap != 1`,
    /// which recurses its sole branch at `scale/overlap` to compress it into the
    /// node's effective window (see the `Max` arm).
    fn place(&self, w: &mut TraceWriter, track: u64, idx: usize, t0: i64, scale: f64) -> i64 {
        match &self.manifest.nodes[idx] {
            FlatCostNode::Leaf(slot) => {
                let leaf_ns = self.slot_ns.get(*slot).copied().unwrap_or(0);
                let dur = ((leaf_ns as f64) * scale).round() as i64;
                let desc = &self.manifest.slots[*slot];
                let mut anns = vec![
                    Annotation::str("kind", desc.kind.clone()),
                    Annotation::str("config", desc.config.clone()),
                    Annotation::dbl("dur_ms", dur as f64 / 1e6),
                ];
                // Achieved throughput from the *natural* leaf time (`leaf_ns`, not
                // the `scale`-compressed display `dur`): a kernel's physical rate
                // is independent of how the critical-path view squeezes the slice.
                // `0` flops/bytes (profile row had no rate) or `0` time → skip.
                let leaf_s = leaf_ns as f64 / 1e9;
                if leaf_s > 0.0 {
                    if let Some(&flops) = self.slot_flops.get(*slot) {
                        if flops > 0.0 {
                            anns.push(Annotation::dbl("tflops", flops / leaf_s / 1e12));
                        }
                    }
                    if let Some(&bytes) = self.slot_bytes.get(*slot) {
                        if bytes > 0.0 {
                            anns.push(Annotation::dbl("gbps", bytes / leaf_s / 1e9));
                        }
                    }
                }
                if let Some(input) = self.slot_input.get(*slot) {
                    if !input.is_empty() {
                        anns.push(Annotation::str("input", input.clone()));
                    }
                }
                w.begin(track, t0, leaf_short_name(&desc.name), &anns);
                w.end(track, t0 + dur);
                dur
            }
            FlatCostNode::Sum { children } => {
                let label = self.label_or(idx, "sum");
                w.begin(track, t0, &label, &[]);
                let mut t = t0;
                for c in children.clone() {
                    t += self.place(w, track, c, t, scale);
                }
                w.end(track, t);
                t - t0
            }
            FlatCostNode::Scale { n, children } => {
                // Single child range (the homogeneous repeat subtree); repeat it
                // `n` times, each under its own wrapper so the repeat index lives
                // on the wrapper (the leaves share one slot, INV-3). The repeat-
                // unit noun comes from the manifest label the arch attached (e.g.
                // "layer"); with no label we stay generic and assume nothing.
                let unit = self.node_label(idx);
                let wrapper = match unit {
                    Some(u) => format!("{u} ×{n}"),
                    None => format!("×{n}"),
                };
                w.begin(track, t0, &wrapper, &[]);
                let child = children.start;
                let mut t = t0;
                for l in 0..*n {
                    let name = match unit {
                        Some(u) => format!("{u} {l}"),
                        None => format!("[{l}]"),
                    };
                    w.begin(track, t, &name, &[]);
                    let d = self.place(w, track, child, t, scale);
                    w.end(track, t + d);
                    t += d;
                }
                w.end(track, t);
                t - t0
            }
            FlatCostNode::Max { overlap, children } => {
                let label = self.label_or(idx, "max");
                w.begin(track, t0, &label, &[]);
                let dd = if self.expanded {
                    // Legacy fan-out: each branch on its own child-track lane, all
                    // from t0. Parent dur = max(child)/overlap; lane slices render
                    // their full length on separate rows (they may overhang the
                    // wrapper — fine, a lane is its own track).
                    let mut maxd = 0i64;
                    for (i, c) in children.clone().enumerate() {
                        let lane = w.child_track(track, &format!("lane {i}"), i as u64);
                        maxd = maxd.max(self.place(w, lane, c, t0, scale));
                    }
                    ((maxd as f64) / (*overlap as f64)).round() as i64
                } else {
                    // Critical-path collapse: place ONLY the bottleneck branch
                    // (largest `node_time`) inline on this same track — no lanes,
                    // so compute stays one row. With `overlap != 1` the branch's
                    // real length exceeds the node's effective time, so we recurse
                    // it at `scale × eff_nat/crit_nat` (= `scale/overlap`) to
                    // compress its whole subtree into the effective window. The
                    // wrapper then ends at the child's own returned length, so
                    // wrapper == child exactly (no sub-ns overhang / mis-nesting).
                    let crit = children
                        .clone()
                        .max_by_key(|&c| node_time(self.manifest, c, self.slot_ns))
                        .expect("Max node has at least one child");
                    let crit_nat = node_time(self.manifest, crit, self.slot_ns).max(1);
                    let eff_nat = ((crit_nat as f64) / (*overlap as f64)).round() as i64;
                    let child_scale = scale * (eff_nat as f64) / (crit_nat as f64);
                    self.place(w, track, crit, t0, child_scale)
                };
                w.end(track, t0 + dd);
                dd
            }
        }
    }
}

/// Last dotted segment of a slot name for a compact slice label
/// (`unified.pre_attn.qkv_proj` → `qkv_proj`).
fn leaf_short_name(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or(name)
}

/// Count the BEGIN/END slice pairs one iteration places, so the caller can cap
/// at iteration granularity (never truncating mid-iteration). Mirrors `place`'s
/// emission structure: each node is one pair; `Scale{n}` multiplies its child
/// subtree by `n` and adds a `layer` wrapper pair per repeat.
pub fn slice_pairs_per_iter(manifest: &Manifest) -> usize {
    fn count(manifest: &Manifest, idx: usize) -> usize {
        match &manifest.nodes[idx] {
            FlatCostNode::Leaf(_) => 1,
            FlatCostNode::Sum { children } | FlatCostNode::Max { children, .. } => {
                1 + children.clone().map(|c| count(manifest, c)).sum::<usize>()
            }
            FlatCostNode::Scale { n, children } => {
                let child = children.start;
                // wrapper + n × (layer wrapper + child subtree)
                1 + (*n as usize) * (1 + count(manifest, child))
            }
        }
    }
    // + 1 for the outer per-iter wrapper the run loop emits around the root.
    1 + count(manifest, 0)
}

/// Critical-path (default mode) counterpart of [`slice_pairs_per_iter`]: a `Max`
/// contributes `1 (wrapper) + count(bottleneck child)` instead of summing all
/// branches, matching the collapsed placement. Which child is the bottleneck is
/// data-dependent (largest `node_time`), so this needs `slot_ns`. Used as the
/// `max_slices` cap unit when `--expanded` is off.
pub fn critical_pairs_per_iter(manifest: &Manifest, slot_ns: &[i64]) -> usize {
    fn count(manifest: &Manifest, idx: usize, slot_ns: &[i64]) -> usize {
        match &manifest.nodes[idx] {
            FlatCostNode::Leaf(_) => 1,
            FlatCostNode::Sum { children } => {
                1 + children
                    .clone()
                    .map(|c| count(manifest, c, slot_ns))
                    .sum::<usize>()
            }
            FlatCostNode::Max { children, .. } => {
                let crit = children
                    .clone()
                    .max_by_key(|&c| node_time(manifest, c, slot_ns))
                    .expect("Max node has at least one child");
                1 + count(manifest, crit, slot_ns)
            }
            FlatCostNode::Scale { n, children } => {
                let child = children.start;
                1 + (*n as usize) * (1 + count(manifest, child, slot_ns))
            }
        }
    }
    1 + count(manifest, 0, slot_ns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::manifest::Manifest;

    fn sample() -> Manifest {
        serde_json::from_str(
            r#"{
              "slots": [
                {"name": "m.embedding", "kind": "elementwise", "config": "h=4096"},
                {"name": "m.lm_head", "kind": "single_gemm", "config": "n=128256"}
              ],
              "nodes": [
                {"Sum": {"children": {"start": 1, "end": 3}}},
                {"Leaf": 0},
                {"Scale": {"n": 4, "children": {"start": 3, "end": 4}}},
                {"Leaf": 1}
              ],
              "node_labels": [null, null, null, null]
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn placed_root_dur_matches_summed_leaves() {
        let m = sample();
        // embed=10ns, lm_head=5ns scaled ×4 → 10 + 4*5 = 30ns.
        let slot_ns = [10i64, 5];
        let inputs: [String; 0] = [];
        // No Max in this tree → mode is irrelevant; use the default (critical).
        let placer = Placer::new(&m, &slot_ns, &inputs, &[], &[], false);
        let mut w = TraceWriter::new();
        let p = w.process_track(0, "w");
        let t = w.thread_track(p, 0, 0, "n");
        let dur = placer.place_root(&mut w, t, 0);
        assert_eq!(dur, 30, "Sum(embed, Scale{{4}}(lm_head)) = 10 + 4*5");
    }

    #[test]
    fn slice_pairs_counts_scale_expansion() {
        let m = sample();
        // outer(1) + root Sum(1) + embed Leaf(1) + Scale wrapper(1)
        //   + 4 × (layer wrapper(1) + lm_head Leaf(1)) = 1+1+1+1+8 = 12
        assert_eq!(slice_pairs_per_iter(&m), 12);
    }

    /// A hand-built tree mixing all three composites — the dense vertical never
    /// emits `Max`, so this is the only coverage of `Max{overlap}` placement and
    /// of `Max` nested inside `Scale` inside `Sum`:
    ///   Sum( Leaf0, Scale{3}( Max{2}[ Leaf1, Leaf2 ] ), Leaf3 )
    /// BFS-flat layout (parent index < its contiguous child range):
    ///   0 Sum{1..4}  1 Leaf0  2 Scale{3,4..5}  3 Leaf3  4 Max{2,5..7}  5 Leaf1  6 Leaf2
    fn mixed() -> Manifest {
        serde_json::from_str(
            r#"{
              "slots": [
                {"name": "a", "kind": "k", "config": "c"},
                {"name": "b", "kind": "k", "config": "c"},
                {"name": "c", "kind": "k", "config": "c"},
                {"name": "d", "kind": "k", "config": "c"}
              ],
              "nodes": [
                {"Sum": {"children": {"start": 1, "end": 4}}},
                {"Leaf": 0},
                {"Scale": {"n": 3, "children": {"start": 4, "end": 5}}},
                {"Leaf": 3},
                {"Max": {"overlap": 2.0, "children": {"start": 5, "end": 7}}},
                {"Leaf": 1},
                {"Leaf": 2}
              ],
              "node_labels": ["root", null, "layer", null, "attn", null, null]
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn placed_dur_mixes_sum_scale_max_expanded() {
        let m = mixed();
        // slots: a=10, b=8, c=4, d=5.
        // Max{2}[b=8, c=4] = max(8,4)/2 = 4; Scale{3}(4) = 12; Sum(10, 12, 5) = 27.
        let slot_ns = [10i64, 8, 4, 5];
        let inputs: [String; 0] = [];
        let placer = Placer::new(&m, &slot_ns, &inputs, &[], &[], true); // expanded (lanes)
        let mut w = TraceWriter::new();
        let p = w.process_track(0, "w");
        let t = w.thread_track(p, 0, 0, "n");
        let dur = placer.place_root(&mut w, t, 0);
        assert_eq!(dur, 27, "Sum(10, Scale{{3}}(Max{{/2}}(8,4)=4)=12, 5)");
        // Expanded mode fans the Max into child-track lanes.
        let bytes = w.into_gzip().unwrap();
        assert_eq!(&bytes[..2], &[0x1f, 0x8b]);
        let raw = decompress(&bytes);
        assert!(raw.windows(5).any(|win| win == b"lane "), "expanded emits lanes");
    }

    /// Critical mode collapses the `Max` onto the parent track: same root dur
    /// (27), but the bottleneck branch `b` is compressed by `1/overlap` (8→4) and
    /// NO `lane` child-track is emitted, so compute stays one row.
    #[test]
    fn placed_dur_critical_collapses_max() {
        let m = mixed();
        let slot_ns = [10i64, 8, 4, 5];
        let inputs: [String; 0] = [];
        let placer = Placer::new(&m, &slot_ns, &inputs, &[], &[], false); // critical (default)
        let mut w = TraceWriter::new();
        let p = w.process_track(0, "w");
        let t = w.thread_track(p, 0, 0, "n");
        let dur = placer.place_root(&mut w, t, 0);
        // Root dur is identical to expanded: had `b` NOT been compressed to 4,
        // Scale{3}(8)=24 → Sum=39 ≠ 27. So 27 proves the 1/overlap compression.
        assert_eq!(dur, 27, "critical Max still contributes eff=4 → Sum=27");
        let raw = decompress(&w.into_gzip().unwrap());
        assert!(
            !raw.windows(5).any(|win| win == b"lane "),
            "critical mode must not emit child-track lanes"
        );
    }

    #[test]
    fn slice_pairs_counts_mixed_tree() {
        let m = mixed();
        // count(Max)=1+1+1=3; count(Scale{3})=1+3*(1+3)=13;
        // count(Sum)=1+1+13+1=16; +1 outer wrapper = 17.
        assert_eq!(slice_pairs_per_iter(&m), 17);
    }

    #[test]
    fn critical_pairs_counts_mixed_tree() {
        let m = mixed();
        let slot_ns = [10i64, 8, 4, 5];
        // Max collapses to its bottleneck child b: count(Max)=1+1=2;
        // count(Scale{3})=1+3*(1+2)=10; count(Sum)=1+1+10+1=13; +1 outer = 14.
        assert_eq!(critical_pairs_per_iter(&m, &slot_ns), 14);
    }

    /// Gunzip helper for the lane-presence assertions.
    fn decompress(bytes: &[u8]) -> Vec<u8> {
        use std::io::Read;
        let mut d = flate2::read::GzDecoder::new(bytes);
        let mut out = Vec::new();
        d.read_to_end(&mut out).unwrap();
        out
    }
}
