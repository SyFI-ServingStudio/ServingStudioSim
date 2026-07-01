//! Analyzer-side mirror of each sim
//! `cost_manifest/worker_<pool_tag>_<worker_id>.json` (the serializable
//! [`CostManifest`] from `simulator/src/timing/cost_tree.rs`). The analyzer has
//! no `simulator` dependency, so we re-declare the shapes as plain serde structs
//! that must stay wire-compatible with the sim's `#[derive(Serialize)]`. The
//! [`tests::sample_manifest_round_trips`] drift guard pins a real sample so a
//! schema change on the sim side fails here loudly instead of silently
//! mis-placing slices.

use std::ops::Range;

use serde::Deserialize;

/// Per-slot leaf identity (kernel kind + one-line config). Slot index = position
/// in `slot_time_ms` / `slot_input` parquet list columns.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct LeafDesc {
    pub name: String,
    pub kind: String,
    pub config: String,
}

/// Flattened cost-tree node. `children` is a contiguous range into `nodes`; a
/// parent index always precedes its children (BFS layout), so a placement walk
/// from index 0 visits the whole tree. `Range<usize>` deserializes from the
/// `{"start":_,"end":_}` JSON the sim's serde emits for `std::ops::Range`.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub enum FlatCostNode {
    Leaf(usize),
    Sum { children: Range<usize> },
    Max { overlap: f32, children: Range<usize> },
    Scale { n: u32, children: Range<usize> },
}

/// One section's manifest: ordered leaf slots, the flattened aggregation tree,
/// and the composite labels index-aligned to `nodes` (`node_labels[i]` is the
/// worklet/model label that wrapped node `i`, or `None`).
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Manifest {
    pub slots: Vec<LeafDesc>,
    pub nodes: Vec<FlatCostNode>,
    pub node_labels: Vec<Option<String>>,
}

/// A named building-block section within a worker's manifest doc. The sim emits
/// `{"section": "...", "slots": [...], "nodes": [...], "node_labels": [...]}` —
/// the `section` tag plus a flattened [`Manifest`] (sim-side `CostManifestSection`).
/// A `cost_log` row's `section` field selects which one interprets its slots.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ManifestSection {
    pub section: String,
    #[serde(flatten)]
    pub manifest: Manifest,
}

/// The whole per-worker manifest: an ordered list of named sections (sim-side
/// `CostManifestDoc`). An iter-wise worker has a single `iter` section; an AFD
/// layer-wise worker has several (`attn`, or `prologue` / `pre_attn` / `post_attn`
/// / `post_attn_last` / `epilogue`), each its own CostTree with its own slots.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ManifestDoc {
    pub sections: Vec<ManifestSection>,
}

impl ManifestDoc {
    /// The sub-manifest for `section`, or `None` if this worker has no such
    /// section (a `cost_log` row tagged with a section its manifest lacks).
    pub fn section(&self, name: &str) -> Option<&Manifest> {
        self.sections
            .iter()
            .find(|s| s.section == name)
            .map(|s| &s.manifest)
    }
}

/// Aggregate duration (ns) of the subtree rooted at node `idx`, folding this
/// tree the same way the sim did to produce `total_time_ms`: Leaf = its slot,
/// Sum = Σ children, Max = max(children)/overlap, Scale = n × child. Ancestor
/// `Scale`s are NOT applied (this is the node's own local fold). The root's
/// value reproduces `total_time_ms` up to per-leaf ns rounding.
///
/// Shared by both consumers of a [`Manifest`]: `breakdown` (critical-path gutter
/// + `Max` bottleneck-member pick) and `trace::place` (critical-path collapse of
/// `Max`). `trace::place::tests` / `breakdown::tests` pin that this equals
/// `place`'s emission fold, so the three stay in lockstep.
pub(crate) fn node_time(m: &Manifest, idx: usize, slot_ns: &[i64]) -> i64 {
    match &m.nodes[idx] {
        FlatCostNode::Leaf(slot) => slot_ns.get(*slot).copied().unwrap_or(0),
        FlatCostNode::Sum { children } => children.clone().map(|c| node_time(m, c, slot_ns)).sum(),
        FlatCostNode::Max { overlap, children } => {
            let maxd = children
                .clone()
                .map(|c| node_time(m, c, slot_ns))
                .max()
                .unwrap_or(0);
            let ov = (*overlap as f64).max(1e-9);
            (maxd as f64 / ov).round() as i64
        }
        FlatCostNode::Scale { n, children } => (*n as i64) * node_time(m, children.start, slot_ns),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trimmed but structurally real per-worker cost manifest doc (dense Llama3,
    /// a single `iter` section, 2 slots + a Scale fold), pinning the exact JSON the
    /// sim emits. If the sim's serde representation drifts (the `sections` wrapper,
    /// the flattened section tag, enum tag style, Range shape, field names), this
    /// fails — that's the drift guard.
    const SAMPLE: &str = r#"{
      "sections": [
        {
          "section": "iter",
          "slots": [
            {"name": "m.embedding", "kind": "elementwise", "config": "hidden=4096"},
            {"name": "m.lm_head", "kind": "single_gemm", "config": "n=128256 k=4096"}
          ],
          "nodes": [
            {"Sum": {"children": {"start": 1, "end": 3}}},
            {"Leaf": 0},
            {"Scale": {"n": 32, "children": {"start": 3, "end": 4}}},
            {"Leaf": 1}
          ],
          "node_labels": ["m [dense local, 32 layers]", null, null, null]
        }
      ]
    }"#;

    #[test]
    fn sample_manifest_round_trips() {
        let doc: ManifestDoc = serde_json::from_str(SAMPLE).expect("deserialize sample manifest doc");
        assert_eq!(doc.sections.len(), 1);
        assert_eq!(doc.sections[0].section, "iter");
        let m = doc.section("iter").expect("iter section present");
        assert!(doc.section("missing").is_none());
        assert_eq!(m.slots.len(), 2);
        assert_eq!(m.slots[1].kind, "single_gemm");
        assert_eq!(
            m.nodes[0],
            FlatCostNode::Sum { children: 1..3 },
            "Sum children must deserialize from {{start,end}}"
        );
        assert_eq!(m.nodes[1], FlatCostNode::Leaf(0));
        assert_eq!(
            m.nodes[2],
            FlatCostNode::Scale { n: 32, children: 3..4 }
        );
        assert_eq!(
            m.node_labels[0].as_deref(),
            Some("m [dense local, 32 layers]")
        );
    }
}
