//! Analyzer-side mirror of the sim's `cost_manifest.json` (the serializable
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

/// The whole manifest: ordered leaf slots, the flattened aggregation tree, and
/// the composite labels index-aligned to `nodes` (`node_labels[i]` is the
/// worklet/model label that wrapped node `i`, or `None`).
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Manifest {
    pub slots: Vec<LeafDesc>,
    pub nodes: Vec<FlatCostNode>,
    pub node_labels: Vec<Option<String>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trimmed but structurally real `cost_manifest.json` (dense Llama3, 2
    /// slots + a Scale fold), pinning the exact JSON the sim emits. If the sim's
    /// serde representation drifts (enum tag style, Range shape, field names),
    /// this fails — that's the drift guard.
    const SAMPLE: &str = r#"{
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
    }"#;

    #[test]
    fn sample_manifest_round_trips() {
        let m: Manifest = serde_json::from_str(SAMPLE).expect("deserialize sample manifest");
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
