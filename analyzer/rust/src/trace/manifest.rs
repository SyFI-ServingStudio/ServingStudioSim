//! Analyzer-side mirror of each sim
//! `cost_manifest/worker_<pool_tag>_<worker_id>.json` (the serializable
//! [`CostManifest`] from `simulator/src/timing/cost_tree.rs`). The analyzer has
//! no `simulator` dependency, so we re-declare the shapes as plain serde structs
//! that must stay wire-compatible with the sim's `#[derive(Serialize)]`. The
//! [`tests::sample_manifest_round_trips`] drift guard pins a real sample so a
//! schema change on the sim side fails here loudly instead of silently
//! mis-placing slices.

use std::ops::Range;

use serde::{de, Deserialize, Deserializer};
use serde_json::{Map, Value};

/// Per-slot leaf identity (kernel kind + structured config). Slot index = position
/// in `slot_time_ms` / `slot_input` parquet list columns.
///
#[derive(Debug, Clone, PartialEq)]
pub struct LeafDesc {
    pub name: String,
    pub kind: String,
    pub kernel_config: Value,
}

#[derive(Deserialize)]
struct WireLeafDesc {
    name: String,
    kind: String,
    #[serde(default)]
    kernel_config: Option<Value>,
    #[serde(default)]
    config: Option<String>,
    #[serde(default)]
    backends: Vec<String>,
}

impl<'de> Deserialize<'de> for LeafDesc {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = WireLeafDesc::deserialize(deserializer)?;
        let mut kernel_config = match wire.kernel_config {
            Some(Value::Object(config)) => config,
            Some(_) => {
                return Err(de::Error::custom(
                    "field `kernel_config` must be a JSON object",
                ));
            }
            None => {
                let legacy_config = wire.config.ok_or_else(|| {
                    de::Error::custom("missing field `kernel_config` (or legacy `config`)")
                })?;
                let mut config = Map::new();
                // The retired field is display text, not a parseable identity. Keep
                // it verbatim so archived manifests remain inspectable without
                // guessing structure from strings such as `n=... k=...`.
                config.insert("config".to_owned(), Value::String(legacy_config));
                config
            }
        };

        // Some transition-era manifests carry a structured kernel config but
        // still keep the slot-backend index space at the old top level. Normalize
        // that representation at the read boundary; current manifests remain
        // authoritative when they already contain `kernel_config.backends`.
        if !kernel_config.contains_key("backends") && !wire.backends.is_empty() {
            kernel_config.insert(
                "backends".to_owned(),
                Value::Array(wire.backends.into_iter().map(Value::String).collect()),
            );
        }

        Ok(Self {
            name: wire.name,
            kind: wire.kind,
            kernel_config: Value::Object(kernel_config),
        })
    }
}

impl LeafDesc {
    /// Ordered best-of-N candidates. Their JSON array position is the index space
    /// of `cost_log.slot_backend`; no duplicate manifest field is maintained.
    pub fn backends(&self) -> Vec<String> {
        self.kernel_config
            .get("backends")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect()
    }
}

/// Flattened cost-tree node. `children` is a contiguous range into `nodes`; a
/// parent index always precedes its children (BFS layout), so a placement walk
/// from index 0 visits the whole tree. `Range<usize>` deserializes from the
/// `{"start":_,"end":_}` JSON the sim's serde emits for `std::ops::Range`.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub enum FlatCostNode {
    Leaf(usize),
    Sum {
        children: Range<usize>,
    },
    Max {
        overlap: f32,
        children: Range<usize>,
    },
    Scale {
        n: u32,
        children: Range<usize>,
    },
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
///   `Max`). `trace::place::tests` / `breakdown::tests` pin that this equals
///   `place`'s emission fold, so the three stay in lockstep.
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
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                reason = "maxd is a leaf duration in ns (a real trace stays far under 2^52 ns / ~52 \
                          days), and the f64/f64 divide is rounded back into the same ns range before \
                          re-truncating to i64"
            )]
            let result = (maxd as f64 / ov).round() as i64;
            result
        }
        FlatCostNode::Scale { n, children } => (*n as i64) * node_time(m, children.start, slot_ns),
    }
}

/// Balanced (perfect-parallelism) fold of the subtree at `idx`, streamed as
/// per-leaf `(slot, weight)` visits rather than a single number. This is the
/// **mean-mode** sibling of [`node_time`]: where `node_time` collapses a `Max`
/// (parallel shards) to its straggler (`max/overlap`), this collapses it to the
/// balanced average (`mean/overlap`) — modelling every shard's GPU doing an equal
/// share, i.e. the imbalance-free lower bound the `optimality` subject wants.
///
/// The fold is **linear** in the leaf values, so it factors into a per-leaf
/// weight `α` (`∏ 1/(child_count·overlap)` over `Max` ancestors × `∏ n` over
/// `Scale` ancestors) times that leaf's value. Emitting `(slot, α)` lets one walk
/// serve every rung (real time, per-config-best, hardware roofline) and every
/// aggregate (worker total + per-kernel attribution) — the caller multiplies `α`
/// by whichever leaf value that rung/bucket needs. `visit` is called exactly once
/// per `Leaf` node in the subtree.
pub(crate) fn fold_mean<F: FnMut(usize, f64)>(
    m: &Manifest,
    idx: usize,
    weight: f64,
    visit: &mut F,
) {
    match &m.nodes[idx] {
        FlatCostNode::Leaf(slot) => visit(*slot, weight),
        FlatCostNode::Sum { children } => {
            for c in children.clone() {
                fold_mean(m, c, weight, visit);
            }
        }
        FlatCostNode::Max { overlap, children } => {
            #[allow(
                clippy::cast_precision_loss,
                reason = "count is a cost-tree fan-out (a handful of parallel children), far under \
                          f64's 52-bit exact integer range"
            )]
            let count = children.len().max(1) as f64;
            let ov = (*overlap as f64).max(1e-9);
            let child_weight = weight / (count * ov);
            for c in children.clone() {
                fold_mean(m, c, child_weight, visit);
            }
        }
        // `Scale` applies its subtree `n` times on one timeline (num_layers); like
        // `node_time` it recurses through `children.start` (the single child).
        FlatCostNode::Scale { n, children } => {
            fold_mean(m, children.start, weight * (*n as f64), visit);
        }
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
            {"name": "m.embedding", "kind": "elementwise", "kernel_config": {"hidden": {"value": 4096, "expression": "hidden", "bindings": {"hidden": 4096}}, "backends": ["torch"]}},
            {"name": "m.lm_head", "kind": "single_gemm", "kernel_config": {"n": {"value": 128256, "expression": null, "bindings": {}}, "k": {"value": 4096, "expression": null, "bindings": {}}, "backends": ["torch_linear"]}}
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
        let doc: ManifestDoc =
            serde_json::from_str(SAMPLE).expect("deserialize sample manifest doc");
        assert_eq!(doc.sections.len(), 1);
        assert_eq!(doc.sections[0].section, "iter");
        let m = doc.section("iter").expect("iter section present");
        assert!(doc.section("missing").is_none());
        assert_eq!(m.slots.len(), 2);
        assert_eq!(m.slots[1].kind, "single_gemm");
        assert_eq!(m.slots[0].backends(), vec!["torch".to_owned()]);
        assert_eq!(m.slots[1].backends(), vec!["torch_linear".to_owned()]);
        assert_eq!(
            m.nodes[0],
            FlatCostNode::Sum { children: 1..3 },
            "Sum children must deserialize from {{start,end}}"
        );
        assert_eq!(m.nodes[1], FlatCostNode::Leaf(0));
        assert_eq!(
            m.nodes[2],
            FlatCostNode::Scale {
                n: 32,
                children: 3..4
            }
        );
        assert_eq!(
            m.node_labels[0].as_deref(),
            Some("m [dense local, 32 layers]")
        );
    }

    #[test]
    fn fold_mean_is_linear_and_collapses_max_to_mean() {
        // Sum[ Max{overlap:1}[ Leaf0, Leaf1 ], Scale{n:3}[ Leaf2 ] ].
        let m = Manifest {
            slots: Vec::new(), // slot descs unused by the fold
            nodes: vec![
                FlatCostNode::Sum { children: 1..3 },
                FlatCostNode::Max {
                    overlap: 1.0,
                    children: 3..5,
                },
                FlatCostNode::Scale {
                    n: 3,
                    children: 5..6,
                },
                FlatCostNode::Leaf(0),
                FlatCostNode::Leaf(1),
                FlatCostNode::Leaf(2),
            ],
            node_labels: Vec::new(),
        };
        // α: Leaf0/Leaf1 sit under Max/2 → 0.5 each; Leaf2 under Scale×3 → 3.
        let mut alpha = [0.0f64; 3];
        fold_mean(&m, 0, 1.0, &mut |slot, w| alpha[slot] += w);
        assert_eq!(alpha, [0.5, 0.5, 3.0]);

        // Mean-fold value: 0.5·t0 + 0.5·t1 + 3·t2 (mean over the Max pair), whereas
        // node_time takes the Max (straggler) → max(t0,t1) + 3·t2.
        let slot_ns = [10i64, 20, 5];
        let mean: f64 = (0..3).map(|i| alpha[i] * slot_ns[i] as f64).sum();
        assert_eq!(mean, 0.5 * 10.0 + 0.5 * 20.0 + 3.0 * 5.0); // 30.0
        assert_eq!(node_time(&m, 0, &slot_ns), 20 + 3 * 5); // 35 (max branch)
    }

    #[test]
    fn legacy_leaf_normalizes_config_and_top_level_backends() {
        let leaf: LeafDesc = serde_json::from_str(
            r#"{
                "name": "m.qkv",
                "kind": "single_gemm",
                "config": "backends=torch,torch_linear n=6144 k=4096",
                "backends": ["torch", "torch_linear"]
            }"#,
        )
        .expect("deserialize legacy leaf");

        assert_eq!(
            leaf.kernel_config,
            serde_json::json!({
                "config": "backends=torch,torch_linear n=6144 k=4096",
                "backends": ["torch", "torch_linear"],
            })
        );
        assert_eq!(
            leaf.backends(),
            vec!["torch".to_owned(), "torch_linear".to_owned()]
        );
    }

    #[test]
    fn structured_config_wins_while_legacy_backends_fill_only_a_missing_list() {
        let transitional: LeafDesc = serde_json::from_str(
            r#"{
                "name": "m.qkv",
                "kind": "single_gemm",
                "kernel_config": {"n": 6144},
                "config": "retired display text",
                "backends": ["torch_linear"]
            }"#,
        )
        .expect("deserialize transitional leaf");
        assert_eq!(
            transitional.kernel_config,
            serde_json::json!({"n": 6144, "backends": ["torch_linear"]})
        );

        let current: LeafDesc = serde_json::from_str(
            r#"{
                "name": "m.qkv",
                "kind": "single_gemm",
                "kernel_config": {"n": 6144, "backends": ["cutlass"]},
                "backends": ["torch_linear"]
            }"#,
        )
        .expect("deserialize current leaf with a redundant legacy field");
        assert_eq!(current.backends(), vec!["cutlass".to_owned()]);
    }
}
