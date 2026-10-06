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
    Parallel {
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
/// Sum = Σ children, Max/Parallel = max(children)/overlap, Scale = n × child. Ancestor
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
        FlatCostNode::Max { overlap, children } | FlatCostNode::Parallel { overlap, children } => {
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

/// The child of a `Max`/`Parallel` with the largest [`node_time`]: the
/// critical path through it. On an exact tie the last such child, as
/// `max_by_key` picks. Placement, the worker CostTree's `critical` and every
/// other critical-path pick share this so they name the same child.
pub(crate) fn critical_child(m: &Manifest, children: Range<usize>, slot_ns: &[i64]) -> usize {
    children
        .max_by_key(|&c| node_time(m, c, slot_ns))
        .expect("Max/Parallel node has at least one child")
}

/// Balanced fold of a subtree for `R` rungs at once: each rung substitutes its
/// own leaf values and re-evaluates the tree, with
/// - `Max` (rank fan-out) folded to `mean/overlap`: every rank does an equal
///   share, so the gap to `node_time`'s straggler is load imbalance;
/// - `Parallel` (streams on one device) kept at `max/overlap`: nothing to
///   balance, so the streams' overlap never reads as imbalance;
/// - `Sum` = Σ, `Scale{n}` = n × child, as in [`node_time`].
///
/// [`Self::fold`] returns the per-rung subtree value and attributes it to the
/// leaves: `visit(slot, contribution)` runs once per `Leaf`, and the
/// contributions sum to the value. Under a `Parallel` each rung's share goes to
/// that rung's slowest child (exact ties split evenly), so a stream hidden
/// behind another contributes 0 for that rung. The scratch is reused across
/// calls; a fold allocates only when the tree grows.
#[derive(Default)]
pub(crate) struct BalancedFold<const R: usize> {
    value_by_node: Vec<[f64; R]>,
}

impl<const R: usize> BalancedFold<R> {
    pub(crate) fn fold(
        &mut self,
        m: &Manifest,
        root: usize,
        scale: f64,
        leaf_value: &impl Fn(usize) -> [f64; R],
        visit: &mut impl FnMut(usize, [f64; R]),
    ) -> [f64; R] {
        if self.value_by_node.len() < m.nodes.len() {
            self.value_by_node.resize(m.nodes.len(), [0.0; R]);
        }
        let value = self.evaluate(m, root, leaf_value);
        self.attribute(m, root, [scale; R], visit);
        value.map(|v| v * scale)
    }

    fn evaluate(
        &mut self,
        m: &Manifest,
        idx: usize,
        leaf_value: &impl Fn(usize) -> [f64; R],
    ) -> [f64; R] {
        let value = match &m.nodes[idx] {
            FlatCostNode::Leaf(slot) => leaf_value(*slot),
            FlatCostNode::Sum { children } => {
                let mut sum = [0.0; R];
                for c in children.clone() {
                    add(&mut sum, self.evaluate(m, c, leaf_value));
                }
                sum
            }
            FlatCostNode::Scale { n, children } => self
                .evaluate(m, children.start, leaf_value)
                .map(|v| v * f64::from(*n)),
            FlatCostNode::Max { overlap, children } => {
                let mut sum = [0.0; R];
                for c in children.clone() {
                    add(&mut sum, self.evaluate(m, c, leaf_value));
                }
                let divisor = children.len().max(1) as f64 * overlap_divisor(*overlap);
                sum.map(|v| v / divisor)
            }
            FlatCostNode::Parallel { overlap, children } => {
                let mut max = [0.0f64; R];
                for c in children.clone() {
                    let child = self.evaluate(m, c, leaf_value);
                    for r in 0..R {
                        max[r] = max[r].max(child[r]);
                    }
                }
                let divisor = overlap_divisor(*overlap);
                max.map(|v| v / divisor)
            }
        };
        self.value_by_node[idx] = value;
        value
    }

    /// Push `weight` (the ancestors' multiplier per rung) down to the leaves.
    fn attribute(
        &self,
        m: &Manifest,
        idx: usize,
        weight: [f64; R],
        visit: &mut impl FnMut(usize, [f64; R]),
    ) {
        match &m.nodes[idx] {
            FlatCostNode::Leaf(slot) => {
                let value = self.value_by_node[idx];
                visit(*slot, std::array::from_fn(|r| weight[r] * value[r]));
            }
            FlatCostNode::Sum { children } => {
                for c in children.clone() {
                    self.attribute(m, c, weight, visit);
                }
            }
            FlatCostNode::Scale { n, children } => {
                self.attribute(m, children.start, weight.map(|w| w * f64::from(*n)), visit);
            }
            FlatCostNode::Max { overlap, children } => {
                let divisor = children.len().max(1) as f64 * overlap_divisor(*overlap);
                let child_weight = weight.map(|w| w / divisor);
                for c in children.clone() {
                    self.attribute(m, c, child_weight, visit);
                }
            }
            FlatCostNode::Parallel { overlap, children } => {
                let divisor = overlap_divisor(*overlap);
                let mut max = [f64::NEG_INFINITY; R];
                let mut ties = [0u32; R];
                for c in children.clone() {
                    let child = self.value_by_node[c];
                    for r in 0..R {
                        if child[r] > max[r] {
                            max[r] = child[r];
                            ties[r] = 1;
                        } else if child[r] == max[r] {
                            ties[r] += 1;
                        }
                    }
                }
                for c in children.clone() {
                    let child = self.value_by_node[c];
                    let child_weight = std::array::from_fn(|r| {
                        if child[r] == max[r] {
                            weight[r] / divisor / f64::from(ties[r])
                        } else {
                            0.0
                        }
                    });
                    self.attribute(m, c, child_weight, visit);
                }
            }
        }
    }
}

fn add<const R: usize>(sum: &mut [f64; R], value: [f64; R]) {
    for r in 0..R {
        sum[r] += value[r];
    }
}

fn overlap_divisor(overlap: f32) -> f64 {
    f64::from(overlap).max(1e-9)
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
    fn balanced_fold_takes_the_mean_over_ranks() {
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
        let slot_ns = [10i64, 20, 5];
        let mut contribution = [0.0f64; 3];
        let [balanced] = BalancedFold::<1>::default().fold(
            &m,
            0,
            1.0,
            &|slot| [slot_ns[slot] as f64],
            &mut |slot, [c]| contribution[slot] += c,
        );
        // Mean over the Max pair: 0.5·t0 + 0.5·t1 + 3·t2, attributed per leaf,
        // whereas node_time takes the Max (straggler) → max(t0,t1) + 3·t2.
        assert_eq!(balanced, 0.5 * 10.0 + 0.5 * 20.0 + 3.0 * 5.0); // 30.0
        assert_eq!(contribution, [5.0, 10.0, 15.0]);
        assert_eq!(node_time(&m, 0, &slot_ns), 20 + 3 * 5); // 35 (max branch)
    }

    #[test]
    fn balanced_fold_keeps_parallel_streams_at_their_wallclock_per_rung() {
        // Scale{2}[ Sum[ Parallel{overlap:0.5}[ Leaf0, Sum[Leaf1, Leaf2] ], Leaf3 ] ].
        let m = Manifest {
            slots: Vec::new(),
            nodes: vec![
                FlatCostNode::Scale {
                    n: 2,
                    children: 1..2,
                },
                FlatCostNode::Sum { children: 2..4 },
                FlatCostNode::Parallel {
                    overlap: 0.5,
                    children: 4..6,
                },
                FlatCostNode::Leaf(3),
                FlatCostNode::Leaf(0),
                FlatCostNode::Sum { children: 6..8 },
                FlatCostNode::Leaf(1),
                FlatCostNode::Leaf(2),
            ],
            node_labels: Vec::new(),
        };
        // Rung 0: stream A (slot 0) = 8 is slower than B (1+2 = 3).
        // Rung 1: A drops to 1, so B (3) becomes the critical stream.
        // Rung 2: an exact tie (3 vs 3) splits evenly.
        let leaf = [
            [8.0, 1.0, 3.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [4.0, 4.0, 4.0],
        ];
        let mut contribution = [[0.0f64; 3]; 4];
        let value =
            BalancedFold::<3>::default().fold(&m, 0, 1.0, &|slot| leaf[slot], &mut |slot, c| {
                for r in 0..3 {
                    contribution[slot][r] += c[r];
                }
            });
        assert_eq!(
            value,
            [
                2.0 * (8.0 / 0.5 + 4.0),
                2.0 * (3.0 / 0.5 + 4.0),
                2.0 * (3.0 / 0.5 + 4.0)
            ]
        );
        // The node equals node_time's wallclock: Parallel never reads as imbalance.
        let slot_ns = [8i64, 1, 2, 4];
        assert_eq!(node_time(&m, 0, &slot_ns) as f64, value[0]);
        assert_eq!(contribution[0], [32.0, 0.0, 6.0]);
        assert_eq!(contribution[1], [0.0, 4.0, 2.0]);
        assert_eq!(contribution[2], [0.0, 8.0, 4.0]);
        assert_eq!(contribution[3], [8.0, 8.0, 8.0]);
        for r in 0..3 {
            let total: f64 = contribution.iter().map(|c| c[r]).sum();
            assert_eq!(total, value[r], "rung {r} contributions reconcile");
        }
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
