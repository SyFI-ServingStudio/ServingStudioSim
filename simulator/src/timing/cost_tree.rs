//! CostTree — compile the cost-model *structure* once, separate from the
//! per-iter numbers. See `COST_TREE.md`.
//!
//! The structure of a cost query (which primitives, how they compose, the
//! homogeneous-layer repeat) is stable across iterations; only the leaf metrics
//! vary with batch shape. So we compile the structure once into a `CostTree`,
//! stream per-iter [`LeafMetrics`] into a flat `buf[slot]`, and aggregate the
//! flat nodes with a caller-owned scratch buffer. The same slot order backs the
//! per-worker `CostManifest` sidecar and the `slot_input` cost-log column.
//!
//! The structure is assembled as a recursive [`CostNode`] tree, then lowered by
//! [`CostTree::flatten`] to the doc's flat [`FlatCostNode`] array with contiguous
//! child ranges. [`CostTree::aggregate`] consumes that flat form in one reverse
//! pass with no allocation when the caller reuses `scratch`.
//!
//! Names live only here (compile time / the slot list), never on the hot path or
//! in log rows (INV-5). `Max`/`Scale` are defined for the full algebra even
//! though the dense vertical only emits `Leaf`/`Sum`/`Scale`.

use std::collections::{BTreeMap, VecDeque};
use std::fmt::Write;
use std::ops::Range;

use serde::{Deserialize, Deserializer, Serialize};

use crate::timing::slot_input::SlotInput;
use crate::timing::LeafMetrics;

/// A node of the recursive cost structure (the build-time form). Composites are
/// anonymous — the dotted leaf names in [`CostTree::slots`] carry identity.
#[derive(Clone, Debug, PartialEq)]
pub enum CostNode {
    /// A materialized leaf = one L1 primitive; the `usize` indexes
    /// [`CostTree::slots`] and the per-iter [`LeafMetrics`] buffer.
    Leaf(usize),
    /// `Σ children` (serial composition; time/flops/bytes all sum).
    Sum(Vec<CostNode>),
    /// Synchronized fan-out: wallclock `= max(children)`. The build API exposes
    /// no overlap coefficient: protocol v1 has only this pure-max semantic. The
    /// compatibility field lives exclusively in [`FlatCostNode::Max`].
    Max { children: Vec<CostNode> },
    /// Fold: `n ×` an identical child subtree — the homogeneous-layer repeat,
    /// evaluated once and scaled, never materialized `n` times (INV-3).
    Scale { n: u32, child: Box<CostNode> },
    /// Render-only wrapper: a composite identity line (worklet type + partition
    /// annotation, the model header, …) attached to its child for [`CostTree::describe`].
    /// Cost-transparent — [`CostTree::flatten`] unwraps it (the label never reaches
    /// the flat array or the hot-path aggregate, honoring INV-5).
    Labeled { label: String, child: Box<CostNode> },
}

/// Flat, topologically-laid-out form (the lowered product of [`CostTree::flatten`]).
/// Each composite references its *direct* children by a contiguous range into the
/// same `Vec<FlatCostNode>`; a parent's index always precedes its children's, so
/// aggregate is a single bottom-up (reverse) pass with no recursion or
/// allocation. Mirrors the flatten pass in `COST_TREE.md`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FlatCostNode {
    Leaf(usize),
    Sum {
        children: Range<usize>,
    },
    Max {
        /// Protocol-v1 wire compatibility only. [`CostTree::flatten`] always
        /// writes `1.0`, and deserialization rejects every other value. A
        /// measured overlap model requires a new protocol version.
        #[serde(deserialize_with = "deserialize_v1_overlap")]
        overlap: f32,
        children: Range<usize>,
    },
    Scale {
        n: u32,
        children: Range<usize>,
    },
}

/// Decode the protocol-v1 compatibility field without admitting a second
/// authoring semantic through manifests produced outside this crate.
fn deserialize_v1_overlap<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: Deserializer<'de>,
{
    let overlap = f32::deserialize(deserializer)?;
    if overlap == 1.0 {
        Ok(overlap)
    } else {
        Err(serde::de::Error::custom(
            "CostTree protocol v1 requires FlatCostNode::Max overlap to equal 1.0",
        ))
    }
}

/// Per-slot manifest entry: the dotted leaf name plus the kernel identity folded
/// in from the old `Describe` trait — `kind` is the kernel KIND tag and `config`
/// the one-line shape/dtype summary (`KernelConfig::describe_config`). Captured at
/// compile so [`CostTree::describe`] is the sole shape renderer.
///
/// `backends` is the leaf's ordered candidate backend list, the structured form of
/// what `config` renders inline. Its order is the index space of the `cost_log`
/// `slot_backend` column (`backends[slot_backend]` names the selected backend), so
/// an analyzer never parses the human-readable `config` string.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LeafDesc {
    pub name: String,
    pub kind: String,
    pub config: String,
    pub backends: Vec<String>,
    /// The leaf's `symbol -> value` legend: every named input in this leaf's
    /// `Dim` formulas (`{num_qo_heads: 64, attn_tp: 4, head_dim: 128, …}`), so a
    /// consumer can resolve the `config` expression to its concrete parts (the
    /// expression↔value toggle). Empty for comm / no-shape leaves. Skipped from
    /// JSON when empty so existing manifests are byte-compatible.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub symbols: BTreeMap<String, u32>,
}

/// Serializable description of a compiled [`CostTree`] — the per-worker
/// `cost_manifest/worker_<pool_tag>_<worker_id>.json` sidecar. Pairs the ordered
/// leaf [`slots`](Self::slots) (the position→name/kind/config map for the
/// parquet `slot_*` list columns) with the flattened aggregation
/// [`nodes`](Self::nodes), so a consumer reading a row from `cost_log/` can
/// re-run [`CostTree::aggregate`] over that row's `slot_time_ms` to reproduce
/// `total_time_ms`: the `Scale{n}` fold, `Sum`, and `Max` operators are all
/// present (slot names alone can't reconstruct the total). The flat `Max` node
/// retains its fixed protocol-v1 compatibility field on the wire.
///
/// [`node_labels`](Self::node_labels) recovers the composite identity that
/// [`CostTree::flatten`] drops: it is index-aligned to [`nodes`](Self::nodes) —
/// `node_labels[i]` is the [`CostNode::Labeled`] line (worklet kind + partition
/// annotation) that wrapped the node now at flat index `i`, or `None`. This is
/// what lets a downstream analyzer group slots into semantic subtrees (e.g.
/// "the attention composite") *structurally*, without parsing dotted slot names.
/// The label rides only this sidecar — never `FlatCostNode` or the hot-path
/// aggregate (INV-5: names stay off the hot path).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CostManifest {
    pub slots: Vec<LeafDesc>,
    pub nodes: Vec<FlatCostNode>,
    pub node_labels: Vec<Option<String>>,
}

/// The `cost_manifest/worker_<pool>_<id>.json` sidecar: one or more named
/// building-block **sections**, each a [`CostManifest`]. The iter-wise path writes
/// a single `iter` section (the whole fused iteration); the AFD layer-wise path
/// writes one section per cost group (`attn` on the attn side; `prologue` /
/// `pre_attn` / `post_attn` / `epilogue` on the ffn side), because those are
/// distinct compiled CostTrees with different slot sets. A `cost_log` row's
/// `section` field selects which section's `slots`/`nodes` interpret that row's
/// `slot_*` lists — so different-shaped sections coexist in one per-worker stream
/// (the `slot_time_ms` list is already variable-length per row).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CostManifestDoc {
    pub sections: Vec<CostManifestSection>,
}

/// One named section of a [`CostManifestDoc`]. `manifest` is `#[serde(flatten)]`,
/// so a section serializes as `{"section": "attn", "slots": [...], "nodes": [...],
/// "node_labels": [...]}` — the [`CostManifest`] fields sit alongside `section`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CostManifestSection {
    pub section: String,
    #[serde(flatten)]
    pub manifest: CostManifest,
}

impl CostManifestDoc {
    /// A single-section doc — the iter-wise form (`section = "iter"`), and the
    /// back-compat wrapper for any model exposing one CostTree.
    pub fn single(section: impl Into<String>, manifest: CostManifest) -> Self {
        Self {
            sections: vec![CostManifestSection {
                section: section.into(),
                manifest,
            }],
        }
    }

    /// An empty doc — a model with no compiled CostTree (cost_log disabled).
    pub fn empty() -> Self {
        Self {
            sections: Vec::new(),
        }
    }

    /// Append a named section (the AFD ffn side builds up `prologue` / `pre_attn`
    /// / `post_attn` / `epilogue`).
    pub fn push(&mut self, section: impl Into<String>, manifest: CostManifest) {
        self.sections.push(CostManifestSection {
            section: section.into(),
            manifest,
        });
    }
}

/// Compiled-once structure: the recursive node tree plus the ordered leaf slots
/// it indexes (slot index = compile traversal order, INV-2).
#[derive(Clone, Debug)]
pub struct CostTree {
    pub root: CostNode,
    pub slots: Vec<LeafDesc>,
}

/// Mints leaf slots in visit order while the per-layer `compile` methods assemble
/// the [`CostNode`] tree around them. One builder threads through the whole
/// compile walk so slot indices are assigned by the order leaves are visited
/// (INV-2).
#[derive(Default)]
pub struct CostTreeBuilder {
    slots: Vec<LeafDesc>,
}

impl CostTreeBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate the next slot for a materialized leaf and return its [`CostNode`].
    /// `kind`/`config` carry the kernel identity for the shape render (the old
    /// `Describe` leaf line); `backends` is the leaf's ordered candidate list (the
    /// index space of the `cost_log` `slot_backend` column).
    pub fn leaf(
        &mut self,
        name: impl Into<String>,
        kind: impl Into<String>,
        config: impl Into<String>,
        backends: Vec<String>,
    ) -> CostNode {
        self.leaf_with_symbols(name, kind, config, backends, BTreeMap::new())
    }

    /// [`Self::leaf`] plus the leaf's `symbol -> value` legend (from the kernel
    /// config's `Dim` fields, via `Probe::symbol_bindings`). Keys are the
    /// `&'static str` symbol names, owned here for the serializable [`LeafDesc`].
    pub fn leaf_with_symbols(
        &mut self,
        name: impl Into<String>,
        kind: impl Into<String>,
        config: impl Into<String>,
        backends: Vec<String>,
        symbols: BTreeMap<&'static str, u32>,
    ) -> CostNode {
        let slot = self.slots.len();
        self.slots.push(LeafDesc {
            name: name.into(),
            kind: kind.into(),
            config: config.into(),
            backends,
            symbols: symbols
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        });
        CostNode::Leaf(slot)
    }

    /// Close the build: pair the assembled `root` with the slots minted along the way.
    pub fn finish(self, root: CostNode) -> CostTree {
        CostTree {
            root,
            slots: self.slots,
        }
    }
}

/// Fills the per-iter leaf buffer in visit order, the eval-time counterpart of
/// [`CostTreeBuilder`]: where the builder mints slots as `compile` walks the
/// children, the evaluator writes them as `eval` walks the same children in the
/// same order (INV-2). Threading one `&mut Evaluator` down the eval recursion
/// replaces the parallel `buf` + `cursor` pair (mirrors `compile`'s `&mut
/// CostTreeBuilder`); `push` hides the slot-cursor bookkeeping.
pub struct Evaluator<'a> {
    buf: &'a mut [LeafMetrics],
    cursor: usize,
    /// `Some` only on the input-capturing path ([`Self::with_inputs`]); each leaf's
    /// input is pushed here in slot/visit order. `None` on the no-logger path, so
    /// [`Self::push`] never invokes its closure → no clone.
    inputs: Option<&'a mut Vec<SlotInput>>,
}

impl<'a> Evaluator<'a> {
    pub fn new(buf: &'a mut [LeafMetrics]) -> Self {
        Self {
            buf,
            cursor: 0,
            inputs: None,
        }
    }

    /// Input-capturing evaluator: like [`Self::new`] but also records each leaf's
    /// typed input into `inputs` (cleared first), aligned to the slot buffer, for
    /// the `cost_log` `slot_input` column.
    pub fn with_inputs(buf: &'a mut [LeafMetrics], inputs: &'a mut Vec<SlotInput>) -> Self {
        inputs.clear();
        Self {
            buf,
            cursor: 0,
            inputs: Some(inputs),
        }
    }

    /// Write the next leaf's metrics into its slot and advance. Slot index =
    /// visit order, so `eval` must push in the order `compile` minted slots.
    /// Every leaf supplies its typed input here; `make_input` is invoked **only**
    /// when recording, so the no-logger path (`inputs: None`) does not clone.
    pub fn push(&mut self, metrics: LeafMetrics, make_input: impl FnOnce() -> SlotInput) {
        self.buf[self.cursor] = metrics;
        if let Some(inputs) = self.inputs.as_deref_mut() {
            inputs.push(make_input());
        }
        self.cursor += 1;
    }

    /// Slots filled so far — used to assert the eval walk covered every slot.
    pub fn filled(&self) -> usize {
        self.cursor
    }
}

impl CostTree {
    /// Number of materialized leaf slots = the per-iter `buf` length. A folded
    /// (`Scale`) subtree is counted once, not `×n`.
    pub fn n_slots(&self) -> usize {
        self.slots.len()
    }

    /// Serializable manifest for the `cost_log` sidecar: the ordered slots plus
    /// the flattened aggregation nodes. Lets a consumer reproduce `total_time_ms`
    /// from a row's per-slot `slot_time_ms` by re-running [`Self::aggregate`].
    pub fn manifest(&self) -> CostManifest {
        let (nodes, node_labels) = self.flatten_labeled();
        CostManifest {
            slots: self.slots.clone(),
            nodes,
            node_labels,
        }
    }

    /// Lower the recursive tree to the flat [`FlatCostNode`] array (doc §4). BFS
    /// layout: each composite reserves a contiguous block for its direct children,
    /// so `children` is a valid `Range` and every parent index precedes its
    /// children's — making the aggregate a reverse linear pass. Labels are dropped
    /// (this is the form the hot-path aggregate consumes — no `String`, INV-5).
    pub fn flatten(&self) -> Vec<FlatCostNode> {
        self.flatten_labeled().0
    }

    /// [`Self::flatten`] plus the composite labels it drops, recovered for the
    /// off-hot-path manifest. One BFS produces both so the label vec can never
    /// drift out of index-alignment with the flat `nodes`: `labels[i]` is the
    /// [`CostNode::Labeled`] line wrapping the node now at flat index `i`, else
    /// `None`. Only [`Self::manifest`] calls this; the runtime aggregate takes the
    /// labelless `nodes` from [`Self::flatten`].
    fn flatten_labeled(&self) -> (Vec<FlatCostNode>, Vec<Option<String>>) {
        let mut out: Vec<Option<FlatCostNode>> = vec![None]; // root at index 0
        let mut labels: Vec<Option<String>> = vec![None];
        let mut queue: VecDeque<(&CostNode, usize)> = VecDeque::from([(&self.root, 0usize)]);
        while let Some((node, idx)) = queue.pop_front() {
            let op = match node {
                CostNode::Leaf(slot) => FlatCostNode::Leaf(*slot),
                CostNode::Sum(children) => {
                    let range = Self::reserve(&mut out, &mut labels, &mut queue, children);
                    FlatCostNode::Sum { children: range }
                }
                CostNode::Max { children } => {
                    let range = Self::reserve(&mut out, &mut labels, &mut queue, children);
                    FlatCostNode::Max {
                        // Keep the v1 manifest shape stable while making any
                        // other value unrepresentable through the build API.
                        overlap: 1.0,
                        children: range,
                    }
                }
                CostNode::Scale { n, child } => {
                    let range = Self::reserve(
                        &mut out,
                        &mut labels,
                        &mut queue,
                        std::slice::from_ref(child),
                    );
                    FlatCostNode::Scale {
                        n: *n,
                        children: range,
                    }
                }
                // Render-only for the flat tree: splice the child into this slot.
                // Keep the label on this index for the manifest; outermost wins if
                // labels ever nest (set only when the index is still empty).
                CostNode::Labeled { label, child } => {
                    if labels[idx].is_none() {
                        labels[idx] = Some(label.clone());
                    }
                    queue.push_back((child, idx));
                    continue;
                }
            };
            out[idx] = Some(op);
        }
        let nodes = out
            .into_iter()
            .map(|o| o.expect("every reserved flat node is filled"))
            .collect();
        (nodes, labels)
    }

    /// Reserve a contiguous block for `children` (growing `labels` in lockstep so
    /// it stays index-aligned to `out`) and enqueue each child to be filled.
    fn reserve<'a>(
        out: &mut Vec<Option<FlatCostNode>>,
        labels: &mut Vec<Option<String>>,
        queue: &mut VecDeque<(&'a CostNode, usize)>,
        children: &'a [CostNode],
    ) -> Range<usize> {
        let start = out.len();
        out.extend(children.iter().map(|_| None));
        labels.resize(out.len(), None);
        for (i, child) in children.iter().enumerate() {
            queue.push_back((child, start + i));
        }
        start..start + children.len()
    }

    /// Aggregate a per-iter leaf buffer up the flattened tree to one
    /// [`LeafMetrics`] (doc §6, INV-4/6). `buf[slot]` holds each leaf's metrics +
    /// coverage for this iteration; `flat` is [`Self::flatten`]'s output.
    /// Composites combine:
    ///   - `Sum`  — field-wise sum of children (coverage flags unioned);
    ///   - `Scale{n}` — child subtree × `n` (the homogeneous-layer fold);
    ///   - `Max` — `time = max(child.time)`, other fields summed
    ///     (flops/bytes/energy always add — work doesn't overlap away, INV-4).
    /// Coverage flags always OR up the tree, so a warning anywhere surfaces at
    /// the root.
    ///
    /// Single reverse pass: BFS layout puts every parent before its children, so
    /// iterating high→low index has each child's subtree result ready when its
    /// parent is reached. `scratch[i]` holds node `i`'s rolled-up metrics; the
    /// root (index 0) is the answer.
    ///
    /// `buf` holds the per-leaf metrics (`buf[slot]`); `scratch` is a caller-owned
    /// node-metrics buffer reused across calls. This runs once per worker iteration
    /// (millions of times), so the caller threads in a persistent `scratch` rather
    /// than allocating a fresh `Vec` each call: we `clear` + `resize(flat.len())`
    /// in place, keeping the capacity. Steady-state aggregation is allocation-free.
    pub fn aggregate(
        flat: &[FlatCostNode],
        buf: &[LeafMetrics],
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        if flat.is_empty() {
            return LeafMetrics::ZERO;
        }
        scratch.clear();
        scratch.resize(flat.len(), LeafMetrics::ZERO);
        for i in (0..flat.len()).rev() {
            scratch[i] = match &flat[i] {
                FlatCostNode::Leaf(slot) => buf[*slot],
                FlatCostNode::Sum { children } => {
                    let mut acc = LeafMetrics::ZERO;
                    for c in children.clone() {
                        acc.add(scratch[c]);
                    }
                    acc
                }
                FlatCostNode::Scale { n, children } => {
                    let mut acc = LeafMetrics::ZERO;
                    for c in children.clone() {
                        acc.add(scratch[c]);
                    }
                    acc.scale(*n as f32);
                    acc
                }
                FlatCostNode::Max { children, .. } => {
                    let mut acc = LeafMetrics::ZERO;
                    let mut max_time = 0.0f32;
                    for c in children.clone() {
                        let cm = scratch[c];
                        max_time = max_time.max(cm.m.time_ms);
                        acc.add(cm);
                    }
                    acc.m.time_ms = max_time;
                    acc
                }
            };
        }
        scratch[0]
    }

    /// Indented render of the compiled structure for inspection ("print after
    /// build"). Leaves show `Leaf#<slot> <name>`; composites show their op.
    pub fn describe(&self) -> String {
        let mut out = String::new();
        self.write_node(&self.root, 0, &mut out);
        out
    }

    fn write_node(&self, node: &CostNode, depth: usize, out: &mut String) {
        let ind = "│  ".repeat(depth);
        match node {
            CostNode::Leaf(slot) => {
                let d = &self.slots[*slot];
                writeln!(out, "{ind}Leaf#{slot} {} ({}) {}", d.name, d.kind, d.config).unwrap()
            }
            CostNode::Sum(children) => {
                writeln!(out, "{ind}Sum").unwrap();
                for c in children {
                    self.write_node(c, depth + 1, out);
                }
            }
            CostNode::Max { children } => {
                writeln!(out, "{ind}Max").unwrap();
                for c in children {
                    self.write_node(c, depth + 1, out);
                }
            }
            CostNode::Scale { n, child } => {
                writeln!(out, "{ind}Scale{{n={n}}}").unwrap();
                self.write_node(child, depth + 1, out);
            }
            CostNode::Labeled { label, child } => {
                writeln!(out, "{ind}{label}").unwrap();
                self.write_node(child, depth + 1, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build `Sum( leaf a, Scale{3}( Sum(leaf b, leaf c) ), leaf d )` by hand —
    /// the dense shape in miniature (a fold wrapping a 2-leaf subtree).
    fn sample() -> CostTree {
        let mut b = CostTreeBuilder::new();
        let a = b.leaf("a", "ka", "x=1", vec![]);
        let layer = CostNode::Scale {
            n: 3,
            child: Box::new(CostNode::Sum(vec![
                b.leaf("b", "kb", "x=2", vec![]),
                b.leaf("c", "kc", "x=3", vec![]),
            ])),
        };
        let d = b.leaf("d", "kd", "x=4", vec![]);
        b.finish(CostNode::Sum(vec![a, layer, d]))
    }

    #[test]
    fn leaf_desc_carries_symbols_and_skips_when_empty() {
        // `leaf_with_symbols` captures the per-leaf legend; it serializes under a
        // `symbols` key and is omitted entirely when empty (manifest byte-compat).
        let mut b = CostTreeBuilder::new();
        let mut syms = BTreeMap::new();
        syms.insert("attn_tp", 4u32);
        syms.insert("num_qo_heads", 64u32);
        b.leaf_with_symbols("m.qkv", "single_gemm", "n=…", vec![], syms);
        b.leaf("m.comm", "p2p_inter", "", vec![]); // no Dim fields → empty
        let tree = b.finish(CostNode::Sum(vec![CostNode::Leaf(0), CostNode::Leaf(1)]));

        assert_eq!(tree.slots[0].symbols.get("attn_tp"), Some(&4));
        assert_eq!(tree.slots[0].symbols.get("num_qo_heads"), Some(&64));
        assert!(tree.slots[1].symbols.is_empty());

        let with = serde_json::to_string(&tree.slots[0]).unwrap();
        assert!(with.contains(r#""symbols""#) && with.contains(r#""attn_tp":4"#));
        let empty = serde_json::to_string(&tree.slots[1]).unwrap();
        assert!(!empty.contains("symbols")); // skip_serializing_if empty
    }

    #[test]
    fn slots_minted_in_visit_order() {
        let t = sample();
        // a, b, c, d — folded subtree leaves counted once, not ×3.
        assert_eq!(t.n_slots(), 4);
        let names: Vec<&str> = t.slots.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["a", "b", "c", "d"]);
    }

    #[test]
    fn describe_renders_fold_and_leaves() {
        let expected = "\
Sum
│  Leaf#0 a (ka) x=1
│  Scale{n=3}
│  │  Sum
│  │  │  Leaf#1 b (kb) x=2
│  │  │  Leaf#2 c (kc) x=3
│  Leaf#3 d (kd) x=4
";
        assert_eq!(sample().describe(), expected);
    }

    #[test]
    fn describe_renders_labeled_header_then_child() {
        // A `Labeled` wrapper prints its identity line (worklet header / partition
        // annotation) above its child — the "no less than describe" guard: the
        // leaf still carries its kernel kind + config.
        let mut b = CostTreeBuilder::new();
        let leaf = b.leaf("w.norm", "rms_norm", "hidden=4096, dtype=Bf16", vec![]);
        let tree = b.finish(CostNode::Labeled {
            label: "w (PreAttnLocalWorklet) [local (1 GPU); qkv n=6144, k=4096]".to_string(),
            child: Box::new(CostNode::Sum(vec![leaf])),
        });
        let expected = "\
w (PreAttnLocalWorklet) [local (1 GPU); qkv n=6144, k=4096]
│  Sum
│  │  Leaf#0 w.norm (rms_norm) hidden=4096, dtype=Bf16
";
        assert_eq!(tree.describe(), expected);
        // Labeled is cost-transparent: flatten drops it, leaving Sum→Leaf only.
        let flat = tree.flatten();
        assert!(matches!(flat[0], FlatCostNode::Sum { .. }));
        assert_eq!(
            flat.iter()
                .filter(|n| matches!(n, FlatCostNode::Leaf(_)))
                .count(),
            1
        );
        // …but the manifest recovers the label at the index the Labeled spliced
        // into (the Sum took flat index 0), index-aligned to `nodes`.
        let manifest = tree.manifest();
        assert_eq!(manifest.node_labels.len(), manifest.nodes.len());
        assert_eq!(
            manifest.node_labels[0].as_deref(),
            Some("w (PreAttnLocalWorklet) [local (1 GPU); qkv n=6144, k=4096]")
        );
    }

    #[test]
    fn manifest_labels_tag_composite_subtrees_not_leaves() {
        // Mirror the real dense tree: a labeled "attn" composite wrapping its
        // leaves, so the analyzer can find the attention subtree structurally.
        let mut b = CostTreeBuilder::new();
        let pre = CostNode::Labeled {
            label: "m.pre_attn (PreAttnLocalWorklet)".to_string(),
            child: Box::new(CostNode::Sum(vec![b.leaf(
                "m.pre_attn.norm",
                "rms_norm",
                "",
                vec![],
            )])),
        };
        let attn = CostNode::Labeled {
            label: "m.attn (AttnLocalWorklet)".to_string(),
            child: Box::new(CostNode::Sum(vec![
                b.leaf("m.attn.prefill", "flashinfer_attn_prefill", "", vec![]),
                b.leaf("m.attn.decode", "flashinfer_attn_decode", "", vec![]),
            ])),
        };
        let tree = b.finish(CostNode::Sum(vec![pre, attn]));
        let m = tree.manifest();
        assert_eq!(m.node_labels.len(), m.nodes.len());
        // The labels sit on the composite Sum nodes, never on leaves.
        for (node, label) in m.nodes.iter().zip(&m.node_labels) {
            if matches!(node, FlatCostNode::Leaf(_)) {
                assert!(label.is_none(), "leaf nodes carry no composite label");
            }
        }
        // Exactly the two worklet labels are present, somewhere on Sum nodes.
        let labels: Vec<&str> = m.node_labels.iter().flatten().map(String::as_str).collect();
        assert!(labels.contains(&"m.pre_attn (PreAttnLocalWorklet)"));
        assert!(labels.contains(&"m.attn (AttnLocalWorklet)"));
        assert_eq!(labels.len(), 2);
    }

    use crate::timing::{CoverageFlags, Metrics4};

    fn leaf(time_ms: f32, flops: f32, bytes: f32) -> LeafMetrics {
        LeafMetrics {
            m: Metrics4 {
                time_ms,
                flops,
                bytes,
                energy_j: 0.0,
            },
            coverage: CoverageFlags::EMPTY,
            backend_index: LeafMetrics::NO_BACKEND,
        }
    }

    #[test]
    fn manifest_round_trips_and_reproduces_total() {
        // The sidecar must let a consumer reproduce the aggregate from the per-slot
        // buffer alone: serialize → deserialize the manifest, then re-run
        // `aggregate` over the same slot buffer and check it matches the direct
        // total (the `Scale{3}` fold is what slot names alone can't reconstruct).
        let tree = sample();
        let json = serde_json::to_string(&tree.manifest()).unwrap();
        let back: CostManifest = serde_json::from_str(&json).unwrap();
        let buf = [
            leaf(1.0, 0.0, 0.0), // a
            leaf(2.0, 0.0, 0.0), // b
            leaf(3.0, 0.0, 0.0), // c
            leaf(4.0, 0.0, 0.0), // d
        ];
        let mut scratch = Vec::new();
        let from_manifest = CostTree::aggregate(&back.nodes, &buf, &mut scratch)
            .m
            .time_ms;
        let direct = CostTree::aggregate(&tree.flatten(), &buf, &mut scratch)
            .m
            .time_ms;
        assert_eq!(from_manifest, direct);
        assert_eq!(from_manifest, 1.0 + 3.0 * (2.0 + 3.0) + 4.0);
        // The manifest carries the fold + slot names (reproducibility, not just labels).
        assert!(back
            .nodes
            .iter()
            .any(|n| matches!(n, FlatCostNode::Scale { n: 3, .. })));
        assert_eq!(back.slots.len(), 4);
    }

    #[test]
    fn aggregate_folds_scale_and_sums_siblings() {
        // sample = Sum( a, Scale{3}(Sum(b, c)), d ); slots [a, b, c, d].
        let flat = sample().flatten();
        let buf = [
            leaf(1.0, 10.0, 100.0), // a
            leaf(2.0, 20.0, 200.0), // b
            leaf(3.0, 30.0, 300.0), // c
            leaf(4.0, 40.0, 400.0), // d
        ];
        let mut scratch = Vec::new();
        let total = CostTree::aggregate(&flat, &buf, &mut scratch).m;
        // time/flops/bytes = a + 3·(b+c) + d (the fold multiplies the layer).
        assert_eq!(total.time_ms, 1.0 + 3.0 * (2.0 + 3.0) + 4.0);
        assert_eq!(total.flops, 10.0 + 3.0 * (20.0 + 30.0) + 40.0);
        assert_eq!(total.bytes, 100.0 + 3.0 * (200.0 + 300.0) + 400.0);
    }

    #[test]
    fn aggregate_max_takes_slowest_time_but_sums_work() {
        // Build Max(leaf x, leaf y): time = max(x,y), flops/bytes sum. Its flat
        // protocol-v1 representation retains the fixed `overlap=1` field.
        let mut b = CostTreeBuilder::new();
        let root = CostNode::Max {
            children: vec![b.leaf("x", "kx", "", vec![]), b.leaf("y", "ky", "", vec![])],
        };
        let tree = b.finish(root);
        assert!(tree.describe().starts_with("Max\n"));
        let flat = tree.flatten();
        assert!(matches!(flat[0], FlatCostNode::Max { overlap: 1.0, .. }));
        let wire_json = serde_json::to_string(&flat[0]).unwrap();
        assert!(wire_json.contains(r#""overlap":1.0"#));
        assert_eq!(
            serde_json::from_str::<FlatCostNode>(&wire_json).unwrap(),
            flat[0]
        );
        let buf = [leaf(4.0, 10.0, 100.0), leaf(6.0, 20.0, 200.0)];
        let mut scratch = Vec::new();
        let total = CostTree::aggregate(&flat, &buf, &mut scratch).m;
        assert_eq!(total.time_ms, 6.0); // synchronized fan-out waits for the slowest branch
        assert_eq!(total.flops, 30.0); // work still sums (INV-4)
        assert_eq!(total.bytes, 300.0);
    }

    #[test]
    fn wire_decoder_rejects_unversioned_overlap_semantics() {
        let json = r#"{"Max":{"overlap":1.01,"children":{"start":1,"end":3}}}"#;
        let error = serde_json::from_str::<FlatCostNode>(json).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("FlatCostNode::Max overlap to equal 1.0"),
            "unexpected decode error: {error}"
        );
    }

    #[test]
    fn aggregate_unions_coverage_flags_up_the_tree() {
        // One extrapolated leaf deep in the folded layer must surface at the root.
        let flat = sample().flatten();
        let mut buf = [
            leaf(1.0, 0.0, 0.0),
            leaf(2.0, 0.0, 0.0),
            leaf(3.0, 0.0, 0.0),
            leaf(4.0, 0.0, 0.0),
        ];
        buf[2].coverage = CoverageFlags::EXTRAPOLATED; // leaf c, inside Scale{3}
        let mut scratch = Vec::new();
        let total = CostTree::aggregate(&flat, &buf, &mut scratch);
        assert!(total.coverage.contains(CoverageFlags::EXTRAPOLATED));
        // A clean buffer leaves the root flag empty.
        buf[2].coverage = CoverageFlags::EMPTY;
        assert!(CostTree::aggregate(&flat, &buf, &mut scratch)
            .coverage
            .is_empty());
    }

    #[test]
    fn flatten_child_ranges_are_contiguous_and_forward() {
        let nodes = sample().flatten();
        // root Sum has 3 direct children in one contiguous block, all after root.
        let FlatCostNode::Sum { children } = &nodes[0] else {
            panic!("root must be Sum, got {:?}", nodes[0]);
        };
        assert_eq!(children.len(), 3);
        assert!(children.start > 0, "children come after the parent");
        // Exactly one Scale{n=3}; one leaf per slot.
        let scales = nodes
            .iter()
            .filter(|n| matches!(n, FlatCostNode::Scale { n: 3, .. }))
            .count();
        assert_eq!(scales, 1);
        let leaves = nodes
            .iter()
            .filter(|n| matches!(n, FlatCostNode::Leaf(_)))
            .count();
        assert_eq!(leaves, 4);
        // Every child range is in-bounds and strictly after its parent (BFS).
        for (i, node) in nodes.iter().enumerate() {
            let range = match node {
                FlatCostNode::Sum { children }
                | FlatCostNode::Max { children, .. }
                | FlatCostNode::Scale { children, .. } => children.clone(),
                FlatCostNode::Leaf(_) => continue,
            };
            assert!(
                range.start > i,
                "parent {i} precedes its children {range:?}"
            );
            assert!(range.end <= nodes.len(), "child range {range:?} in bounds");
        }
    }
}
