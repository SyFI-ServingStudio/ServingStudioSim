//! CostTree (milestone 1) — compile the cost-model *structure* once, separate
//! from the per-iter numbers. See `docs/cost_tree.md` §2/§5.
//!
//! The structure of a cost query (which primitives, how they compose, the
//! homogeneous-layer repeat) is stable across iterations; only the leaf metrics
//! vary with batch shape. So we compile the structure once into a `CostTree` and
//! (later milestones) stream per-iter `Metrics4` into a flat `buf[slot]`.
//!
//! This milestone builds the structure and prints it — no per-iter eval, no
//! clock wiring, no logging. The structure is assembled as a recursive
//! [`CostNode`] tree (each layer's hand-written `compile` mirrors its `lookup`),
//! then lowered by [`CostTree::flatten`] to the doc's flat [`FlatCostNode`] array
//! with contiguous child ranges (the form the future zero-alloc aggregate walk
//! consumes).
//!
//! Names live only here (compile time / the slot list), never on the hot path or
//! in log rows (INV-5). `Max`/`Scale` are defined for the full algebra even
//! though the dense vertical only emits `Leaf`/`Sum`/`Scale`.

use std::collections::VecDeque;
use std::fmt::Write;
use std::ops::Range;

use crate::timing::LeafMetrics;

/// A node of the recursive cost structure (the build-time form). Composites are
/// anonymous — the dotted leaf names in [`CostTree::slots`] carry identity.
#[derive(Clone, Debug, PartialEq)]
pub enum CostNode {
    /// A materialized leaf = one L1 primitive; the `usize` indexes
    /// [`CostTree::slots`] (and, later, the per-iter `Metrics4` buffer).
    Leaf(usize),
    /// `Σ children` (serial composition; time/flops/bytes all sum).
    Sum(Vec<CostNode>),
    /// Fan-out: wallclock `= max(children)/overlap`. Unused by the dense vertical
    /// (no collective); present for future HP/EP fan-out (INV-3).
    Max { overlap: f32, children: Vec<CostNode> },
    /// Fold: `n ×` an identical child subtree — the homogeneous-layer repeat,
    /// evaluated once and scaled, never materialized `n` times (INV-3).
    Scale { n: u32, child: Box<CostNode> },
}

/// Flat, topologically-laid-out form (the lowered product of [`CostTree::flatten`]).
/// Each composite references its *direct* children by a contiguous range into the
/// same `Vec<FlatCostNode>`; a parent's index always precedes its children's, so
/// the future aggregate is a single bottom-up (reverse) pass with no recursion or
/// allocation. Mirrors `docs/cost_tree.md` §4.
#[derive(Clone, Debug, PartialEq)]
pub enum FlatCostNode {
    Leaf(usize),
    Sum { children: Range<usize> },
    Max { overlap: f32, children: Range<usize> },
    Scale { n: u32, children: Range<usize> },
}

/// Per-slot manifest entry. Milestone 1 carries the dotted name only; the leaf's
/// kernel `kind` and its fitted-kernel binding (for per-iter eval) land with the
/// eval/logging milestones.
#[derive(Clone, Debug, PartialEq)]
pub struct LeafDesc {
    pub name: String,
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
    pub fn leaf(&mut self, name: impl Into<String>) -> CostNode {
        let slot = self.slots.len();
        self.slots.push(LeafDesc { name: name.into() });
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

impl CostTree {
    /// Number of materialized leaf slots = the per-iter `buf` length. A folded
    /// (`Scale`) subtree is counted once, not `×n`.
    pub fn n_slots(&self) -> usize {
        self.slots.len()
    }

    /// Lower the recursive tree to the flat [`FlatCostNode`] array (doc §4). BFS
    /// layout: each composite reserves a contiguous block for its direct children,
    /// so `children` is a valid `Range` and every parent index precedes its
    /// children's — making the future aggregate a reverse linear pass.
    pub fn flatten(&self) -> Vec<FlatCostNode> {
        let mut out: Vec<Option<FlatCostNode>> = vec![None]; // root at index 0
        let mut queue: VecDeque<(&CostNode, usize)> = VecDeque::from([(&self.root, 0usize)]);
        while let Some((node, idx)) = queue.pop_front() {
            let op = match node {
                CostNode::Leaf(slot) => FlatCostNode::Leaf(*slot),
                CostNode::Sum(children) => {
                    let range = Self::reserve(&mut out, &mut queue, children);
                    FlatCostNode::Sum { children: range }
                }
                CostNode::Max { overlap, children } => {
                    let range = Self::reserve(&mut out, &mut queue, children);
                    FlatCostNode::Max {
                        overlap: *overlap,
                        children: range,
                    }
                }
                CostNode::Scale { n, child } => {
                    let range = Self::reserve(&mut out, &mut queue, std::slice::from_ref(child));
                    FlatCostNode::Scale {
                        n: *n,
                        children: range,
                    }
                }
            };
            out[idx] = Some(op);
        }
        out.into_iter()
            .map(|o| o.expect("every reserved flat node is filled"))
            .collect()
    }

    /// Reserve a contiguous block for `children` and enqueue each to be filled.
    fn reserve<'a>(
        out: &mut Vec<Option<FlatCostNode>>,
        queue: &mut VecDeque<(&'a CostNode, usize)>,
        children: &'a [CostNode],
    ) -> Range<usize> {
        let start = out.len();
        out.extend(children.iter().map(|_| None));
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
    ///   - `Max{overlap}` — `time = max(child.time)/overlap`, other fields summed
    ///     (flops/bytes/energy always add — work doesn't overlap away, INV-4).
    /// Coverage flags always OR up the tree, so a warning anywhere surfaces at
    /// the root.
    ///
    /// Single reverse pass: BFS layout puts every parent before its children, so
    /// iterating high→low index has each child's subtree result ready when its
    /// parent is reached. `scratch[i]` holds node `i`'s rolled-up metrics; the
    /// root (index 0) is the answer.
    pub fn aggregate(flat: &[FlatCostNode], buf: &[LeafMetrics]) -> LeafMetrics {
        let mut scratch = vec![LeafMetrics::ZERO; flat.len()];
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
                FlatCostNode::Max { overlap, children } => {
                    let mut acc = LeafMetrics::ZERO;
                    let mut max_time = 0.0f32;
                    for c in children.clone() {
                        let cm = scratch[c];
                        max_time = max_time.max(cm.m.time_ms);
                        acc.add(cm);
                    }
                    acc.m.time_ms = max_time / overlap;
                    acc
                }
            };
        }
        scratch.into_iter().next().unwrap_or(LeafMetrics::ZERO)
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
                writeln!(out, "{ind}Leaf#{slot} {}", self.slots[*slot].name).unwrap()
            }
            CostNode::Sum(children) => {
                writeln!(out, "{ind}Sum").unwrap();
                for c in children {
                    self.write_node(c, depth + 1, out);
                }
            }
            CostNode::Max { overlap, children } => {
                writeln!(out, "{ind}Max{{overlap={overlap}}}").unwrap();
                for c in children {
                    self.write_node(c, depth + 1, out);
                }
            }
            CostNode::Scale { n, child } => {
                writeln!(out, "{ind}Scale{{n={n}}}").unwrap();
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
        let a = b.leaf("a");
        let layer = CostNode::Scale {
            n: 3,
            child: Box::new(CostNode::Sum(vec![b.leaf("b"), b.leaf("c")])),
        };
        let d = b.leaf("d");
        b.finish(CostNode::Sum(vec![a, layer, d]))
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
│  Leaf#0 a
│  Scale{n=3}
│  │  Sum
│  │  │  Leaf#1 b
│  │  │  Leaf#2 c
│  Leaf#3 d
";
        assert_eq!(sample().describe(), expected);
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
        }
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
        let total = CostTree::aggregate(&flat, &buf).m;
        // time/flops/bytes = a + 3·(b+c) + d (the fold multiplies the layer).
        assert_eq!(total.time_ms, 1.0 + 3.0 * (2.0 + 3.0) + 4.0);
        assert_eq!(total.flops, 10.0 + 3.0 * (20.0 + 30.0) + 40.0);
        assert_eq!(total.bytes, 100.0 + 3.0 * (200.0 + 300.0) + 400.0);
    }

    #[test]
    fn aggregate_max_takes_overlapped_time_but_sums_work() {
        // Max{overlap=2}( leaf x, leaf y ): time = max(x,y)/2, flops/bytes sum.
        let mut b = CostTreeBuilder::new();
        let root = CostNode::Max {
            overlap: 2.0,
            children: vec![b.leaf("x"), b.leaf("y")],
        };
        let flat = b.finish(root).flatten();
        let buf = [leaf(4.0, 10.0, 100.0), leaf(6.0, 20.0, 200.0)];
        let total = CostTree::aggregate(&flat, &buf).m;
        assert_eq!(total.time_ms, 6.0 / 2.0); // max(4, 6)/overlap
        assert_eq!(total.flops, 30.0); // work still sums (INV-4)
        assert_eq!(total.bytes, 300.0);
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
        let total = CostTree::aggregate(&flat, &buf);
        assert!(total.coverage.contains(CoverageFlags::EXTRAPOLATED));
        // A clean buffer leaves the root flag empty.
        buf[2].coverage = CoverageFlags::EMPTY;
        assert!(CostTree::aggregate(&flat, &buf).coverage.is_empty());
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
            assert!(range.start > i, "parent {i} precedes its children {range:?}");
            assert!(range.end <= nodes.len(), "child range {range:?} in bounds");
        }
    }
}
