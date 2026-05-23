//! L2 (Operation) — names one or more L1 kernels into an op with `lookup` /
//! `dry_run_init` / `describe` entry points. See `docs/detailed_design/L2/`.
//!
//! This module holds the generic single-kernel wrapper `Op<K>` (L2 design §2.2).
//! Atomic ops (qkv / o_proj / gate_up / down / lm_head / rms_norm …) are all
//! `Op<SomeKernel>` and so occupy *no file* — L4 wiring instantiates them with
//! `Op::new(name, kernel)`. Only compound ops (multi-kernel + custom cost math,
//! e.g. attention / moe) get their own files under `op/<family>/`.

pub mod attention;
pub mod comm;
pub mod moe;
pub mod ssm;

use std::sync::Arc;

use crate::common::time::Time;
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Describe, DryRun, JitPlan, LeafMetrics, LookupResult,
    PerfApiBridge, Probe,
};

/// Generic single-kernel atomic op: names an L1 kernel and forwards its lookup.
/// `name` is the owned dotted path injected at the L4/L3 wiring point; the same
/// path is given to the kernel's `init`, so the op node and its one child share
/// it (L2 design §2.3).
pub struct Op<K> {
    pub name: String,
    pub kernel: Arc<K>,
}

impl<K: Probe> Op<K> {
    pub fn new(name: String, kernel: Arc<K>) -> Self {
        Self { name, kernel }
    }

    /// Wrap the kernel result as this op's single-child breakdown. A one-part
    /// `LookupResult::sum` is exactly the design §2.2 passthrough: time / flops /
    /// bytes / energy equal the child, warnings carry up, breakdown = `[child]`.
    pub fn lookup(&self, input: &K::Input) -> LookupResult {
        LookupResult::sum(self.name.clone(), vec![self.kernel.lookup(input)])
    }

    /// Hot-path time-only forward: an atomic op adds no time of its own, so skip
    /// the wrap and return the kernel's `lookup_time` directly. Stays consistent
    /// with `lookup().time` (sum-of-one == the child's time).
    pub fn lookup_time(&self, input: &K::Input) -> Time {
        self.kernel.lookup_time(input)
    }

    /// CostTree compile (M1): an atomic op is one leaf, named by its dotted path.
    /// The leaf's per-iter metrics + kernel `kind` bind in the eval milestone; the
    /// structure here needs only the name (mirrors the single-child `lookup`).
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        builder.leaf(self.name.clone())
    }

    /// CostTree eval: write this op's one leaf into `buf[*cursor]` and advance the
    /// cursor — the inverse of `compile`'s single `leaf()`. Walking `eval` in the
    /// same child order `compile` minted slots keeps `cursor` aligned with the
    /// slot index (INV-2). The leaf metric is the kernel's best-of-N `Metrics4`.
    pub fn eval(&self, input: &K::Input, buf: &mut [LeafMetrics], cursor: &mut usize) {
        buf[*cursor] = self.kernel.lookup_metrics(input);
        *cursor += 1;
    }
}

impl<K: DryRun> Op<K> {
    /// Build-time sibling of `new` (no `self`, constructs nothing): wrap the
    /// kernel's dry-run plan as this op's single-child `JitPlan`.
    pub fn dry_run_init(
        name: String,
        cfg: &K::Config,
        bridge: &PerfApiBridge,
    ) -> Result<JitPlan, BuildError> {
        let inner = K::dry_run(&name, cfg, bridge)?;
        Ok(JitPlan::sum(name, vec![inner]))
    }
}

impl<K: Describe> Describe for Op<K> {
    /// Op header at `depth`, then delegate to the kernel at `depth + 1`.
    fn describe(&self, depth: usize, out: &mut String) {
        use std::fmt::Write;
        writeln!(out, "{}{}", "│  ".repeat(depth), self.name).unwrap();
        self.kernel.describe(depth + 1, out);
    }
}

#[cfg(test)]
mod tests {
    use super::Op;
    use crate::common::time::Time;
    use crate::timing::{
        CostNode, CostTree, CostTreeBuilder, Describe, LeafMetrics, LookupResult, Probe,
    };
    use std::sync::Arc;

    /// Mock `Probe` + `Describe` standing in for an L1 kernel, so the op-wrapping
    /// logic is tested without a `PerfApiBridge` / Python (L2 design §15).
    struct FakeKernel {
        name: &'static str,
        time_ms: f64,
        flops: u64,
        bytes: u64,
    }

    impl Probe for FakeKernel {
        type Input = ();
        fn lookup(&self, _input: &()) -> LookupResult {
            LookupResult::leaf(
                self.name,
                Time::from_ms(self.time_ms),
                self.flops,
                self.bytes,
                0.0,
                Vec::new(),
            )
        }
    }

    impl Describe for FakeKernel {
        fn describe(&self, depth: usize, out: &mut String) {
            use std::fmt::Write;
            writeln!(out, "{}{} (FakeKernel)", "│  ".repeat(depth), self.name).unwrap();
        }
    }

    fn fake_op() -> Op<FakeKernel> {
        Op::new(
            "model.attn.o_proj".to_string(),
            Arc::new(FakeKernel {
                name: "o_proj_kernel",
                time_ms: 2.5,
                flops: 100,
                bytes: 200,
            }),
        )
    }

    #[test]
    fn op_lookup_wraps_kernel_as_single_child() {
        let op = fake_op();
        let result = op.lookup(&());

        // Parent node carries the op name; the kernel result is its one child.
        assert_eq!(&*result.name, "model.attn.o_proj");
        assert_eq!(result.breakdown.len(), 1);
        assert_eq!(&*result.breakdown[0].name, "o_proj_kernel");

        // Single-child sum passes the child's metrics straight through.
        assert_eq!(result.time.as_ms(), 2.5);
        assert_eq!(result.flops, 100);
        assert_eq!(result.bytes, 200);
        assert_eq!(result.time, result.breakdown[0].time);
    }

    #[test]
    fn op_lookup_time_matches_lookup_time_field() {
        let op = fake_op();
        assert_eq!(op.lookup_time(&()), op.lookup(&()).time);
    }

    #[test]
    fn op_compile_emits_one_leaf_named_by_path() {
        use crate::timing::{CostNode, CostTreeBuilder};
        let op = fake_op();
        let mut b = CostTreeBuilder::new();
        let node = op.compile(&mut b);
        assert_eq!(node, CostNode::Leaf(0));
        let tree = b.finish(node);
        assert_eq!(tree.n_slots(), 1);
        assert_eq!(tree.slots[0].name, "model.attn.o_proj");
    }

    #[test]
    fn op_describe_renders_two_level_tree() {
        let op = fake_op();
        let mut out = String::new();
        op.describe(0, &mut out);

        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 2);
        // Op header at depth 0 (no indent), kernel leaf at depth 1 (one indent).
        assert_eq!(lines[0], "model.attn.o_proj");
        assert_eq!(lines[1], "│  o_proj_kernel (FakeKernel)");
    }

    /// Stand-in compound op: two GEMM atomic ops under one parent (e.g. an FFN's
    /// `gate_up` + `down`). A real compound op (§3/§4) lives in its own file, but
    /// this mirrors its shape to exercise `Describe` recursion over multiple
    /// children and a third nesting level, plus `LookupResult::sum` composition.
    struct FakeFfn {
        name: String,
        gate_up: Op<FakeKernel>,
        down: Op<FakeKernel>,
    }

    impl FakeFfn {
        fn lookup(&self) -> LookupResult {
            LookupResult::sum(
                self.name.clone(),
                vec![self.gate_up.lookup(&()), self.down.lookup(&())],
            )
        }
    }

    impl FakeFfn {
        fn compile(&self, b: &mut CostTreeBuilder) -> CostNode {
            CostNode::Sum(vec![self.gate_up.compile(b), self.down.compile(b)])
        }

        fn eval(&self, buf: &mut [LeafMetrics], cursor: &mut usize) {
            self.gate_up.eval(&(), buf, cursor);
            self.down.eval(&(), buf, cursor);
        }
    }

    impl Describe for FakeFfn {
        fn describe(&self, depth: usize, out: &mut String) {
            use std::fmt::Write;
            writeln!(out, "{}{} (FakeFfn)", "│  ".repeat(depth), self.name).unwrap();
            self.gate_up.describe(depth + 1, out);
            self.down.describe(depth + 1, out);
        }
    }

    fn fake_ffn() -> FakeFfn {
        FakeFfn {
            name: "model.ffn".to_string(),
            gate_up: Op::new(
                "model.ffn.gate_up".to_string(),
                Arc::new(FakeKernel {
                    name: "gate_up_kernel",
                    time_ms: 3.0,
                    flops: 10,
                    bytes: 20,
                }),
            ),
            down: Op::new(
                "model.ffn.down".to_string(),
                Arc::new(FakeKernel {
                    name: "down_kernel",
                    time_ms: 1.5,
                    flops: 5,
                    bytes: 8,
                }),
            ),
        }
    }

    #[test]
    fn compound_op_sums_two_gemm_children() {
        let result = fake_ffn().lookup();
        assert_eq!(&*result.name, "model.ffn");
        // Two op children, each wrapping one kernel leaf.
        assert_eq!(result.breakdown.len(), 2);
        // sum over the two ops: time 3.0 + 1.5, flops 10 + 5, bytes 20 + 8.
        assert_eq!(result.time.as_ms(), 4.5);
        assert_eq!(result.flops, 15);
        assert_eq!(result.bytes, 28);
    }

    #[test]
    fn compound_op_eval_aggregate_matches_lookup() {
        // The compile→eval→aggregate triangle must reproduce `lookup`'s totals
        // with no bridge: cursor walks slots in `compile` order, `aggregate` sums.
        let ffn = fake_ffn();
        let mut b = CostTreeBuilder::new();
        let root = ffn.compile(&mut b);
        let tree = b.finish(root);
        let flat = tree.flatten();

        let mut buf = vec![LeafMetrics::ZERO; tree.n_slots()];
        let mut cursor = 0;
        ffn.eval(&mut buf, &mut cursor);
        assert_eq!(cursor, tree.n_slots(), "eval must fill every slot");

        let agg = CostTree::aggregate(&flat, &buf).m;
        let want = ffn.lookup();
        assert_eq!(agg.time_ms as f64, want.time.as_ms());
        assert_eq!(agg.flops as u64, want.flops);
        assert_eq!(agg.bytes as u64, want.bytes);
    }

    #[test]
    fn compound_op_describe_renders_nested_tree() {
        let mut out = String::new();
        fake_ffn().describe(0, &mut out);

        // Three levels: compound (depth 0) → each Op (depth 1) → kernel (depth 2).
        let expected = "\
model.ffn (FakeFfn)
│  model.ffn.gate_up
│  │  gate_up_kernel (FakeKernel)
│  model.ffn.down
│  │  down_kernel (FakeKernel)
";
        assert_eq!(out, expected);
    }
}
