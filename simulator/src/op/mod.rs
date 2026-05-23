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

use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, DryRun, JitPlan, LeafMetrics, PerfApiBridge, Probe,
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

    /// CostTree compile: an atomic op is one leaf, named by its dotted path and
    /// carrying the kernel's `kind`/`config` for the shape render (the old
    /// `Describe` leaf line).
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        builder.leaf(
            self.name.clone(),
            self.kernel.kind(),
            self.kernel.describe_config(),
        )
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

#[cfg(test)]
mod tests {
    use super::Op;
    use crate::timing::cache::interp::{CoverageFlags, Metrics4};
    use crate::timing::{CostNode, CostTree, CostTreeBuilder, LeafMetrics, Probe};
    use std::sync::Arc;

    /// Mock `Probe` standing in for an L1 kernel, so the op-wrapping logic is
    /// tested without a `PerfApiBridge` / Python (L2 design §15).
    struct FakeKernel {
        kind: &'static str,
        config: &'static str,
        time_ms: f32,
        flops: f32,
        bytes: f32,
    }

    impl Probe for FakeKernel {
        type Input = ();
        fn lookup_metrics(&self, _input: &()) -> LeafMetrics {
            LeafMetrics {
                m: Metrics4 {
                    time_ms: self.time_ms,
                    flops: self.flops,
                    bytes: self.bytes,
                    energy_j: 0.0,
                },
                coverage: CoverageFlags::EMPTY,
            }
        }
        fn kind(&self) -> &'static str {
            self.kind
        }
        fn describe_config(&self) -> String {
            self.config.to_string()
        }
    }

    fn fake_op() -> Op<FakeKernel> {
        Op::new(
            "model.attn.o_proj".to_string(),
            Arc::new(FakeKernel {
                kind: "single_gemm",
                config: "n=4096, k=4096",
                time_ms: 2.5,
                flops: 100.0,
                bytes: 200.0,
            }),
        )
    }

    #[test]
    fn op_compile_emits_one_leaf_with_name_kind_config() {
        let op = fake_op();
        let mut b = CostTreeBuilder::new();
        let node = op.compile(&mut b);
        assert_eq!(node, CostNode::Leaf(0));
        let tree = b.finish(node);
        assert_eq!(tree.n_slots(), 1);
        // The leaf carries the op's dotted name plus the kernel's kind + config
        // (the merged-in `Describe` leaf line).
        assert_eq!(tree.slots[0].name, "model.attn.o_proj");
        assert_eq!(tree.slots[0].kind, "single_gemm");
        assert_eq!(tree.slots[0].config, "n=4096, k=4096");
    }

    #[test]
    fn op_eval_writes_one_leaf_metric() {
        let op = fake_op();
        let mut buf = vec![LeafMetrics::ZERO; 1];
        let mut cursor = 0;
        op.eval(&(), &mut buf, &mut cursor);
        assert_eq!(cursor, 1);
        assert_eq!(buf[0].m.time_ms, 2.5);
        assert_eq!(buf[0].m.flops, 100.0);
    }

    /// Stand-in compound op: two GEMM atomic ops under one parent (e.g. an FFN's
    /// `gate_up` + `down`). A real compound op (§3/§4) lives in its own file, but
    /// this mirrors its shape to exercise the compile→eval→aggregate triangle over
    /// multiple children.
    struct FakeFfn {
        gate_up: Op<FakeKernel>,
        down: Op<FakeKernel>,
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

    fn fake_ffn() -> FakeFfn {
        FakeFfn {
            gate_up: Op::new(
                "model.ffn.gate_up".to_string(),
                Arc::new(FakeKernel {
                    kind: "single_gemm",
                    config: "n=14336, k=4096",
                    time_ms: 3.0,
                    flops: 10.0,
                    bytes: 20.0,
                }),
            ),
            down: Op::new(
                "model.ffn.down".to_string(),
                Arc::new(FakeKernel {
                    kind: "single_gemm",
                    config: "n=4096, k=14336",
                    time_ms: 1.5,
                    flops: 5.0,
                    bytes: 8.0,
                }),
            ),
        }
    }

    #[test]
    fn compound_op_eval_aggregate_sums_children() {
        // The compile→eval→aggregate triangle sums the two leaves: cursor walks
        // slots in `compile` order, `aggregate` rolls up the Sum node.
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
        // sum over the two ops: time 3.0 + 1.5, flops 10 + 5, bytes 20 + 8.
        assert_eq!(agg.time_ms, 4.5);
        assert_eq!(agg.flops, 15.0);
        assert_eq!(agg.bytes, 28.0);
    }
}
