//! `MoeCombineOp` — compound L2 op (L2 design §3) for MoE token combine. One
//! combine call prices the four combine network stages (`combine_intra_reduce`,
//! `combine_inter_reduce`, `combine_inter_bcast`, `combine_intra_fanout`) against
//! the `p2p_intra` / `p2p_inter` L1 kernels.
//!
//! All four stages are read off the SAME unified [`super::sim::simulate_moe_comm`]
//! curves that drive [`super::MoeDispatchOp`] — built once at op init from the
//! config's routing distribution + placement. Combine is the exact mirror of
//! dispatch (the per-token reduce tree is the dispatch fan-out reversed), so this
//! op simply reads stage slots `2..6` of the same six-curve set. Under the v1
//! single-home placement the bcast/fanout curves are identically zero (no
//! replicated output to deliver); they are kept as leaves so the op's slot
//! taxonomy stays stable for a future replicated placement.
//!
//! The runtime input is just the token count `T`: each stage's prebuilt `E[max]`
//! bottleneck curve is interpolated at `T` and looked up on its tier's p2p
//! kernel. The four leaves sum (the stages are sequential).

use std::sync::Arc;

use super::sim::simulate_moe_comm;
use super::{BottleneckCurve, MoeNetConfig, MoeNetInput};
use crate::timing::kernels::{P2pInterKernel, P2pIntraKernel};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge, Probe};

pub struct MoeCombineOp {
    pub name: String,
    pub p2p_intra: Arc<P2pIntraKernel>,
    pub p2p_inter: Arc<P2pInterKernel>,
    /// Prebuilt bottleneck curves `[intra_reduce, inter_reduce, inter_bcast,
    /// intra_fanout]` (stage slots `2..6` of the unified six-curve set).
    combine: [BottleneckCurve; 4],
}

impl MoeCombineOp {
    pub fn build(
        name: String,
        cfg: MoeNetConfig,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let p2p_intra = Arc::new(P2pIntraKernel::build(
            format!("{name}.intra"),
            cfg.p2p_intra_config(),
            bridge,
        )?);
        let p2p_inter = Arc::new(P2pInterKernel::build(
            format!("{name}.inter"),
            cfg.p2p_inter_config(),
            bridge,
        )?);
        let [_, _, intra_reduce_curve, inter_reduce_curve, inter_bcast_curve, intra_fanout_curve] =
            simulate_moe_comm(
                cfg.routing.ppm(),
                cfg.top_k,
                &cfg.net_params(),
                cfg.placement,
                &super::SIM_T_GRID,
                super::SIM_TRIALS,
                super::SIM_SEED,
            );
        Ok(Self {
            name,
            p2p_intra,
            p2p_inter,
            combine: [
                intra_reduce_curve,
                inter_reduce_curve,
                inter_bcast_curve,
                intra_fanout_curve,
            ],
        })
    }

    /// CostTree compile: four leaves in stage order (intra_reduce, inter_reduce,
    /// inter_bcast, intra_fanout), matching the eval push order.
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let intra_kind = self.p2p_intra.kind();
        let intra_cfg = self.p2p_intra.describe_config();
        let intra_backends = self.p2p_intra.backends();
        let inter_kind = self.p2p_inter.kind();
        let inter_cfg = self.p2p_inter.describe_config();
        let inter_backends = self.p2p_inter.backends();
        CostNode::Sum(vec![
            builder.leaf(
                format!("{}.combine_intra_reduce", self.name),
                intra_kind,
                intra_cfg.clone(),
                intra_backends.clone(),
            ),
            builder.leaf(
                format!("{}.combine_inter_reduce", self.name),
                inter_kind,
                inter_cfg.clone(),
                inter_backends.clone(),
            ),
            builder.leaf(
                format!("{}.combine_inter_bcast", self.name),
                inter_kind,
                inter_cfg,
                inter_backends,
            ),
            builder.leaf(
                format!("{}.combine_intra_fanout", self.name),
                intra_kind,
                intra_cfg,
                intra_backends,
            ),
        ])
    }

    /// CostTree eval: interpolate each combine stage's bottleneck curve at the
    /// token count and look it up on the stage's tier kernel.
    pub fn eval(&self, input: &MoeNetInput, ev: &mut Evaluator) {
        for curve in &self.combine {
            curve.push_to(input.tokens, &self.p2p_intra, &self.p2p_inter, ev);
        }
    }
}
