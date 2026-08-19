//! `MoeDispatchOp` — compound L2 op (L2 design §3) for `MoE` token dispatch. One
//! dispatch call prices the two dispatch network stages (`dispatch_inter`,
//! `dispatch_intra`) against the `p2p_inter` / `p2p_intra` L1 kernels.
//!
//! Both stages are read off the unified [`super::sim::simulate_moe_comm`] curves
//! built once at op init from the config's routing distribution + placement (the
//! op identity, the L2 analogue of an L1 kernel's static dims). The runtime input
//! is just the token count `T`: each stage's prebuilt `E[max]` bottleneck curve is
//! interpolated at `T` and looked up on its tier's p2p kernel. The two leaves sum
//! (the stages are sequential), mirroring ref's `Σ p2p_curve(max_gpu(...))`.

use std::sync::Arc;

use super::sim::simulate_moe_comm;
use super::{BottleneckCurve, MoeNetConfig, MoeNetInput};
use crate::timing::kernels::{P2pInterKernel, P2pIntraKernel};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge, Probe};

pub struct MoeDispatchOp {
    pub name: String,
    pub p2p_intra: Arc<P2pIntraKernel>,
    pub p2p_inter: Arc<P2pInterKernel>,
    /// Prebuilt bottleneck curves `[dispatch_inter, dispatch_intra]`.
    dispatch: [BottleneckCurve; 2],
}

impl MoeDispatchOp {
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
        let [dispatch_inter_curve, dispatch_intra_curve, _, _, _, _] = simulate_moe_comm(
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
            dispatch: [dispatch_inter_curve, dispatch_intra_curve],
        })
    }

    /// `CostTree` compile: two fixed leaves — `inter` (stage `dispatch_inter`) then
    /// `intra` (stage `dispatch_intra`) — matching the eval push order.
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Sum(vec![
            builder.leaf(
                format!("{}.dispatch_inter", self.name),
                self.p2p_inter.kind(),
                self.p2p_inter.describe_config(),
            ),
            builder.leaf(
                format!("{}.dispatch_intra", self.name),
                self.p2p_intra.kind(),
                self.p2p_intra.describe_config(),
            ),
        ])
    }

    /// `CostTree` eval: interpolate each dispatch stage's bottleneck curve at the
    /// token count and look it up on the stage's tier kernel.
    pub fn eval(&self, input: &MoeNetInput, ev: &mut Evaluator) {
        for curve in &self.dispatch {
            curve.push_to(input.tokens, &self.p2p_intra, &self.p2p_inter, ev);
        }
    }
}
