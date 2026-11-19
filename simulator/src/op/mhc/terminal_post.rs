//! The last layer's standalone mHC post, priced from the two measured mHC kinds.
//!
//! Every mHC boundary inside the stack is one fused launch group: the previous
//! sublayer's `mhc_post` followed by the next sublayer's `mhc_pre` (prenorm
//! GEMM + fused Sinkhorn/norm), measured as `mhc_fused_post_pre_rms_norm`. The
//! first boundary has no post and is measured alone as `mhc_pre_rms_norm`. The
//! final boundary is the opposite case -- a post with no following pre -- and
//! has no measured kind of its own, so this op prices it as the difference
//! `fused(T') - pre(T')` of two rows at the same token count.
//!
//! `T' = max(T, 17)`: below 17 tokens vLLM's fused path runs the pre and post
//! as one small-batch kernel whose split is not observable, so the difference
//! is taken on the T > 16 branch where the post is its own launch. The result
//! is clamped at zero so an interpolation crossing cannot mint negative time.
//!
//! After an MoE layer whose shared + routed add is folded into the post kernel
//! ([`MhcTerminalPostOp::build_after_moe_sum`]) the fused side is that kernel's
//! `mhc_fused_sum_post_pre_rms_norm` row instead: `fused_sum(T') - pre(T')`.
//!
//! One leaf, carrying the fused kind's identity: the slot is the post launch
//! that kind contains.

use std::sync::Arc;

use crate::timing::kernels::{
    MhcFusedPostPreRmsNormKernel, MhcFusedSumKernelConfig, MhcFusedSumPostPreRmsNormKernel,
    MhcPreRmsNormKernel, MhcRmsNormKernelConfig, MhcRmsNormKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Evaluator, LeafMetrics, PerfApiBridge, Probe,
};

/// Smallest token count at which the fused boundary launches its post alone.
pub const SPLIT_POST_MIN_TOKENS: u32 = 17;

#[derive(Clone, Debug)]
pub struct MhcTerminalPostConfig {
    /// One identity for both measured kinds: they share hidden, hc_mult, dtype.
    pub mhc: MhcRmsNormKernelConfig,
    pub pre_backends: Vec<&'static str>,
    pub fused_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct MhcTerminalPostInput {
    pub num_tokens: u32,
}

/// The measured boundary the post is the fused side of.
pub enum TerminalFused {
    PostPre(Arc<MhcFusedPostPreRmsNormKernel>),
    /// The post also adds the MoE's shared and routed outputs.
    SumPostPre(Arc<MhcFusedSumPostPreRmsNormKernel>),
}

pub struct MhcTerminalPostOp {
    pub name: String,
    pub fused: TerminalFused,
    pub pre: Arc<MhcPreRmsNormKernel>,
}

impl MhcTerminalPostOp {
    pub fn build(
        name: String,
        cfg: MhcTerminalPostConfig,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let fused = MhcRmsNormKernelConfig {
            backends: cfg.fused_backends.clone(),
            ..cfg.mhc.clone()
        };
        let fused = TerminalFused::PostPre(Arc::new(MhcFusedPostPreRmsNormKernel::build(
            format!("{name}.fused_reference"),
            fused,
            bridge,
        )?));
        Self::with_fused(name, cfg, fused, bridge)
    }

    /// The last layer's post after an MoE layer with the shared + routed add
    /// in the post kernel: priced as `fused_sum(T') - pre(T')`. `cfg`'s
    /// `fused_backends` are unused; `sum` carries the fused side.
    pub fn build_after_moe_sum(
        name: String,
        cfg: MhcTerminalPostConfig,
        sum: MhcFusedSumKernelConfig,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let fused = TerminalFused::SumPostPre(Arc::new(MhcFusedSumPostPreRmsNormKernel::build(
            format!("{name}.fused_sum_reference"),
            sum,
            bridge,
        )?));
        Self::with_fused(name, cfg, fused, bridge)
    }

    fn with_fused(
        name: String,
        cfg: MhcTerminalPostConfig,
        fused: TerminalFused,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let pre = MhcRmsNormKernelConfig {
            backends: cfg.pre_backends.clone(),
            ..cfg.mhc
        };
        Ok(Self {
            fused,
            pre: Arc::new(MhcPreRmsNormKernel::build(
                format!("{name}.pre_reference"),
                pre,
                bridge,
            )?),
            name,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        match &self.fused {
            TerminalFused::PostPre(k) => {
                builder.leaf(self.name.clone(), k.kind(), k.describe_config())
            }
            TerminalFused::SumPostPre(k) => {
                builder.leaf(self.name.clone(), k.kind(), k.describe_config())
            }
        }
    }

    pub fn eval(&self, input: &MhcTerminalPostInput, ev: &mut Evaluator) {
        let shape = MhcRmsNormKernelInput {
            num_tokens: input.num_tokens.max(SPLIT_POST_MIN_TOKENS),
        };
        let metrics = if input.num_tokens == 0 {
            LeafMetrics::ZERO
        } else {
            let fused = match &self.fused {
                TerminalFused::PostPre(k) => k.eval(&shape),
                TerminalFused::SumPostPre(k) => k.eval(&shape),
            };
            post_only(fused, self.pre.eval(&shape))
        };
        ev.push(metrics, || shape.clone().into());
    }
}

/// `fused - pre`, field-wise, clamped at zero; coverage and backend follow the
/// fused row the slot identifies with.
fn post_only(fused: LeafMetrics, pre: LeafMetrics) -> LeafMetrics {
    let mut post = fused;
    post.m.add_scaled(pre.m, -1.0);
    post.m.time_ms = post.m.time_ms.max(0.0);
    post.m.flops = post.m.flops.max(0.0);
    post.m.bytes = post.m.bytes.max(0.0);
    post.m.energy_j = post.m.energy_j.max(0.0);
    post.coverage |= pre.coverage;
    post
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::cache::interp::{CoverageFlags, Metrics4};

    fn metrics(time_ms: f32) -> LeafMetrics {
        LeafMetrics {
            m: Metrics4 {
                time_ms,
                flops: 10.0 * time_ms,
                bytes: 100.0 * time_ms,
                energy_j: 0.0,
            },
            coverage: CoverageFlags::EMPTY,
            backend_index: 0,
        }
    }

    #[test]
    fn post_is_fused_minus_pre_and_never_negative() {
        let post = post_only(metrics(0.030), metrics(0.012));
        assert!((post.m.time_ms - 0.018).abs() < 1e-6);
        assert!((post.m.bytes - 1.8).abs() < 1e-4);
        assert_eq!(post.backend_index, 0);
        let crossed = post_only(metrics(0.010), metrics(0.012));
        assert_eq!(crossed.m.time_ms, 0.0);
    }

    fn cfg() -> MhcTerminalPostConfig {
        MhcTerminalPostConfig {
            mhc: MhcRmsNormKernelConfig {
                backends: Vec::new(),
                gpu_name: "NVIDIA B200".into(),
                hidden_size: 4096.into(),
                hc_mult: 4,
                hidden_dtype: crate::timing::bridge::DType::Bf16,
            },
            pre_backends: vec!["vllm_tilelang"],
            fused_backends: vec!["vllm_tilelang"],
        }
    }

    fn slots(op: &MhcTerminalPostOp) -> Vec<(String, String)> {
        let mut builder = CostTreeBuilder::new();
        let root = op.compile(&mut builder);
        let tree = builder.finish(root);
        tree.slots
            .iter()
            .map(|s| (s.name.clone(), s.kind.clone()))
            .collect()
    }

    #[test]
    fn compile_mints_one_slot_under_the_op_name() {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let op = MhcTerminalPostOp::build("m.final_mhc_post".into(), cfg(), &bridge).unwrap();
        assert_eq!(
            slots(&op),
            [(
                "m.final_mhc_post".to_string(),
                "mhc_fused_post_pre_rms_norm".to_string()
            )]
        );
    }

    /// Catches the post after a summing MoE layer keeping the plain fused
    /// boundary's identity.
    #[test]
    fn after_a_summing_moe_the_slot_is_the_sum_kind() {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let sum = MhcFusedSumKernelConfig {
            backends: vec!["vllm_tilelang"],
            gpu_name: "NVIDIA B200".into(),
            hidden_size: 4096.into(),
            hc_mult: 4,
            hidden_dtype: crate::timing::bridge::DType::Bf16,
            num_partials: 1,
            shared_addend: true,
        };
        let op =
            MhcTerminalPostOp::build_after_moe_sum("m.final_mhc_post".into(), cfg(), sum, &bridge)
                .unwrap();
        assert_eq!(
            slots(&op),
            [(
                "m.final_mhc_post".to_string(),
                "mhc_fused_sum_post_pre_rms_norm".to_string()
            )]
        );
    }
}
