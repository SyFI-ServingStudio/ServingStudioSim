//! `MoeExpertComputeLocalWorklet` — per-EP-rank MoE expert compute (the work
//! between MoE dispatch and MoE combine): grouped-up_gate → SwiGLU activation →
//! grouped-down. `Local` group suffix (L3 §1.5): one sync section on ONE GPU,
//! no collective inside. The arch wraps `ep_size` independent instances of this
//! worklet under a `Max{1.0}` so the slowest EP rank's expert compute is the
//! cell's wallclock (L4 §3.3 fan-out for per-GPU imbalance).
//!
//! `local_ppm` is this rank's SHARD of the global routing distribution (one
//! ppm value per local expert; `len() == num_experts / ep_size`). The grouped
//! GEMM kernel is **distribution-sensitive** (its cache identity bakes
//! `local_ppm` in), so different ranks with different shards become different
//! cached kernels — this is what makes the `Max` non-trivial under skewed
//! routing. For the v1 uniform shape every rank's shard is identical and the
//! Max degenerates, but the structure is ready for skew.
//!
//! Shape per leaf (`moe_intermediate = m_inter`, full-hidden = `h`):
//!   - gate_up : `(2·m_inter) × h` GEMM, one row per routed (token,expert) pair
//!     in this rank's experts; the runtime input is `global_expert_selections =
//!     num_tokens × top_k` and the kernel splits it across the local experts by
//!     `local_ppm`.
//!   - activation : SwiGLU elementwise on a routed-pair count = `global_expert_
//!     selections / ep_size` (this rank's share, on `2·m_inter`-wide partials).
//!   - down : `h × m_inter` GEMM, same global routed-pair input.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, GroupedGemmKernel,
    GroupedGemmKernelConfig, GroupedGemmKernelInput,
};
use crate::timing::routing::RoutingDistribution;
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge};

/// Raw config + this rank's `local_ppm` shard. Partition is derivable
/// (`experts_per_gpu = num_experts / ep_size`).
#[derive(Clone, Debug)]
pub struct MoeExpertComputeLocalWorkletConfig {
    pub hidden: u32,
    pub moe_intermediate: u32,
    pub num_experts: u32,
    pub ep_size: u16,
    pub dtype: DType,
    pub gpu_name: String,
    pub act_backends: Vec<&'static str>,
    pub grouped_gemm_backends: Vec<&'static str>,
    /// This rank's slice of the global routing distribution — one ppm value per
    /// local expert. `Σ(local_ppm) < TOTAL_PPM` (it is a shard). v1 callers use
    /// the uniform shard `[TOTAL_PPM / num_experts; experts_per_gpu]`.
    pub local_ppm: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct MoeExpertComputeLocalWorkletResolved {
    pub raw_cfg: MoeExpertComputeLocalWorkletConfig,
    pub gate_up: GroupedGemmKernelConfig,
    pub act: ElementwiseKernelConfig,
    pub down: GroupedGemmKernelConfig,
    pub experts_per_gpu: u32,
    pub dtype_bytes: u32,
}

/// Per-call shape: the GLOBAL routed-pair count for the iteration (`num_tokens
/// × top_k`). Grouped GEMMs consume it directly; the activation consumes its
/// per-rank share (`global / ep_size`), derived inside `eval`.
#[derive(Clone, Debug, Default)]
pub struct MoeExpertComputeLocalWorkletInput {
    pub global_expert_selections: u32,
}

pub struct MoeExpertComputeLocalWorklet {
    pub name: String,
    pub gate_up: Op<GroupedGemmKernel>,
    pub act: Op<ElementwiseKernel>,
    pub down: Op<GroupedGemmKernel>,
    resolved: MoeExpertComputeLocalWorkletResolved,
}

impl MoeExpertComputeLocalWorklet {
    pub fn resolve_config(
        cfg: &MoeExpertComputeLocalWorkletConfig,
    ) -> MoeExpertComputeLocalWorkletResolved {
        let ep = cfg.ep_size as u32;
        assert!(ep > 0, "ep_size must be non-zero");
        assert!(
            cfg.num_experts % ep == 0,
            "num_experts {} not divisible by ep_size {}",
            cfg.num_experts,
            ep,
        );
        let experts_per_gpu = cfg.num_experts / ep;
        assert_eq!(
            cfg.local_ppm.len() as u32,
            experts_per_gpu,
            "local_ppm len ({}) must equal experts_per_gpu ({})",
            cfg.local_ppm.len(),
            experts_per_gpu,
        );
        let dtype_bytes = cfg.dtype.size_bytes();
        MoeExpertComputeLocalWorkletResolved {
            gate_up: GroupedGemmKernelConfig {
                backends: cfg.grouped_gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: 2 * cfg.moe_intermediate,
                k: cfg.hidden,
                dtype: cfg.dtype,
                local_ppm: cfg.local_ppm.clone(),
            },
            act: ElementwiseKernelConfig {
                // SwiGLU on the gate_up output: reads 2·moe_intermediate, writes
                // moe_intermediate elements per routed (token,expert) pair.
                backends: cfg.act_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: 2 * cfg.moe_intermediate * dtype_bytes,
                output_bytes_per_token: cfg.moe_intermediate * dtype_bytes,
            },
            down: GroupedGemmKernelConfig {
                backends: cfg.grouped_gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.hidden,
                k: cfg.moe_intermediate,
                dtype: cfg.dtype,
                local_ppm: cfg.local_ppm.clone(),
            },
            experts_per_gpu,
            dtype_bytes,
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: MoeExpertComputeLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let gu_name = format!("{name}.gate_up");
        let act_name = format!("{name}.activation");
        let dn_name = format!("{name}.down");
        let gate_up = Op::new(
            gu_name.clone(),
            Arc::new(GroupedGemmKernel::build(gu_name, resolved.gate_up.clone(), bridge)?),
        );
        let act = Op::new(
            act_name.clone(),
            Arc::new(ElementwiseKernel::build(act_name, resolved.act.clone(), bridge)?),
        );
        let down = Op::new(
            dn_name.clone(),
            Arc::new(GroupedGemmKernel::build(dn_name, resolved.down.clone(), bridge)?),
        );
        Ok(Self {
            name,
            gate_up,
            act,
            down,
            resolved,
        })
    }

    /// CostTree compile: `Sum(gate_up, activation, down)`.
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let r = &self.resolved;
        let label = format!(
            "{} (MoeExpertComputeLocalWorklet) [hidden={}, m_inter={}, experts_per_gpu={}/{}]",
            self.name,
            r.raw_cfg.hidden,
            r.raw_cfg.moe_intermediate,
            r.experts_per_gpu,
            r.raw_cfg.num_experts,
        );
        CostNode::Labeled {
            label,
            child: Box::new(CostNode::Sum(vec![
                self.gate_up.compile(builder),
                self.act.compile(builder),
                self.down.compile(builder),
            ])),
        }
    }

    /// CostTree eval: fill slots in `compile` order — gate_up, activation, down.
    /// `global_expert_selections` flows straight into the grouped GEMMs (their
    /// caches split it by `local_ppm`); the activation gets this rank's share
    /// `global / ep_size` (uniform v1 assumption — even routing across EP ranks).
    pub fn eval(&self, input: &MoeExpertComputeLocalWorkletInput, ev: &mut Evaluator) {
        let global = input.global_expert_selections;
        let ep = u32::from(self.resolved.raw_cfg.ep_size);
        let routed_per_rank = global / ep;
        self.gate_up.eval(
            &GroupedGemmKernelInput {
                global_expert_selections: global,
            },
            ev,
        );
        self.act.eval(
            &ElementwiseKernelInput {
                num_tokens: routed_per_rank,
            },
            ev,
        );
        self.down.eval(
            &GroupedGemmKernelInput {
                global_expert_selections: global,
            },
            ev,
        );
    }
}

/// Build a v1 uniform `local_ppm` shard: each of the `experts_per_gpu` experts
/// on this rank receives `TOTAL_PPM / num_experts` ppm (the global uniform per-
/// expert weight). Bake-time helper used by the L4 arch's `build_configs`; not
/// part of the worklet contract, but co-located so the v1 shape stays in one
/// file.
pub fn uniform_local_ppm(num_experts: u32, ep_size: u16) -> Vec<u32> {
    assert!(ep_size > 0, "ep_size must be non-zero");
    let ep = u32::from(ep_size);
    assert!(
        num_experts % ep == 0,
        "num_experts {num_experts} not divisible by ep_size {ep}"
    );
    let experts_per_gpu = (num_experts / ep) as usize;
    let per_expert = RoutingDistribution::TOTAL_PPM / num_experts;
    vec![per_expert; experts_per_gpu]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(ep_size: u16) -> MoeExpertComputeLocalWorkletConfig {
        let num_experts = 128u32;
        MoeExpertComputeLocalWorkletConfig {
            hidden: 4096,
            moe_intermediate: 3072,
            num_experts,
            ep_size,
            dtype: DType::Bf16,
            gpu_name: "H100".to_string(),
            act_backends: vec!["triton"],
            grouped_gemm_backends: vec!["deepgemm"],
            local_ppm: uniform_local_ppm(num_experts, ep_size),
        }
    }

    #[test]
    fn uniform_local_ppm_distributes_evenly() {
        let s = uniform_local_ppm(128, 8);
        assert_eq!(s.len(), 16); // 128 / 8
        // Each expert gets TOTAL_PPM / num_experts (rounded toward zero).
        assert_eq!(s[0], 1_000_000 / 128);
        assert!(s.iter().all(|&v| v == s[0]));
    }

    #[test]
    fn resolve_shards_experts_and_threads_local_ppm_into_grouped_gemms() {
        let r = MoeExpertComputeLocalWorklet::resolve_config(&cfg(8));
        assert_eq!(r.experts_per_gpu, 16); // 128 / 8
        assert_eq!(r.gate_up.n, 2 * 3072); // fused gate||up output
        assert_eq!(r.gate_up.k, 4096); // hidden
        assert_eq!(r.down.n, 4096); // hidden
        assert_eq!(r.down.k, 3072); // moe_intermediate
        assert_eq!(r.gate_up.local_ppm.len(), 16);
        // bf16 = 2 bytes/elem.
        assert_eq!(r.act.input_bytes_per_token, 2 * 3072 * 2);
        assert_eq!(r.act.output_bytes_per_token, 3072 * 2);
    }

    #[test]
    #[should_panic(expected = "num_experts")]
    fn ep_indivisible_num_experts_panics() {
        let mut c = cfg(8);
        c.ep_size = 9; // 128 % 9 != 0
        c.local_ppm = vec![0; 1]; // ignored — panic fires first
        let _ = MoeExpertComputeLocalWorklet::resolve_config(&c);
    }

    #[test]
    #[should_panic(expected = "local_ppm len")]
    fn local_ppm_length_mismatch_panics() {
        let mut c = cfg(8);
        c.local_ppm = vec![0; 4]; // expected 16 for ep=8, num_experts=128
        let _ = MoeExpertComputeLocalWorklet::resolve_config(&c);
    }
}
