//! vLLM-aligned local (non-EP) `MoE` expert compute, mirroring the four launches
//! nsys observes per `MoE` layer: per-token-group FP8 quant, Triton
//! `fused_moe_kernel` gate/up, `SwiGLU`, quant, Triton `fused_moe_kernel` down.
//!
//! "vLLM-aligned" here means the *local* path specifically. vLLM has a second
//! expert-compute realization -- TRT-LLM blockscale grouped GEMM over
//! DeepEP-permuted tokens -- which `moe_expert_compute_local` beside this file
//! keeps. Choosing between them is a modeling decision about which code path
//! the target deployment takes, not a tuning knob.

use std::sync::Arc;

use crate::op::moe::{
    GroupedFp8GemmWithQuantConfig, GroupedFp8GemmWithQuantInput, GroupedFp8GemmWithQuantOp,
    GroupedGemmConfig, GroupedQuantConfig, QuantRows,
};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput,
    Fp8PerTokenGroupQuantKernelConfig, VllmFusedMoeKernelConfig, LAUNCH_ROLE_DOWN,
    LAUNCH_ROLE_GATE_UP,
};

/// FP8 activation scaling granularity, matching every other vLLM quant leaf in
/// this model. Duplicated from the shared-expert worklet rather than shared:
/// each worklet owns the contract of the kernels it launches.
const FP8_GROUP_SIZE: u32 = 128;
const SCALE_FORMAT: &str = "ue8m0_column_major";
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

#[derive(Clone, Debug)]
pub struct VllmFp8MoeExpertComputeLocalWorkletConfig {
    pub hidden: Dim,
    pub moe_intermediate: Dim,
    pub num_experts: Dim,
    pub ep_size: u16,
    pub top_k: u32,
    pub activation_dtype: DType,
    pub gpu_name: String,
    pub act_backends: Vec<&'static str>,
    pub fp8_quant_backends: Vec<&'static str>,
    pub fp8_grouped_gemm_backends: Vec<&'static str>,
    pub local_ppm: Vec<u32>,
}

impl VllmFp8MoeExpertComputeLocalWorkletConfig {
    #[must_use]
    pub fn split_for_ep(mut template: Self, global_ppm: &[u32]) -> Vec<Self> {
        let ep_size = usize::from(template.ep_size);
        assert!(ep_size > 0, "ep_size must be non-zero");
        let num_experts = template.num_experts.get() as usize;
        assert_eq!(global_ppm.len(), num_experts);
        assert_eq!(num_experts % ep_size, 0);
        let experts_per_rank = num_experts / ep_size;
        template.local_ppm.clear();
        (0..ep_size)
            .map(|rank| {
                let mut rank_config = template.clone();
                let start = rank * experts_per_rank;
                rank_config.local_ppm = global_ppm[start..start + experts_per_rank].to_vec();
                rank_config
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct VllmFp8MoeExpertComputeLocalWorkletResolved {
    pub raw_cfg: VllmFp8MoeExpertComputeLocalWorkletConfig,
    pub gate_up: GroupedFp8GemmWithQuantConfig,
    pub act: ElementwiseKernelConfig,
    pub down: GroupedFp8GemmWithQuantConfig,
    pub experts_per_gpu: Dim,
}

#[derive(Clone, Debug, Default)]
pub struct VllmFp8MoeExpertComputeLocalWorkletInput {
    pub global_expert_selections: u32,
}

pub struct VllmFp8MoeExpertComputeLocalWorklet {
    pub name: String,
    pub gate_up: GroupedFp8GemmWithQuantOp,
    pub act: Op<ElementwiseKernel>,
    pub down: GroupedFp8GemmWithQuantOp,
    resolved: VllmFp8MoeExpertComputeLocalWorkletResolved,
}

impl VllmFp8MoeExpertComputeLocalWorklet {
    #[must_use]
    pub fn resolve_config(
        cfg: &VllmFp8MoeExpertComputeLocalWorkletConfig,
    ) -> VllmFp8MoeExpertComputeLocalWorkletResolved {
        assert_eq!(cfg.activation_dtype, DType::Bf16);
        let ep_size = u32::from(cfg.ep_size);
        assert!(ep_size > 0, "ep_size must be non-zero");
        assert_eq!(cfg.num_experts.get() % ep_size, 0);
        let experts_per_gpu = cfg.num_experts.clone() / Dim::param("ep", ep_size);
        assert_eq!(cfg.local_ppm.len(), experts_per_gpu.get() as usize);
        // vLLM quantizes the routed activation with the very same
        // `per_token_group_quant_8bit_kernel` it uses for every dense
        // projection -- nsys shows that exact kernel on both routed quants, not
        // FlashInfer's grouped `scale_1x128_kernel`. See `GroupedQuantConfig`
        // for the measured cost of getting this backwards.
        let quant_config = |hidden_size: Dim| {
            GroupedQuantConfig::PerTokenGroup(Fp8PerTokenGroupQuantKernelConfig {
                backends: cfg.fp8_quant_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden_size,
                group_size: FP8_GROUP_SIZE,
                input_dtype: cfg.activation_dtype,
                scale_format: SCALE_FORMAT.to_string(),
            })
        };
        // One Triton `fused_moe_kernel` launch per half, matching what nsys sees
        // on this path -- not the TRT-LLM grouped GEMM, which is the realization
        // vLLM picks only when an EP prepare/finalize has already permuted
        // tokens into per-expert order.
        let gemm_config = |n: Dim, k: Dim, launch_role: &str| {
            GroupedGemmConfig::VllmFusedMoe(VllmFusedMoeKernelConfig {
                backends: cfg.fp8_grouped_gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n,
                k,
                dtype: DType::Fp8E4m3,
                experts_per_token: cfg.top_k,
                launch_role: launch_role.to_string(),
                block_size: FP8_GROUP_SIZE,
                local_ppm: cfg.local_ppm.clone(),
            })
        };
        let bytes = Dim::param("activation_bytes", cfg.activation_dtype.size_bytes());
        VllmFp8MoeExpertComputeLocalWorkletResolved {
            gate_up: GroupedFp8GemmWithQuantConfig {
                quant: quant_config(cfg.hidden.clone()),
                gemm: gemm_config(
                    2 * cfg.moe_intermediate.clone(),
                    cfg.hidden.clone(),
                    LAUNCH_ROLE_GATE_UP,
                ),
                // vLLM quantizes the hidden states once and lets fused_moe_kernel
                // gather rows per expert from that buffer, so top-k never
                // multiplies this launch. Verified against nsys: the measured
                // gate_up quant is *smaller* than the down quant (218 ms vs
                // 287 ms), which only holds on the unexpanded layout.
                quant_rows: QuantRows::PerToken,
            },
            act: ElementwiseKernelConfig {
                backends: cfg.act_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: 2 * cfg.moe_intermediate.clone() * bytes.clone(),
                output_bytes_per_token: cfg.moe_intermediate.clone() * bytes,
            },
            down: GroupedFp8GemmWithQuantConfig {
                quant: quant_config(cfg.moe_intermediate.clone()),
                gemm: gemm_config(
                    cfg.hidden.clone(),
                    cfg.moe_intermediate.clone(),
                    LAUNCH_ROLE_DOWN,
                ),
                // The intermediate activation physically exists once per
                // (token, selected expert), so this one really is expanded.
                quant_rows: QuantRows::PerSelection,
            },
            experts_per_gpu,
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: VllmFp8MoeExpertComputeLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let gate_up = GroupedFp8GemmWithQuantOp::build(
            format!("{name}.gate_up"),
            resolved.gate_up.clone(),
            bridge,
        )?;
        let act_name = format!("{name}.activation");
        let act = Op::new(
            act_name.clone(),
            Arc::new(ElementwiseKernel::build(
                act_name,
                resolved.act.clone(),
                bridge,
            )?),
        );
        let down = GroupedFp8GemmWithQuantOp::build(
            format!("{name}.down"),
            resolved.down.clone(),
            bridge,
        )?;
        Ok(Self {
            name,
            gate_up,
            act,
            down,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: format!(
                "{} (VllmFp8MoeExpertComputeLocalWorklet) [local; experts={:?}]",
                self.name, self.resolved.experts_per_gpu
            ),
            child: Box::new(CostNode::Sum(vec![
                self.gate_up.compile(builder),
                self.act.compile(builder),
                self.down.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &VllmFp8MoeExpertComputeLocalWorkletInput, ev: &mut Evaluator) {
        let global_expert_selections = input.global_expert_selections;
        self.gate_up.eval(
            &GroupedFp8GemmWithQuantInput {
                global_expert_selections,
            },
            ev,
        );
        let local_ppm_sum: u64 = self
            .resolved
            .raw_cfg
            .local_ppm
            .iter()
            .map(|&x| u64::from(x))
            .sum();
        #[allow(
            clippy::cast_possible_truncation,
            reason = "local_ppm sums a per-million routing share of global_expert_selections (a u32), so the routed subset cannot exceed u32::MAX"
        )]
        let local_routed_tokens =
            (u64::from(global_expert_selections) * local_ppm_sum).div_ceil(1_000_000) as u32;
        self.act.eval(
            &ElementwiseKernelInput {
                num_tokens: local_routed_tokens,
            },
            ev,
        );
        self.down.eval(
            &GroupedFp8GemmWithQuantInput {
                global_expert_selections,
            },
            ev,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolved() -> VllmFp8MoeExpertComputeLocalWorkletResolved {
        VllmFp8MoeExpertComputeLocalWorklet::resolve_config(
            &VllmFp8MoeExpertComputeLocalWorkletConfig {
                hidden: 4096.into(),
                moe_intermediate: 1536.into(),
                num_experts: 8.into(),
                ep_size: 2,
                top_k: 2,
                activation_dtype: DType::Bf16,
                gpu_name: "H100".into(),
                act_backends: vec!["triton"],
                fp8_quant_backends: vec!["vllm_cuda"],
                fp8_grouped_gemm_backends: vec!["vllm_triton"],
                local_ppm: vec![125_000; 4],
            },
        )
    }

    /// Pins the realization, not just the shapes. The two halves are the same
    /// kernel in different launch roles, and the roles are not interchangeable:
    /// swapping them silently applies routed weights to the wrong projection.
    #[test]
    fn resolves_two_triton_fused_moe_launches_in_opposite_roles() {
        let resolved = resolved();
        let GroupedGemmConfig::VllmFusedMoe(gate_up) = &resolved.gate_up.gemm else {
            panic!("the local vLLM path must use the Triton fused_moe kernel");
        };
        let GroupedGemmConfig::VllmFusedMoe(down) = &resolved.down.gemm else {
            panic!("the local vLLM path must use the Triton fused_moe kernel");
        };
        assert_eq!(gate_up.dtype, DType::Fp8E4m3);
        assert_eq!(down.dtype, DType::Fp8E4m3);
        assert_eq!(gate_up.launch_role, LAUNCH_ROLE_GATE_UP);
        assert_eq!(down.launch_role, LAUNCH_ROLE_DOWN);
        assert_eq!(gate_up.n.get(), 2 * 1536);
        assert_eq!(gate_up.k.get(), 4096);
        assert_eq!(down.n.get(), 4096);
        assert_eq!(down.k.get(), 1536);
    }

    /// The quant halves differ in row layout, and only the gate_up side is
    /// unexpanded. Asserted here because the two are configured by one shared
    /// closure and are easy to make accidentally symmetric.
    #[test]
    fn only_the_gate_up_quant_reads_the_unexpanded_layout() {
        let resolved = resolved();
        assert_eq!(resolved.gate_up.quant_rows, QuantRows::PerToken);
        assert_eq!(resolved.down.quant_rows, QuantRows::PerSelection);
        assert!(matches!(
            resolved.gate_up.quant,
            GroupedQuantConfig::PerTokenGroup(_)
        ));
    }
}
