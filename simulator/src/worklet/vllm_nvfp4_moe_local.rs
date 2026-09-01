//! Device-local NVFP4 routed-MoE section used by vLLM on B200.
//!
//! The input activation quantization precedes one whole-callable fused-MoE L1
//! leaf. The fused leaf already owns routing, both expert GEMMs, SwiGLU, and
//! finalize routing, including their PDL overlap; exposing those internal
//! launches again here would double-count the callable.
//!
//! With `tp_size > 1` this callable produces one rank's partial hidden output.
//! The L4 consumer owns the later TP all-reduce; this local worklet does not
//! claim that synchronization boundary.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    Nvfp4FusedMoeKernel, Nvfp4FusedMoeKernelConfig, Nvfp4FusedMoeKernelInput, Nvfp4QuantKernel,
    Nvfp4QuantKernelConfig, Nvfp4QuantKernelInput,
};
use crate::timing::routing::RoutingDistribution;
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

#[derive(Clone, Debug)]
pub struct VllmNvfp4MoeLocalWorkletConfig {
    pub hidden: Dim,
    pub moe_intermediate: Dim,
    pub num_experts: Dim,
    pub ep_size: u16,
    /// TP shards each expert's intermediate width; EP instead shards the
    /// expert axis. A pure-EP deployment sets this to one.
    pub tp_size: u16,
    pub top_k: u32,
    pub activation_dtype: DType,
    pub gpu_name: String,
    pub quant_backends: Vec<&'static str>,
    pub moe_backends: Vec<&'static str>,
    pub weight_format: String,
    pub group_size: u32,
    pub routing_method: String,
    pub n_group: u32,
    pub topk_group: u32,
    pub routed_scaling_numerator: u32,
    pub routed_scaling_denominator: u32,
    pub layerwise_global_ppm: Vec<Vec<u32>>,
    /// Position in the active-count-ranked EP workload list produced by the
    /// shared complete-layer fold. It is not a physical rank identity.
    pub folded_rank_position: u32,
}

impl VllmNvfp4MoeLocalWorkletConfig {
    /// Build one local config per ranked EP workload from the same global
    /// layerwise routing evidence. Sampling and folding remain inside L1 so all
    /// children share one deterministic routing law.
    pub fn split_for_ep(
        mut template: Self,
        routing: &RoutingDistribution,
        num_moe_layers: u32,
    ) -> Vec<Self> {
        let ep = usize::from(template.ep_size);
        assert!(ep > 0, "ep_size must be non-zero");
        let num_experts = template.num_experts.get() as usize;
        assert_eq!(num_experts % ep, 0, "experts must evenly partition EP");
        assert_eq!(
            routing.num_experts() as usize,
            num_experts,
            "routing width must match num_experts"
        );
        template.layerwise_global_ppm = routing.layerwise_ppm(num_moe_layers);

        (0..ep)
            .map(|position| {
                let mut rank_config = template.clone();
                rank_config.folded_rank_position = position as u32;
                rank_config
            })
            .collect()
    }

    /// Build the one rank-symmetric config for pure tensor parallelism.
    pub fn replicated_for_tp(
        mut template: Self,
        routing: &RoutingDistribution,
        num_moe_layers: u32,
    ) -> Self {
        assert_eq!(
            template.ep_size, 1,
            "pure tensor parallelism leaves no expert parallelism"
        );
        assert!(template.tp_size > 0, "tp_size must be non-zero");
        assert_eq!(
            routing.num_experts(),
            template.num_experts.get(),
            "routing width must match num_experts"
        );
        template.layerwise_global_ppm = routing.layerwise_ppm(num_moe_layers);
        template.folded_rank_position = 0;
        template
    }
}

#[derive(Clone, Debug)]
pub struct VllmNvfp4MoeLocalWorkletResolved {
    pub raw_cfg: VllmNvfp4MoeLocalWorkletConfig,
    pub quant: Nvfp4QuantKernelConfig,
    pub fused_moe: Nvfp4FusedMoeKernelConfig,
    pub experts_per_device: Dim,
    pub intermediate_per_rank: Dim,
}

#[derive(Clone, Debug, Default)]
pub struct VllmNvfp4MoeLocalWorkletInput {
    pub num_tokens: u32,
}

pub struct VllmNvfp4MoeLocalWorklet {
    pub name: String,
    pub quant: Op<Nvfp4QuantKernel>,
    pub fused_moe: Op<Nvfp4FusedMoeKernel>,
    resolved: VllmNvfp4MoeLocalWorkletResolved,
}

impl VllmNvfp4MoeLocalWorklet {
    pub fn resolve_config(
        cfg: &VllmNvfp4MoeLocalWorkletConfig,
    ) -> VllmNvfp4MoeLocalWorkletResolved {
        assert_eq!(
            cfg.activation_dtype,
            DType::Bf16,
            "NVFP4 activation input must be BF16"
        );
        assert!(cfg.ep_size > 0, "ep_size must be non-zero");
        assert!(cfg.top_k > 0 && cfg.top_k <= cfg.num_experts.get());
        assert_eq!(
            cfg.num_experts.get() % u32::from(cfg.ep_size),
            0,
            "experts must evenly partition EP"
        );
        assert!(cfg.tp_size > 0, "tp_size must be non-zero");
        assert_eq!(
            cfg.moe_intermediate.get() % u32::from(cfg.tp_size),
            0,
            "moe_intermediate must evenly partition TP"
        );
        assert!(
            cfg.folded_rank_position < u32::from(cfg.ep_size),
            "folded rank position must select an EP workload"
        );
        assert!(!cfg.layerwise_global_ppm.is_empty());
        assert!(
            cfg.layerwise_global_ppm
                .iter()
                .all(|layer| layer.len() == cfg.num_experts.get() as usize),
            "every routing layer must match num_experts"
        );

        let experts_per_device = cfg.num_experts.clone() / Dim::param("ep", u32::from(cfg.ep_size));
        let intermediate_per_rank =
            cfg.moe_intermediate.clone() / Dim::param("moe_tp", u32::from(cfg.tp_size));
        VllmNvfp4MoeLocalWorkletResolved {
            quant: Nvfp4QuantKernelConfig {
                backends: cfg.quant_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden_size: cfg.hidden.clone(),
                group_size: cfg.group_size,
                input_dtype: cfg.activation_dtype,
                scale_format: "linear_e4m3".to_string(),
            },
            fused_moe: Nvfp4FusedMoeKernelConfig {
                backends: cfg.moe_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden_size: cfg.hidden.clone(),
                intermediate_size: intermediate_per_rank.clone(),
                num_experts: cfg.num_experts.clone(),
                num_local_experts: experts_per_device.clone(),
                top_k: cfg.top_k,
                input_dtype: cfg.activation_dtype,
                weight_format: cfg.weight_format.clone(),
                group_size: cfg.group_size,
                routing_method: cfg.routing_method.clone(),
                n_group: cfg.n_group,
                topk_group: cfg.topk_group,
                routed_scaling_numerator: cfg.routed_scaling_numerator,
                routed_scaling_denominator: cfg.routed_scaling_denominator,
                layerwise_global_ppm: cfg.layerwise_global_ppm.clone(),
                folded_rank_position: cfg.folded_rank_position,
            },
            experts_per_device,
            intermediate_per_rank,
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: VllmNvfp4MoeLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let quant_name = format!("{name}.input_quant");
        let fused_moe_name = format!("{name}.fused_moe");
        let quant = Op::new(
            quant_name.clone(),
            Arc::new(Nvfp4QuantKernel::build(
                quant_name,
                resolved.quant.clone(),
                bridge,
            )?),
        );
        let fused_moe = Op::new(
            fused_moe_name.clone(),
            Arc::new(Nvfp4FusedMoeKernel::build(
                fused_moe_name,
                resolved.fused_moe.clone(),
                bridge,
            )?),
        );
        Ok(Self {
            name,
            quant,
            fused_moe,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: format!(
                "{} (VllmNvfp4MoeLocalWorklet) [ep={}; tp={}; experts={:?}; intermediate={:?}]",
                self.name,
                self.resolved.raw_cfg.ep_size,
                self.resolved.raw_cfg.tp_size,
                self.resolved.experts_per_device,
                self.resolved.intermediate_per_rank,
            ),
            child: Box::new(CostNode::Sum(vec![
                self.quant.compile(builder),
                self.fused_moe.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &VllmNvfp4MoeLocalWorkletInput, ev: &mut Evaluator) {
        self.quant.eval(
            &Nvfp4QuantKernelInput {
                num_tokens: input.num_tokens,
            },
            ev,
        );
        self.fused_moe.eval(
            &Nvfp4FusedMoeKernelInput {
                num_tokens: input.num_tokens,
            },
            ev,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn template(ep_size: u16, tp_size: u16) -> VllmNvfp4MoeLocalWorkletConfig {
        VllmNvfp4MoeLocalWorkletConfig {
            hidden: 6144.into(),
            moe_intermediate: 2048.into(),
            num_experts: 256.into(),
            ep_size,
            tp_size,
            top_k: 8,
            activation_dtype: DType::Bf16,
            gpu_name: "NVIDIA B200".to_string(),
            quant_backends: vec!["vllm_cuda"],
            moe_backends: vec!["flashinfer_trtllm_sm100"],
            weight_format: "nvfp4_e2m1".to_string(),
            group_size: 16,
            routing_method: "minimax2".to_string(),
            n_group: 1,
            topk_group: 1,
            routed_scaling_numerator: 5,
            routed_scaling_denominator: 2,
            layerwise_global_ppm: Vec::new(),
            folded_rank_position: 0,
        }
    }

    #[test]
    fn ep_split_preserves_one_global_routing_sample_and_partitions_experts() {
        let popularity = RoutingDistribution::from_profile(
            &(0..256)
                .map(|expert| (expert + 1) as f32)
                .collect::<Vec<_>>(),
        );
        let configs = VllmNvfp4MoeLocalWorkletConfig::split_for_ep(template(4, 1), &popularity, 2);

        assert_eq!(configs.len(), 4);
        assert_eq!(
            configs
                .iter()
                .map(|config| config.folded_rank_position)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert!(configs
            .windows(2)
            .all(|pair| pair[0].layerwise_global_ppm == pair[1].layerwise_global_ppm));

        let resolved = VllmNvfp4MoeLocalWorklet::resolve_config(&configs[3]);
        assert_eq!(resolved.experts_per_device, 64);
        assert_eq!(resolved.quant.scale_format, "linear_e4m3");
        assert_eq!(resolved.fused_moe.folded_rank_position, 3);
    }

    #[test]
    #[should_panic(expected = "routing width must match num_experts")]
    fn ep_split_rejects_wrong_routing_width() {
        let _ = VllmNvfp4MoeLocalWorkletConfig::split_for_ep(
            template(4, 1),
            &RoutingDistribution::uniform(128),
            2,
        );
    }

    #[test]
    fn ep_deployment_leaves_the_intermediate_dimension_whole() {
        let configs = VllmNvfp4MoeLocalWorkletConfig::split_for_ep(
            template(4, 1),
            &RoutingDistribution::uniform(256),
            2,
        );
        let resolved = VllmNvfp4MoeLocalWorklet::resolve_config(&configs[0]);

        assert_eq!(resolved.intermediate_per_rank, 2048);
        assert_eq!(resolved.fused_moe.intermediate_size, 2048);
        assert_eq!(resolved.fused_moe.num_local_experts, 64);
    }

    #[test]
    fn pure_tp_shards_intermediate_and_keeps_every_expert() {
        let config = VllmNvfp4MoeLocalWorkletConfig::replicated_for_tp(
            template(1, 4),
            &RoutingDistribution::uniform(256),
            2,
        );
        let resolved = VllmNvfp4MoeLocalWorklet::resolve_config(&config);

        assert_eq!(resolved.intermediate_per_rank, 512);
        assert_eq!(resolved.fused_moe.intermediate_size, 512);
        assert_eq!(resolved.experts_per_device, 256);
        assert_eq!(resolved.fused_moe.num_local_experts, 256);
        assert_eq!(resolved.fused_moe.folded_rank_position, 0);
    }

    #[test]
    fn hybrid_ep_tp_partitions_the_two_orthogonal_axes() {
        let configs = VllmNvfp4MoeLocalWorkletConfig::split_for_ep(
            template(4, 4),
            &RoutingDistribution::uniform(256),
            2,
        );
        let resolved = VllmNvfp4MoeLocalWorklet::resolve_config(&configs[3]);

        assert_eq!(resolved.experts_per_device, 64);
        assert_eq!(resolved.intermediate_per_rank, 512);
        assert_eq!(resolved.fused_moe.folded_rank_position, 3);
    }

    #[test]
    #[should_panic(expected = "pure tensor parallelism leaves no expert parallelism")]
    fn replicated_for_tp_rejects_expert_parallelism() {
        let _ = VllmNvfp4MoeLocalWorkletConfig::replicated_for_tp(
            template(4, 4),
            &RoutingDistribution::uniform(256),
            2,
        );
    }

    #[test]
    #[should_panic(expected = "moe_intermediate must evenly partition TP")]
    fn resolve_rejects_an_indivisible_tp_degree() {
        let _ = VllmNvfp4MoeLocalWorklet::resolve_config(&template(1, 3));
    }
}
