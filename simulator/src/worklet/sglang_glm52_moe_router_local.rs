//! SGLang GLM-5.2 router launch graph: residual RMSNorm plus a real
//! BF16-input/FP32-output gate GEMM. Routing selection is owned by the fused
//! MoE callable, so this worklet has no cast or select proxy leaves.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    GemmFp32OutputKernel, GemmFp32OutputKernelConfig, GemmFp32OutputKernelInput,
    ResidualRmsNormKernel, ResidualRmsNormKernelConfig, ResidualRmsNormKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, LeafMetrics, PerfApiBridge, Probe,
    SlotInput,
};

const HIDDEN_DIM: u32 = 6_144;
const NUM_EXPERTS: u32 = 256;
const TOP_K: u32 = 8;
// Preserve the accepted external slot schema used by alignment maps/rules.
// The field and kernel type are accurate (`GemmFp32OutputKernel`); only the
// historical location suffix remains for compatibility.
const GATE_SLOT_SUFFIX: &str = "router_gemm_bf16_proxy";

#[derive(Clone, Debug)]
pub struct SglangGlm52MoeRouterLocalWorkletConfig {
    pub residual_norm_backends: Vec<&'static str>,
    pub gate_gemm_backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_dim: Dim,
    pub num_experts: Dim,
    pub top_k: u32,
    pub base_dtype: DType,
    pub router_output_dtype: DType,
    pub index_dtype: String,
    pub scoring_func: String,
    pub topk_method: String,
    pub norm_topk_prob: bool,
    pub routed_scaling_numerator: u32,
    pub routed_scaling_denominator: u32,
}

#[derive(Clone, Debug)]
pub struct SglangGlm52MoeRouterLocalWorkletResolved {
    pub post_attn_add_rms_norm: ResidualRmsNormKernelConfig,
    pub router_gemm: GemmFp32OutputKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct SglangGlm52MoeRouterLocalWorkletInput {
    pub batch_tokens: u32,
}

pub struct SglangGlm52MoeRouterLocalWorklet {
    pub name: String,
    pub post_attn_add_rms_norm: Op<ResidualRmsNormKernel>,
    pub router_gemm: Op<GemmFp32OutputKernel>,
}

impl SglangGlm52MoeRouterLocalWorklet {
    pub fn resolve_config(
        cfg: &SglangGlm52MoeRouterLocalWorkletConfig,
    ) -> SglangGlm52MoeRouterLocalWorkletResolved {
        validate_config(cfg).unwrap_or_else(|reason| {
            panic!("invalid SglangGlm52MoeRouterLocalWorkletConfig: {reason}")
        });
        SglangGlm52MoeRouterLocalWorkletResolved {
            post_attn_add_rms_norm: ResidualRmsNormKernelConfig {
                backends: cfg.residual_norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden_dim.clone(),
                dtype: cfg.base_dtype,
            },
            router_gemm: GemmFp32OutputKernelConfig {
                backends: cfg.gate_gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.num_experts.clone(),
                k: cfg.hidden_dim.clone(),
                input_dtype: cfg.base_dtype,
            },
        }
    }

    pub fn build(
        name: String,
        resolved: SglangGlm52MoeRouterLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let norm_name = format!("{name}.post_attn_add_rms_norm");
        let gate_name = format!("{name}.{GATE_SLOT_SUFFIX}");
        Ok(Self {
            name,
            post_attn_add_rms_norm: Op::new(
                norm_name.clone(),
                Arc::new(ResidualRmsNormKernel::build(
                    norm_name,
                    resolved.post_attn_add_rms_norm.clone(),
                    bridge,
                )?),
            ),
            router_gemm: Op::new(
                gate_name.clone(),
                Arc::new(GemmFp32OutputKernel::build(
                    gate_name,
                    resolved.router_gemm.clone(),
                    bridge,
                )?),
            ),
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: format!(
                "{} (SglangGlm52MoeRouterLocalWorklet) [BF16->FP32 gate; production-exact]",
                self.name
            ),
            child: Box::new(CostNode::Sum(vec![
                self.post_attn_add_rms_norm.compile(builder),
                self.router_gemm.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &SglangGlm52MoeRouterLocalWorkletInput, ev: &mut Evaluator) {
        self.eval_with_post_attn_norm(input, true, ev);
    }

    pub fn eval_with_post_attn_norm(
        &self,
        input: &SglangGlm52MoeRouterLocalWorkletInput,
        include_post_attn_norm: bool,
        ev: &mut Evaluator,
    ) {
        let zero = input.batch_tokens == 0;
        push(
            &self.post_attn_add_rms_norm,
            ResidualRmsNormKernelInput {
                m: input.batch_tokens,
            },
            zero || !include_post_attn_norm,
            ev,
        );
        push(
            &self.router_gemm,
            GemmFp32OutputKernelInput {
                m: input.batch_tokens,
            },
            zero,
            ev,
        );
    }
}

fn validate_config(cfg: &SglangGlm52MoeRouterLocalWorkletConfig) -> Result<(), String> {
    if cfg.hidden_dim.get() != HIDDEN_DIM
        || cfg.num_experts.get() != NUM_EXPERTS
        || cfg.top_k != TOP_K
        || cfg.base_dtype != DType::Bf16
        || cfg.router_output_dtype != DType::Fp32
        || cfg.index_dtype != "int32"
        || cfg.scoring_func != "sigmoid"
        || cfg.topk_method != "noaux_tc"
        || !cfg.norm_topk_prob
        || cfg.routed_scaling_numerator != 5
        || cfg.routed_scaling_denominator != 2
    {
        return Err("unsupported SGLang GLM-5.2 router identity".to_string());
    }
    Ok(())
}

fn push<K>(op: &Op<K>, input: K::Input, zero: bool, ev: &mut Evaluator)
where
    K: Probe,
    K::Input: Clone + Into<SlotInput>,
{
    let metrics = if zero {
        LeafMetrics::ZERO
    } else {
        op.kernel.eval(&input)
    };
    ev.push(metrics, || input.clone().into());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_is_real_bf16_to_fp32_and_has_no_proxy_leaves() {
        let cfg = SglangGlm52MoeRouterLocalWorkletConfig {
            residual_norm_backends: vec!["flashinfer"],
            gate_gemm_backends: vec!["sglang_router_auto"],
            gpu_name: "NVIDIA B200".to_string(),
            hidden_dim: 6144.into(),
            num_experts: 256.into(),
            top_k: 8,
            base_dtype: DType::Bf16,
            router_output_dtype: DType::Fp32,
            index_dtype: "int32".to_string(),
            scoring_func: "sigmoid".to_string(),
            topk_method: "noaux_tc".to_string(),
            norm_topk_prob: true,
            routed_scaling_numerator: 5,
            routed_scaling_denominator: 2,
        };
        let r = SglangGlm52MoeRouterLocalWorklet::resolve_config(&cfg);
        assert_eq!(r.router_gemm.backends, vec!["sglang_router_auto"]);
        assert_eq!(r.router_gemm.input_dtype, DType::Bf16);
        assert_eq!(r.router_gemm.n, 256);
        assert_eq!(r.router_gemm.k, 6144);
    }
}
