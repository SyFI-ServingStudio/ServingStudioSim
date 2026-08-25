//! One DeepSeek V4 EP rank's routed expert compute after dispatch.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::mxfp4_marlin_moe_gemm::marlin_block_size_m;
use crate::timing::kernels::{
    ClampedSwigluKernel, ClampedSwigluKernelConfig, ClampedSwigluKernelInput, ElementwiseKernel,
    ElementwiseKernelConfig, ElementwiseKernelInput, MoeAlignBlockSizeKernel,
    MoeAlignBlockSizeKernelConfig, MoeAlignBlockSizeKernelInput, MoeSumKernel, MoeSumKernelConfig,
    MoeSumKernelInput, MoeTopkSoftplusSqrtKernel, MoeTopkSoftplusSqrtKernelConfig,
    MoeTopkSoftplusSqrtKernelInput, Mxfp4MarlinMoeFcRole, Mxfp4MarlinMoeGemmKernel,
    Mxfp4MarlinMoeGemmKernelConfig, Mxfp4MarlinMoeGemmKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, LeafMetrics, PerfApiBridge, Probe,
    SlotInput,
};

#[derive(Clone, Debug)]
pub struct DeepseekV4MoeExpertComputeLocalWorkletConfig {
    pub hidden_size: Dim,
    pub intermediate_size: Dim,
    pub selection_mode: String,
    pub num_experts: Dim,
    pub top_k: u32,
    pub hash_vocab_size: u32,
    pub hidden_dtype: DType,
    /// Popularity of this rank's contiguous 64-expert shard.
    pub local_ppm: Vec<u32>,
    pub gpu_name: String,
    pub selection_backends: Vec<&'static str>,
    pub align_backends: Vec<&'static str>,
    pub marlin_backends: Vec<&'static str>,
    pub activation_backends: Vec<&'static str>,
    pub zero_fill_backends: Vec<&'static str>,
    pub sum_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct DeepseekV4MoeExpertComputeLocalWorkletResolved {
    pub raw_cfg: DeepseekV4MoeExpertComputeLocalWorkletConfig,
    pub selection: MoeTopkSoftplusSqrtKernelConfig,
    pub align: MoeAlignBlockSizeKernelConfig,
    pub fc1: Mxfp4MarlinMoeGemmKernelConfig,
    pub activation: ClampedSwigluKernelConfig,
    pub fc2_output_zero: ElementwiseKernelConfig,
    pub fc2: Mxfp4MarlinMoeGemmKernelConfig,
    pub sum: MoeSumKernelConfig,
}

#[derive(Clone, Debug)]
pub struct DeepseekV4MoeExpertComputeLocalWorkletInput {
    pub num_gathered_tokens: u32,
}

pub struct DeepseekV4MoeExpertComputeLocalWorklet {
    pub name: String,
    pub selection: Op<MoeTopkSoftplusSqrtKernel>,
    pub align: Op<MoeAlignBlockSizeKernel>,
    pub fc1: Op<Mxfp4MarlinMoeGemmKernel>,
    pub activation: Op<ClampedSwigluKernel>,
    pub fc2_output_zero: Op<ElementwiseKernel>,
    pub fc2: Op<Mxfp4MarlinMoeGemmKernel>,
    pub sum: Op<MoeSumKernel>,
    resolved: DeepseekV4MoeExpertComputeLocalWorkletResolved,
}

impl DeepseekV4MoeExpertComputeLocalWorklet {
    pub fn resolve_config(
        config: &DeepseekV4MoeExpertComputeLocalWorkletConfig,
    ) -> DeepseekV4MoeExpertComputeLocalWorkletResolved {
        assert_eq!(config.hidden_size.get(), 4096);
        assert_eq!(config.intermediate_size.get(), 2048);
        assert!(matches!(config.selection_mode.as_str(), "learned" | "hash"));
        assert_eq!(config.num_experts.get(), 256);
        assert_eq!(config.top_k, 6);
        assert_eq!(
            config.hash_vocab_size,
            if config.selection_mode == "hash" {
                129_280
            } else {
                0
            }
        );
        assert_eq!(config.hidden_dtype, DType::Bf16);
        assert_eq!(config.local_ppm.len(), 64);
        assert!(config.local_ppm.iter().any(|&mass| mass > 0));
        let marlin = |fc_role, n, k| Mxfp4MarlinMoeGemmKernelConfig {
            backends: config.marlin_backends.clone(),
            gpu_name: config.gpu_name.clone(),
            fc_role,
            n,
            k,
            dtype: config.hidden_dtype,
            routing_top_k: config.top_k,
            local_ppm: config.local_ppm.clone(),
        };
        DeepseekV4MoeExpertComputeLocalWorkletResolved {
            selection: MoeTopkSoftplusSqrtKernelConfig {
                backends: config.selection_backends.clone(),
                gpu_name: config.gpu_name.clone(),
                selection_mode: config.selection_mode.clone(),
                num_experts: config.num_experts.clone(),
                top_k: config.top_k,
                hash_vocab_size: config.hash_vocab_size,
                logits_dtype: DType::Fp32,
            },
            align: MoeAlignBlockSizeKernelConfig {
                backends: config.align_backends.clone(),
                gpu_name: config.gpu_name.clone(),
                num_experts: config.num_experts.clone(),
                top_k: config.top_k,
            },
            fc1: marlin(
                Mxfp4MarlinMoeFcRole::Fc1,
                config.intermediate_size.clone() * 2,
                config.hidden_size.clone(),
            ),
            activation: ClampedSwigluKernelConfig {
                backends: config.activation_backends.clone(),
                gpu_name: config.gpu_name.clone(),
                hidden_dim: config.intermediate_size.clone(),
                dtype: config.hidden_dtype,
            },
            // vLLM zeros [tokens, top_k, hidden] before the scatter-style FC2.
            fc2_output_zero: ElementwiseKernelConfig {
                backends: config.zero_fill_backends.clone(),
                gpu_name: config.gpu_name.clone(),
                input_bytes_per_token: 0.into(),
                output_bytes_per_token: (config.top_k * config.hidden_size.get() * 2).into(),
            },
            fc2: marlin(
                Mxfp4MarlinMoeFcRole::Fc2,
                config.hidden_size.clone(),
                config.intermediate_size.clone(),
            ),
            sum: MoeSumKernelConfig {
                backends: config.sum_backends.clone(),
                gpu_name: config.gpu_name.clone(),
                top_k: config.top_k,
                hidden_dim: config.hidden_size.clone(),
                dtype: config.hidden_dtype,
            },
            raw_cfg: config.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: DeepseekV4MoeExpertComputeLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        macro_rules! build_op {
            ($field:ident, $kernel:ty, $suffix:literal) => {{
                let op_name = format!("{name}.{}", $suffix);
                Op::new(
                    op_name.clone(),
                    Arc::new(<$kernel>::build(op_name, resolved.$field.clone(), bridge)?),
                )
            }};
        }
        Ok(Self {
            name: name.clone(),
            selection: build_op!(selection, MoeTopkSoftplusSqrtKernel, "topk_softplus_sqrt"),
            align: build_op!(align, MoeAlignBlockSizeKernel, "align_block_size"),
            fc1: build_op!(fc1, Mxfp4MarlinMoeGemmKernel, "fc1_mxfp4_marlin"),
            activation: build_op!(activation, ClampedSwigluKernel, "clamped_swiglu"),
            fc2_output_zero: build_op!(fc2_output_zero, ElementwiseKernel, "fc2_output_zero"),
            fc2: build_op!(fc2, Mxfp4MarlinMoeGemmKernel, "fc2_mxfp4_marlin"),
            sum: build_op!(sum, MoeSumKernel, "moe_sum"),
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: format!("{} (DeepseekV4MoeExpertComputeLocalWorklet)", self.name),
            child: Box::new(CostNode::Sum(vec![
                self.selection.compile(builder),
                self.align.compile(builder),
                self.fc1.compile(builder),
                self.activation.compile(builder),
                self.fc2_output_zero.compile(builder),
                self.fc2.compile(builder),
                self.sum.compile(builder),
            ])),
        }
    }

    pub fn eval(
        &self,
        input: &DeepseekV4MoeExpertComputeLocalWorkletInput,
        evaluator: &mut Evaluator,
    ) {
        let work = derive_work(input.num_gathered_tokens, self.resolved.raw_cfg.top_k);
        let zero = input.num_gathered_tokens == 0;
        eval_or_zero(&self.selection, work.selection, zero, evaluator);
        eval_or_zero(&self.align, work.align, zero, evaluator);
        eval_or_zero(&self.fc1, work.marlin.clone(), zero, evaluator);
        eval_or_zero(&self.activation, work.activation, zero, evaluator);
        eval_or_zero(&self.fc2_output_zero, work.fc2_output_zero, zero, evaluator);
        eval_or_zero(&self.fc2, work.marlin, zero, evaluator);
        eval_or_zero(&self.sum, work.sum, zero, evaluator);
    }
}

struct WorkInputs {
    selection: MoeTopkSoftplusSqrtKernelInput,
    align: MoeAlignBlockSizeKernelInput,
    marlin: Mxfp4MarlinMoeGemmKernelInput,
    activation: ClampedSwigluKernelInput,
    fc2_output_zero: ElementwiseKernelInput,
    sum: MoeSumKernelInput,
}

fn derive_work(num_gathered_tokens: u32, top_k: u32) -> WorkInputs {
    let routed_rows = num_gathered_tokens
        .checked_mul(top_k)
        .expect("num_gathered_tokens * top_k must fit u32");
    WorkInputs {
        selection: MoeTopkSoftplusSqrtKernelInput {
            num_tokens: num_gathered_tokens,
        },
        align: MoeAlignBlockSizeKernelInput {
            num_tokens: num_gathered_tokens,
            block_size: marlin_block_size_m(num_gathered_tokens, top_k, 64),
        },
        marlin: Mxfp4MarlinMoeGemmKernelInput {
            num_input_tokens: num_gathered_tokens,
        },
        activation: ClampedSwigluKernelInput {
            num_rows: routed_rows,
        },
        fc2_output_zero: ElementwiseKernelInput {
            num_tokens: num_gathered_tokens,
        },
        sum: MoeSumKernelInput {
            num_tokens: num_gathered_tokens,
        },
    }
}

fn eval_or_zero<K>(op: &Op<K>, input: K::Input, zero: bool, evaluator: &mut Evaluator)
where
    K: Probe,
    K::Input: Clone + Into<SlotInput>,
{
    if zero {
        evaluator.push(LeafMetrics::ZERO, || input.into());
    } else {
        op.eval(&input, evaluator);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gathered_tokens_drive_selection_alignment_and_routed_rows_once() {
        let work = derive_work(128, 6);
        assert_eq!(work.selection.num_tokens, 128);
        assert_eq!(work.align.num_tokens, 128);
        assert_eq!(work.align.block_size, 16);
        assert_eq!(work.marlin.num_input_tokens, 128);
        assert_eq!(work.activation.num_rows, 768);
        assert_eq!(work.fc2_output_zero.num_tokens, 128);
        assert_eq!(work.sum.num_tokens, 128);
    }
}
