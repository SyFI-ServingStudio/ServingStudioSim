//! FP8 grouped GEMM together with its required BF16 input quantization.
//!
//! The two launches form one semantic operation: the quantized activation and
//! block scales exist only for the immediately following grouped GEMM. Keeping
//! their row-collapse logic here prevents L3 worklets from seeing sub-kernels.

use std::sync::Arc;

use crate::timing::kernels::{
    Fp8BlockQuantKernel, Fp8BlockQuantKernelConfig, Fp8BlockQuantKernelInput,
    Fp8BlockscaleGroupedGemmKernel, Fp8BlockscaleGroupedGemmKernelConfig,
    Fp8BlockscaleGroupedGemmKernelInput, Fp8PerTokenGroupQuantKernel,
    Fp8PerTokenGroupQuantKernelConfig, Fp8PerTokenGroupQuantKernelInput, VllmFusedMoeKernel,
    VllmFusedMoeKernelConfig, VllmFusedMoeKernelInput,
};
use crate::timing::bridge::DType;
use crate::timing::routing::RoutingDistribution;
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge, Probe};

/// Whether the activation this op quantizes has already been expanded to one
/// row per (token, selected expert).
///
/// The grouped GEMM always reads an expanded layout, but the *quantize* before
/// it does not always produce one. vLLM quantizes the gate_up input straight
/// off the hidden states — one row per token — and lets `fused_moe_kernel`
/// gather rows per expert from that; only the intermediate activation, which
/// physically exists once per selection, is quantized expanded. Realizations
/// that permute tokens into per-expert order *before* quantizing (DeepEP into
/// DeepGEMM) quantize the expanded layout on both sides.
///
/// Getting this wrong is a silent top-k-fold error, not a small one: measured
/// against vLLM the expanded reading over-predicted the gate_up quant by 8.9x
/// (1937 ms vs 218 ms over 966 iterations), and it showed up as the largest
/// single term in the whole comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantRows {
    /// One row per token, quantized before the grouped GEMM gathers per expert.
    PerToken,
    /// One row per (token, selected expert).
    PerSelection,
}

/// Which quantize kernel the framework actually launches ahead of the grouped
/// GEMM. The two are not interchangeable and the difference is measurable.
///
/// `Block` is FlashInfer/TensorRT-LLM's `scale_1x128_kernel`, whose cache
/// identity carries `num_problems` because it is launched with the grouped
/// per-expert batch layout in mind. `PerTokenGroup` is vLLM's own
/// `per_token_group_quant_8bit_kernel` — one flat launch over rows, no group
/// count in its identity — and it is what a vLLM server running the Triton
/// `fused_moe_kernel` path emits, the same kernel it uses for every dense
/// projection in the model.
///
/// Picking the wrong one is not a small error: against a measured vLLM
/// Qwen3.6-35B-A3B-FP8 run the `Block` reading over-predicted the routed
/// gate_up quant by 58% and the down quant by 105%, while the identical
/// `PerTokenGroup` kernel modeling the shared expert's quant sat within 20%.
#[derive(Clone, Debug)]
pub enum GroupedQuantConfig {
    Block(Fp8BlockQuantKernelConfig),
    PerTokenGroup(Fp8PerTokenGroupQuantKernelConfig),
}

/// Built counterpart of [`GroupedQuantConfig`]. Kept as one enum rather than two
/// optional fields so a realization can never accidentally launch both.
enum GroupedQuant {
    Block(Arc<Fp8BlockQuantKernel>),
    PerTokenGroup(Arc<Fp8PerTokenGroupQuantKernel>),
}

/// Which expert-compute kernel follows the quantize.
///
/// `TrtllmBlockscale` permutes tokens into per-expert contiguous blocks and
/// hands each group to a dense GEMM. `VllmFusedMoe` never materializes that
/// permutation: one Triton launch gathers `BLOCK_SIZE_M` rows at a time off a
/// padded `sorted_token_ids` list. They are different algorithms, not different
/// tunings, and at a routed decode batch (~2 rows per expert) the grouped curve
/// over-predicted the measured down projection by 45.8%.
///
/// Both variants keep the same `.gemm` slot name so the operation identity that
/// labeling and analysis join on does not move when a realization changes.
#[derive(Clone, Debug)]
pub enum GroupedGemmConfig {
    TrtllmBlockscale(Fp8BlockscaleGroupedGemmKernelConfig),
    VllmFusedMoe(VllmFusedMoeKernelConfig),
}

impl GroupedGemmConfig {
    /// Compute dtype, which both realizations agree on. Exposed so callers that
    /// only care about the numeric contract need not match on the variant;
    /// callers that care *which* kernel runs should match instead.
    pub fn dtype(&self) -> DType {
        match self {
            Self::TrtllmBlockscale(config) => config.dtype,
            Self::VllmFusedMoe(config) => config.dtype,
        }
    }

    fn local_ppm(&self) -> &[u32] {
        match self {
            Self::TrtllmBlockscale(config) => &config.local_ppm,
            Self::VllmFusedMoe(config) => &config.local_ppm,
        }
    }

    fn experts_per_token(&self) -> u32 {
        match self {
            Self::TrtllmBlockscale(config) => config.experts_per_token,
            Self::VllmFusedMoe(config) => config.experts_per_token,
        }
    }
}

/// Built counterpart of [`GroupedGemmConfig`].
enum GroupedGemm {
    TrtllmBlockscale(Arc<Fp8BlockscaleGroupedGemmKernel>),
    VllmFusedMoe(Arc<VllmFusedMoeKernel>),
}

#[derive(Clone, Debug)]
pub struct GroupedFp8GemmWithQuantConfig {
    pub quant: GroupedQuantConfig,
    pub gemm: GroupedGemmConfig,
    /// The layout the quantize kernel sees. Owned here rather than by the
    /// worklet because the two launches are one operation and this is the
    /// row-collapse logic L3 is not supposed to see (see the module header).
    pub quant_rows: QuantRows,
}

#[derive(Clone, Debug, Default)]
pub struct GroupedFp8GemmWithQuantInput {
    pub global_expert_selections: u32,
}

pub struct GroupedFp8GemmWithQuantOp {
    pub name: String,
    quant: GroupedQuant,
    gemm: GroupedGemm,
    local_ppm: Vec<u32>,
    experts_per_token: u32,
    quant_rows: QuantRows,
}

impl GroupedFp8GemmWithQuantOp {
    pub fn build(
        name: String,
        config: GroupedFp8GemmWithQuantConfig,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let quant = match config.quant {
            GroupedQuantConfig::Block(block) => {
                assert_eq!(
                    block.num_problems.get() as usize,
                    config.gemm.local_ppm().len(),
                    "grouped FP8 quant num_problems must equal local expert count"
                );
                GroupedQuant::Block(Arc::new(Fp8BlockQuantKernel::build(
                    format!("{name}.input_quant"),
                    block,
                    bridge,
                )?))
            }
            // No `num_problems` cross-check here: this kernel is launched flat
            // over rows and its cache identity does not carry the group count,
            // so there is nothing to keep consistent with `local_ppm`.
            GroupedQuantConfig::PerTokenGroup(per_token_group) => {
                GroupedQuant::PerTokenGroup(Arc::new(Fp8PerTokenGroupQuantKernel::build(
                    format!("{name}.input_quant"),
                    per_token_group,
                    bridge,
                )?))
            }
        };
        let local_ppm = config.gemm.local_ppm().to_vec();
        let experts_per_token = config.gemm.experts_per_token();
        let quant_rows = config.quant_rows;
        let gemm_name = format!("{name}.gemm");
        let gemm = match config.gemm {
            GroupedGemmConfig::TrtllmBlockscale(gemm_config) => {
                GroupedGemm::TrtllmBlockscale(Arc::new(Fp8BlockscaleGroupedGemmKernel::build(
                    gemm_name,
                    gemm_config,
                    bridge,
                )?))
            }
            GroupedGemmConfig::VllmFusedMoe(gemm_config) => GroupedGemm::VllmFusedMoe(Arc::new(
                VllmFusedMoeKernel::build(gemm_name, gemm_config, bridge)?,
            )),
        };
        Ok(Self {
            name,
            quant,
            gemm,
            local_ppm,
            experts_per_token,
            quant_rows,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let (quant_kind, quant_config) = match &self.quant {
            GroupedQuant::Block(kernel) => (kernel.kind(), kernel.describe_config()),
            GroupedQuant::PerTokenGroup(kernel) => (kernel.kind(), kernel.describe_config()),
        };
        let (gemm_kind, gemm_config) = match &self.gemm {
            GroupedGemm::TrtllmBlockscale(kernel) => (kernel.kind(), kernel.describe_config()),
            GroupedGemm::VllmFusedMoe(kernel) => (kernel.kind(), kernel.describe_config()),
        };
        CostNode::Sum(vec![
            builder.leaf(
                format!("{}.input_quant", self.name),
                quant_kind,
                quant_config,
            ),
            builder.leaf(format!("{}.gemm", self.name), gemm_kind, gemm_config),
        ])
    }

    pub fn eval(&self, input: &GroupedFp8GemmWithQuantInput, ev: &mut Evaluator) {
        assert!(
            self.experts_per_token > 0
                && input.global_expert_selections % self.experts_per_token == 0,
            "global expert selections must be divisible by experts_per_token"
        );
        let num_tokens = input.global_expert_selections / self.experts_per_token;
        let expanded_rows = local_quant_rows(input.global_expert_selections, &self.local_ppm);
        let quant_num_tokens = match self.quant_rows {
            // Under EP the rank quantizes the tokens it was dispatched, and
            // how many *distinct* tokens those are needs the joint routing
            // structure (which tokens picked several of this rank's experts)
            // that a marginal per-expert distribution cannot carry. `min`
            // is the reachable bound: it is exact at EP=1, where every token
            // is local and the expanded count is num_tokens x top_k.
            QuantRows::PerToken => num_tokens.min(expanded_rows),
            QuantRows::PerSelection => expanded_rows,
        };
        // The row count is a property of the layout, not of which kernel does
        // the quantizing, so it is resolved once above and only the input type
        // differs per realization.
        match &self.quant {
            GroupedQuant::Block(kernel) => {
                let quant_input = Fp8BlockQuantKernelInput {
                    num_tokens: quant_num_tokens,
                };
                ev.push(kernel.eval(&quant_input), || quant_input.into());
            }
            GroupedQuant::PerTokenGroup(kernel) => {
                let quant_input = Fp8PerTokenGroupQuantKernelInput {
                    num_tokens: quant_num_tokens,
                };
                ev.push(kernel.eval(&quant_input), || quant_input.into());
            }
        }

        // Both realizations key off the same token count; only the field name
        // differs, because each mirrors its own Python args schema.
        match &self.gemm {
            GroupedGemm::TrtllmBlockscale(kernel) => {
                let gemm_input = Fp8BlockscaleGroupedGemmKernelInput {
                    num_input_tokens: num_tokens,
                };
                ev.push(kernel.eval(&gemm_input), || gemm_input.into());
            }
            GroupedGemm::VllmFusedMoe(kernel) => {
                let gemm_input = VllmFusedMoeKernelInput { num_tokens };
                ev.push(kernel.eval(&gemm_input), || gemm_input.into());
            }
        }
    }
}

fn local_quant_rows(global_expert_selections: u32, local_ppm: &[u32]) -> u32 {
    RoutingDistribution::to_per_expert_counts(global_expert_selections, local_ppm)
        .into_iter()
        .sum()
}

#[cfg(test)]
mod tests {
    use super::local_quant_rows;

    #[test]
    fn quant_rows_are_the_grouped_gemm_rank_share() {
        assert_eq!(local_quant_rows(512, &[250_000, 125_000]), 192);
    }
}
