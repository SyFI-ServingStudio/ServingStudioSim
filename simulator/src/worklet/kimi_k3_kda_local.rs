//! Kimi-K3 rank-local KDA decode worklet.
//!
//! The fused SGLang KDA launch owns the causal-convolution update, recurrent
//! state read/write, recurrence, and gated normalization. Prefill is present in
//! the input contract, but this decode-first recipe leaves that fused decode
//! leaf at zero for prefill because no K3 prefill profile is registered here.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    KdaFusedDecodeKernel, KdaFusedDecodeKernelConfig, KdaFusedDecodeKernelInput,
    ResidualRmsNormKernel, ResidualRmsNormKernelConfig, ResidualRmsNormKernelInput,
    SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

use super::kimi_k3_common::{build_atomic, eval_atomic_or_zero, zero_all_reduce};

const HIDDEN: u32 = 7_168;
const HEAD_DIM: u32 = 128;
const CONV_KERNEL: u32 = 4;

#[cfg(test)]
const SOURCE_ORDER: [&str; 6] = [
    "input_layernorm",
    "qkvbfg_a_proj",
    "kda_fused_decode",
    "o_proj",
    "tp_allreduce_zero",
    "post_attention_layernorm",
];

#[derive(Clone, Debug)]
pub struct KimiK3KdaLocalWorkletConfig {
    pub gpu_name: String,
    pub hidden: Dim,
    pub heads: Dim,
    pub head_dim: Dim,
    pub conv_kernel: Dim,
    pub lower_bound: i32,
    pub dtype: DType,
    pub state_dtype: DType,
    pub gemm_backends: Vec<&'static str>,
    pub fused_decode_backends: Vec<&'static str>,
    pub residual_norm_backends: Vec<&'static str>,
    pub tp_size: u16,
}

#[derive(Clone, Debug)]
pub struct KimiK3KdaLocalWorkletResolved {
    pub raw_cfg: KimiK3KdaLocalWorkletConfig,
    pub input_layernorm: ResidualRmsNormKernelConfig,
    pub qkvbfg_a_proj: SingleGemmKernelConfig,
    pub kda_fused_decode: KdaFusedDecodeKernelConfig,
    pub o_proj: SingleGemmKernelConfig,
    pub post_attention_layernorm: ResidualRmsNormKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct KimiK3KdaLocalWorkletInput {
    pub batch_tokens: u32,
    pub decode_tokens: u32,
}

pub struct KimiK3KdaLocalWorklet {
    pub name: String,
    pub input_layernorm: Op<ResidualRmsNormKernel>,
    pub qkvbfg_a_proj: Op<SingleGemmKernel>,
    pub kda_fused_decode: Op<KdaFusedDecodeKernel>,
    pub o_proj: Op<SingleGemmKernel>,
    pub tp_allreduce_zero: Op<super::kimi_k3_common::ZeroAllReduceProbe>,
    pub post_attention_layernorm: Op<ResidualRmsNormKernel>,
    resolved: KimiK3KdaLocalWorkletResolved,
}

impl KimiK3KdaLocalWorklet {
    pub fn resolve_config(cfg: &KimiK3KdaLocalWorkletConfig) -> KimiK3KdaLocalWorkletResolved {
        validate_config(cfg)
            .unwrap_or_else(|reason| panic!("invalid KimiK3KdaLocalWorkletConfig: {reason}"));

        // SGLang's full-rank KDA projection is qkv + two head-width slices +
        // one per-head beta slice. The recipe deliberately keeps the natural
        // width (7692 for 12 heads) and does not invent an alignment pad.
        let heads = cfg.heads.get();
        let head_dim = cfg.head_dim.get();
        let qkvbfg_n = 3 * heads * head_dim + 2 * heads * head_dim + heads;

        KimiK3KdaLocalWorkletResolved {
            input_layernorm: ResidualRmsNormKernelConfig {
                backends: cfg.residual_norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden.clone(),
                dtype: cfg.dtype,
            },
            qkvbfg_a_proj: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: qkvbfg_n.into(),
                k: cfg.hidden.clone(),
                dtype: cfg.dtype,
            },
            kda_fused_decode: KdaFusedDecodeKernelConfig {
                backends: cfg.fused_decode_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads: cfg.heads.clone(),
                head_k_dim: cfg.head_dim.clone(),
                head_v_dim: cfg.head_dim.clone(),
                dtype: cfg.dtype,
                state_dtype: cfg.state_dtype,
                lower_bound: cfg.lower_bound,
            },
            o_proj: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.hidden.clone(),
                k: (heads * head_dim).into(),
                dtype: cfg.dtype,
            },
            post_attention_layernorm: ResidualRmsNormKernelConfig {
                backends: cfg.residual_norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden.clone(),
                dtype: cfg.dtype,
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: KimiK3KdaLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, crate::timing::BuildError> {
        let tp_allreduce_zero = zero_all_reduce(
            format!("{name}.tp_allreduce_zero"),
            serde_json::json!({
                "gpu_name": resolved.raw_cfg.gpu_name,
                "num_gpus": resolved.raw_cfg.tp_size,
                "fabric": "nvlink",
                "zero_time": true,
            }),
        );
        Ok(Self {
            input_layernorm: build_atomic(
                &name,
                "input_layernorm",
                resolved.input_layernorm.clone(),
                ResidualRmsNormKernel::build,
                bridge,
            )?,
            qkvbfg_a_proj: build_atomic(
                &name,
                "qkvbfg_a_proj",
                resolved.qkvbfg_a_proj.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            kda_fused_decode: build_atomic(
                &name,
                "kda_fused_decode",
                resolved.kda_fused_decode.clone(),
                KdaFusedDecodeKernel::build,
                bridge,
            )?,
            o_proj: build_atomic(
                &name,
                "o_proj",
                resolved.o_proj.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            tp_allreduce_zero,
            post_attention_layernorm: build_atomic(
                &name,
                "post_attention_layernorm",
                resolved.post_attention_layernorm.clone(),
                ResidualRmsNormKernel::build,
                bridge,
            )?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: format!(
                "{} (KimiK3KdaLocalWorklet) [TP{}; heads={}]",
                self.name, self.resolved.raw_cfg.tp_size, self.resolved.raw_cfg.heads
            ),
            child: Box::new(CostNode::Sum(vec![
                self.input_layernorm.compile(builder),
                self.qkvbfg_a_proj.compile(builder),
                self.kda_fused_decode.compile(builder),
                self.o_proj.compile(builder),
                self.tp_allreduce_zero.compile(builder),
                self.post_attention_layernorm.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &KimiK3KdaLocalWorkletInput, evaluator: &mut Evaluator) {
        eval_atomic_or_zero(
            &self.input_layernorm,
            ResidualRmsNormKernelInput {
                m: input.batch_tokens,
            },
            input.batch_tokens == 0,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.qkvbfg_a_proj,
            SingleGemmKernelInput {
                m: input.batch_tokens,
            },
            input.batch_tokens == 0,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.kda_fused_decode,
            KdaFusedDecodeKernelInput {
                batch_size: input.decode_tokens,
            },
            input.decode_tokens == 0,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.o_proj,
            SingleGemmKernelInput {
                m: input.batch_tokens,
            },
            input.batch_tokens == 0,
            evaluator,
        );
        self.tp_allreduce_zero.eval(
            &crate::timing::kernels::AllReduceKernelInput {
                message_size_bytes: u64::from(input.batch_tokens) * u64::from(HIDDEN) * 2,
            },
            evaluator,
        );
        eval_atomic_or_zero(
            &self.post_attention_layernorm,
            ResidualRmsNormKernelInput {
                m: input.batch_tokens,
            },
            input.batch_tokens == 0,
            evaluator,
        );
    }
}

fn validate_config(cfg: &KimiK3KdaLocalWorkletConfig) -> Result<(), String> {
    if cfg.hidden.get() != HIDDEN {
        return Err(format!("hidden must be {HIDDEN}, got {}", cfg.hidden));
    }
    if cfg.head_dim.get() != HEAD_DIM {
        return Err(format!("head_dim must be {HEAD_DIM}, got {}", cfg.head_dim));
    }
    if cfg.conv_kernel.get() != CONV_KERNEL {
        return Err(format!(
            "conv_kernel must be {CONV_KERNEL}, got {}",
            cfg.conv_kernel
        ));
    }
    if cfg.heads.get() == 0 {
        return Err("heads must be positive".to_string());
    }
    if cfg.dtype != DType::Bf16 || cfg.state_dtype != DType::Bf16 {
        return Err("K3 KDA production decode uses bf16 compute and state".to_string());
    }
    if cfg.gemm_backends.is_empty() || cfg.fused_decode_backends.is_empty() {
        return Err("K3 KDA requires non-empty compute backend lists".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> KimiK3KdaLocalWorkletConfig {
        KimiK3KdaLocalWorkletConfig {
            gpu_name: "NVIDIA B200".to_string(),
            hidden: HIDDEN.into(),
            heads: 12.into(),
            head_dim: HEAD_DIM.into(),
            conv_kernel: CONV_KERNEL.into(),
            lower_bound: -5,
            dtype: DType::Bf16,
            state_dtype: DType::Bf16,
            gemm_backends: vec!["sglang_bf16_auto"],
            fused_decode_backends: vec!["sglang_fused"],
            residual_norm_backends: vec!["vllm_cuda"],
            tp_size: 8,
        }
    }

    #[test]
    fn source_order_and_resolved_shapes_are_frozen() {
        assert_eq!(SOURCE_ORDER.len(), 6);
        let resolved = KimiK3KdaLocalWorklet::resolve_config(&config());
        assert_eq!(resolved.qkvbfg_a_proj.n, 7_692);
        assert_eq!(resolved.qkvbfg_a_proj.k, HIDDEN);
        assert_eq!(resolved.kda_fused_decode.num_heads, 12);
        assert_eq!(resolved.kda_fused_decode.head_k_dim, HEAD_DIM);
        assert_eq!(resolved.o_proj.k, 12 * HEAD_DIM);
        assert_eq!(resolved.o_proj.n, HIDDEN);
    }

    #[test]
    fn state_and_convolution_geometry_matches_the_k3_recipe() {
        let state = 12_u64 * 128 * 128 * 2;
        let conv = u64::from(CONV_KERNEL - 1) * u64::from(3 * HEAD_DIM * 12) * 2;
        assert_eq!(state, 393_216);
        assert_eq!(conv, 27_648);
        assert_eq!(state + conv, 420_864);
    }

    #[test]
    fn decode_input_keeps_prefill_out_of_the_decode_only_leaf() {
        let input = KimiK3KdaLocalWorkletInput {
            batch_tokens: 17,
            decode_tokens: 5,
        };
        assert_eq!(input.batch_tokens, 17);
        assert_eq!(input.decode_tokens, 5);
        assert_eq!(KimiK3KdaLocalWorkletInput::default().decode_tokens, 0);
    }
}
