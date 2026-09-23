//! Kimi-K3 rank-local KDA decode worklet.
//!
//! SGLang uses the fused KDA launch for an FP32 recurrent state. Its cookbook
//! BF16-state path launches the causal-convolution update, recurrent update, and
//! gated RMS norm as three sequential Triton kernels. Prefill is present in the
//! input contract, but this decode-first recipe leaves the decode leaves at zero
//! for prefill because no K3 prefill profile is registered here.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    GdnCausalConvDecodeKernel, GdnCausalConvDecodeKernelConfig, GdnCausalConvDecodeKernelInput,
    GdnGatedRmsNormKernel, GdnGatedRmsNormKernelConfig, GdnGatedRmsNormKernelInput,
    KdaFusedDecodeKernel, KdaFusedDecodeKernelConfig, KdaFusedDecodeKernelInput,
    KdaRecurrentDecodeKernel, KdaRecurrentDecodeKernelConfig, KdaRecurrentDecodeKernelInput,
    ResidualRmsNormKernel, ResidualRmsNormKernelConfig, ResidualRmsNormKernelInput,
    SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

use super::kimi_k3_common::{build_atomic, eval_atomic_or_zero, zero_all_reduce};

const HIDDEN: u32 = 7_168;
const HEAD_DIM: u32 = 128;
const CONV_KERNEL: u32 = 4;

#[cfg(test)]
const FUSED_SOURCE_ORDER: [&str; 7] = [
    "input_layernorm",
    "qkvbfg_a_proj",
    "qkvbfg_a_proj_bfa",
    "kda_fused_decode",
    "o_proj",
    "tp_allreduce_zero",
    "post_attention_layernorm",
];

#[cfg(test)]
const SPLIT_SOURCE_ORDER: [&str; 9] = [
    "input_layernorm",
    "qkvbfg_a_proj",
    "qkvbfg_a_proj_bfa",
    "kda_conv_decode",
    "kda_recurrent_decode",
    "kda_gated_norm",
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
    pub kda_state_dtype: DType,
    pub gemm_backends: Vec<&'static str>,
    pub fused_decode_backends: Vec<&'static str>,
    pub causal_conv_decode_backends: Vec<&'static str>,
    pub recurrent_decode_backends: Vec<&'static str>,
    pub gated_norm_backends: Vec<&'static str>,
    pub residual_norm_backends: Vec<&'static str>,
    pub tp_size: u16,
}

#[derive(Clone, Debug)]
pub struct KimiK3KdaLocalWorkletResolved {
    pub raw_cfg: KimiK3KdaLocalWorkletConfig,
    pub input_layernorm: ResidualRmsNormKernelConfig,
    pub qkvbfg_a_proj: SingleGemmKernelConfig,
    pub qkvbfg_a_proj_bfa: SingleGemmKernelConfig,
    pub kda_fused_decode: Option<KdaFusedDecodeKernelConfig>,
    pub kda_conv_decode: Option<GdnCausalConvDecodeKernelConfig>,
    pub kda_recurrent_decode: Option<KdaRecurrentDecodeKernelConfig>,
    pub kda_gated_norm: Option<GdnGatedRmsNormKernelConfig>,
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
    pub qkvbfg_a_proj_bfa: Op<SingleGemmKernel>,
    pub kda_fused_decode: Option<Op<KdaFusedDecodeKernel>>,
    pub kda_conv_decode: Option<Op<GdnCausalConvDecodeKernel>>,
    pub kda_recurrent_decode: Option<Op<KdaRecurrentDecodeKernel>>,
    pub kda_gated_norm: Option<Op<GdnGatedRmsNormKernel>>,
    pub o_proj: Op<SingleGemmKernel>,
    pub tp_allreduce_zero: Op<super::kimi_k3_common::ZeroAllReduceProbe>,
    pub post_attention_layernorm: Op<ResidualRmsNormKernel>,
    resolved: KimiK3KdaLocalWorkletResolved,
}

impl KimiK3KdaLocalWorklet {
    pub fn resolve_config(cfg: &KimiK3KdaLocalWorkletConfig) -> KimiK3KdaLocalWorkletResolved {
        validate_config(cfg)
            .unwrap_or_else(|reason| panic!("invalid KimiK3KdaLocalWorkletConfig: {reason}"));

        // SGLang emits q/k/v/g as one 4*projection_size GEMM (6144 columns for
        // 12 heads), while the [f_a|b] GEMV is issued on the alternate stream.
        // The latter is padded to the kernel's 8-column alignment (128+12 ->
        // 144), so the two production shapes must remain separate leaves.
        let heads = cfg.heads.get();
        let head_dim = cfg.head_dim.get();
        let projection_size = heads * head_dim;
        let qkvbfg_n = 4 * projection_size;
        let qkvbfg_bfa_n = (head_dim + heads + 7) / 8 * 8;

        let kda_fused_decode =
            (cfg.kda_state_dtype == DType::Fp32).then(|| KdaFusedDecodeKernelConfig {
                backends: cfg.fused_decode_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads: cfg.heads.clone(),
                head_k_dim: cfg.head_dim.clone(),
                head_v_dim: cfg.head_dim.clone(),
                dtype: cfg.dtype,
                state_dtype: cfg.kda_state_dtype,
                lower_bound: cfg.lower_bound,
            });
        let kda_conv_decode =
            (cfg.kda_state_dtype == DType::Bf16).then(|| GdnCausalConvDecodeKernelConfig {
                backends: cfg.causal_conv_decode_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                channels: (3 * heads * head_dim).into(),
                kernel_size: cfg.conv_kernel.clone(),
                dtype: cfg.dtype,
                state_dtype: DType::Bf16,
            });
        let kda_recurrent_decode =
            (cfg.kda_state_dtype == DType::Bf16).then(|| KdaRecurrentDecodeKernelConfig {
                backends: cfg.recurrent_decode_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads: cfg.heads.clone(),
                head_k_dim: cfg.head_dim.clone(),
                head_v_dim: cfg.head_dim.clone(),
                dtype: cfg.dtype,
                state_dtype: cfg.kda_state_dtype,
                lower_bound: cfg.lower_bound,
            });
        let kda_gated_norm =
            (cfg.kda_state_dtype == DType::Bf16).then(|| GdnGatedRmsNormKernelConfig {
                backends: cfg.gated_norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.head_dim.clone(),
                dtype: cfg.dtype,
            });

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
            qkvbfg_a_proj_bfa: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: qkvbfg_bfa_n.into(),
                k: cfg.hidden.clone(),
                dtype: cfg.dtype,
            },
            kda_fused_decode,
            kda_conv_decode,
            kda_recurrent_decode,
            kda_gated_norm,
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
        let kda_fused_decode = resolved
            .kda_fused_decode
            .clone()
            .map(|config| {
                build_atomic(
                    &name,
                    "kda_fused_decode",
                    config,
                    KdaFusedDecodeKernel::build,
                    bridge,
                )
            })
            .transpose()?;
        let kda_conv_decode = resolved
            .kda_conv_decode
            .clone()
            .map(|config| {
                build_atomic(
                    &name,
                    "kda_conv_decode",
                    config,
                    GdnCausalConvDecodeKernel::build,
                    bridge,
                )
            })
            .transpose()?;
        let kda_recurrent_decode = resolved
            .kda_recurrent_decode
            .clone()
            .map(|config| {
                build_atomic(
                    &name,
                    "kda_recurrent_decode",
                    config,
                    KdaRecurrentDecodeKernel::build,
                    bridge,
                )
            })
            .transpose()?;
        let kda_gated_norm = resolved
            .kda_gated_norm
            .clone()
            .map(|config| {
                build_atomic(
                    &name,
                    "kda_gated_norm",
                    config,
                    GdnGatedRmsNormKernel::build,
                    bridge,
                )
            })
            .transpose()?;
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
            qkvbfg_a_proj_bfa: build_atomic(
                &name,
                "qkvbfg_a_proj_bfa",
                resolved.qkvbfg_a_proj_bfa.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            kda_fused_decode,
            kda_conv_decode,
            kda_recurrent_decode,
            kda_gated_norm,
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
        let input_layernorm = self.input_layernorm.compile(builder);
        let qkvbfg_a_proj = self.qkvbfg_a_proj.compile(builder);
        let qkvbfg_a_proj_bfa = self.qkvbfg_a_proj_bfa.compile(builder);
        let qkvbfg = CostNode::Labeled {
            label: format!(
                "{}.qkvbfg [wide qkvg + alternate-stream bfa GEMV]",
                self.name
            ),
            child: Box::new(CostNode::Max {
                overlap: 1.0,
                children: vec![qkvbfg_a_proj, qkvbfg_a_proj_bfa],
            }),
        };
        let mut children = vec![input_layernorm, qkvbfg];
        if let Some(kda_fused_decode) = &self.kda_fused_decode {
            children.push(kda_fused_decode.compile(builder));
        } else {
            children.push(
                self.kda_conv_decode
                    .as_ref()
                    .expect("BF16 KDA mode must build causal-conv decode")
                    .compile(builder),
            );
            children.push(
                self.kda_recurrent_decode
                    .as_ref()
                    .expect("BF16 KDA mode must build recurrent decode")
                    .compile(builder),
            );
            children.push(
                self.kda_gated_norm
                    .as_ref()
                    .expect("BF16 KDA mode must build gated norm")
                    .compile(builder),
            );
        }
        children.extend([
            self.o_proj.compile(builder),
            self.tp_allreduce_zero.compile(builder),
            self.post_attention_layernorm.compile(builder),
        ]);
        CostNode::Labeled {
            label: format!(
                "{} (KimiK3KdaLocalWorklet) [TP{}; heads={}]",
                self.name, self.resolved.raw_cfg.tp_size, self.resolved.raw_cfg.heads
            ),
            child: Box::new(CostNode::Sum(children)),
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
            &self.qkvbfg_a_proj_bfa,
            SingleGemmKernelInput {
                m: input.batch_tokens,
            },
            input.batch_tokens == 0,
            evaluator,
        );
        if let Some(kda_fused_decode) = &self.kda_fused_decode {
            eval_atomic_or_zero(
                kda_fused_decode,
                KdaFusedDecodeKernelInput {
                    batch_size: input.decode_tokens,
                },
                input.decode_tokens == 0,
                evaluator,
            );
        } else {
            eval_atomic_or_zero(
                self.kda_conv_decode
                    .as_ref()
                    .expect("BF16 KDA mode must build causal-conv decode"),
                GdnCausalConvDecodeKernelInput {
                    batch_size: input.decode_tokens,
                },
                input.decode_tokens == 0,
                evaluator,
            );
            eval_atomic_or_zero(
                self.kda_recurrent_decode
                    .as_ref()
                    .expect("BF16 KDA mode must build recurrent decode"),
                KdaRecurrentDecodeKernelInput {
                    batch_size: input.decode_tokens,
                },
                input.decode_tokens == 0,
                evaluator,
            );
            let m = input
                .decode_tokens
                .checked_mul(self.resolved.raw_cfg.heads.get())
                .expect("KDA gated norm row count must fit u32");
            eval_atomic_or_zero(
                self.kda_gated_norm
                    .as_ref()
                    .expect("BF16 KDA mode must build gated norm"),
                GdnGatedRmsNormKernelInput { m },
                input.decode_tokens == 0,
                evaluator,
            );
        }
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
    if cfg.dtype != DType::Bf16 {
        return Err("K3 KDA production decode uses bf16 compute".to_string());
    }
    if !matches!(cfg.kda_state_dtype, DType::Bf16 | DType::Fp32) {
        return Err("K3 KDA recurrent state must use bf16 or fp32".to_string());
    }
    if cfg.gemm_backends.is_empty() {
        return Err("K3 KDA requires a non-empty GEMM backend list".to_string());
    }
    if cfg.kda_state_dtype == DType::Fp32 && cfg.fused_decode_backends.is_empty() {
        return Err("K3 KDA FP32 state requires a fused decode backend".to_string());
    }
    if cfg.kda_state_dtype == DType::Bf16
        && (cfg.causal_conv_decode_backends.is_empty()
            || cfg.recurrent_decode_backends.is_empty()
            || cfg.gated_norm_backends.is_empty())
    {
        return Err("K3 KDA BF16 state requires split decode backends".to_string());
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
            kda_state_dtype: DType::Bf16,
            gemm_backends: vec!["sglang_bf16_auto"],
            fused_decode_backends: vec!["sglang_fused"],
            causal_conv_decode_backends: vec!["sglang_triton"],
            recurrent_decode_backends: vec!["sglang_triton"],
            gated_norm_backends: vec!["sglang_triton"],
            residual_norm_backends: vec!["vllm_cuda"],
            tp_size: 8,
        }
    }

    #[test]
    fn source_order_and_resolved_shapes_are_frozen() {
        let resolved = KimiK3KdaLocalWorklet::resolve_config(&config());
        assert_eq!(resolved.qkvbfg_a_proj.n, 4 * 12 * HEAD_DIM);
        assert_eq!(resolved.qkvbfg_a_proj.k, HIDDEN);
        assert_eq!(resolved.qkvbfg_a_proj_bfa.n, 144);
        assert_eq!(resolved.qkvbfg_a_proj_bfa.k, HIDDEN);
        assert_eq!(SPLIT_SOURCE_ORDER.len(), 9);
        assert!(resolved.kda_fused_decode.is_none());
        let conv = resolved.kda_conv_decode.as_ref().unwrap();
        assert_eq!(conv.channels, 3 * 12 * HEAD_DIM);
        assert_eq!(conv.state_dtype, DType::Bf16);
        let recurrent = resolved.kda_recurrent_decode.as_ref().unwrap();
        assert_eq!(recurrent.num_heads, 12);
        assert_eq!(recurrent.head_k_dim, HEAD_DIM);
        assert_eq!(recurrent.state_dtype, DType::Bf16);
        assert_eq!(resolved.kda_gated_norm.as_ref().unwrap().hidden, HEAD_DIM);
        assert_eq!(resolved.o_proj.k, 12 * HEAD_DIM);
        assert_eq!(resolved.o_proj.n, HIDDEN);

        let mut fp32 = config();
        fp32.kda_state_dtype = DType::Fp32;
        let resolved = KimiK3KdaLocalWorklet::resolve_config(&fp32);
        assert_eq!(FUSED_SOURCE_ORDER.len(), 7);
        assert!(resolved.kda_conv_decode.is_none());
        assert!(resolved.kda_recurrent_decode.is_none());
        assert!(resolved.kda_gated_norm.is_none());
        let fused = resolved.kda_fused_decode.as_ref().unwrap();
        assert_eq!(fused.num_heads, 12);
        assert_eq!(fused.head_k_dim, HEAD_DIM);
        assert_eq!(fused.state_dtype, DType::Fp32);
    }

    #[test]
    fn state_and_convolution_geometry_matches_the_k3_recipe() {
        let state = 12_u64 * 128 * 128 * DType::Bf16.size_bytes() as u64;
        let conv = u64::from(CONV_KERNEL - 1) * u64::from(3 * HEAD_DIM * 12) * 2;
        assert_eq!(state, 393_216);
        assert_eq!(conv, 27_648);
        assert_eq!(state + conv, 420_864);
        assert_eq!(
            12_u64 * 128 * 128 * DType::Fp32.size_bytes() as u64 + conv,
            814_080
        );
    }

    #[test]
    fn split_and_fused_leaf_names_are_frozen() {
        let build_names = |cfg: KimiK3KdaLocalWorkletConfig| {
            let bridge = PerfApiBridge::new_uninit_for_test();
            bridge.enable_enumerate();
            let worklet = KimiK3KdaLocalWorklet::build(
                "model.kda".into(),
                KimiK3KdaLocalWorklet::resolve_config(&cfg),
                &bridge,
            )
            .unwrap();
            let mut builder = CostTreeBuilder::new();
            let root = worklet.compile(&mut builder);
            let tree = builder.finish(root);
            tree.slots
                .iter()
                .map(|slot| (slot.name.clone(), slot.kind.clone()))
                .collect::<Vec<_>>()
        };

        let split = build_names(config());
        assert_eq!(
            split
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            [
                "model.kda.input_layernorm",
                "model.kda.qkvbfg_a_proj",
                "model.kda.qkvbfg_a_proj_bfa",
                "model.kda.kda_conv_decode",
                "model.kda.kda_recurrent_decode",
                "model.kda.kda_gated_norm",
                "model.kda.o_proj",
                "model.kda.tp_allreduce_zero",
                "model.kda.post_attention_layernorm",
            ]
        );
        assert_eq!(split[3].1, "gdn_causal_conv_decode");
        assert_eq!(split[4].1, "kda_recurrent_decode");
        assert_eq!(split[5].1, "gdn_gated_rms_norm");

        let mut fp32 = config();
        fp32.kda_state_dtype = DType::Fp32;
        let fused = build_names(fp32);
        assert_eq!(
            fused
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            [
                "model.kda.input_layernorm",
                "model.kda.qkvbfg_a_proj",
                "model.kda.qkvbfg_a_proj_bfa",
                "model.kda.kda_fused_decode",
                "model.kda.o_proj",
                "model.kda.tp_allreduce_zero",
                "model.kda.post_attention_layernorm",
            ]
        );
        assert_eq!(fused[3].1, "kda_fused_decode");
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
