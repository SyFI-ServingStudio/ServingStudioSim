//! Native FP8 AFD pre-attention projection: BF16 input `RMSNorm` followed by
//! per-token-group quantization and a column-parallel FP8 fused-QKV GEMM.

use std::sync::Arc;

use crate::op::gemm::{
    SingleFp8GemmWithQuantConfig, SingleFp8GemmWithQuantInput, SingleFp8GemmWithQuantOp,
};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    Fp8PerTokenGroupQuantKernelConfig, RmsNormKernel, RmsNormKernelConfig, RmsNormKernelInput,
    SingleGemmKernelConfig,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

#[derive(Clone, Debug)]
pub struct Fp8PreAttnProjTpWorkletConfig {
    pub hidden: Dim,
    pub num_qo_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    pub activation_dtype: DType,
    pub tp_size: u16,
    pub tp_name: &'static str,
    pub gpu_name: String,
    pub norm_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
    pub fp8_quant_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct Fp8PreAttnProjTpWorkletResolved {
    pub raw_cfg: Fp8PreAttnProjTpWorkletConfig,
    pub input_norm: RmsNormKernelConfig,
    pub qkv: SingleFp8GemmWithQuantConfig,
    pub num_qo_heads_per_rank: Dim,
    pub num_kv_heads_per_rank: Dim,
}

#[derive(Clone, Debug, Default)]
pub struct Fp8PreAttnProjTpWorkletInput {
    pub batch_tokens: u32,
}

pub struct Fp8PreAttnProjTpWorklet {
    pub name: String,
    pub input_norm: Op<RmsNormKernel>,
    pub qkv: SingleFp8GemmWithQuantOp,
    resolved: Fp8PreAttnProjTpWorkletResolved,
}

impl Fp8PreAttnProjTpWorklet {
    #[must_use]
    pub fn resolve_config(cfg: &Fp8PreAttnProjTpWorkletConfig) -> Fp8PreAttnProjTpWorkletResolved {
        let tp_size = u32::from(cfg.tp_size);
        assert!(tp_size > 0);
        assert_eq!(cfg.num_qo_heads.get() % tp_size, 0);
        assert_eq!(cfg.num_kv_heads.get() % tp_size, 0);
        assert!(tp_size <= cfg.num_kv_heads.get());
        let tp_dim = Dim::param(cfg.tp_name, tp_size);
        let num_qo_heads_per_rank = cfg.num_qo_heads.clone() / tp_dim.clone();
        let num_kv_heads_per_rank = cfg.num_kv_heads.clone() / tp_dim;
        Fp8PreAttnProjTpWorkletResolved {
            input_norm: RmsNormKernelConfig {
                backends: cfg.norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden.clone(),
                dtype: cfg.activation_dtype,
            },
            qkv: SingleFp8GemmWithQuantConfig {
                quant: Fp8PerTokenGroupQuantKernelConfig {
                    backends: cfg.fp8_quant_backends.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    hidden_size: cfg.hidden.clone(),
                    group_size: 128,
                    input_dtype: cfg.activation_dtype,
                    scale_format: "ue8m0_column_major".to_string(),
                },
                gemm: SingleGemmKernelConfig {
                    backends: cfg.gemm_backends.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    n: (num_qo_heads_per_rank.clone() + 2 * num_kv_heads_per_rank.clone())
                        * cfg.head_dim.clone(),
                    k: cfg.hidden.clone(),
                    dtype: DType::Fp8E4m3,
                },
            },
            num_qo_heads_per_rank,
            num_kv_heads_per_rank,
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: Fp8PreAttnProjTpWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let norm_name = format!("{name}.input_norm");
        let input_norm = Op::new(
            norm_name.clone(),
            Arc::new(RmsNormKernel::build(
                norm_name,
                resolved.input_norm.clone(),
                bridge,
            )?),
        );
        let qkv = SingleFp8GemmWithQuantOp::build(
            format!("{name}.qkv_proj"),
            resolved.qkv.clone(),
            bridge,
        )?;
        Ok(Self {
            name,
            input_norm,
            qkv,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: format!(
                "{} (Fp8PreAttnProjTpWorklet) [tp={}; qo {:?}, kv {:?}]",
                self.name,
                self.resolved.raw_cfg.tp_size,
                self.resolved.num_qo_heads_per_rank,
                self.resolved.num_kv_heads_per_rank
            ),
            child: Box::new(CostNode::Sum(vec![
                self.input_norm.compile(builder),
                self.qkv.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &Fp8PreAttnProjTpWorkletInput, ev: &mut Evaluator) {
        self.input_norm.eval(
            &RmsNormKernelInput {
                m: input.batch_tokens,
            },
            ev,
        );
        self.qkv.eval(
            &SingleFp8GemmWithQuantInput {
                num_tokens: input.batch_tokens,
            },
            ev,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_only_fp8_qkv() {
        let resolved = Fp8PreAttnProjTpWorklet::resolve_config(&Fp8PreAttnProjTpWorkletConfig {
            hidden: 4096.into(),
            num_qo_heads: 32.into(),
            num_kv_heads: 8.into(),
            head_dim: 128.into(),
            activation_dtype: DType::Bf16,
            tp_size: 4,
            tp_name: "tp",
            gpu_name: "H100".into(),
            norm_backends: vec!["flashinfer"],
            gemm_backends: vec!["deepgemm"],
            fp8_quant_backends: vec!["vllm_cuda"],
        });
        assert_eq!(resolved.qkv.gemm.dtype, DType::Fp8E4m3);
        assert_eq!(resolved.qkv.gemm.n, 1536);
    }
}
