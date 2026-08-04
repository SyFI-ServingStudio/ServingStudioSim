//! Native FP8 AFD post-attention section: quantized row-parallel o_proj,
//! optional pure TP all-reduce, then native FP8 post-norm + router.

use std::sync::Arc;

use crate::common::Fabric;
use crate::op::gemm::{
    SingleFp8GemmWithQuantConfig, SingleFp8GemmWithQuantInput, SingleFp8GemmWithQuantOp,
};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    AllReduceKernel, AllReduceKernelConfig, AllReduceKernelInput,
    Fp8PerTokenGroupQuantKernelConfig, SingleGemmKernelConfig,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

use super::native_fp8_moe_router_local::{
    NativeFp8MoeRouterLocalWorklet, NativeFp8MoeRouterLocalWorkletConfig,
    NativeFp8MoeRouterLocalWorkletInput, NativeFp8MoeRouterLocalWorkletResolved,
};

#[derive(Clone, Debug)]
pub struct Fp8PostAttnRouterTpWorkletConfig {
    pub hidden: Dim,
    pub num_qo_heads: Dim,
    pub head_dim: Dim,
    pub num_experts: Dim,
    pub activation_dtype: DType,
    pub tp_size: u16,
    pub tp_name: &'static str,
    pub allreduce_fabric: Fabric,
    pub gpu_name: String,
    pub norm_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
    pub fp8_quant_backends: Vec<&'static str>,
    pub allreduce_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct Fp8PostAttnRouterTpWorkletResolved {
    pub raw_cfg: Fp8PostAttnRouterTpWorkletConfig,
    pub o_proj: SingleFp8GemmWithQuantConfig,
    pub tp_ar: Option<AllReduceKernelConfig>,
    pub router: NativeFp8MoeRouterLocalWorkletResolved,
    pub num_qo_heads_per_rank: Dim,
}

#[derive(Clone, Debug, Default)]
pub struct Fp8PostAttnRouterTpWorkletInput {
    pub batch_tokens: u32,
}

pub struct Fp8PostAttnRouterTpWorklet {
    pub name: String,
    pub o_proj: SingleFp8GemmWithQuantOp,
    pub tp_ar: Option<Op<AllReduceKernel>>,
    pub router: NativeFp8MoeRouterLocalWorklet,
    resolved: Fp8PostAttnRouterTpWorkletResolved,
}

impl Fp8PostAttnRouterTpWorklet {
    pub fn resolve_config(
        cfg: &Fp8PostAttnRouterTpWorkletConfig,
    ) -> Fp8PostAttnRouterTpWorkletResolved {
        let tp_size = u32::from(cfg.tp_size);
        assert!(tp_size > 0);
        assert_eq!(cfg.num_qo_heads.get() % tp_size, 0);
        let num_qo_heads_per_rank = cfg.num_qo_heads.clone() / Dim::param(cfg.tp_name, tp_size);
        let o_proj_input_width = num_qo_heads_per_rank.clone() * cfg.head_dim.clone();
        Fp8PostAttnRouterTpWorkletResolved {
            o_proj: SingleFp8GemmWithQuantConfig {
                quant: Fp8PerTokenGroupQuantKernelConfig {
                    backends: cfg.fp8_quant_backends.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    hidden_size: o_proj_input_width.clone(),
                    group_size: 128,
                    input_dtype: cfg.activation_dtype,
                    scale_format: "ue8m0_column_major".to_string(),
                },
                gemm: SingleGemmKernelConfig {
                    backends: cfg.gemm_backends.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    n: cfg.hidden.clone(),
                    k: o_proj_input_width,
                    dtype: DType::Fp8E4m3,
                },
            },
            tp_ar: (cfg.tp_size > 1).then(|| AllReduceKernelConfig {
                backends: cfg.allreduce_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_gpus: tp_size,
                fabric: cfg.allreduce_fabric,
            }),
            router: NativeFp8MoeRouterLocalWorklet::resolve_config(
                &NativeFp8MoeRouterLocalWorkletConfig {
                    hidden: cfg.hidden.clone(),
                    num_experts: cfg.num_experts.clone(),
                    activation_dtype: cfg.activation_dtype,
                    gpu_name: cfg.gpu_name.clone(),
                    norm_backends: cfg.norm_backends.clone(),
                    gemm_backends: cfg.gemm_backends.clone(),
                    fp8_quant_backends: cfg.fp8_quant_backends.clone(),
                },
            ),
            num_qo_heads_per_rank,
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: Fp8PostAttnRouterTpWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let o_proj = SingleFp8GemmWithQuantOp::build(
            format!("{name}.o_proj"),
            resolved.o_proj.clone(),
            bridge,
        )?;
        let tp_ar = resolved
            .tp_ar
            .as_ref()
            .map(|config| {
                let allreduce_name = format!("{name}.tp_allreduce");
                AllReduceKernel::build(allreduce_name.clone(), config.clone(), bridge)
                    .map(|kernel| Op::new(allreduce_name, Arc::new(kernel)))
            })
            .transpose()?;
        let router = NativeFp8MoeRouterLocalWorklet::build(
            format!("{name}.moe_router"),
            resolved.router.clone(),
            bridge,
        )?;
        Ok(Self {
            name,
            o_proj,
            tp_ar,
            router,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let mut parts = vec![self.o_proj.compile(builder)];
        if let Some(tp_ar) = &self.tp_ar {
            parts.push(tp_ar.compile(builder));
        }
        parts.push(self.router.compile(builder));
        CostNode::Labeled {
            label: format!(
                "{} (Fp8PostAttnRouterTpWorklet) [tp={}; qo={:?}]",
                self.name, self.resolved.raw_cfg.tp_size, self.resolved.num_qo_heads_per_rank
            ),
            child: Box::new(CostNode::Sum(parts)),
        }
    }

    pub fn eval(&self, input: &Fp8PostAttnRouterTpWorkletInput, ev: &mut Evaluator) {
        let batch_tokens = input.batch_tokens;
        self.o_proj.eval(
            &SingleFp8GemmWithQuantInput {
                num_tokens: batch_tokens,
            },
            ev,
        );
        if let Some(tp_ar) = &self.tp_ar {
            tp_ar.eval(
                &AllReduceKernelInput {
                    message_size_bytes: u64::from(batch_tokens)
                        * u64::from(self.resolved.raw_cfg.hidden.get())
                        * u64::from(DType::Fp8E4m3.size_bytes()),
                },
                ev,
            );
        }
        self.router
            .eval(&NativeFp8MoeRouterLocalWorkletInput { batch_tokens }, ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_only_fp8_o_proj_and_router() {
        let resolved =
            Fp8PostAttnRouterTpWorklet::resolve_config(&Fp8PostAttnRouterTpWorkletConfig {
                hidden: 4096.into(),
                num_qo_heads: 32.into(),
                head_dim: 128.into(),
                num_experts: 128.into(),
                activation_dtype: DType::Bf16,
                tp_size: 4,
                tp_name: "tp",
                allreduce_fabric: Fabric::Nvlink,
                gpu_name: "H100".into(),
                norm_backends: vec!["flashinfer"],
                gemm_backends: vec!["deepgemm"],
                fp8_quant_backends: vec!["vllm_cuda"],
                allreduce_backends: vec!["nvshmem"],
            });
        assert_eq!(resolved.o_proj.gemm.dtype, DType::Fp8E4m3);
        assert_eq!(resolved.router.router.gemm.dtype, DType::Fp8E4m3);
        assert!(resolved.tp_ar.is_some());
    }
}
