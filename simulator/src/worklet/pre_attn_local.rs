//! `PreAttnLocalWorklet` — single-GPU pre-attention section of a dense decoder
//! layer: input RMSNorm → fused QKV projection. `Local` group suffix (L3 §1.5):
//! one GPU, self-synced, no collective.
//!
//! `gpu_name` rides in `*Config` (baked into each sub-kernel cfg at
//! `resolve_config`); `*Input` is pure shape (supersedes L3 INV-13, see the
//! `gpu_name in *KernelConfig` decision).

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    RmsNormKernel, RmsNormKernelConfig, RmsNormKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

/// Raw global config (no partition: `Local` = 1 GPU). Backend strings pass
/// straight through to L1 (config-level polymorphism, L3 §1.6).
#[derive(Clone, Debug)]
pub struct PreAttnLocalWorkletConfig {
    pub hidden: Dim,
    pub num_qo_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    pub dtype: DType,
    pub gpu_name: String,
    pub norm_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
}

/// Post-resolve: sub-kernel cfgs fully baked. Consumed by `build`.
/// `raw_cfg` kept for the partition pre-image.
#[derive(Clone, Debug)]
pub struct PreAttnLocalWorkletResolved {
    pub raw_cfg: PreAttnLocalWorkletConfig,
    pub input_norm: RmsNormKernelConfig,
    pub qkv: SingleGemmKernelConfig,
}

/// Runtime per-call shape (no `gpu_name` — that lives in the config).
#[derive(Clone, Debug)]
pub struct PreAttnLocalWorkletInput {
    pub batch_tokens: u32,
}

pub struct PreAttnLocalWorklet {
    pub name: String,
    pub input_norm: Op<RmsNormKernel>,
    pub qkv: Op<SingleGemmKernel>,
    resolved: PreAttnLocalWorkletResolved,
}

impl PreAttnLocalWorklet {
    pub fn resolve_config(cfg: &PreAttnLocalWorkletConfig) -> PreAttnLocalWorkletResolved {
        // Fused QKV output dim: q heads + 2× kv heads (GQA), each `head_dim` wide.
        let qkv_n =
            (cfg.num_qo_heads.clone() + 2 * cfg.num_kv_heads.clone()) * cfg.head_dim.clone();
        PreAttnLocalWorkletResolved {
            input_norm: RmsNormKernelConfig {
                backends: cfg.norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden.clone(),
                dtype: cfg.dtype,
            },
            qkv: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: qkv_n,
                k: cfg.hidden.clone(),
                dtype: cfg.dtype,
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: PreAttnLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let norm_name = format!("{name}.input_norm");
        let qkv_name = format!("{name}.qkv_proj");
        let input_norm = Op::new(
            norm_name.clone(),
            Arc::new(RmsNormKernel::build(
                norm_name,
                resolved.input_norm.clone(),
                bridge,
            )?),
        );
        let qkv = Op::new(
            qkv_name.clone(),
            Arc::new(SingleGemmKernel::build(
                qkv_name,
                resolved.qkv.clone(),
                bridge,
            )?),
        );
        Ok(Self {
            name,
            input_norm,
            qkv,
            resolved,
        })
    }

    /// CostTree compile: sum over the two atomic ops, wrapped in a `Labeled` node
    /// carrying the worklet identity + partition/shape annotation (the old
    /// `Describe` header lines).
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let label = format!(
            "{} (PreAttnLocalWorklet) [local (1 GPU); qkv n={}, k={}]",
            self.name, self.resolved.qkv.n, self.resolved.qkv.k
        );
        CostNode::Labeled {
            label,
            child: Box::new(CostNode::Sum(vec![
                self.input_norm.compile(builder),
                self.qkv.compile(builder),
            ])),
        }
    }

    /// CostTree eval: fill the input_norm then qkv slots in the same child order
    /// as `compile`, so `cursor` tracks the minted slot indices.
    pub fn eval(&self, input: &PreAttnLocalWorkletInput, ev: &mut Evaluator) {
        let norm_in = RmsNormKernelInput {
            m: input.batch_tokens,
        };
        let gemm_in = SingleGemmKernelInput {
            m: input.batch_tokens,
        };
        self.input_norm.eval(&norm_in, ev);
        self.qkv.eval(&gemm_in, ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> PreAttnLocalWorkletConfig {
        PreAttnLocalWorkletConfig {
            hidden: 4096.into(),
            num_qo_heads: 32.into(),
            num_kv_heads: 8.into(),
            head_dim: 128.into(),
            dtype: DType::Bf16,
            gpu_name: "H100".to_string(),
            norm_backends: vec!["flashinfer"],
            gemm_backends: vec!["torch"],
        }
    }

    #[test]
    fn resolve_bakes_fused_qkv_shape() {
        let r = PreAttnLocalWorklet::resolve_config(&cfg());
        // (32 + 2·8) · 128 = 48 · 128 = 6144
        assert_eq!(r.qkv.n, 6144);
        assert_eq!(r.qkv.k, 4096);
        assert_eq!(r.input_norm.hidden, 4096);
        // gpu_name baked into every sub-kernel cfg.
        assert_eq!(r.qkv.gpu_name, "H100");
        assert_eq!(r.input_norm.gpu_name, "H100");
    }
}
