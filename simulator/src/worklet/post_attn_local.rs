//! `PostAttnLocalWorklet` — single-GPU post-attention section of a dense decoder
//! layer: o_proj → post-attn RMSNorm → MLP (fused up_gate GEMM → SwiGLU
//! activation → down GEMM). `Local` group suffix (L3 §1.5): 1 GPU, no
//! collective.
//!
//! SwiGLU activation is modeled with the byte-keyed `ElementwiseKernel`: the
//! fused up_gate GEMM emits `2·intermediate` elements/token (gate‖up concat),
//! the activation reads those and writes `intermediate` elements/token. So
//! `input_bytes_per_token = 2·intermediate·dtype_bytes`,
//! `output_bytes_per_token = intermediate·dtype_bytes`.
//!
//! `gpu_name` rides in `*Config`; `*Input` is pure shape.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, RmsNormKernel,
    RmsNormKernelConfig, RmsNormKernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge};

#[derive(Clone, Debug)]
pub struct PostAttnLocalWorkletConfig {
    pub hidden: u32,
    pub intermediate: u32,
    pub num_qo_heads: u32,
    pub head_dim: u32,
    pub dtype: DType,
    pub gpu_name: String,
    pub norm_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
    pub act_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct PostAttnLocalWorkletResolved {
    pub raw_cfg: PostAttnLocalWorkletConfig,
    pub o_proj: SingleGemmKernelConfig,
    pub post_norm: RmsNormKernelConfig,
    pub up_gate: SingleGemmKernelConfig,
    pub act: ElementwiseKernelConfig,
    pub down: SingleGemmKernelConfig,
}

#[derive(Clone, Debug)]
pub struct PostAttnLocalWorkletInput {
    pub batch_tokens: u32,
}

pub struct PostAttnLocalWorklet {
    pub name: String,
    pub o_proj: Op<SingleGemmKernel>,
    pub post_norm: Op<RmsNormKernel>,
    pub up_gate: Op<SingleGemmKernel>,
    pub act: Op<ElementwiseKernel>,
    pub down: Op<SingleGemmKernel>,
    resolved: PostAttnLocalWorkletResolved,
}

impl PostAttnLocalWorklet {
    pub fn resolve_config(cfg: &PostAttnLocalWorkletConfig) -> PostAttnLocalWorkletResolved {
        let dtype_bytes = cfg.dtype.size_bytes();
        PostAttnLocalWorkletResolved {
            o_proj: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.hidden,
                k: cfg.num_qo_heads * cfg.head_dim,
                dtype: cfg.dtype,
            },
            post_norm: RmsNormKernelConfig {
                backends: cfg.norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden,
                dtype: cfg.dtype,
            },
            up_gate: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: 2 * cfg.intermediate, // gate ‖ up concat
                k: cfg.hidden,
                dtype: cfg.dtype,
            },
            act: ElementwiseKernelConfig {
                backends: cfg.act_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: 2 * cfg.intermediate * dtype_bytes,
                output_bytes_per_token: cfg.intermediate * dtype_bytes,
            },
            down: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.hidden,
                k: cfg.intermediate,
                dtype: cfg.dtype,
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: PostAttnLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let o_name = format!("{name}.o_proj");
        let pn_name = format!("{name}.post_norm");
        let ug_name = format!("{name}.up_gate_proj");
        let act_name = format!("{name}.activation");
        let down_name = format!("{name}.down_proj");

        let o_proj = Op::new(
            o_name.clone(),
            Arc::new(SingleGemmKernel::build(o_name, resolved.o_proj.clone(), bridge)?),
        );
        let post_norm = Op::new(
            pn_name.clone(),
            Arc::new(RmsNormKernel::build(pn_name, resolved.post_norm.clone(), bridge)?),
        );
        let up_gate = Op::new(
            ug_name.clone(),
            Arc::new(SingleGemmKernel::build(ug_name, resolved.up_gate.clone(), bridge)?),
        );
        let act = Op::new(
            act_name.clone(),
            Arc::new(ElementwiseKernel::build(act_name, resolved.act.clone(), bridge)?),
        );
        let down = Op::new(
            down_name.clone(),
            Arc::new(SingleGemmKernel::build(down_name, resolved.down.clone(), bridge)?),
        );
        Ok(Self {
            name,
            o_proj,
            post_norm,
            up_gate,
            act,
            down,
            resolved,
        })
    }

    /// CostTree compile: sum over the five atomic ops (o_proj, post_norm,
    /// up_gate, act, down), wrapped in a `Labeled` node with the worklet identity
    /// + partition/shape annotation (the old `Describe` header lines).
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let label = format!(
            "{} (PostAttnLocalWorklet) [local (1 GPU); up_gate n={}, down k={}]",
            self.name, self.resolved.up_gate.n, self.resolved.down.k
        );
        CostNode::Labeled {
            label,
            child: Box::new(CostNode::Sum(vec![
                self.o_proj.compile(builder),
                self.post_norm.compile(builder),
                self.up_gate.compile(builder),
                self.act.compile(builder),
                self.down.compile(builder),
            ])),
        }
    }

    /// CostTree eval: fill o_proj, post_norm, up_gate, act, down slots in that
    /// order — mirrors `compile`/`lookup` so `cursor` tracks the minted slots.
    pub fn eval(&self, input: &PostAttnLocalWorkletInput, ev: &mut Evaluator) {
        let m = input.batch_tokens;
        let gemm_in = SingleGemmKernelInput { m };
        let norm_in = RmsNormKernelInput { m };
        let act_in = ElementwiseKernelInput { num_tokens: m };
        self.o_proj.eval(&gemm_in, ev);
        self.post_norm.eval(&norm_in, ev);
        self.up_gate.eval(&gemm_in, ev);
        self.act.eval(&act_in, ev);
        self.down.eval(&gemm_in, ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> PostAttnLocalWorkletConfig {
        PostAttnLocalWorkletConfig {
            hidden: 4096,
            intermediate: 14336,
            num_qo_heads: 32,
            head_dim: 128,
            dtype: DType::Bf16,
            gpu_name: "H100".to_string(),
            norm_backends: vec!["flashinfer"],
            gemm_backends: vec!["torch"],
            act_backends: vec!["triton"],
        }
    }

    #[test]
    fn resolve_bakes_mlp_and_oproj_shapes() {
        let r = PostAttnLocalWorklet::resolve_config(&cfg());
        assert_eq!(r.o_proj.n, 4096);
        assert_eq!(r.o_proj.k, 4096); // 32 · 128
        assert_eq!(r.up_gate.n, 28672); // 2 · 14336
        assert_eq!(r.up_gate.k, 4096);
        assert_eq!(r.down.n, 4096);
        assert_eq!(r.down.k, 14336);
        // bf16 = 2 bytes/elem.
        assert_eq!(r.act.input_bytes_per_token, 2 * 14336 * 2);
        assert_eq!(r.act.output_bytes_per_token, 14336 * 2);
    }
}
