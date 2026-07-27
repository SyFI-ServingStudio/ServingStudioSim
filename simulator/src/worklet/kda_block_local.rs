//! `KdaBlockLocalWorklet` — one KDA (Kimi Delta Attention, gated linear
//! attention) decoder-layer block on ONE GPU: input RMSNorm → fused q/k/v
//! projection → gate/beta projection → short-conv + gating (elementwise
//! placeholder) → `kda_scan` (the chunked delta-rule scan kernel) → out
//! projection. `Local` group suffix (L3 §1.5): one sync section, single GPU,
//! no collective — the arch replicates it per attention-DP shard (attn_tp=1).
//!
//! GEMM shapes (Kimi-K3 dims in parens, `d_inner = num_heads·head_dim = 12288`):
//!   - `qkv_proj`  : hidden → 3·d_inner            (7168 → 36864)
//!   - `gate_proj` : hidden → d_inner + num_heads   (7168 → 12384; output gate
//!     per channel + β per head — the low-rank decay projections are folded in
//!     as an approximation until the reference implementation dims are public)
//!   - `out_proj`  : d_inner → hidden               (12288 → 7168)
//!
//! The short causal conv (width `short_conv_kernel_size`) + gating activation
//! are modeled as ONE byte-keyed elementwise placeholder: per token, read the
//! conv window over q/k/v (`conv_k · 3·d_inner` elems) and the gates, write the
//! convolved/gated `3·d_inner`. The KDA recurrent state (per-request
//! `num_heads·head_dim²` matrix) is constant per request — it is NOT per-token
//! KV cache and is excluded from the arch's KV accounting.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, KdaScanKernel,
    KdaScanKernelConfig, KdaScanKernelInput, RmsNormKernel, RmsNormKernelConfig,
    RmsNormKernelInput, SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge};

/// Raw global config. No parallelism degree: the block is `Local`.
#[derive(Clone, Debug)]
pub struct KdaBlockLocalWorkletConfig {
    pub hidden: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub short_conv_kernel_size: u32,
    /// Base (16-bit) dtype — RMSNorm and the scan state keep it.
    pub dtype: DType,
    /// FP8 run: the projections move to fp8 (via `gemm_backends` = deepgemm);
    /// the scan itself stays at the base dtype (state update precision).
    pub fp8: bool,
    pub gpu_name: String,
    pub norm_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
    pub act_backends: Vec<&'static str>,
    pub kda_scan_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct KdaBlockLocalWorkletResolved {
    pub raw_cfg: KdaBlockLocalWorkletConfig,
    pub input_norm: RmsNormKernelConfig,
    pub qkv: SingleGemmKernelConfig,
    pub gate: SingleGemmKernelConfig,
    pub conv_gate: ElementwiseKernelConfig,
    pub scan: KdaScanKernelConfig,
    pub out_proj: SingleGemmKernelConfig,
    pub dtype_bytes: u32,
}

/// Per-call shape: the pooled token count drives every leaf (the scan walks
/// all tokens once; per-request chunk boundaries are folded into its curve).
#[derive(Clone, Debug, Default)]
pub struct KdaBlockLocalWorkletInput {
    pub batch_tokens: u32,
}

pub struct KdaBlockLocalWorklet {
    pub name: String,
    pub input_norm: Op<RmsNormKernel>,
    pub qkv: Op<SingleGemmKernel>,
    pub gate: Op<SingleGemmKernel>,
    pub conv_gate: Op<ElementwiseKernel>,
    pub scan: Op<KdaScanKernel>,
    pub out_proj: Op<SingleGemmKernel>,
    resolved: KdaBlockLocalWorkletResolved,
}

impl KdaBlockLocalWorklet {
    pub fn resolve_config(cfg: &KdaBlockLocalWorkletConfig) -> KdaBlockLocalWorkletResolved {
        let d_inner = cfg.num_heads * cfg.head_dim;
        let compute = if cfg.fp8 { DType::Fp8E4m3 } else { cfg.dtype };
        // Base-dtype bytes for the conv/gate placeholder — the conv window and
        // gate math run on 16-bit activations even in an fp8 run.
        let base_bytes = cfg.dtype.size_bytes();
        KdaBlockLocalWorkletResolved {
            input_norm: RmsNormKernelConfig {
                backends: cfg.norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden,
                dtype: cfg.dtype,
            },
            qkv: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: 3 * d_inner,
                k: cfg.hidden,
                dtype: compute,
            },
            gate: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: d_inner + cfg.num_heads,
                k: cfg.hidden,
                dtype: compute,
            },
            conv_gate: ElementwiseKernelConfig {
                backends: cfg.act_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                // Read the conv window over q/k/v + the gate row; write the
                // convolved/gated q/k/v. Placeholder byte model (see header).
                input_bytes_per_token: (cfg.short_conv_kernel_size * 3 * d_inner
                    + d_inner
                    + cfg.num_heads)
                    * base_bytes,
                output_bytes_per_token: 3 * d_inner * base_bytes,
            },
            scan: KdaScanKernelConfig {
                backends: cfg.kda_scan_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads: cfg.num_heads,
                head_dim: cfg.head_dim,
                short_conv_kernel_size: cfg.short_conv_kernel_size,
                dtype: cfg.dtype,
            },
            out_proj: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.hidden,
                k: d_inner,
                dtype: compute,
            },
            dtype_bytes: compute.size_bytes(),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: KdaBlockLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let norm_name = format!("{name}.input_norm");
        let qkv_name = format!("{name}.qkv_proj");
        let gate_name = format!("{name}.gate_proj");
        let cg_name = format!("{name}.conv_gate");
        let scan_name = format!("{name}.kda_scan");
        let out_name = format!("{name}.out_proj");
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
        let gate = Op::new(
            gate_name.clone(),
            Arc::new(SingleGemmKernel::build(
                gate_name,
                resolved.gate.clone(),
                bridge,
            )?),
        );
        let conv_gate = Op::new(
            cg_name.clone(),
            Arc::new(ElementwiseKernel::build(
                cg_name,
                resolved.conv_gate.clone(),
                bridge,
            )?),
        );
        let scan = Op::new(
            scan_name.clone(),
            Arc::new(KdaScanKernel::build(
                scan_name,
                resolved.scan.clone(),
                bridge,
            )?),
        );
        let out_proj = Op::new(
            out_name.clone(),
            Arc::new(SingleGemmKernel::build(
                out_name,
                resolved.out_proj.clone(),
                bridge,
            )?),
        );
        Ok(Self {
            name,
            input_norm,
            qkv,
            gate,
            conv_gate,
            scan,
            out_proj,
            resolved,
        })
    }

    /// CostTree compile: Sum(input_norm, qkv, gate, conv_gate, kda_scan,
    /// out_proj) under a labeled header.
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let r = &self.resolved;
        let label = format!(
            "{} (KdaBlockLocalWorklet) [heads={}, head_dim={}, conv_k={}]",
            self.name, r.raw_cfg.num_heads, r.raw_cfg.head_dim, r.raw_cfg.short_conv_kernel_size,
        );
        CostNode::Labeled {
            label,
            child: Box::new(CostNode::Sum(vec![
                self.input_norm.compile(builder),
                self.qkv.compile(builder),
                self.gate.compile(builder),
                self.conv_gate.compile(builder),
                self.scan.compile(builder),
                self.out_proj.compile(builder),
            ])),
        }
    }

    /// CostTree eval: fill slots in the exact `compile` child order.
    pub fn eval(&self, input: &KdaBlockLocalWorkletInput, ev: &mut Evaluator) {
        let m = input.batch_tokens;
        self.input_norm.eval(&RmsNormKernelInput { m }, ev);
        self.qkv.eval(&SingleGemmKernelInput { m }, ev);
        self.gate.eval(&SingleGemmKernelInput { m }, ev);
        self.conv_gate
            .eval(&ElementwiseKernelInput { num_tokens: m }, ev);
        self.scan.eval(&KdaScanKernelInput { total_tokens: m }, ev);
        self.out_proj.eval(&SingleGemmKernelInput { m }, ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(fp8: bool) -> KdaBlockLocalWorkletConfig {
        KdaBlockLocalWorkletConfig {
            hidden: 7168,
            num_heads: 96,
            head_dim: 128,
            short_conv_kernel_size: 4,
            dtype: DType::Bf16,
            fp8,
            gpu_name: "NVIDIA H200".to_string(),
            norm_backends: vec!["flashinfer"],
            gemm_backends: vec!["torch", "torch_linear"],
            act_backends: vec!["triton"],
            kda_scan_backends: vec!["triton"],
        }
    }

    #[test]
    fn resolve_bakes_kimi_k3_kda_shapes() {
        let r = KdaBlockLocalWorklet::resolve_config(&cfg(false));
        let d_inner = 96 * 128;
        assert_eq!((r.qkv.n, r.qkv.k), (3 * d_inner, 7168));
        assert_eq!((r.gate.n, r.gate.k), (d_inner + 96, 7168));
        assert_eq!((r.out_proj.n, r.out_proj.k), (7168, d_inner));
        assert_eq!(r.scan.num_heads, 96);
        assert_eq!(r.scan.head_dim, 128);
        assert_eq!(r.scan.short_conv_kernel_size, 4);
        assert_eq!(r.scan.backends, vec!["triton"]);
        // Conv/gate placeholder: window (4·3·d_inner) + gates, out 3·d_inner (bf16).
        assert_eq!(
            r.conv_gate.input_bytes_per_token,
            (4 * 3 * d_inner + d_inner + 96) * 2
        );
        assert_eq!(r.conv_gate.output_bytes_per_token, 3 * d_inner * 2);
    }

    #[test]
    fn fp8_moves_projections_but_scan_and_norm_stay_base() {
        let mut c = cfg(true);
        c.gemm_backends = vec!["deepgemm"];
        let r = KdaBlockLocalWorklet::resolve_config(&c);
        assert_eq!(r.qkv.dtype, DType::Fp8E4m3);
        assert_eq!(r.out_proj.dtype, DType::Fp8E4m3);
        assert_eq!(r.input_norm.dtype, DType::Bf16);
        assert_eq!(r.scan.dtype, DType::Bf16);
    }
}
