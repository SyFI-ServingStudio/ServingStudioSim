//! `MlaBlockLocalWorklet` — one MLA (multi-head latent attention) decoder-layer
//! attention block on ONE GPU: input RMSNorm → q LoRA down/up → kv LoRA down →
//! MLA attention (compressed-KV cache) → o_proj. `Local` group suffix (L3
//! §1.5): a single sync section with no collective — MLA's single shared
//! compressed-KV head cannot be TP-split, so the arch runs this block with
//! `attn_tp_size` fixed at 1 (pure DP attention; per-shard == full shapes).
//!
//! GEMM shapes (Kimi-K3 dims in parens):
//!   - `q_down` : hidden → q_lora_rank            (7168 → 1536)
//!   - `q_up`   : q_lora_rank → heads·qk_head_dim (1536 → 96·192)
//!   - `kv_down`: hidden → kv_compressed_dim      (7168 → 576, incl. rope part)
//!   - `o_proj` : heads·head_dim → hidden         (96·128 → 7168)
//!
//! v1 deviation: the prefill KV **decompression** up-projection
//! (kv_lora_rank → heads·(nope+v) per prefilled token) is not billed as a
//! separate GEMM — the decompressed-MHA prefill kernel is the profiling proxy
//! for the whole prefill path. Revisit with the profiling phase.

use std::sync::Arc;

use crate::op::attention::{MlaAttentionConfig, MlaAttentionInput, MlaAttentionOp};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    RmsNormKernel, RmsNormKernelConfig, RmsNormKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge};

/// Raw global config. No parallelism degree: the block is `Local` (attn_tp=1).
#[derive(Clone, Debug)]
pub struct MlaBlockLocalWorkletConfig {
    pub hidden: u32,
    pub num_heads: u32,
    /// Decompressed value / nope head dim (128) — the prefill MHA head_dim and
    /// the o_proj input head width.
    pub head_dim: u32,
    /// Full q/k head dim = nope + rope (192) — the q_up output head width.
    pub qk_head_dim: u32,
    /// Query LoRA rank (1536).
    pub q_lora_rank: u32,
    /// Compressed-KV width = kv_lora_rank + rope (576).
    pub kv_compressed_dim: u32,
    /// Base (16-bit) dtype.
    pub dtype: DType,
    /// FP8 run: GEMMs + attention presets go fp8.
    pub fp8: bool,
    pub gpu_name: String,
    pub norm_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
    pub attn_backends: Vec<&'static str>,
    pub kv_cache_append_backends: Vec<&'static str>,
    pub kv_cache_block_size: u32,
    pub kv_cache_layout: String,
    pub kv_scale_granularity: String,
}

#[derive(Clone, Debug)]
pub struct MlaBlockLocalWorkletResolved {
    pub raw_cfg: MlaBlockLocalWorkletConfig,
    pub input_norm: RmsNormKernelConfig,
    pub q_down: SingleGemmKernelConfig,
    pub q_up: SingleGemmKernelConfig,
    pub kv_down: SingleGemmKernelConfig,
    pub attn: MlaAttentionConfig,
    pub o_proj: SingleGemmKernelConfig,
    pub dtype_bytes: u32,
}

/// Per-call shape: `batch_tokens` drives the norm/GEMMs; the prefill
/// `(prefix_len, append_len)` pairs and decode KV lengths drive attention.
#[derive(Clone, Debug, Default)]
pub struct MlaBlockLocalWorkletInput {
    pub batch_tokens: u32,
    pub prefill_chunk_pairs: Vec<(u32, u32)>,
    pub decode_kv_lens: Vec<u32>,
}

pub struct MlaBlockLocalWorklet {
    pub name: String,
    pub input_norm: Op<RmsNormKernel>,
    pub q_down: Op<SingleGemmKernel>,
    pub q_up: Op<SingleGemmKernel>,
    pub kv_down: Op<SingleGemmKernel>,
    pub attn: MlaAttentionOp,
    pub o_proj: Op<SingleGemmKernel>,
    resolved: MlaBlockLocalWorkletResolved,
}

impl MlaBlockLocalWorkletConfig {
    /// Compressed-KV cache dtype: fp8 in an fp8 run, else the base dtype. Used
    /// by the arch's KV-byte accounting via `raw_cfg`.
    pub fn kv_dtype(&self) -> DType {
        if self.fp8 {
            DType::Fp8E4m3
        } else {
            self.dtype
        }
    }
}

impl MlaBlockLocalWorklet {
    pub fn resolve_config(cfg: &MlaBlockLocalWorkletConfig) -> MlaBlockLocalWorkletResolved {
        // Compute dtype: FP8 for the GEMMs; RMSNorm and the attention output
        // keep the base `dtype`.
        let compute = if cfg.fp8 { DType::Fp8E4m3 } else { cfg.dtype };
        MlaBlockLocalWorkletResolved {
            input_norm: RmsNormKernelConfig {
                backends: cfg.norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden,
                dtype: cfg.dtype,
            },
            q_down: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.q_lora_rank,
                k: cfg.hidden,
                dtype: compute,
            },
            q_up: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.num_heads * cfg.qk_head_dim,
                k: cfg.q_lora_rank,
                dtype: compute,
            },
            kv_down: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.kv_compressed_dim,
                k: cfg.hidden,
                dtype: compute,
            },
            attn: MlaAttentionConfig {
                gpu_name: cfg.gpu_name.clone(),
                num_heads: cfg.num_heads,
                head_dim: cfg.head_dim,
                kv_compressed_dim: cfg.kv_compressed_dim,
                dtype: cfg.dtype,
                fp8: cfg.fp8,
                prefill_backends: cfg.attn_backends.clone(),
                decode_backends: cfg.attn_backends.clone(),
                kv_cache_append_backends: cfg.kv_cache_append_backends.clone(),
                kv_cache_block_size: cfg.kv_cache_block_size,
                kv_cache_layout: cfg.kv_cache_layout.clone(),
                kv_scale_granularity: cfg.kv_scale_granularity.clone(),
            },
            o_proj: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.hidden,
                k: cfg.num_heads * cfg.head_dim,
                dtype: compute,
            },
            dtype_bytes: compute.size_bytes(),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: MlaBlockLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let norm_name = format!("{name}.input_norm");
        let qd_name = format!("{name}.q_down_proj");
        let qu_name = format!("{name}.q_up_proj");
        let kvd_name = format!("{name}.kv_down_proj");
        let o_name = format!("{name}.o_proj");
        let input_norm = Op::new(
            norm_name.clone(),
            Arc::new(RmsNormKernel::build(
                norm_name,
                resolved.input_norm.clone(),
                bridge,
            )?),
        );
        let q_down = Op::new(
            qd_name.clone(),
            Arc::new(SingleGemmKernel::build(
                qd_name,
                resolved.q_down.clone(),
                bridge,
            )?),
        );
        let q_up = Op::new(
            qu_name.clone(),
            Arc::new(SingleGemmKernel::build(
                qu_name,
                resolved.q_up.clone(),
                bridge,
            )?),
        );
        let kv_down = Op::new(
            kvd_name.clone(),
            Arc::new(SingleGemmKernel::build(
                kvd_name,
                resolved.kv_down.clone(),
                bridge,
            )?),
        );
        let attn = MlaAttentionOp::build(format!("{name}.attn"), resolved.attn.clone(), bridge)?;
        let o_proj = Op::new(
            o_name.clone(),
            Arc::new(SingleGemmKernel::build(
                o_name,
                resolved.o_proj.clone(),
                bridge,
            )?),
        );
        Ok(Self {
            name,
            input_norm,
            q_down,
            q_up,
            kv_down,
            attn,
            o_proj,
            resolved,
        })
    }

    /// CostTree compile: Sum(input_norm, q_down, q_up, kv_down, attn (3
    /// leaves), o_proj) under a labeled partition header.
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let r = &self.resolved;
        let label = format!(
            "{} (MlaBlockLocalWorklet) [heads={}, qk_head_dim={}, kv_compressed={}, q_lora={}]",
            self.name,
            r.raw_cfg.num_heads,
            r.raw_cfg.qk_head_dim,
            r.raw_cfg.kv_compressed_dim,
            r.raw_cfg.q_lora_rank,
        );
        CostNode::Labeled {
            label,
            child: Box::new(CostNode::Sum(vec![
                self.input_norm.compile(builder),
                self.q_down.compile(builder),
                self.q_up.compile(builder),
                self.kv_down.compile(builder),
                self.attn.compile(builder),
                self.o_proj.compile(builder),
            ])),
        }
    }

    /// CostTree eval: fill slots in the exact `compile` child order.
    pub fn eval(&self, input: &MlaBlockLocalWorkletInput, ev: &mut Evaluator) {
        let m = input.batch_tokens;
        self.input_norm.eval(&RmsNormKernelInput { m }, ev);
        self.q_down.eval(&SingleGemmKernelInput { m }, ev);
        self.q_up.eval(&SingleGemmKernelInput { m }, ev);
        self.kv_down.eval(&SingleGemmKernelInput { m }, ev);
        self.attn.eval(
            &MlaAttentionInput {
                prefill_chunk_pairs: input.prefill_chunk_pairs.clone(),
                decode_kv_lens: input.decode_kv_lens.clone(),
            },
            ev,
        );
        self.o_proj.eval(&SingleGemmKernelInput { m }, ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(fp8: bool) -> MlaBlockLocalWorkletConfig {
        MlaBlockLocalWorkletConfig {
            hidden: 7168,
            num_heads: 96,
            head_dim: 128,
            qk_head_dim: 192,
            q_lora_rank: 1536,
            kv_compressed_dim: 576,
            dtype: DType::Bf16,
            fp8,
            gpu_name: "NVIDIA H200".to_string(),
            norm_backends: vec!["flashinfer"],
            gemm_backends: vec!["torch", "torch_linear"],
            attn_backends: vec!["fa2", "fa3"],
            kv_cache_append_backends: vec!["vllm_cuda"],
            kv_cache_block_size: 16,
            kv_cache_layout: "NHD".to_string(),
            kv_scale_granularity: "tensor".to_string(),
        }
    }

    #[test]
    fn resolve_bakes_the_real_kimi_k3_lora_dims() {
        let r = MlaBlockLocalWorklet::resolve_config(&cfg(false));
        // q: 7168 → 1536 → 96·192.
        assert_eq!((r.q_down.n, r.q_down.k), (1536, 7168));
        assert_eq!((r.q_up.n, r.q_up.k), (96 * 192, 1536));
        // kv: 7168 → 576 (compressed, incl. rope).
        assert_eq!((r.kv_down.n, r.kv_down.k), (576, 7168));
        // o: 96·128 → 7168.
        assert_eq!((r.o_proj.n, r.o_proj.k), (7168, 96 * 128));
        // attention: prefill MHA 96/96/128, decode MQA 96/1/576 (via op cfg).
        assert_eq!(r.attn.num_heads, 96);
        assert_eq!(r.attn.head_dim, 128);
        assert_eq!(r.attn.kv_compressed_dim, 576);
    }

    #[test]
    fn fp8_moves_gemms_to_compute_dtype_but_norm_stays_base() {
        let mut c = cfg(true);
        c.gemm_backends = vec!["deepgemm"];
        let r = MlaBlockLocalWorklet::resolve_config(&c);
        assert_eq!(r.q_down.dtype, DType::Fp8E4m3);
        assert_eq!(r.o_proj.dtype, DType::Fp8E4m3);
        assert_eq!(r.input_norm.dtype, DType::Bf16);
        assert!(r.attn.fp8);
        assert_eq!(c.kv_dtype(), DType::Fp8E4m3);
    }
}
