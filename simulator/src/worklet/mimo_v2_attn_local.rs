//! MiMo-V2 attention on one TP rank, up to the reduction
//! (`MiMoV2Attention.forward`, vllm/model_executor/models/mimo_v2.py), one
//! layer type (global or sliding window) per configuration.
//!
//! Six launches in order:
//!
//! 1. the per-token-group FP8 quant of the normed input (packed UE8M0 scales);
//! 2. the fused QKV projection, a DeepGEMM block-FP8 GEMM whose output packs
//!    Q (`q_heads x head_dim`), K (`kv_heads x head_dim`) and V
//!    (`kv_heads x v_head_dim`);
//! 3. RoPE over the first `rotary_dim` channels of Q and K together with
//!    `v * attention_value_scale`, one compiled Triton launch over the packed
//!    QKV rows (positions and the cos/sin row read alongside), modeled as an
//!    elementwise byte rate;
//! 4. the KV insert into the paged cache that packs each token's K and V side
//!    by side (`triton_reshape_and_cache_flash_diffkv`);
//! 5. the FA4 varlen call over the whole step (window and sink logits on
//!    sliding-window layers);
//! 6. the BF16 O projection (`RowParallelLinear`, the checkpoint leaves
//!    `o_proj` unquantized). Its all-reduce belongs to the arch's fused
//!    all-reduce + residual + RMSNorm.
//!
//! Every launch but attention runs on the graph-padded rows; attention reads
//! the real requests.

use crate::op::gemm::{
    SingleFp8GemmWithQuantConfig, SingleFp8GemmWithQuantInput, SingleFp8GemmWithQuantOp,
};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, Fa4VarlenAttentionKernel,
    Fa4VarlenAttentionKernelConfig, Fa4VarlenAttentionKernelInput,
    Fp8PerTokenGroupQuantKernelConfig, KvCacheAppendKernel, KvCacheAppendKernelConfig,
    KvCacheAppendKernelInput, SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

use super::glm53_common::{atomic, elementwise, push_or_zero};
use super::glm53_mlp_local::PACKED_SCALE_FORMAT;

/// vLLM's paged layout for different K and V head dims: K and V of one
/// (token, head) side by side in a page.
const DIFFKV_KV_LAYOUT: &str = "lbnhc";
/// The insert kernel's write view of that page (`[blocks, block, heads, K+V]`).
const DIFFKV_INSERT_LAYOUT: &str = "NHD";
/// One scale per K / V tensor (no FP8 cache).
const KV_SCALE_GRANULARITY: &str = "tensor";
/// RoPE positions are int64.
const POSITION_BYTES: u32 = 8;

#[derive(Clone, Debug)]
pub struct MimoV2AttnLocalWorkletConfig {
    pub hidden: Dim,
    /// Query heads on this rank.
    pub num_q_heads: u32,
    /// KV heads on this rank.
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub v_head_dim: u32,
    pub rotary_dim: u32,
    /// Keys a query sees, itself included; 0 = full causal.
    pub window: u32,
    pub sinks: bool,
    /// Tokens per KV cache page.
    pub page_size: u32,
    /// FP8 activation quant group (the checkpoint's weight block edge).
    pub quant_group_size: u32,
    pub max_model_len: u32,
    /// Decode-only steps replay a FULL CUDA graph.
    pub cuda_graph: bool,
    pub dtype: DType,
    pub gpu_name: String,
    pub quant_backends: Vec<&'static str>,
    pub fp8_gemm_backends: Vec<&'static str>,
    pub rope_backends: Vec<&'static str>,
    pub kv_append_backends: Vec<&'static str>,
    pub attention_backends: Vec<&'static str>,
    pub o_proj_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct MimoV2AttnLocalWorkletResolved {
    pub raw_cfg: MimoV2AttnLocalWorkletConfig,
    pub qkv_proj: SingleFp8GemmWithQuantConfig,
    pub rope: ElementwiseKernelConfig,
    pub kv_append: KvCacheAppendKernelConfig,
    pub attention: Fa4VarlenAttentionKernelConfig,
    pub o_proj: SingleGemmKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct MimoV2AttnLocalWorkletInput {
    /// Graph-padded rows every launch but attention runs on.
    pub num_tokens: u32,
    /// The step's real requests as `(query_len, context_len)`.
    pub query_context_pairs: Vec<(u32, u32)>,
    /// Requests attention is launched over when a FULL CUDA graph pads the
    /// batch with zero-length requests; 0 = unpadded.
    pub padded_requests: u32,
}

pub struct MimoV2AttnLocalWorklet {
    pub name: String,
    pub qkv_proj: SingleFp8GemmWithQuantOp,
    pub rope: Op<ElementwiseKernel>,
    pub kv_append: Op<KvCacheAppendKernel>,
    pub attention: Op<Fa4VarlenAttentionKernel>,
    pub o_proj: Op<SingleGemmKernel>,
    resolved: MimoV2AttnLocalWorkletResolved,
}

impl MimoV2AttnLocalWorklet {
    pub fn resolve_config(cfg: &MimoV2AttnLocalWorkletConfig) -> MimoV2AttnLocalWorkletResolved {
        assert!(
            cfg.num_kv_heads > 0 && cfg.num_q_heads.is_multiple_of(cfg.num_kv_heads),
            "query heads must group evenly over KV heads"
        );
        assert!(
            cfg.rotary_dim <= cfg.head_dim,
            "RoPE cannot rotate more channels than a head has"
        );
        let elem = cfg.dtype.size_bytes();
        let qkv_width =
            (cfg.num_q_heads + cfg.num_kv_heads) * cfg.head_dim + cfg.num_kv_heads * cfg.v_head_dim;
        let qkv_bytes = qkv_width * elem;
        // The packed QKV row in and out, plus the position and its cos/sin
        // cache row (`rotary_dim` values: half cos, half sin).
        let rope_input = qkv_bytes + POSITION_BYTES + cfg.rotary_dim * elem;
        MimoV2AttnLocalWorkletResolved {
            qkv_proj: SingleFp8GemmWithQuantConfig {
                quant: Fp8PerTokenGroupQuantKernelConfig {
                    backends: cfg.quant_backends.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    hidden_size: cfg.hidden.clone(),
                    group_size: cfg.quant_group_size,
                    input_dtype: cfg.dtype,
                    scale_format: PACKED_SCALE_FORMAT.to_string(),
                },
                gemm: SingleGemmKernelConfig {
                    backends: cfg.fp8_gemm_backends.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    n: qkv_width.into(),
                    k: cfg.hidden.clone(),
                    dtype: DType::Fp8E4m3,
                },
            },
            rope: elementwise(&cfg.rope_backends, &cfg.gpu_name, rope_input, qkv_bytes),
            kv_append: KvCacheAppendKernelConfig {
                backends: cfg.kv_append_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_kv_heads: cfg.num_kv_heads.into(),
                head_dim: cfg.head_dim.into(),
                head_dim_v: Some(cfg.v_head_dim.into()),
                block_size: cfg.page_size,
                input_dtype: cfg.dtype,
                kv_dtype: cfg.dtype,
                cache_layout: DIFFKV_INSERT_LAYOUT.to_string(),
                scale_granularity: KV_SCALE_GRANULARITY.to_string(),
            },
            attention: Fa4VarlenAttentionKernelConfig {
                backends: cfg.attention_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_qo_heads: cfg.num_q_heads.into(),
                num_kv_heads: cfg.num_kv_heads.into(),
                head_dim_qk: cfg.head_dim.into(),
                head_dim_v: cfg.v_head_dim.into(),
                window: cfg.window,
                sinks: cfg.sinks,
                page_size: cfg.page_size,
                kv_layout: DIFFKV_KV_LAYOUT.to_string(),
                q_dtype: cfg.dtype,
                kv_dtype: cfg.dtype,
                max_model_len: cfg.max_model_len,
                cuda_graph: cfg.cuda_graph,
            },
            o_proj: SingleGemmKernelConfig {
                backends: cfg.o_proj_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.hidden.clone(),
                k: (cfg.num_q_heads * cfg.v_head_dim).into(),
                dtype: cfg.dtype,
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: MimoV2AttnLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let n = name.as_str();
        Ok(Self {
            qkv_proj: SingleFp8GemmWithQuantOp::build(format!("{n}.qkv_proj"), r.qkv_proj, bridge)?,
            rope: atomic(n, "rope", r.rope, ElementwiseKernel::build, bridge)?,
            kv_append: atomic(
                n,
                "kv_append",
                r.kv_append,
                KvCacheAppendKernel::build,
                bridge,
            )?,
            attention: atomic(
                n,
                "attention",
                r.attention,
                Fa4VarlenAttentionKernel::build,
                bridge,
            )?,
            o_proj: atomic(n, "o_proj", r.o_proj, SingleGemmKernel::build, bridge)?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        CostNode::Labeled {
            label: format!(
                "{} (MimoV2AttnLocalWorklet) [TP rank; {}; q/kv heads {}/{}; qk/v dim {}/{}]",
                self.name,
                if cfg.window > 0 {
                    format!("window {}", cfg.window)
                } else {
                    "global".to_string()
                },
                cfg.num_q_heads,
                cfg.num_kv_heads,
                cfg.head_dim,
                cfg.v_head_dim
            ),
            child: Box::new(CostNode::Sum(vec![
                self.qkv_proj.compile(builder),
                self.rope.compile(builder),
                self.kv_append.compile(builder),
                self.attention.compile(builder),
                self.o_proj.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &MimoV2AttnLocalWorkletInput, ev: &mut Evaluator) {
        let tokens = input.num_tokens;
        self.qkv_proj
            .eval(&SingleFp8GemmWithQuantInput { num_tokens: tokens }, ev);
        push_or_zero(
            &self.rope,
            ElementwiseKernelInput { num_tokens: tokens },
            false,
            ev,
        );
        push_or_zero(
            &self.kv_append,
            KvCacheAppendKernelInput { num_tokens: tokens },
            false,
            ev,
        );
        push_or_zero(
            &self.attention,
            Fa4VarlenAttentionKernelInput {
                query_context_pairs: input.query_context_pairs.clone(),
                padded_requests: input.padded_requests,
            },
            false,
            ev,
        );
        push_or_zero(&self.o_proj, SingleGemmKernelInput { m: tokens }, false, ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(window: u32, num_kv_heads: u32, page_size: u32) -> MimoV2AttnLocalWorkletConfig {
        MimoV2AttnLocalWorkletConfig {
            hidden: 4096.into(),
            num_q_heads: 16,
            num_kv_heads,
            head_dim: 192,
            v_head_dim: 128,
            rotary_dim: 64,
            window,
            sinks: window > 0,
            page_size,
            quant_group_size: 128,
            max_model_len: 131072,
            cuda_graph: true,
            dtype: DType::Bf16,
            gpu_name: "NVIDIA B200".into(),
            quant_backends: vec!["vllm_cuda"],
            fp8_gemm_backends: vec!["deepgemm"],
            rope_backends: vec!["triton"],
            kv_append_backends: vec!["vllm_triton_diffkv"],
            attention_backends: vec!["vllm_fa4"],
            o_proj_backends: vec!["torch_linear_vllm"],
        }
    }

    #[test]
    fn tp4_shapes_match_the_capture() {
        // MiMo-V2.6-Flash at TP4: 16 query heads per rank; global layers keep
        // one of four KV heads, window layers two of eight.
        let global = MimoV2AttnLocalWorklet::resolve_config(&cfg(0, 1, 32));
        assert_eq!(global.qkv_proj.gemm.n.get(), 16 * 192 + 192 + 128);
        assert_eq!(global.qkv_proj.gemm.n.get(), 3392);
        let swa = MimoV2AttnLocalWorklet::resolve_config(&cfg(128, 2, 16));
        assert_eq!(swa.qkv_proj.gemm.n.get(), 3712);
        // RoPE + V scale: the packed row in and out, plus an int64 position
        // and a 64-wide bf16 cos/sin row in.
        assert_eq!(
            (
                global.rope.input_bytes_per_token.get(),
                global.rope.output_bytes_per_token.get()
            ),
            (6920, 6784)
        );
        assert_eq!(swa.rope.input_bytes_per_token.get(), 7560);
        assert_eq!((swa.o_proj.n.get(), swa.o_proj.k.get()), (4096, 2048));
        assert_eq!(swa.kv_append.head_dim_v.as_ref().map(Dim::get), Some(128));
        assert_eq!((swa.attention.window, swa.attention.sinks), (128, true));
        assert_eq!(
            (global.attention.window, global.attention.page_size),
            (0, 32)
        );
    }
}
