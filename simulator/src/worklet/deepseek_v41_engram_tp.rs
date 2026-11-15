//! DeepSeek-V4.1-Flash Engram block on one TP rank (vLLM), layers 1 and 14.
//!
//! The Engram block sits between the previous layer's FFN and this layer's
//! attention (fork `models/deepseek_v41/nvidia/model.py:465-480`,
//! `common/engram.py:960-1030`). One sync section, main stream only; capture 2
//! decode iteration 1337 (48 rows, device 0), layer 1:
//!
//! | launch | capture | leaf |
//! |---|---|---|
//! | `mhc_post_tilelang_kernel` 508.26-518.18 | previous FFN's hc post (non-fused because the next pre is replaced) | `mhc_post` placeholder |
//! | `ncclDevKernel_AllGather_RING_LL` 520.99-536.22 | gather the 6 local lookup heads to all 24 | `all_gather` (`all_reduce` proxy) |
//! | `elementwise_kernel` 536.83-540.54 | `[T,4,6,256]` -> `[T,24,256]` contiguous copy | `gather_copy` placeholder |
//! | `mxfp8_quant` + `MXFP8GEMM` 540.51-573.86 | replicated `wkv` 6144 -> 25600 | `wkv` (flashinfer MXFP8) |
//! | `_fused_engram_post_wkv_kernel` 574.50-578.82 | gate, conv, residual add into the hc streams | `post_wkv` placeholder |
//! | `sm100_tf32_hc_prenorm_gemm` + `mhc_pre_big_fuse_with_norm` 578.75-591.81 | this layer's attention pre | `hc_prenorm`, `pre_big_fuse` placeholders |
//!
//! The lookups that produce the gathered rows run on side streams from the
//! iteration start and are modelled by `DeepseekV41EngramPrefetchLocalWorklet`.
//!
//! With `mhc_unfused_backends` (DeepInfer `hc.py`), the hc post is
//! TileKernels' `mhc_post_fwd` and the attention pre its five-launch shifted
//! pre (`mhc_shifted_pre`), in place of the two placeholders. The RMSNorm
//! DeepInfer runs after the shifted pre (`_norm_kernel`) has no kind and no
//! leaf.

use std::sync::Arc;

use super::deepseek_v41_common::{
    all_gather_as_all_reduce_bytes, eval_or_zero, nccl_all_gather_proxy, placeholder,
};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    AllReduceKernel, AllReduceKernelConfig, AllReduceKernelInput, ElementwiseKernel,
    ElementwiseKernelConfig, ElementwiseKernelInput, EngramGateKernel, EngramGateKernelConfig,
    EngramGateKernelInput, MhcPostKernel, MhcRmsNormKernelConfig, MhcRmsNormKernelInput,
    MhcShiftedPreKernel, SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

#[derive(Clone, Debug)]
pub struct DeepseekV41EngramTpWorkletConfig {
    pub tp_size: u32,
    pub hidden_size: Dim,
    pub hc_mult: u32,
    /// Hash columns = gathered Engram heads (24).
    pub engram_num_heads: u32,
    pub engram_head_dim: u32,
    /// `wkv` output width (25600).
    pub wkv_out_dim: Dim,
    pub gpu_name: String,
    pub gemm_backends: Vec<&'static str>,
    /// NCCL `all_reduce` curve used as the AllGather proxy.
    pub all_gather_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
    /// `Some` when the engine gates the `wkv` output into the streams with
    /// TileKernels' `engram_gate_fwd` (DeepInfer `v4_modules/engram.py`)
    /// instead of vLLM's `_fused_engram_post_wkv`.
    pub engram_gate_backends: Option<Vec<&'static str>>,
    /// Largest per-rank token count (bounds the `engram_gate` token axis).
    pub max_rank_tokens: u32,
    /// `Some` when the engine runs the hc post and the shifted pre as
    /// TileKernels' unfused mHC kernels (DeepInfer), not as placeholders.
    pub mhc_unfused_backends: Option<Vec<&'static str>>,
}

/// The launch that folds the `wkv` output into the hc streams.
#[derive(Clone, Debug)]
pub enum EngramPostWkvConfig {
    /// vLLM `_fused_engram_post_wkv` (placeholder).
    Fused(ElementwiseKernelConfig),
    /// DeepInfer `engram_gate_fwd`.
    Gate(EngramGateKernelConfig),
}

pub enum EngramPostWkv {
    Fused(Op<ElementwiseKernel>),
    Gate(Op<EngramGateKernel>),
}

impl EngramPostWkv {
    fn compile(&self, b: &mut CostTreeBuilder) -> CostNode {
        match self {
            Self::Fused(op) => op.compile(b),
            Self::Gate(op) => op.compile(b),
        }
    }

    fn eval(&self, rows: u32, ev: &mut Evaluator) {
        let zero = rows == 0;
        match self {
            Self::Fused(op) => {
                eval_or_zero(op, ElementwiseKernelInput { num_tokens: rows }, zero, ev)
            }
            Self::Gate(op) => {
                eval_or_zero(op, EngramGateKernelInput { num_tokens: rows }, zero, ev)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct DeepseekV41EngramTpWorkletResolved {
    pub raw_cfg: DeepseekV41EngramTpWorkletConfig,
    /// Gathered bf16 bytes per token on every rank.
    pub gathered_bytes_per_token: u64,
    /// vLLM's hc post placeholder; `None` with `mhc_unfused_backends`.
    pub mhc_post: Option<ElementwiseKernelConfig>,
    /// TileKernels' `mhc_post_fwd`.
    pub tile_mhc_post: Option<MhcRmsNormKernelConfig>,
    pub all_gather: Option<AllReduceKernelConfig>,
    /// `None` at TP1: no all-gather, so no head-major copy.
    pub gather_copy: Option<ElementwiseKernelConfig>,
    pub wkv: SingleGemmKernelConfig,
    pub post_wkv: EngramPostWkvConfig,
    pub hc_prenorm: Option<ElementwiseKernelConfig>,
    pub pre_big_fuse: Option<ElementwiseKernelConfig>,
    /// TileKernels' shifted pre in place of `hc_prenorm` + `pre_big_fuse`.
    pub shifted_pre: Option<MhcRmsNormKernelConfig>,
}

#[derive(Clone, Debug, Default)]
pub struct DeepseekV41EngramTpWorkletInput {
    pub num_tokens: u32,
}

pub struct DeepseekV41EngramTpWorklet {
    pub name: String,
    pub mhc_post: Option<Op<ElementwiseKernel>>,
    pub tile_mhc_post: Option<Op<MhcPostKernel>>,
    pub all_gather: Option<Op<AllReduceKernel>>,
    pub gather_copy: Option<Op<ElementwiseKernel>>,
    pub wkv: Op<SingleGemmKernel>,
    pub post_wkv: EngramPostWkv,
    pub hc_prenorm: Option<Op<ElementwiseKernel>>,
    pub pre_big_fuse: Option<Op<ElementwiseKernel>>,
    pub shifted_pre: Option<Op<MhcShiftedPreKernel>>,
    resolved: DeepseekV41EngramTpWorkletResolved,
}

impl DeepseekV41EngramTpWorklet {
    pub fn resolve_config(
        cfg: &DeepseekV41EngramTpWorkletConfig,
    ) -> DeepseekV41EngramTpWorkletResolved {
        assert!(cfg.tp_size > 0, "tp_size must be positive");
        assert_eq!(
            cfg.engram_num_heads % cfg.tp_size,
            0,
            "Engram heads must divide tp_size"
        );
        let gpu = cfg.gpu_name.as_str();
        let ew =
            |input: u32, output: u32| placeholder(&cfg.elementwise_backends, gpu, input, output);
        let hidden_bytes = cfg.hidden_size.get() * 2;
        let hc_bytes = cfg.hc_mult * hidden_bytes;
        let mix_bytes = (2 + cfg.hc_mult) * cfg.hc_mult * 4;
        let engram_width = cfg.engram_num_heads * cfg.engram_head_dim;
        let gathered = u64::from(engram_width) * 2;
        let placeholders = cfg.mhc_unfused_backends.is_none();
        let tile = |backends: &Vec<&'static str>| MhcRmsNormKernelConfig {
            backends: backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            hidden_size: cfg.hidden_size.clone(),
            hc_mult: cfg.hc_mult,
            hidden_dtype: DType::Bf16,
        };
        DeepseekV41EngramTpWorkletResolved {
            gathered_bytes_per_token: gathered,
            // nvidia/model.py:472 `mhc_post`: hc streams + block output in,
            // updated hc streams out.
            mhc_post: placeholders.then(|| ew(hc_bytes + hidden_bytes, hc_bytes)),
            tile_mhc_post: cfg.mhc_unfused_backends.as_ref().map(tile),
            // common/engram.py:992 `tensor_model_parallel_all_gather`.
            all_gather: (cfg.tp_size > 1)
                .then(|| nccl_all_gather_proxy(&cfg.all_gather_backends, gpu, cfg.tp_size)),
            // The all-gather's rank-major -> head-major contiguous copy
            // (`tensor_model_parallel_all_gather(out, dim=1)`, only at TP > 1).
            gather_copy: (cfg.tp_size > 1).then(|| ew(engram_width * 2, engram_width * 2)),
            wkv: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.wkv_out_dim.clone(),
                k: engram_width.into(),
                dtype: DType::Mxfp8E4m3,
            },
            // common/engram.py:1027 `_fused_engram_post_wkv`: wkv output and
            // hc streams in, hc streams out.
            post_wkv: match &cfg.engram_gate_backends {
                None => {
                    EngramPostWkvConfig::Fused(ew(cfg.wkv_out_dim.get() * 2 + hc_bytes, hc_bytes))
                }
                Some(backends) => EngramPostWkvConfig::Gate(EngramGateKernelConfig {
                    backends: backends.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    hc_mult: cfg.hc_mult,
                    hidden_size: cfg.hidden_size.clone(),
                    dtype: DType::Bf16,
                    max_tokens_per_rank: cfg.max_rank_tokens,
                }),
            },
            // nvidia/model.py:480 `mhc_pre_delayed_tilelang` (non-fused pre).
            hc_prenorm: placeholders.then(|| ew(hc_bytes, mix_bytes)),
            pre_big_fuse: placeholders.then(|| ew(hc_bytes + mix_bytes, hidden_bytes)),
            shifted_pre: cfg.mhc_unfused_backends.as_ref().map(tile),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: DeepseekV41EngramTpWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        macro_rules! op {
            ($kernel:ty, $cfg:expr, $suffix:literal) => {{
                let op_name = format!("{name}.{}", $suffix);
                Op::new(
                    op_name.clone(),
                    Arc::new(<$kernel>::build(op_name, $cfg, bridge)?),
                )
            }};
        }
        macro_rules! maybe {
            ($kernel:ty, $cfg:expr, $suffix:literal) => {
                match $cfg {
                    Some(config) => Some(op!($kernel, config, $suffix)),
                    None => None,
                }
            };
        }
        let r = resolved.clone();
        Ok(Self {
            mhc_post: maybe!(ElementwiseKernel, r.mhc_post, "mhc_post"),
            tile_mhc_post: maybe!(MhcPostKernel, r.tile_mhc_post, "mhc_post"),
            all_gather: match r.all_gather {
                Some(config) => Some(op!(AllReduceKernel, config, "all_gather")),
                None => None,
            },
            gather_copy: match r.gather_copy {
                Some(config) => Some(op!(ElementwiseKernel, config, "gather_copy")),
                None => None,
            },
            wkv: op!(SingleGemmKernel, r.wkv, "wkv"),
            post_wkv: match r.post_wkv {
                EngramPostWkvConfig::Fused(config) => {
                    EngramPostWkv::Fused(op!(ElementwiseKernel, config, "post_wkv"))
                }
                EngramPostWkvConfig::Gate(config) => {
                    EngramPostWkv::Gate(op!(EngramGateKernel, config, "engram_gate"))
                }
            },
            hc_prenorm: maybe!(ElementwiseKernel, r.hc_prenorm, "mhc_hc_prenorm"),
            pre_big_fuse: maybe!(ElementwiseKernel, r.pre_big_fuse, "mhc_pre_big_fuse"),
            shifted_pre: maybe!(MhcShiftedPreKernel, r.shifted_pre, "mhc_shifted_pre"),
            name,
            resolved,
        })
    }

    pub fn compile(&self, b: &mut CostTreeBuilder) -> CostNode {
        let mut children = Vec::new();
        if let Some(op) = &self.mhc_post {
            children.push(op.compile(b));
        }
        if let Some(op) = &self.tile_mhc_post {
            children.push(op.compile(b));
        }
        if let Some(all_gather) = &self.all_gather {
            children.push(all_gather.compile(b));
        }
        if let Some(gather_copy) = &self.gather_copy {
            children.push(gather_copy.compile(b));
        }
        children.push(self.wkv.compile(b));
        children.push(self.post_wkv.compile(b));
        for op in [&self.hc_prenorm, &self.pre_big_fuse].into_iter().flatten() {
            children.push(op.compile(b));
        }
        if let Some(op) = &self.shifted_pre {
            children.push(op.compile(b));
        }
        CostNode::Labeled {
            label: format!(
                "{} (DeepseekV41EngramTpWorklet) [tp={}; heads={}x{}]",
                self.name,
                self.resolved.raw_cfg.tp_size,
                self.resolved.raw_cfg.engram_num_heads,
                self.resolved.raw_cfg.engram_head_dim,
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    pub fn eval(&self, input: &DeepseekV41EngramTpWorkletInput, ev: &mut Evaluator) {
        let rows = input.num_tokens;
        let zero = rows == 0;
        let ew = ElementwiseKernelInput { num_tokens: rows };
        let mhc = MhcRmsNormKernelInput { num_tokens: rows };
        if let Some(op) = &self.mhc_post {
            eval_or_zero(op, ew.clone(), zero, ev);
        }
        if let Some(op) = &self.tile_mhc_post {
            eval_or_zero(op, mhc.clone(), zero, ev);
        }
        if let Some(all_gather) = &self.all_gather {
            eval_or_zero(
                all_gather,
                AllReduceKernelInput {
                    message_size_bytes: all_gather_as_all_reduce_bytes(
                        u64::from(rows) * self.resolved.gathered_bytes_per_token,
                    ),
                },
                zero,
                ev,
            );
        }
        if let Some(gather_copy) = &self.gather_copy {
            eval_or_zero(gather_copy, ew.clone(), zero, ev);
        }
        eval_or_zero(&self.wkv, SingleGemmKernelInput { m: rows }, zero, ev);
        self.post_wkv.eval(rows, ev);
        for op in [&self.hc_prenorm, &self.pre_big_fuse].into_iter().flatten() {
            eval_or_zero(op, ew.clone(), zero, ev);
        }
        if let Some(op) = &self.shifted_pre {
            eval_or_zero(op, mhc, zero, ev);
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn config() -> DeepseekV41EngramTpWorkletConfig {
        DeepseekV41EngramTpWorkletConfig {
            tp_size: 4,
            hidden_size: 5120.into(),
            hc_mult: 4,
            engram_num_heads: 24,
            engram_head_dim: 256,
            wkv_out_dim: 25600.into(),
            gpu_name: "NVIDIA B200".into(),
            gemm_backends: vec!["flashinfer_mxfp8"],
            all_gather_backends: vec!["nccl"],
            elementwise_backends: vec!["triton"],
            engram_gate_backends: None,
            max_rank_tokens: 8192,
            mhc_unfused_backends: None,
        }
    }

    #[test]
    fn resolves_the_captured_wkv_shape_and_gather_bytes() {
        let r = DeepseekV41EngramTpWorklet::resolve_config(&config());
        assert_eq!((r.wkv.k.get(), r.wkv.n.get()), (6144, 25600));
        assert_eq!(r.gathered_bytes_per_token, 12_288);
        assert_eq!(r.mhc_post.as_ref().unwrap().input_bytes_per_token.get(), 51_200);
        let EngramPostWkvConfig::Fused(post_wkv) = &r.post_wkv else {
            panic!("vLLM runs the fused post-wkv")
        };
        assert_eq!(post_wkv.output_bytes_per_token.get(), 40_960);
        assert!(r.all_gather.is_some());
        let mut single = config();
        single.tp_size = 1;
        let single = DeepseekV41EngramTpWorklet::resolve_config(&single);
        assert!(single.all_gather.is_none() && single.gather_copy.is_none());
        assert!(r.gather_copy.is_some());
    }
}
