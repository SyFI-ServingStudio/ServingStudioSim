//! TensorFold's DFlash2 drafter pass on GB10: `DFlash2.launch_blocks` over
//! every TREE-mode stream of a round, the `tensorfold.draft.launch` span
//! (`tensorfold/families/qwen3_5/cuda/dflash2.py`, `multi.py` `_verify`).
//!
//! **Rows.** Each drafted stream runs one block of `block_rows` rows (its
//! pending token, then mask tokens). GB10 (sm_121) is not in `DEPTH_CHIPS`, so
//! the block is always `max_rows` = 16 (`multi.py:550`). All streams' blocks
//! run as one batch: every row-wise leaf sees `R = block_rows x streams` rows.
//! The head drops each block's first row: `H = (block_rows - 1) x streams`.
//! A stream drafts only when its drafter context is not empty
//! (`launch_blocks`' `live`); its context is the last `window` committed taps,
//! `min(window, committed_len)`.
//!
//! **Leaves, in launch order** (n8 capture, one launch with S streams):
//!
//! | Slot | Kernel (capture name) | Kind / backend | Launches |
//! |---|---|---|---|
//! | `embed` | `vectorized_gather_kernel` + id cast | `embedding_row_gather` / tensorfold_torch (pending) | 1 x R |
//! | `rotary` | `_rotary`: arange, mul, cos, sin | `elementwise` / tensorfold_torch_rotary (pending) | S x block_rows |
//! | `rotary_cat` | `CatArrayBatchedCopy_vectorized` (cos, sin) | `elementwise` / tensorfold_torch_cat (pending) | 2 x R |
//! | per layer (`Scale{layers}`): | | | |
//! | `norm` | `vectorized_layer_norm_kernel` (input, post-attention) | `rms_norm` / tensorfold_torch | 2 x R |
//! | `group_sums` | `_group_sums` of the four k = hidden qmm inputs | `row_group_sums` (pending) | 4 x R |
//! | `o_group_sums` | `_group_sums` of o_proj's input | `row_group_sums` (pending) | 1 x R |
//! | `kernel_projection` | `group_kernel` n = conv_kernel_width (attention, mlp conv) | `single_gemm` / tensorfold_qmm | 2 x R |
//! | `dconv` | `_dconv_kernel`, no residual (before attention, before MLP) | `dynamic_causal_conv` (pending) | 2 x R |
//! | `qkv` | `group_kernel` n = qkv_width | `single_gemm` / tensorfold_qmm | 1 x R |
//! | `prep` | `_prep_kernel` (q/k norm + RoPE, v copy) | `head_major_qk_norm_rope` (pending) | 1 x R |
//! | `lens_cast` | `unrolled_elementwise_kernel`: context lengths to int32 | `elementwise` / tensorfold_torch_cast (pending) | 1 x S |
//! | `attention` | `_block_attention` | `sliding_block_attention` / tensorfold_triton | 1 (context_lens) |
//! | `o_proj` | `group_kernel` n = hidden, k = q_width | `single_gemm` / tensorfold_qmm | 1 x R |
//! | `dconv_residual` | `_dconv_kernel` + residual (after attention, after MLP) | `dynamic_causal_conv` (pending) | 2 x R |
//! | `gate_up` | `group_kernel`, gate and up in one grouped launch | `single_gemm` / tensorfold_qmm n = 2 x intermediate | 1 x R |
//! | `swiglu` | `_swiglu` (gate, up as two tensors; same kernel) | `silu_and_mul` / tensorfold_triton | 1 x R |
//! | `down` | `group_kernel`, sums from `_swiglu` | `single_gemm` / tensorfold_qmm | 1 x R |
//! | head: | | | |
//! | `head_rows` | `elementwise_kernel`: `x.view(S, L, -1)[:, 1:].reshape` copy | `elementwise` / tensorfold_torch_strided_copy (pending) | 1 x H |
//! | `final_norm` | `vectorized_layer_norm_kernel` | `rms_norm` / tensorfold_torch | 1 x H |
//! | `selector_sums` | `_group_sums` of hidden_projection's input | `row_group_sums` (pending) | 1 x H |
//! | `hidden_projection` | `group_kernel` n = selector_rank | `single_gemm` / tensorfold_qmm | 1 x H |
//! | `selector_cast` | `unrolled_elementwise_kernel`: projected `.float()` | `elementwise` / tensorfold_torch_cast (pending) | 1 x H |
//! | `head_quant` | `quant4_kernel` (+ scale fill), one per vocabulary span | `nvfp4_quant` / tensorfold_cuda | spans x H |
//! | `head_lane_<i>` | `lane_kernel` on the span's 64-column tiles | `single_gemm` / tensorfold_lane nvfp4 | 1 x H each |
//! | `logits_cat` | `CatArrayBatchedCopy` of the span columns | `elementwise` / tensorfold_torch_cat_strided (pending) | 1 x H |
//! | `logits_cast` | `unrolled_elementwise_kernel`: logits `.float()` | `elementwise` / tensorfold_torch_cast (pending) | 1 x H |
//! | `topk` | `torch.topk(k=16)` radix select | `logits_topk` / tensorfold_torch | 1 x H |
//! | `ids_gather` | `index_elementwise_kernel`: `head_ids[local_ids]` | `elementwise` / tensorfold_torch (pending) | 1 x H |
//! | `scores_cat` | `CatArrayBatchedCopy_vectorized`: values with projected | `elementwise` / tensorfold_torch_cat (pending) | 1 x H |
//!
//! Every qmm launch without precomputed sums runs `_group_sums` first
//! (`tensorfold/cuda/kernels/qmm.py` `matmul`), and `single_gemm`
//! tensorfold_qmm times the `group_kernel` alone, so the sums are their own
//! leaves: 5 a layer and 1 for `hidden_projection`, 26 a launch with 5 layers
//! (capture: 5070 / 195). `group_kernel` launches: 6 a layer + 1 = 31 (6045 /
//! 195).
//!
//! **Pending slots** (lane tf-copies): each is built from an `Option` of
//! backends that defaults to absent, so the slot is not minted until the
//! orchestrator forwards a measured backend. Byte-pass slots are already typed
//! as `elementwise` (tf-copies registers its torch passes under that kind) and
//! their byte rates are derived here; `group_sums`, `dconv` and `prep` are typed
//! `elementwise` for now with the bytes of their tf-copies kinds
//! (`row_group_sums`, `dynamic_causal_conv`, `head_major_qk_norm_rope`) and are
//! retyped to those kernels when their Rust specs land.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    DynamicCausalConvKernel, DynamicCausalConvKernelConfig, DynamicCausalConvKernelInput,
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, EmbeddingRowGatherKernel,
    EmbeddingRowGatherKernelConfig, EmbeddingRowGatherKernelInput, HeadMajorQkNormRopeKernel,
    HeadMajorQkNormRopeKernelConfig, HeadMajorQkNormRopeKernelInput, LogitsTopkKernel,
    LogitsTopkKernelConfig, LogitsTopkKernelInput, Nvfp4QuantKernel, Nvfp4QuantKernelConfig,
    Nvfp4QuantKernelInput, RmsNormKernel, RmsNormKernelConfig, RmsNormKernelInput,
    RowGroupSumsKernel, RowGroupSumsKernelConfig, RowGroupSumsKernelInput, SiluAndMulKernel,
    SiluAndMulKernelConfig, SiluAndMulKernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput, SlidingBlockAttentionKernel, SlidingBlockAttentionKernelConfig,
    SlidingBlockAttentionKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge, Probe, SlotInput,
};

use super::tensorfold_common::{
    atomic, elementwise, gemm, push_or_zero, push_sum, repeated, VerifyRound, WindowMode,
    TENSORFOLD_CUDA, TENSORFOLD_LANE, TENSORFOLD_QMM, TENSORFOLD_TORCH, TENSORFOLD_TRITON,
};
use crate::arch::qwen35_dflash2_draft_cfg::Qwen35Dflash2DraftCfg;
use crate::arch::qwen35_model_cfg::Qwen35ModelCfg;

/// TensorFold's `max_rows` default (`multi.py` `MultiEngine(max_rows=16)`):
/// the drafting block on chips outside `DEPTH_CHIPS`, GB10 included.
pub const TENSORFOLD_MAX_ROWS: u32 = 16;

/// The target vocabulary the drafter's head scores (`dflash2.py:223`): token
/// ids `[a, b)` of each span, clipped to the target vocabulary.
pub const TENSORFOLD_DFLASH2_VOCAB_SPANS: [(u32, u32); 2] = [(0, 98304), (248032, 248320)];

/// Inputs a 4-bit affine qmm group (`QLinear.gs`): `_group_sums` writes one
/// fp32 a group.
pub const QMM_GROUP_SIZE: u32 = 64;

/// NVFP4 activation group (`quant4_kernel`).
const NVFP4_GROUP_SIZE: u32 = 16;
/// `nvfp4_quant` tensorfold_cuda's scale layout.
const NVFP4_SCALE_FORMAT: &str = "kslice_e4m3";

const BF16: u32 = 2;
const FP32: u32 = 4;
const I32: u32 = 4;
const I64: u32 = 8;

/// The NVFP4 lm_head's columns for one vocabulary span: the whole 64-column
/// tiles that hold `[a, b)` (`_sub_parts`, `Fp4Linear.tiles`), so the lane
/// GEMM's `n`. `vocab` clips the last tile.
pub fn span_tile_columns((a, b): (u32, u32), vocab: u32) -> u32 {
    let (t0, t1) = (a / 64, b.div_ceil(64));
    (64 * t1).min(vocab) - 64 * t0
}

/// Backends for the slots lane tf-copies is still profiling. `None` leaves the
/// slot out of the tree; the orchestrator forwards each measured backend.
#[derive(Clone, Debug, Default)]
pub struct TensorfoldDflash2DraftPending {
    /// `embedding_row_gather` (`glue.embedding`: id cast and row gather).
    pub embed: Option<Vec<&'static str>>,
    /// `elementwise` tensorfold_torch_rotary: one `_rotary(start, rows)` a stream.
    pub rotary: Option<Vec<&'static str>>,
    /// `elementwise` tensorfold_torch_cat (contiguous cats).
    pub cat: Option<Vec<&'static str>>,
    /// `elementwise` tensorfold_torch_cast (`.float()` and int casts).
    pub cast: Option<Vec<&'static str>>,
    /// `elementwise` tensorfold_torch_strided_copy (the head's row reshape).
    pub strided_copy: Option<Vec<&'static str>>,
    /// `elementwise` tensorfold_torch_cat_strided (the head's span columns).
    pub cat_strided: Option<Vec<&'static str>>,
    /// `elementwise` tensorfold_torch (small index launches).
    pub index: Option<Vec<&'static str>>,
    /// `_group_sums` (`row_group_sums` tensorfold_triton once wired).
    pub group_sums: Option<Vec<&'static str>>,
    /// `_dconv_kernel` (`dynamic_causal_conv` tensorfold_triton once wired).
    pub dconv: Option<Vec<&'static str>>,
    /// `_prep_kernel` (`head_major_qk_norm_rope` tensorfold_triton once wired).
    pub prep: Option<Vec<&'static str>>,
}

#[derive(Clone, Debug)]
pub struct TensorfoldDflash2DraftLocalWorkletConfig {
    pub gpu_name: String,
    pub hidden: u32,
    pub num_layers: u32,
    pub num_qo_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub intermediate: u32,
    pub conv_kernel_width: u32,
    /// Channels per `_dconv` group (`dflash_config.conv_group_size`).
    pub conv_group_size: u32,
    pub selector_rank: u32,
    pub selector_top_k: u32,
    /// Context rows a stream keeps (`sliding_window - 1`).
    pub window: u32,
    pub causal: bool,
    /// Rows a stream's block drafts ([`TENSORFOLD_MAX_ROWS`] on GB10).
    pub block_rows: u32,
    /// The target's vocabulary (its embedding table rows).
    pub target_vocab: u32,
    /// Lane GEMM `n` of each vocabulary span ([`span_tile_columns`]).
    pub head_span_columns: Vec<u32>,
    /// Columns the top-k selects over: the spans' token ids.
    pub head_columns: u32,
    pub pending: TensorfoldDflash2DraftPending,
}

impl TensorfoldDflash2DraftLocalWorkletConfig {
    /// The GB10 config from the drafter and target checkpoints; every pending
    /// slot absent.
    pub fn new(draft: &Qwen35Dflash2DraftCfg, target: &Qwen35ModelCfg, gpu_name: &str) -> Self {
        assert_eq!(
            draft.hidden_size, target.hidden,
            "the drafter reads the target's embedding and lm_head"
        );
        let spans: Vec<(u32, u32)> = TENSORFOLD_DFLASH2_VOCAB_SPANS
            .iter()
            .filter(|&&(a, _)| a < target.vocab_size)
            .map(|&(a, b)| (a, b.min(target.vocab_size)))
            .collect();
        Self {
            gpu_name: gpu_name.to_string(),
            hidden: draft.hidden_size,
            num_layers: draft.num_hidden_layers,
            num_qo_heads: draft.num_attention_heads,
            num_kv_heads: draft.num_key_value_heads,
            head_dim: draft.head_dim,
            intermediate: draft.intermediate_size,
            conv_kernel_width: draft.conv_kernel_width(),
            conv_group_size: draft.dflash_config.conv_group_size,
            selector_rank: draft.dflash_config.selector_rank,
            selector_top_k: draft.dflash_config.selector_top_k,
            window: draft.window(),
            causal: draft.is_causal,
            block_rows: TENSORFOLD_MAX_ROWS,
            target_vocab: target.vocab_size,
            head_span_columns: spans
                .iter()
                .map(|&span| span_tile_columns(span, target.vocab_size))
                .collect(),
            head_columns: spans.iter().map(|&(a, b)| b - a).sum(),
            pending: TensorfoldDflash2DraftPending::default(),
        }
    }

    fn q_width(&self) -> u32 {
        self.num_qo_heads * self.head_dim
    }

    fn qkv_width(&self) -> u32 {
        (self.num_qo_heads + 2 * self.num_kv_heads) * self.head_dim
    }
}

/// A qmm `_group_sums` launch (`row_group_sums`) over bf16 rows of width `k`.
pub(crate) fn row_group_sums(
    backends: &[&'static str],
    gpu: &str,
    k: u32,
) -> RowGroupSumsKernelConfig {
    RowGroupSumsKernelConfig {
        backends: backends.to_vec(),
        gpu_name: gpu.to_string(),
        hidden: k.into(),
        group_size: QMM_GROUP_SIZE.into(),
        dtype: DType::Bf16,
    }
}

/// `_prep_kernel` (`head_major_qk_norm_rope`) over `q_heads` query (0 for the
/// taps) and `kv_heads` key/value heads.
pub(crate) fn head_major_prep(
    backends: &[&'static str],
    gpu: &str,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
) -> HeadMajorQkNormRopeKernelConfig {
    HeadMajorQkNormRopeKernelConfig {
        backends: backends.to_vec(),
        gpu_name: gpu.to_string(),
        num_q_heads: q_heads.into(),
        num_kv_heads: kv_heads.into(),
        head_dim: head_dim.into(),
        dtype: DType::Bf16,
    }
}

#[derive(Clone, Debug)]
pub struct TensorfoldDflash2DraftLocalWorkletResolved {
    pub raw_cfg: TensorfoldDflash2DraftLocalWorkletConfig,
    pub embed: Option<EmbeddingRowGatherKernelConfig>,
    pub rotary: Option<ElementwiseKernelConfig>,
    pub rotary_cat: Option<ElementwiseKernelConfig>,
    pub norm: RmsNormKernelConfig,
    pub group_sums: Option<RowGroupSumsKernelConfig>,
    pub o_group_sums: Option<RowGroupSumsKernelConfig>,
    pub kernel_projection: SingleGemmKernelConfig,
    pub dconv: Option<DynamicCausalConvKernelConfig>,
    pub qkv: SingleGemmKernelConfig,
    pub prep: Option<HeadMajorQkNormRopeKernelConfig>,
    pub lens_cast: Option<ElementwiseKernelConfig>,
    pub attention: SlidingBlockAttentionKernelConfig,
    pub o_proj: SingleGemmKernelConfig,
    pub dconv_residual: Option<DynamicCausalConvKernelConfig>,
    pub gate_up: SingleGemmKernelConfig,
    pub swiglu: SiluAndMulKernelConfig,
    pub down: SingleGemmKernelConfig,
    pub head_rows: Option<ElementwiseKernelConfig>,
    pub hidden_projection: SingleGemmKernelConfig,
    pub selector_cast: Option<ElementwiseKernelConfig>,
    pub head_quant: Nvfp4QuantKernelConfig,
    pub head_lanes: Vec<SingleGemmKernelConfig>,
    pub logits_cat: Option<ElementwiseKernelConfig>,
    pub logits_cast: Option<ElementwiseKernelConfig>,
    pub topk: LogitsTopkKernelConfig,
    pub ids_gather: Option<ElementwiseKernelConfig>,
    pub scores_cat: Option<ElementwiseKernelConfig>,
}

/// One drafter pass: each drafted stream's context length, in stream order.
/// Empty means no stream drafts this round.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TensorfoldDflash2DraftLocalWorkletInput {
    pub context_lens: Vec<u32>,
}

impl TensorfoldDflash2DraftLocalWorkletInput {
    /// From a verify round: every [`WindowMode::Tree`] stream with a non-empty
    /// context, whose context is its last `min(window, committed_len)`
    /// committed rows. The draft runs before the verify, on the context the
    /// previous commit left, which is exactly `committed_len` rows long.
    pub fn from_round(round: &VerifyRound, window: u32) -> Self {
        Self {
            context_lens: round
                .streams
                .iter()
                .filter(|s| s.mode == WindowMode::Tree)
                .map(|s| s.committed_len.min(window))
                .filter(|&len| len > 0)
                .collect(),
        }
    }

    pub fn streams(&self) -> u32 {
        self.context_lens.len() as u32
    }
}

pub struct TensorfoldDflash2DraftLocalWorklet {
    pub name: String,
    pub embed: Option<Op<EmbeddingRowGatherKernel>>,
    pub rotary: Option<Op<ElementwiseKernel>>,
    pub rotary_cat: Option<Op<ElementwiseKernel>>,
    pub norm: Op<RmsNormKernel>,
    pub group_sums: Option<Op<RowGroupSumsKernel>>,
    pub o_group_sums: Option<Op<RowGroupSumsKernel>>,
    pub kernel_projection: Op<SingleGemmKernel>,
    pub dconv: Option<Op<DynamicCausalConvKernel>>,
    pub qkv: Op<SingleGemmKernel>,
    pub prep: Option<Op<HeadMajorQkNormRopeKernel>>,
    pub lens_cast: Option<Op<ElementwiseKernel>>,
    pub attention: Op<SlidingBlockAttentionKernel>,
    pub o_proj: Op<SingleGemmKernel>,
    pub dconv_residual: Option<Op<DynamicCausalConvKernel>>,
    pub gate_up: Op<SingleGemmKernel>,
    pub swiglu: Op<SiluAndMulKernel>,
    pub down: Op<SingleGemmKernel>,
    pub head_rows: Option<Op<ElementwiseKernel>>,
    pub final_norm: Op<RmsNormKernel>,
    pub selector_sums: Option<Op<RowGroupSumsKernel>>,
    pub hidden_projection: Op<SingleGemmKernel>,
    pub selector_cast: Option<Op<ElementwiseKernel>>,
    pub head_quant: Op<Nvfp4QuantKernel>,
    pub head_lanes: Vec<Op<SingleGemmKernel>>,
    pub logits_cat: Option<Op<ElementwiseKernel>>,
    pub logits_cast: Option<Op<ElementwiseKernel>>,
    pub topk: Op<LogitsTopkKernel>,
    pub ids_gather: Option<Op<ElementwiseKernel>>,
    pub scores_cat: Option<Op<ElementwiseKernel>>,
    resolved: TensorfoldDflash2DraftLocalWorkletResolved,
}

impl TensorfoldDflash2DraftLocalWorklet {
    pub fn resolve_config(
        cfg: &TensorfoldDflash2DraftLocalWorkletConfig,
    ) -> TensorfoldDflash2DraftLocalWorkletResolved {
        assert!(
            cfg.block_rows >= 2,
            "a drafted block has its root and drafts"
        );
        assert!(
            cfg.num_qo_heads % cfg.num_kv_heads == 0,
            "query heads must group over kv heads"
        );
        assert!(
            cfg.hidden % QMM_GROUP_SIZE == 0 && cfg.q_width() % QMM_GROUP_SIZE == 0,
            "qmm inputs come in groups of {QMM_GROUP_SIZE}"
        );
        assert!(
            !cfg.head_span_columns.is_empty(),
            "the head scores at least one vocabulary span"
        );
        let gpu = cfg.gpu_name.as_str();
        let p = &cfg.pending;
        let pass = |backends: &Option<Vec<&'static str>>, input: u32, output: u32| {
            backends
                .as_ref()
                .map(|b| elementwise(b, gpu, input, output))
        };
        let hidden_row = cfg.hidden * BF16;
        let qmm = |n: u32, k: u32| gemm(TENSORFOLD_QMM, gpu, n, k, DType::W4a16AffineG64);
        let half = cfg.head_dim / 2;
        let dconv = |residual: bool| {
            p.dconv.as_ref().map(|b| DynamicCausalConvKernelConfig {
                backends: b.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden.into(),
                group_size: cfg.conv_group_size.into(),
                residual,
                dtype: DType::Bf16,
            })
        };
        TensorfoldDflash2DraftLocalWorkletResolved {
            embed: p.embed.as_ref().map(|b| EmbeddingRowGatherKernelConfig {
                backends: b.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads: 1,
                head_dim: cfg.hidden,
                table_rows: u64::from(cfg.target_vocab),
                residency: "device".to_string(),
                weight_dtype: DType::Bf16,
            }),
            // fp32 cos and sin tables, `head_dim / 2` wide; nothing read.
            rotary: pass(&p.rotary, 0, 2 * half * FP32),
            rotary_cat: pass(&p.cat, half * FP32, half * FP32),
            norm: RmsNormKernelConfig {
                backends: vec![TENSORFOLD_TORCH],
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden.into(),
                dtype: DType::Bf16,
            },
            group_sums: p
                .group_sums
                .as_ref()
                .map(|b| row_group_sums(b, gpu, cfg.hidden)),
            o_group_sums: p
                .group_sums
                .as_ref()
                .map(|b| row_group_sums(b, gpu, cfg.q_width())),
            kernel_projection: qmm(cfg.conv_kernel_width, cfg.hidden),
            // `dynamic_causal_conv`: the row mixed with its predecessor, one
            // tap pair per `conv_group_size` channels.
            dconv: dconv(false),
            qkv: qmm(cfg.qkv_width(), cfg.hidden),
            prep: p
                .prep
                .as_ref()
                .map(|b| head_major_prep(b, gpu, cfg.num_qo_heads, cfg.num_kv_heads, cfg.head_dim)),
            // `block_attention`'s int length casts: two int32 -> int64 a
            // stream (tf-copies' wiring table: 4/8 bytes, 2 x streams).
            lens_cast: pass(&p.cast, I32, I64),
            attention: SlidingBlockAttentionKernelConfig {
                backends: vec![TENSORFOLD_TRITON],
                gpu_name: cfg.gpu_name.clone(),
                num_qo_heads: cfg.num_qo_heads.into(),
                num_kv_heads: cfg.num_kv_heads.into(),
                head_dim: cfg.head_dim.into(),
                dtype: DType::Bf16,
                block_rows: cfg.block_rows.into(),
                window: cfg.window.into(),
                causal: cfg.causal,
            },
            o_proj: qmm(cfg.hidden, cfg.q_width()),
            dconv_residual: dconv(true),
            gate_up: qmm(2 * cfg.intermediate, cfg.hidden),
            swiglu: SiluAndMulKernelConfig {
                backends: vec![TENSORFOLD_TRITON],
                gpu_name: cfg.gpu_name.clone(),
                hidden_dim: cfg.intermediate.into(),
                dtype: DType::Bf16,
            },
            down: qmm(cfg.hidden, cfg.intermediate),
            head_rows: pass(&p.strided_copy, hidden_row, hidden_row),
            hidden_projection: qmm(cfg.selector_rank, cfg.hidden),
            selector_cast: pass(&p.cast, cfg.selector_rank * BF16, cfg.selector_rank * FP32),
            head_quant: Nvfp4QuantKernelConfig {
                backends: vec![TENSORFOLD_CUDA],
                gpu_name: cfg.gpu_name.clone(),
                hidden_size: cfg.hidden.into(),
                group_size: NVFP4_GROUP_SIZE,
                input_dtype: DType::Bf16,
                scale_format: NVFP4_SCALE_FORMAT.to_string(),
            },
            head_lanes: cfg
                .head_span_columns
                .iter()
                .map(|&n| gemm(TENSORFOLD_LANE, gpu, n, cfg.hidden, DType::Nvfp4E2m1))
                .collect(),
            logits_cat: pass(
                &p.cat_strided,
                cfg.head_columns * BF16,
                cfg.head_columns * BF16,
            ),
            logits_cast: pass(&p.cast, cfg.head_columns * BF16, cfg.head_columns * FP32),
            topk: LogitsTopkKernelConfig {
                backends: vec![TENSORFOLD_TORCH],
                gpu_name: cfg.gpu_name.clone(),
                num_columns: cfg.head_columns.into(),
                top_k: cfg.selector_top_k,
                dtype: DType::Fp32,
            },
            // `head_ids[local_ids]`: int64 ids in, int64 ids out.
            ids_gather: pass(&p.index, cfg.selector_top_k * I64, cfg.selector_top_k * I64),
            scores_cat: pass(
                &p.cat,
                (cfg.selector_top_k + cfg.selector_rank) * FP32,
                (cfg.selector_top_k + cfg.selector_rank) * FP32,
            ),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: TensorfoldDflash2DraftLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = &resolved;
        let n = name.as_str();
        let pass = |suffix: &str, cfg: &Option<ElementwiseKernelConfig>| {
            optional(n, suffix, cfg.clone(), ElementwiseKernel::build, bridge)
        };
        let gemm_op = |suffix: &str, cfg: &SingleGemmKernelConfig| {
            atomic(n, suffix, cfg.clone(), SingleGemmKernel::build, bridge)
        };
        Ok(Self {
            embed: optional(
                n,
                "embed",
                r.embed.clone(),
                EmbeddingRowGatherKernel::build,
                bridge,
            )?,
            rotary: pass("rotary", &r.rotary)?,
            rotary_cat: pass("rotary_cat", &r.rotary_cat)?,
            norm: atomic(n, "norm", r.norm.clone(), RmsNormKernel::build, bridge)?,
            group_sums: optional(
                n,
                "group_sums",
                r.group_sums.clone(),
                RowGroupSumsKernel::build,
                bridge,
            )?,
            o_group_sums: optional(
                n,
                "o_group_sums",
                r.o_group_sums.clone(),
                RowGroupSumsKernel::build,
                bridge,
            )?,
            kernel_projection: gemm_op("kernel_projection", &r.kernel_projection)?,
            dconv: optional(
                n,
                "dconv",
                r.dconv.clone(),
                DynamicCausalConvKernel::build,
                bridge,
            )?,
            qkv: gemm_op("qkv", &r.qkv)?,
            prep: optional(
                n,
                "prep",
                r.prep.clone(),
                HeadMajorQkNormRopeKernel::build,
                bridge,
            )?,
            lens_cast: pass("lens_cast", &r.lens_cast)?,
            attention: atomic(
                n,
                "attention",
                r.attention.clone(),
                SlidingBlockAttentionKernel::build,
                bridge,
            )?,
            o_proj: gemm_op("o_proj", &r.o_proj)?,
            dconv_residual: optional(
                n,
                "dconv_residual",
                r.dconv_residual.clone(),
                DynamicCausalConvKernel::build,
                bridge,
            )?,
            gate_up: gemm_op("gate_up", &r.gate_up)?,
            swiglu: atomic(
                n,
                "swiglu",
                r.swiglu.clone(),
                SiluAndMulKernel::build,
                bridge,
            )?,
            down: gemm_op("down", &r.down)?,
            head_rows: pass("head_rows", &r.head_rows)?,
            final_norm: atomic(
                n,
                "final_norm",
                r.norm.clone(),
                RmsNormKernel::build,
                bridge,
            )?,
            selector_sums: optional(
                n,
                "selector_sums",
                r.group_sums.clone(),
                RowGroupSumsKernel::build,
                bridge,
            )?,
            hidden_projection: gemm_op("hidden_projection", &r.hidden_projection)?,
            selector_cast: pass("selector_cast", &r.selector_cast)?,
            head_quant: atomic(
                n,
                "head_quant",
                r.head_quant.clone(),
                Nvfp4QuantKernel::build,
                bridge,
            )?,
            head_lanes: r
                .head_lanes
                .iter()
                .enumerate()
                .map(|(i, cfg)| gemm_op(&format!("head_lane_{i}"), cfg))
                .collect::<Result<_, _>>()?,
            logits_cat: pass("logits_cat", &r.logits_cat)?,
            logits_cast: pass("logits_cast", &r.logits_cast)?,
            topk: atomic(n, "topk", r.topk.clone(), LogitsTopkKernel::build, bridge)?,
            ids_gather: pass("ids_gather", &r.ids_gather)?,
            scores_cat: pass("scores_cat", &r.scores_cat)?,
            name,
            resolved,
        })
    }

    fn layer_nodes(&self, builder: &mut CostTreeBuilder) -> Vec<CostNode> {
        let mut nodes = vec![repeated(&self.norm, 2, builder)];
        nodes.extend(compile_opt_n(&self.group_sums, 4, builder));
        nodes.extend(compile_opt(&self.o_group_sums, builder));
        nodes.push(repeated(&self.kernel_projection, 2, builder));
        nodes.extend(compile_opt_n(&self.dconv, 2, builder));
        nodes.push(self.qkv.compile(builder));
        nodes.extend(compile_opt(&self.prep, builder));
        nodes.extend(compile_opt(&self.lens_cast, builder));
        nodes.push(self.attention.compile(builder));
        nodes.push(self.o_proj.compile(builder));
        nodes.extend(compile_opt_n(&self.dconv_residual, 2, builder));
        nodes.push(self.gate_up.compile(builder));
        nodes.push(self.swiglu.compile(builder));
        nodes.push(self.down.compile(builder));
        nodes
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let mut children = Vec::new();
        children.extend(compile_opt(&self.embed, builder));
        children.extend(compile_opt(&self.rotary, builder));
        children.extend(compile_opt_n(&self.rotary_cat, 2, builder));
        let layer = CostNode::Sum(self.layer_nodes(builder));
        children.push(CostNode::Scale {
            n: self.resolved.raw_cfg.num_layers,
            child: Box::new(layer),
        });
        children.extend(compile_opt(&self.head_rows, builder));
        children.push(self.final_norm.compile(builder));
        children.extend(compile_opt(&self.selector_sums, builder));
        children.push(self.hidden_projection.compile(builder));
        children.extend(compile_opt(&self.selector_cast, builder));
        children.push(repeated(
            &self.head_quant,
            self.head_lanes.len() as u32,
            builder,
        ));
        children.extend(self.head_lanes.iter().map(|op| op.compile(builder)));
        children.extend(compile_opt(&self.logits_cat, builder));
        children.extend(compile_opt(&self.logits_cast, builder));
        children.push(self.topk.compile(builder));
        children.extend(compile_opt(&self.ids_gather, builder));
        children.extend(compile_opt(&self.scores_cat, builder));
        let cfg = &self.resolved.raw_cfg;
        CostNode::Labeled {
            label: format!(
                "{} (TensorfoldDflash2DraftLocalWorklet) [layers={}; block={}; window={}; \
                 head spans n={:?}]",
                self.name, cfg.num_layers, cfg.block_rows, cfg.window, cfg.head_span_columns
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    pub fn eval(&self, input: &TensorfoldDflash2DraftLocalWorkletInput, ev: &mut Evaluator) {
        let cfg = &self.resolved.raw_cfg;
        let streams = input.streams();
        let zero = streams == 0;
        let rows = cfg.block_rows * streams;
        let head = (cfg.block_rows - 1) * streams;
        let m = |m| SingleGemmKernelInput { m };
        let ew = |num_tokens| ElementwiseKernelInput { num_tokens };

        eval_opt(
            &self.embed,
            EmbeddingRowGatherKernelInput { num_tokens: rows },
            zero,
            ev,
        );
        if let Some(op) = &self.rotary {
            push_sum(op, &vec![ew(cfg.block_rows); streams as usize], ev);
        }
        eval_opt(&self.rotary_cat, ew(rows), zero, ev);

        // One layer, scaled by `num_layers` at compile.
        push_or_zero(&self.norm, RmsNormKernelInput { m: rows }, zero, ev);
        let sums = |num_tokens| RowGroupSumsKernelInput { num_tokens };
        eval_opt(&self.group_sums, sums(rows), zero, ev);
        eval_opt(&self.o_group_sums, sums(rows), zero, ev);
        push_or_zero(&self.kernel_projection, m(rows), zero, ev);
        let conv = DynamicCausalConvKernelInput { num_tokens: rows };
        eval_opt(&self.dconv, conv.clone(), zero, ev);
        push_or_zero(&self.qkv, m(rows), zero, ev);
        eval_opt(
            &self.prep,
            HeadMajorQkNormRopeKernelInput { num_tokens: rows },
            zero,
            ev,
        );
        eval_opt(&self.lens_cast, ew(2 * streams), zero, ev);
        push_or_zero(
            &self.attention,
            SlidingBlockAttentionKernelInput {
                context_lens: input.context_lens.clone(),
            },
            zero,
            ev,
        );
        push_or_zero(&self.o_proj, m(rows), zero, ev);
        eval_opt(&self.dconv_residual, conv, zero, ev);
        push_or_zero(&self.gate_up, m(rows), zero, ev);
        push_or_zero(
            &self.swiglu,
            SiluAndMulKernelInput { num_tokens: rows },
            zero,
            ev,
        );
        push_or_zero(&self.down, m(rows), zero, ev);

        eval_opt(&self.head_rows, ew(head), zero, ev);
        push_or_zero(&self.final_norm, RmsNormKernelInput { m: head }, zero, ev);
        eval_opt(&self.selector_sums, sums(head), zero, ev);
        push_or_zero(&self.hidden_projection, m(head), zero, ev);
        eval_opt(&self.selector_cast, ew(head), zero, ev);
        push_or_zero(
            &self.head_quant,
            Nvfp4QuantKernelInput { num_tokens: head },
            zero,
            ev,
        );
        for lane in &self.head_lanes {
            push_or_zero(lane, m(head), zero, ev);
        }
        eval_opt(&self.logits_cat, ew(head), zero, ev);
        eval_opt(&self.logits_cast, ew(head), zero, ev);
        push_or_zero(
            &self.topk,
            LogitsTopkKernelInput { num_rows: head },
            zero,
            ev,
        );
        eval_opt(&self.ids_gather, ew(head), zero, ev);
        eval_opt(&self.scores_cat, ew(head), zero, ev);
    }
}

// ─── optional (pending) slots ───────────────────────────────────────────────

/// Build an op only when its config is present.
pub(crate) fn optional<K, C, F>(
    prefix: &str,
    suffix: &str,
    config: Option<C>,
    build_kernel: F,
    bridge: &PerfApiBridge,
) -> Result<Option<Op<K>>, BuildError>
where
    K: Probe,
    F: FnOnce(String, C, &PerfApiBridge) -> Result<K, BuildError>,
{
    config
        .map(|config| atomic(prefix, suffix, config, build_kernel, bridge))
        .transpose()
}

/// An optional slot's leaf, if it is built.
pub(crate) fn compile_opt<K: Probe>(
    op: &Option<Op<K>>,
    builder: &mut CostTreeBuilder,
) -> Option<CostNode> {
    op.as_ref().map(|op| op.compile(builder))
}

/// An optional slot launched `n` identical times.
pub(crate) fn compile_opt_n<K: Probe>(
    op: &Option<Op<K>>,
    n: u32,
    builder: &mut CostTreeBuilder,
) -> Option<CostNode> {
    op.as_ref().map(|op| repeated(op, n, builder))
}

/// Fill an optional slot, if it is built (zero when the launch is skipped).
pub(crate) fn eval_opt<K>(op: &Option<Op<K>>, input: K::Input, zero: bool, ev: &mut Evaluator)
where
    K: Probe,
    K::Input: Clone + Into<SlotInput>,
{
    if let Some(op) = op {
        push_or_zero(op, input, zero, ev);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::timing::LeafMetrics;
    use std::path::Path;

    const GB10: &str = "NVIDIA GB10";

    fn cfgs() -> (Qwen35Dflash2DraftCfg, Qwen35ModelCfg) {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        (
            Qwen35Dflash2DraftCfg::from_json(&root.join("model/config/qwen3_8_27b_dflash2.json"))
                .unwrap(),
            Qwen35ModelCfg::from_json(&root.join("model/config/qwen3_8_27b_nvfp4.json")).unwrap(),
        )
    }

    pub(crate) fn config(all_pending: bool) -> TensorfoldDflash2DraftLocalWorkletConfig {
        let (draft, target) = cfgs();
        let mut cfg = TensorfoldDflash2DraftLocalWorkletConfig::new(&draft, &target, GB10);
        if all_pending {
            let b = Some(vec!["tensorfold_pending"]);
            cfg.pending = TensorfoldDflash2DraftPending {
                embed: b.clone(),
                rotary: b.clone(),
                cat: b.clone(),
                cast: b.clone(),
                strided_copy: b.clone(),
                cat_strided: b.clone(),
                index: b.clone(),
                group_sums: b.clone(),
                dconv: b.clone(),
                prep: b,
            };
        }
        cfg
    }

    fn worklet(all_pending: bool) -> (TensorfoldDflash2DraftLocalWorklet, crate::timing::CostTree) {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let w = TensorfoldDflash2DraftLocalWorklet::build(
            "draft".into(),
            TensorfoldDflash2DraftLocalWorklet::resolve_config(&config(all_pending)),
            &bridge,
        )
        .unwrap();
        let mut builder = CostTreeBuilder::new();
        let root = w.compile(&mut builder);
        (w, builder.finish(root))
    }

    #[test]
    fn head_spans_are_the_nvfp4_tile_views() {
        let cfg = config(false);
        // (0, 98304) is whole tiles; (248032, 248320) sits in tiles 3875..3880.
        assert_eq!(cfg.head_span_columns, [98304, 320]);
        assert_eq!(cfg.head_columns, 98592);
        assert_eq!((cfg.block_rows, cfg.window), (16, 2047));
    }

    #[test]
    fn shapes_follow_the_drafter_config() {
        let r = TensorfoldDflash2DraftLocalWorklet::resolve_config(&config(true));
        let nk = |g: &SingleGemmKernelConfig| (g.n.get(), g.k.get());
        assert_eq!(nk(&r.kernel_projection), (1280, 5120));
        assert_eq!(nk(&r.qkv), (6144, 5120));
        assert_eq!(nk(&r.o_proj), (5120, 4096));
        assert_eq!(nk(&r.gate_up), (34816, 5120));
        assert_eq!(nk(&r.down), (5120, 17408));
        assert_eq!(nk(&r.hidden_projection), (256, 5120));
        assert_eq!(r.qkv.dtype, DType::W4a16AffineG64);
        assert_eq!(r.head_lanes[0].dtype, DType::Nvfp4E2m1);
        assert_eq!(r.swiglu.hidden_dim.get(), 17408);
        // `_group_sums` over k = 5120 in groups of 64; `_dconv` groups of 16;
        // `_prep` over 32 query and 8 kv heads.
        let sums = r.group_sums.unwrap();
        assert_eq!((sums.hidden.get(), sums.group_size.get()), (5120, 64));
        assert_eq!(r.o_group_sums.unwrap().hidden.get(), 4096);
        let conv = r.dconv_residual.unwrap();
        assert_eq!((conv.group_size.get(), conv.residual), (16, true));
        let prep = r.prep.unwrap();
        assert_eq!((prep.num_q_heads.get(), prep.num_kv_heads.get()), (32, 8));
    }

    #[test]
    fn leaves_without_pending_slots() {
        let (_, tree) = worklet(false);
        let names: Vec<&str> = tree.slots.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "draft.norm",
                "draft.kernel_projection",
                "draft.qkv",
                "draft.attention",
                "draft.o_proj",
                "draft.gate_up",
                "draft.swiglu",
                "draft.down",
                "draft.final_norm",
                "draft.hidden_projection",
                "draft.head_quant",
                "draft.head_lane_0",
                "draft.head_lane_1",
                "draft.topk",
            ]
        );
    }

    /// Launches a draft pass costs per kernel name, from the compiled tree:
    /// each leaf counts its `Scale` multiplicity.
    fn launches(tree: &crate::timing::CostTree) -> std::collections::BTreeMap<String, u32> {
        fn walk(
            node: &CostNode,
            mult: u32,
            tree: &crate::timing::CostTree,
            out: &mut std::collections::BTreeMap<String, u32>,
        ) {
            match node {
                CostNode::Leaf(slot) => {
                    let name = tree.slots[*slot]
                        .name
                        .trim_start_matches("draft.")
                        .to_string();
                    *out.entry(name).or_default() += mult;
                }
                CostNode::Sum(children)
                | CostNode::Max { children, .. }
                | CostNode::Parallel { children, .. } => {
                    children.iter().for_each(|c| walk(c, mult, tree, out))
                }
                CostNode::Scale { n, child } => walk(child, mult * n, tree, out),
                CostNode::Labeled { child, .. } => walk(child, mult, tree, out),
            }
        }
        let mut out = Default::default();
        walk(&tree.root, 1, tree, &mut out);
        out
    }

    #[test]
    fn launch_counts_match_the_n8_capture() {
        let (_, tree) = worklet(true);
        let n = launches(&tree);
        let qmm = [
            "kernel_projection",
            "qkv",
            "o_proj",
            "gate_up",
            "down",
            "hidden_projection",
        ];
        // group_kernel: 31 per draft launch (6045 / 195).
        assert_eq!(qmm.iter().map(|k| n[*k]).sum::<u32>(), 31);
        // _group_sums: 26 (5070 / 195).
        assert_eq!(n["group_sums"] + n["o_group_sums"] + n["selector_sums"], 26);
        // _dconv_kernel 20 (3900 / 195), _prep 5, _block_attention 5, _swiglu 5.
        assert_eq!(n["dconv"] + n["dconv_residual"], 20);
        assert_eq!((n["prep"], n["attention"], n["swiglu"]), (5, 5, 5));
        // vectorized_layer_norm_kernel 11 (2145 / 195).
        assert_eq!(n["norm"] + n["final_norm"], 11);
        // lane_kernel 2, quant4_kernel 2 (390 / 195 each).
        assert_eq!(n["head_lane_0"] + n["head_lane_1"], 2);
        assert_eq!(n["head_quant"], 2);
        // unrolled_elementwise_kernel 8 (1560 / 195): 5 length casts, the
        // projected and logits casts, and the embedding's id cast.
        assert_eq!(n["lens_cast"] + n["selector_cast"] + n["logits_cast"], 7);
        // CatArrayBatchedCopy_vectorized 3 (585 / 195), CatArrayBatchedCopy 1.
        assert_eq!(n["rotary_cat"] + n["scores_cat"], 3);
        assert_eq!(n["logits_cat"], 1);
        assert_eq!(n["topk"], 1);
    }

    #[test]
    fn an_empty_pass_is_zero_and_fills_every_slot() {
        let (w, tree) = worklet(true);
        let mut buf = vec![LeafMetrics::MISS; tree.n_slots()];
        let mut ev = Evaluator::new(&mut buf);
        w.eval(&TensorfoldDflash2DraftLocalWorkletInput::default(), &mut ev);
        assert_eq!(ev.filled(), tree.n_slots());
        assert!(buf.iter().all(|m| m.m.time_ms == 0.0));
    }

    #[test]
    fn only_tree_streams_with_context_draft() {
        use super::super::tensorfold_common::VerifyStream;
        let s = |committed_len, mode| VerifyStream {
            committed_len,
            rows: 4,
            accepted: 1,
            tree_slots: 0,
            mode,
        };
        let round = VerifyRound {
            streams: vec![
                s(33465, WindowMode::Tree),
                s(500, WindowMode::Copy),
                s(700, WindowMode::Tree),
                s(9, WindowMode::One),
                s(0, WindowMode::Tree),
            ],
        };
        let input = TensorfoldDflash2DraftLocalWorkletInput::from_round(&round, 2047);
        assert_eq!(input.context_lens, [2047, 700]);
    }

    /// One captured round (`wt-tf-spark/logs/20261023_5_tf27b_batch`,
    /// `tf27b_n{n}.sqlite` matched to `iter_records.jsonl`): each stream's
    /// `(committed_len, rows, accepted, mode)` with mode 0 copy / 1 tree /
    /// 2 one, then the measured busy ms of the kernels these worklets price
    /// without pending slots, and of the whole span.
    pub(crate) struct CapturedRound {
        pub n: u32,
        pub streams: &'static [(u32, u32, u32, u8)],
        /// draft.launch: group_kernel, _block_attention, layer norms,
        /// _swiglu, quant4, lane and the top-k kernels.
        pub draft_modeled_ms: f64,
        pub draft_span_ms: f64,
        /// engine.commit replay_kernel + draft.taps group_kernel and _append.
        pub commit_modeled_ms: f64,
        /// engine.commit with its nested draft.taps.
        pub commit_span_ms: f64,
    }

    /// The median no-prefill round with a tree stream of each capture.
    pub(crate) const CAPTURED_ROUNDS: &[CapturedRound] = &[
        CapturedRound {
            n: 1,
            streams: &[(31623, 16, 3, 1)],
            draft_modeled_ms: 6.1985,
            draft_span_ms: 6.3326,
            commit_modeled_ms: 2.2444,
            commit_span_ms: 2.7322,
        },
        CapturedRound {
            n: 2,
            streams: &[(70809, 16, 1, 0), (45243, 16, 4, 1)],
            draft_modeled_ms: 6.2534,
            draft_span_ms: 6.391,
            commit_modeled_ms: 3.9622,
            commit_span_ms: 5.0295,
        },
        CapturedRound {
            n: 4,
            streams: &[
                (24142, 16, 8, 1),
                (32504, 16, 10, 1),
                (34251, 16, 4, 1),
                (18234, 16, 1, 0),
            ],
            draft_modeled_ms: 7.424,
            draft_span_ms: 7.7458,
            commit_modeled_ms: 7.8217,
            commit_span_ms: 9.0098,
        },
        CapturedRound {
            n: 8,
            streams: &[
                (33734, 13, 4, 1),
                (27339, 16, 0, 0),
                (21998, 13, 3, 1),
                (21115, 9, 1, 1),
                (26592, 16, 2, 1),
                (14617, 13, 3, 1),
                (8910, 16, 0, 1),
                (7631, 16, 2, 1),
            ],
            draft_modeled_ms: 12.3394,
            draft_span_ms: 13.1318,
            commit_modeled_ms: 14.7773,
            commit_span_ms: 16.4629,
        },
    ];

    pub(crate) fn round_of(streams: &[(u32, u32, u32, u8)]) -> VerifyRound {
        use super::super::tensorfold_common::VerifyStream;
        VerifyRound {
            streams: streams
                .iter()
                .map(|&(committed_len, rows, accepted, mode)| VerifyStream {
                    committed_len,
                    rows,
                    accepted,
                    tree_slots: 0,
                    mode: match mode {
                        0 => WindowMode::Copy,
                        1 => WindowMode::Tree,
                        _ => WindowMode::One,
                    },
                })
                .collect(),
        }
    }

    /// Every captured round, from the JSON that `TENSORFOLD_CAPTURED_ROUNDS`
    /// names (`[{"n", "streams": [[pos, rows, accepted, mode]], "d_mod",
    /// "d_all", "c_mod", "c_all", "prefill"}]`), as `(n, round, draft
    /// modeled, draft span, commit modeled, commit span)`.
    pub(crate) fn captured_rounds_from_env() -> Vec<(u32, VerifyRound, [f64; 4])> {
        let Ok(path) = std::env::var("TENSORFOLD_CAPTURED_ROUNDS") else {
            return Vec::new();
        };
        let rows: Vec<serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        rows.iter()
            .map(|r| {
                let streams: Vec<(u32, u32, u32, u8)> = r["streams"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|s| {
                        let v = |i: usize| s[i].as_u64().unwrap() as u32;
                        (v(0), v(1), v(2), v(3) as u8)
                    })
                    .collect();
                let f = |k: &str| r[k].as_f64().unwrap();
                (
                    r["n"].as_u64().unwrap() as u32,
                    round_of(&streams),
                    [f("d_mod"), f("d_all"), f("c_mod"), f("c_all")],
                )
            })
            .collect()
    }

    /// Prices draft passes on the GB10 rows in `profiling/profile.db` and
    /// compares them with the captured `tensorfold.draft.launch` time of the
    /// same rounds. Run from the repo root with libpython on
    /// `LD_LIBRARY_PATH`: `PYTHONPATH=$PWD uv run cargo test -p simulator --lib
    /// tensorfold_dflash2_draft_local -- --include-ignored --nocapture`; set
    /// `TENSORFOLD_CAPTURED_ROUNDS` to also sum every captured round.
    #[test]
    #[ignore = "needs profile.db GB10 rows and the venv interpreter"]
    fn gb10_rows_price_captured_draft_passes() {
        let bridge = PerfApiBridge::new().expect("perf_api bridge");
        let cfg = config(false);
        let window = cfg.window;
        let w = TensorfoldDflash2DraftLocalWorklet::build(
            "draft".into(),
            TensorfoldDflash2DraftLocalWorklet::resolve_config(&cfg),
            &bridge,
        )
        .expect("drafter rows on GB10");
        let mut builder = CostTreeBuilder::new();
        let root = w.compile(&mut builder);
        let tree = builder.finish(root);
        let flat = tree.flatten();
        let mut buf = vec![LeafMetrics::MISS; tree.n_slots()];
        let mut price = |round: &VerifyRound| {
            let input = TensorfoldDflash2DraftLocalWorkletInput::from_round(round, window);
            let mut ev = Evaluator::new(&mut buf);
            w.eval(&input, &mut ev);
            assert_eq!(ev.filled(), tree.n_slots());
            f64::from(
                crate::timing::CostTree::aggregate(&flat, &buf, &mut Vec::new())
                    .m
                    .time_ms,
            )
        };
        for c in CAPTURED_ROUNDS {
            let round = round_of(c.streams);
            let ms = price(&round);
            eprintln!(
                "draft n{} S={}: predicted {ms:.3} ms; captured modeled kernels {:.3} ms \
                 ({:+.1}%), span {:.3} ms ({:+.1}%)",
                c.n,
                round.tree_streams(),
                c.draft_modeled_ms,
                100.0 * (ms / c.draft_modeled_ms - 1.0),
                c.draft_span_ms,
                100.0 * (ms / c.draft_span_ms - 1.0),
            );
            assert!(ms > 0.0 && ms.is_finite());
        }
        let mut sums = std::collections::BTreeMap::<u32, [f64; 4]>::new();
        for (n, round, measured) in captured_rounds_from_env() {
            let e = sums.entry(n).or_default();
            e[0] += price(&round);
            e[1] += measured[0];
            e[2] += measured[1];
            e[3] += 1.0;
        }
        for (n, [ms, modeled, span, rounds]) in sums {
            eprintln!(
                "draft n{n} all {rounds} rounds: predicted {ms:.1} ms; modeled kernels \
                 {modeled:.1} ms ({:+.1}%), span {span:.1} ms ({:+.1}%)",
                100.0 * (ms / modeled - 1.0),
                100.0 * (ms / span - 1.0),
            );
        }
    }
}
