//! TensorFold's round commit on GB10 and the DFlash2 taps around it: what a
//! round costs beyond the drafter pass and the verify forward's layers.
//!
//! Sources (`tensorfold/families/qwen3_5/cuda/`): `forward.py`
//! `multi_tree_forward` (the tap capture), `commit_streams` / `_commit`,
//! `path_indices`; `multi.py` `_commit` (span `tensorfold.engine.commit`);
//! `dflash2.py` `add_taps_streams` (span `tensorfold.draft.taps`, nested in the
//! commit).
//!
//! **Input** is the round itself ([`VerifyRound`]). Per stream it reads:
//! `rows` (W = Σ rows), `accepted` (each stream commits its accepted path plus
//! the root: A = Σ(accepted + 1)), `committed_len` (the drafter context before
//! the taps, `min(window, committed_len)`) and `mode`. A stream adds drafter
//! taps when it drafts (`s.draft`): in TensorFold a stream with a drafter is
//! in COPY or TREE mode, and ONE mode means it does not draft. The verify
//! forward captures taps when any stream drafts.
//!
//! **Leaves, in launch order** (n8 capture, a round of S streams, D drafting):
//!
//! | Slot | Kernel (capture name) | Kind / backend | Launches |
//! |---|---|---|---|
//! | verify forward, tap capture (owned here, not by the MLP lane): | | | |
//! | `tap_add` | `(x.float() + pending.float()).to(bf16)`: 2 unrolled + 2 vectorized | `elementwise` / tensorfold_torch_fp32_add (pending) | taps x W |
//! | `tap_cat` | `CatArrayBatchedCopy_vectorized`: `torch.cat(taps, -1)` | `elementwise` / tensorfold_torch_cat (pending) | 1 x W |
//! | `engine.commit`: | | | |
//! | `take_cast` | `unrolled_elementwise_kernel`: each path's rows `.long()` | `elementwise` / tensorfold_torch_cast (pending) | S x (accepted + 1) |
//! | `replay` | `replay_kernel` | `gdn_state_replay` / tensorfold_cuda | 1 (`gdn_replay_input`) |
//! | `conv_cat` | `CatArrayBatchedCopy_vectorized`: the streams' conv windows, S > 1 | `elementwise` / tensorfold_torch_cat (pending) | gdn x S·keep |
//! | `conv_src_cat` | cats of [old windows, QKV rows] (one a layer; or three stacks when W ≤ 32) | `elementwise` / tensorfold_torch_cat (pending) | see `conv_launches` |
//! | `conv_pick` | `vectorized_gather_kernel`: `index_select(0, pick)` a layer, W > 32 | `indexed_row_copy` / tensorfold_torch (pending) | gdn x S·keep |
//! | `conv_pick_stacked` | `index_select(1, pick)` of every layer at once, W ≤ 32 | `indexed_row_copy` / tensorfold_torch (pending), row = gdn x C | 1 x S·keep |
//! | `conv_copy` | `multi_tensor_apply_kernel`: `_foreach_copy_` of the windows | `elementwise` / tensorfold_torch_foreach_copy (pending) | 1 x gdn·S·keep |
//! | `take_cat` | `CatArrayBatchedCopy_alignedK_contig`: the int64 takes, S > 1 | `elementwise` / tensorfold_torch_cat (pending) | 1 x A |
//! | `kv_gather` | `vectorized_gather_kernel`: keys and values `index_select` | `indexed_row_copy` / tensorfold_torch (pending) | 2·gqa x A |
//! | `kv_copy` | `multi_tensor_apply_kernel`: `_foreach_copy_` into every cache | `elementwise` / tensorfold_torch_foreach_copy (pending) | 1 x 2·gqa·A |
//! | `tap_rows` | `indexSelectSmallIndex`: `taps.index_select(0, take)` a drafting stream | `indexed_row_copy` / tensorfold_torch (pending) | D x (accepted + 1) |
//! | `draft.taps` (`add_taps_streams`), A_d = Σ(accepted + 1) of drafting streams: | | | |
//! | `taps_cat` | `CatArrayBatchedCopy_vectorized`, D > 1 | `elementwise` / tensorfold_torch_cat (pending) | 1 x A_d |
//! | `fc_sums` | `_group_sums`, k = tap width | `row_group_sums` (pending) | 1 x A_d |
//! | `fc` | `group_kernel` n = hidden, k = tap width | `single_gemm` / tensorfold_qmm | 1 x A_d |
//! | `fc_norm` | `_norm`: eager fp32 RMSNorm (reduce_kernel + 8 pointwise) | `rms_norm` / tensorfold_torch_eager (pending) | 1 x A_d |
//! | `taps_rotary` | `pos[:, None] * inv_freq`, cos, sin | `elementwise` / tensorfold_torch_rotary (pending) | 1 x A_d |
//! | per drafter layer (`Scale{layers}`): | | | |
//! | `kv_sums` | `_group_sums`, k = hidden | `row_group_sums` (pending) | 1 x A_d |
//! | `kv` | `group_kernel` n = 2 x kv heads x head_dim | `single_gemm` / tensorfold_qmm | 1 x A_d |
//! | `kv_prep` | `_prep_kernel` with no query heads | `head_major_qk_norm_rope` (pending) | 1 x A_d |
//! | `append` | `_append`, keys then values | `sliding_window_kv_append` / tensorfold_triton | 2 (D, Σ context, A_d) |
//!
//! `keep` is the target's `conv_kernel - 1` rows, C its GDN conv channels, gdn
//! and gqa its layer counts. Pending slots follow the drafter worklet's rule
//! (built from an `Option` of backends, absent by default; see
//! `tensorfold_dflash2_draft_local`).
//!
//! Prefill taps (`prefill.py` `draft.add_taps`, one stream, the last ≤ 2047
//! rows of a prompt) are not here: they run the single-stream `_add_taps`
//! (cat + window trim instead of `_append`) inside the prefill step.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, GdnStateReplayKernel,
    GdnStateReplayKernelConfig, HeadMajorQkNormRopeKernel, HeadMajorQkNormRopeKernelConfig,
    HeadMajorQkNormRopeKernelInput, IndexedRowCopyKernel, IndexedRowCopyKernelConfig,
    IndexedRowCopyKernelInput, RmsNormKernel, RmsNormKernelConfig, RmsNormKernelInput,
    RowGroupSumsKernel, RowGroupSumsKernelConfig, RowGroupSumsKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput, SlidingWindowKvAppendKernel,
    SlidingWindowKvAppendKernelConfig, SlidingWindowKvAppendKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge};

use super::tensorfold_common::{
    atomic, elementwise, gemm, push_or_zero, push_sum, VerifyRound, WindowMode, TENSORFOLD_CUDA,
    TENSORFOLD_QMM, TENSORFOLD_TRITON,
};
use super::tensorfold_dflash2_draft_local::{
    compile_opt, compile_opt_n, eval_opt, head_major_prep, optional, row_group_sums,
};
use crate::arch::qwen35_dflash2_draft_cfg::Qwen35Dflash2DraftCfg;
use crate::arch::qwen35_model_cfg::{Qwen35Mixer, Qwen35ModelCfg};

/// `_commit` gathers every GDN layer's conv windows at once when the verify
/// record has at most this many rows (`items[0].qkv.shape[0] <= 32`).
pub const FEW_ROWS: u32 = 32;

const BF16: u32 = 2;
const FP32: u32 = 4;
const I32: u32 = 4;
const I64: u32 = 8;

/// Backends for the slots lane tf-copies is still profiling; `None` leaves
/// the slot out of the tree.
#[derive(Clone, Debug, Default)]
pub struct TensorfoldRoundCommitPending {
    /// `elementwise` tensorfold_torch_fp32_add (one tap: four launches).
    pub fp32_add: Option<Vec<&'static str>>,
    /// `elementwise` tensorfold_torch_cat.
    pub cat: Option<Vec<&'static str>>,
    /// `elementwise` tensorfold_torch_cast.
    pub cast: Option<Vec<&'static str>>,
    /// `indexed_row_copy` tensorfold_torch (`index_select`).
    pub gather: Option<Vec<&'static str>>,
    /// `elementwise` tensorfold_torch_foreach_copy.
    pub foreach_copy: Option<Vec<&'static str>>,
    /// `rms_norm` tensorfold_torch_eager (the taps' `_norm`).
    pub eager_norm: Option<Vec<&'static str>>,
    /// `elementwise` tensorfold_torch_rotary.
    pub rotary: Option<Vec<&'static str>>,
    /// `_group_sums` (`row_group_sums` once wired).
    pub group_sums: Option<Vec<&'static str>>,
    /// `_prep_kernel` (`head_major_qk_norm_rope` once wired).
    pub prep: Option<Vec<&'static str>>,
}

#[derive(Clone, Debug)]
pub struct TensorfoldRoundCommitLocalWorkletConfig {
    pub gpu_name: String,
    pub hidden: u32,
    /// Target layers tapped, and one committed row's tap width.
    pub num_taps: u32,
    pub tap_width: u32,
    pub gdn_layers: u32,
    pub gqa_layers: u32,
    pub gdn_key_heads: u32,
    pub gdn_value_heads: u32,
    pub gdn_key_head_dim: u32,
    pub gdn_value_head_dim: u32,
    pub gdn_state_dtype: DType,
    /// GDN conv channels (C) and the window rows a stream keeps.
    pub conv_channels: u32,
    pub conv_keep: u32,
    /// One cached key (or value) row: target kv heads x head_dim.
    pub kv_row: u32,
    pub draft_layers: u32,
    pub draft_kv_heads: u32,
    pub draft_head_dim: u32,
    pub draft_window: u32,
    pub pending: TensorfoldRoundCommitPending,
}

impl TensorfoldRoundCommitLocalWorkletConfig {
    pub fn new(draft: &Qwen35Dflash2DraftCfg, target: &Qwen35ModelCfg, gpu_name: &str) -> Self {
        assert_eq!(draft.hidden_size, target.hidden, "taps are target rows");
        let count = |mixer| target.mixers.iter().filter(|&&m| m == mixer).count() as u32;
        Self {
            gpu_name: gpu_name.to_string(),
            hidden: target.hidden,
            num_taps: draft.num_taps(),
            tap_width: draft.tap_width(),
            gdn_layers: count(Qwen35Mixer::Gdn),
            gqa_layers: count(Qwen35Mixer::Gqa),
            gdn_key_heads: target.gdn_key_heads,
            gdn_value_heads: target.gdn_value_heads,
            gdn_key_head_dim: target.gdn_key_head_dim,
            gdn_value_head_dim: target.gdn_value_head_dim,
            gdn_state_dtype: target.gdn_state_dtype,
            conv_channels: target.gdn_conv_channels(),
            conv_keep: target.gdn_conv_kernel - 1,
            kv_row: target.gqa_kv_heads * target.gqa_head_dim,
            draft_layers: draft.num_hidden_layers,
            draft_kv_heads: draft.num_key_value_heads,
            draft_head_dim: draft.head_dim,
            draft_window: draft.window(),
            pending: TensorfoldRoundCommitPending::default(),
        }
    }

    fn draft_kv_width(&self) -> u32 {
        2 * self.draft_kv_heads * self.draft_head_dim
    }
}

#[derive(Clone, Debug)]
pub struct TensorfoldRoundCommitLocalWorkletResolved {
    pub raw_cfg: TensorfoldRoundCommitLocalWorkletConfig,
    pub tap_add: Option<ElementwiseKernelConfig>,
    pub tap_cat: Option<ElementwiseKernelConfig>,
    pub take_cast: Option<ElementwiseKernelConfig>,
    pub replay: GdnStateReplayKernelConfig,
    pub conv_cat: Option<ElementwiseKernelConfig>,
    pub conv_pick: Option<IndexedRowCopyKernelConfig>,
    pub conv_pick_stacked: Option<IndexedRowCopyKernelConfig>,
    pub conv_copy: Option<ElementwiseKernelConfig>,
    pub take_cat: Option<ElementwiseKernelConfig>,
    pub kv_gather: Option<IndexedRowCopyKernelConfig>,
    pub kv_copy: Option<ElementwiseKernelConfig>,
    pub tap_rows: Option<IndexedRowCopyKernelConfig>,
    pub fc_sums: Option<RowGroupSumsKernelConfig>,
    pub fc: SingleGemmKernelConfig,
    pub fc_norm: Option<RmsNormKernelConfig>,
    pub taps_rotary: Option<ElementwiseKernelConfig>,
    pub kv_sums: Option<RowGroupSumsKernelConfig>,
    pub kv: SingleGemmKernelConfig,
    pub kv_prep: Option<HeadMajorQkNormRopeKernelConfig>,
    pub append: SlidingWindowKvAppendKernelConfig,
}

/// The commit costs one verify round: [`VerifyRound`] as is.
pub type TensorfoldRoundCommitLocalWorkletInput = VerifyRound;

pub struct TensorfoldRoundCommitLocalWorklet {
    pub name: String,
    pub tap_add: Option<Op<ElementwiseKernel>>,
    pub tap_cat: Option<Op<ElementwiseKernel>>,
    pub take_cast: Option<Op<ElementwiseKernel>>,
    pub replay: Op<GdnStateReplayKernel>,
    pub conv_cat: Option<Op<ElementwiseKernel>>,
    pub conv_src_cat: Option<Op<ElementwiseKernel>>,
    pub conv_pick: Option<Op<IndexedRowCopyKernel>>,
    pub conv_pick_stacked: Option<Op<IndexedRowCopyKernel>>,
    pub conv_copy: Option<Op<ElementwiseKernel>>,
    pub take_cat: Option<Op<ElementwiseKernel>>,
    pub kv_gather: Option<Op<IndexedRowCopyKernel>>,
    pub kv_copy: Option<Op<ElementwiseKernel>>,
    pub tap_rows: Option<Op<IndexedRowCopyKernel>>,
    pub taps_cat: Option<Op<ElementwiseKernel>>,
    pub fc_sums: Option<Op<RowGroupSumsKernel>>,
    pub fc: Op<SingleGemmKernel>,
    pub fc_norm: Option<Op<RmsNormKernel>>,
    pub taps_rotary: Option<Op<ElementwiseKernel>>,
    pub kv_sums: Option<Op<RowGroupSumsKernel>>,
    pub kv: Op<SingleGemmKernel>,
    pub kv_prep: Option<Op<HeadMajorQkNormRopeKernel>>,
    pub append: Op<SlidingWindowKvAppendKernel>,
    resolved: TensorfoldRoundCommitLocalWorkletResolved,
}

/// The per-round shapes the commit's leaves read, from the round.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CommitShape {
    pub streams: u32,
    /// W: verify rows.
    pub rows: u32,
    /// Each stream's committed rows (`accepted + 1`), in stream order.
    pub committed: Vec<u32>,
    /// Each drafting stream's committed rows and its drafter context before
    /// the taps.
    pub drafting: Vec<(u32, u32)>,
}

impl CommitShape {
    pub fn of(round: &VerifyRound, window: u32) -> Self {
        Self {
            streams: round.num_streams(),
            rows: round.total_rows(),
            committed: round.streams.iter().map(|s| s.accepted + 1).collect(),
            drafting: round
                .streams
                .iter()
                .filter(|s| s.mode != WindowMode::One)
                .map(|s| (s.accepted + 1, s.committed_len.min(window)))
                .collect(),
        }
    }

    /// A: rows committed over every stream.
    pub fn committed_rows(&self) -> u32 {
        self.committed.iter().sum()
    }

    /// A_d: tap rows over drafting streams.
    pub fn tap_rows(&self) -> u32 {
        self.drafting.iter().map(|d| d.0).sum()
    }
}

impl TensorfoldRoundCommitLocalWorklet {
    pub fn resolve_config(
        cfg: &TensorfoldRoundCommitLocalWorkletConfig,
    ) -> TensorfoldRoundCommitLocalWorkletResolved {
        assert!(cfg.conv_keep >= 1, "a GDN conv keeps at least one row");
        let gpu = cfg.gpu_name.as_str();
        let p = &cfg.pending;
        let pass = |backends: &Option<Vec<&'static str>>, input: u32, output: u32| {
            backends
                .as_ref()
                .map(|b| elementwise(b, gpu, input, output))
        };
        let select = |row_elements: u32| {
            p.gather.as_ref().map(|b| IndexedRowCopyKernelConfig {
                backends: b.clone(),
                gpu_name: cfg.gpu_name.clone(),
                row_elements: row_elements.into(),
                dtype: DType::Bf16,
                operation: "index_select".to_string(),
            })
        };
        let hidden_row = cfg.hidden * BF16;
        let conv_row = cfg.conv_channels * BF16;
        let tap_row = cfg.tap_width * BF16;
        let half = cfg.draft_head_dim / 2;
        TensorfoldRoundCommitLocalWorkletResolved {
            // Two bf16 rows in, one out (the four launches of one tap).
            tap_add: pass(&p.fp32_add, 2 * hidden_row, hidden_row),
            tap_cat: pass(&p.cat, tap_row, tap_row),
            take_cast: pass(&p.cast, I32, I64),
            replay: GdnStateReplayKernelConfig {
                backends: vec![TENSORFOLD_CUDA],
                gpu_name: cfg.gpu_name.clone(),
                num_layers: cfg.gdn_layers.into(),
                num_qk_heads: cfg.gdn_key_heads.into(),
                num_value_heads: cfg.gdn_value_heads.into(),
                key_head_dim: cfg.gdn_key_head_dim.into(),
                value_head_dim: cfg.gdn_value_head_dim.into(),
                dtype: DType::Bf16,
                state_dtype: cfg.gdn_state_dtype,
            },
            conv_cat: pass(&p.cat, conv_row, conv_row),
            conv_pick: select(cfg.conv_channels),
            conv_pick_stacked: select(cfg.gdn_layers * cfg.conv_channels),
            // One stream's windows of every GDN layer a unit.
            conv_copy: pass(
                &p.foreach_copy,
                cfg.gdn_layers * cfg.conv_keep * conv_row,
                cfg.gdn_layers * cfg.conv_keep * conv_row,
            ),
            take_cat: pass(&p.cat, I64, I64),
            kv_gather: select(cfg.kv_row),
            // One committed row's K and V of every GQA layer a unit.
            kv_copy: pass(
                &p.foreach_copy,
                2 * cfg.gqa_layers * cfg.kv_row * BF16,
                2 * cfg.gqa_layers * cfg.kv_row * BF16,
            ),
            tap_rows: select(cfg.tap_width),
            fc_sums: p
                .group_sums
                .as_ref()
                .map(|b| row_group_sums(b, gpu, cfg.tap_width)),
            fc: gemm(
                TENSORFOLD_QMM,
                gpu,
                cfg.hidden,
                cfg.tap_width,
                DType::W4a16AffineG64,
            ),
            fc_norm: p.eager_norm.as_ref().map(|b| RmsNormKernelConfig {
                backends: b.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden.into(),
                dtype: DType::Bf16,
            }),
            // fp32 cos and sin out (the positions come from the host).
            taps_rotary: pass(&p.rotary, 0, 2 * half * FP32),
            kv_sums: p
                .group_sums
                .as_ref()
                .map(|b| row_group_sums(b, gpu, cfg.hidden)),
            kv: gemm(
                TENSORFOLD_QMM,
                gpu,
                cfg.draft_kv_width(),
                cfg.hidden,
                DType::W4a16AffineG64,
            ),
            kv_prep: p
                .prep
                .as_ref()
                .map(|b| head_major_prep(b, gpu, 0, cfg.draft_kv_heads, cfg.draft_head_dim)),
            append: SlidingWindowKvAppendKernelConfig {
                backends: vec![TENSORFOLD_TRITON],
                gpu_name: cfg.gpu_name.clone(),
                num_kv_heads: cfg.draft_kv_heads.into(),
                head_dim: cfg.draft_head_dim.into(),
                dtype: DType::Bf16,
                window: cfg.draft_window.into(),
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: TensorfoldRoundCommitLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = &resolved;
        let n = name.as_str();
        let pass = |suffix: &str, cfg: &Option<ElementwiseKernelConfig>| {
            optional(n, suffix, cfg.clone(), ElementwiseKernel::build, bridge)
        };
        let select = |suffix: &str, cfg: &Option<IndexedRowCopyKernelConfig>| {
            optional(n, suffix, cfg.clone(), IndexedRowCopyKernel::build, bridge)
        };
        Ok(Self {
            tap_add: pass("tap_add", &r.tap_add)?,
            tap_cat: pass("tap_cat", &r.tap_cat)?,
            take_cast: pass("take_cast", &r.take_cast)?,
            replay: atomic(
                n,
                "replay",
                r.replay.clone(),
                GdnStateReplayKernel::build,
                bridge,
            )?,
            conv_cat: pass("conv_cat", &r.conv_cat)?,
            conv_src_cat: pass("conv_src_cat", &r.conv_cat)?,
            conv_pick: select("conv_pick", &r.conv_pick)?,
            conv_pick_stacked: select("conv_pick_stacked", &r.conv_pick_stacked)?,
            conv_copy: pass("conv_copy", &r.conv_copy)?,
            take_cat: pass("take_cat", &r.take_cat)?,
            kv_gather: select("kv_gather", &r.kv_gather)?,
            kv_copy: pass("kv_copy", &r.kv_copy)?,
            tap_rows: select("tap_rows", &r.tap_rows)?,
            taps_cat: pass("taps_cat", &r.tap_cat)?,
            fc_sums: optional(
                n,
                "fc_sums",
                r.fc_sums.clone(),
                RowGroupSumsKernel::build,
                bridge,
            )?,
            fc: atomic(n, "fc", r.fc.clone(), SingleGemmKernel::build, bridge)?,
            fc_norm: optional(
                n,
                "fc_norm",
                r.fc_norm.clone(),
                RmsNormKernel::build,
                bridge,
            )?,
            taps_rotary: pass("taps_rotary", &r.taps_rotary)?,
            kv_sums: optional(
                n,
                "kv_sums",
                r.kv_sums.clone(),
                RowGroupSumsKernel::build,
                bridge,
            )?,
            kv: atomic(n, "kv", r.kv.clone(), SingleGemmKernel::build, bridge)?,
            kv_prep: optional(
                n,
                "kv_prep",
                r.kv_prep.clone(),
                HeadMajorQkNormRopeKernel::build,
                bridge,
            )?,
            append: atomic(
                n,
                "append",
                r.append.clone(),
                SlidingWindowKvAppendKernel::build,
                bridge,
            )?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        let mut children = Vec::new();
        children.extend(compile_opt_n(&self.tap_add, cfg.num_taps, builder));
        children.extend(compile_opt(&self.tap_cat, builder));
        children.extend(compile_opt(&self.take_cast, builder));
        children.push(self.replay.compile(builder));
        children.extend(compile_opt(&self.conv_cat, builder));
        children.extend(compile_opt(&self.conv_src_cat, builder));
        children.extend(compile_opt(&self.conv_pick, builder));
        children.extend(compile_opt(&self.conv_pick_stacked, builder));
        children.extend(compile_opt(&self.conv_copy, builder));
        children.extend(compile_opt(&self.take_cat, builder));
        children.extend(compile_opt_n(&self.kv_gather, 2 * cfg.gqa_layers, builder));
        children.extend(compile_opt(&self.kv_copy, builder));
        children.extend(compile_opt(&self.tap_rows, builder));
        children.extend(compile_opt(&self.taps_cat, builder));
        children.extend(compile_opt(&self.fc_sums, builder));
        children.push(self.fc.compile(builder));
        children.extend(compile_opt(&self.fc_norm, builder));
        children.extend(compile_opt(&self.taps_rotary, builder));
        let mut layer = Vec::new();
        layer.extend(compile_opt(&self.kv_sums, builder));
        layer.push(self.kv.compile(builder));
        layer.extend(compile_opt(&self.kv_prep, builder));
        layer.push(super::tensorfold_common::repeated(&self.append, 2, builder));
        children.push(CostNode::Scale {
            n: cfg.draft_layers,
            child: Box::new(CostNode::Sum(layer)),
        });
        CostNode::Labeled {
            label: format!(
                "{} (TensorfoldRoundCommitLocalWorklet) [gdn={}; gqa={}; taps={}; \
                 draft layers={}]",
                self.name, cfg.gdn_layers, cfg.gqa_layers, cfg.num_taps, cfg.draft_layers
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    pub fn eval(&self, round: &TensorfoldRoundCommitLocalWorkletInput, ev: &mut Evaluator) {
        let cfg = &self.resolved.raw_cfg;
        let shape = CommitShape::of(round, cfg.draft_window);
        let ew = |num_tokens| ElementwiseKernelInput { num_tokens };
        let sel = |num_rows| IndexedRowCopyKernelInput { num_rows };
        let s = shape.streams;
        let none = s == 0;
        let drafting = shape.drafting.len() as u32;
        let no_taps = drafting == 0;
        let a = shape.committed_rows();
        let a_d = shape.tap_rows();

        // Verify forward: taps captured over every row once any stream drafts.
        eval_opt(&self.tap_add, ew(shape.rows), no_taps, ev);
        eval_opt(&self.tap_cat, ew(shape.rows), no_taps, ev);

        if let Some(op) = &self.take_cast {
            let launches: Vec<_> = shape.committed.iter().map(|&n| ew(n)).collect();
            push_sum(op, &launches, ev);
        }
        push_or_zero(&self.replay, round.gdn_replay_input(), none, ev);
        let windows = s * cfg.conv_keep;
        let gdn = cfg.gdn_layers as usize;
        if let Some(op) = &self.conv_cat {
            let launches = if s > 1 {
                vec![ew(windows); gdn]
            } else {
                Vec::new()
            };
            push_sum(op, &launches, ev);
        }
        let few = shape.rows <= FEW_ROWS;
        if let Some(op) = &self.conv_src_cat {
            let launches = match (none, few) {
                (true, _) => Vec::new(),
                // stack(olds), stack(qkv), cat(dim=1): every layer at once.
                (false, true) => vec![
                    ew(cfg.gdn_layers * windows),
                    ew(cfg.gdn_layers * shape.rows),
                    ew(cfg.gdn_layers * (windows + shape.rows)),
                ],
                (false, false) => vec![ew(windows + shape.rows); gdn],
            };
            push_sum(op, &launches, ev);
        }
        if let Some(op) = &self.conv_pick {
            let launches = if !none && !few {
                vec![sel(windows); gdn]
            } else {
                Vec::new()
            };
            push_sum(op, &launches, ev);
        }
        eval_opt(&self.conv_pick_stacked, sel(windows), none || !few, ev);
        eval_opt(&self.conv_copy, ew(s), none, ev);
        eval_opt(&self.take_cat, ew(a), s <= 1, ev);
        eval_opt(&self.kv_gather, sel(a), none, ev);
        eval_opt(&self.kv_copy, ew(a), none, ev);
        if let Some(op) = &self.tap_rows {
            let launches: Vec<_> = shape.drafting.iter().map(|&(n, _)| sel(n)).collect();
            push_sum(op, &launches, ev);
        }

        // add_taps_streams.
        eval_opt(&self.taps_cat, ew(a_d), drafting <= 1, ev);
        let sums = RowGroupSumsKernelInput { num_tokens: a_d };
        eval_opt(&self.fc_sums, sums.clone(), no_taps, ev);
        push_or_zero(&self.fc, SingleGemmKernelInput { m: a_d }, no_taps, ev);
        eval_opt(&self.fc_norm, RmsNormKernelInput { m: a_d }, no_taps, ev);
        eval_opt(&self.taps_rotary, ew(a_d), no_taps, ev);
        eval_opt(&self.kv_sums, sums, no_taps, ev);
        push_or_zero(&self.kv, SingleGemmKernelInput { m: a_d }, no_taps, ev);
        eval_opt(
            &self.kv_prep,
            HeadMajorQkNormRopeKernelInput { num_tokens: a_d },
            no_taps,
            ev,
        );
        push_or_zero(
            &self.append,
            SlidingWindowKvAppendKernelInput {
                batch_size: drafting,
                context_len: shape.drafting.iter().map(|d| d.1).sum(),
                num_tokens: a_d,
            },
            no_taps,
            ev,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::tensorfold_common::VerifyStream;
    use super::*;
    use crate::timing::{CostTree, LeafMetrics};
    use std::collections::BTreeMap;
    use std::path::Path;

    const GB10: &str = "NVIDIA GB10";

    fn config(all_pending: bool) -> TensorfoldRoundCommitLocalWorkletConfig {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let draft =
            Qwen35Dflash2DraftCfg::from_json(&root.join("model/config/qwen3_8_27b_dflash2.json"))
                .unwrap();
        let target =
            Qwen35ModelCfg::from_json(&root.join("model/config/qwen3_8_27b_nvfp4.json")).unwrap();
        let mut cfg = TensorfoldRoundCommitLocalWorkletConfig::new(&draft, &target, GB10);
        if all_pending {
            let b = Some(vec!["tensorfold_pending"]);
            cfg.pending = TensorfoldRoundCommitPending {
                fp32_add: b.clone(),
                cat: b.clone(),
                cast: b.clone(),
                gather: b.clone(),
                foreach_copy: b.clone(),
                eager_norm: b.clone(),
                rotary: b.clone(),
                group_sums: b.clone(),
                prep: b,
            };
        }
        cfg
    }

    fn worklet(all_pending: bool) -> (TensorfoldRoundCommitLocalWorklet, CostTree) {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let w = TensorfoldRoundCommitLocalWorklet::build(
            "commit".into(),
            TensorfoldRoundCommitLocalWorklet::resolve_config(&config(all_pending)),
            &bridge,
        )
        .unwrap();
        let mut builder = CostTreeBuilder::new();
        let root = w.compile(&mut builder);
        (w, builder.finish(root))
    }

    fn stream(committed_len: u32, rows: u32, mode: WindowMode, accepted: u32) -> VerifyStream {
        VerifyStream {
            committed_len,
            rows,
            accepted,
            tree_slots: 0,
            mode,
        }
    }

    /// The n8 capture round dumped in progress.md: 8 streams, W = 100.
    fn captured_round() -> VerifyRound {
        use WindowMode::*;
        VerifyRound {
            streams: vec![
                stream(33465, 10, Tree, 2),
                stream(26931, 16, Copy, 1),
                stream(21658, 9, Tree, 2),
                stream(20873, 10, Tree, 2),
                stream(26271, 15, Tree, 6),
                stream(7965, 8, Tree, 2),
                stream(14256, 16, Tree, 6),
                stream(6441, 16, Tree, 7),
            ],
        }
    }

    #[test]
    fn shapes_follow_the_target_and_drafter_configs() {
        let r = TensorfoldRoundCommitLocalWorklet::resolve_config(&config(true));
        assert_eq!(r.replay.num_layers.get(), 48);
        assert_eq!((r.fc.n.get(), r.fc.k.get()), (5120, 25600));
        assert_eq!((r.kv.n.get(), r.kv.k.get()), (2048, 5120));
        assert_eq!(r.append.window.get(), 2047);
        assert_eq!(r.append.num_kv_heads.get(), 8);
        assert_eq!(r.kv_gather.unwrap().row_elements.get(), 1024);
        assert_eq!(r.conv_pick_stacked.unwrap().row_elements.get(), 48 * 10240);
        let cfg = &r.raw_cfg;
        assert_eq!((cfg.conv_keep, cfg.gqa_layers, cfg.num_taps), (3, 16, 5));
    }

    #[test]
    fn leaves_without_pending_slots() {
        let (_, tree) = worklet(false);
        let names: Vec<&str> = tree.slots.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(
            names,
            ["commit.replay", "commit.fc", "commit.kv", "commit.append"]
        );
    }

    #[test]
    fn empty_round_is_zero_and_fills_every_slot() {
        let (w, tree) = worklet(true);
        let mut buf = vec![LeafMetrics::MISS; tree.n_slots()];
        let mut ev = Evaluator::new(&mut buf);
        w.eval(&VerifyRound::default(), &mut ev);
        assert_eq!(ev.filled(), tree.n_slots());
        assert!(buf.iter().all(|m| m.m.time_ms == 0.0));
    }

    #[test]
    fn captured_round_shape() {
        let shape = CommitShape::of(&captured_round(), 2047);
        assert_eq!((shape.streams, shape.rows), (8, 100));
        // Σ(accepted + 1) = 36: the capture's `_prep_kernel` grid (36, 16)
        // and `_group_sums` grid (36, 25).
        assert_eq!((shape.committed_rows(), shape.tap_rows()), (36, 36));
        assert_eq!(shape.drafting.len(), 8);
        assert!(shape.drafting.iter().all(|d| d.1 == 2047));
    }

    /// Launch multiplicities per slot from the compiled tree, `Scale`s folded
    /// in. Variable-count slots (`push_sum`) count one leaf.
    fn multiplicity(tree: &CostTree) -> BTreeMap<String, u32> {
        fn walk(node: &CostNode, mult: u32, tree: &CostTree, out: &mut BTreeMap<String, u32>) {
            match node {
                CostNode::Leaf(slot) => {
                    let name = tree.slots[*slot]
                        .name
                        .trim_start_matches("commit.")
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
        let mut out = BTreeMap::new();
        walk(&tree.root, 1, tree, &mut out);
        out
    }

    #[test]
    fn launch_counts_match_the_n8_capture() {
        let (_, tree) = worklet(true);
        let n = multiplicity(&tree);
        // draft.taps a commit: group_kernel 6 (fc + 5 kv), _group_sums 6,
        // _prep 5, _append 10 (1960 / 196).
        assert_eq!(n["fc"] + n["kv"], 6);
        assert_eq!(n["fc_sums"] + n["kv_sums"], 6);
        assert_eq!((n["kv_prep"], n["append"]), (5, 10));
        // engine.commit: replay_kernel 1; per round 32 KV index_selects.
        assert_eq!((n["replay"], n["kv_gather"]), (1, 32));
        // verify: 5 taps of four launches each.
        assert_eq!(n["tap_add"], 5);
    }

    /// Prices captured rounds' commits (with the drafter taps) on the GB10
    /// rows in `profiling/profile.db` against the capture's `replay_kernel`,
    /// taps `group_kernel` and `_append` time and the whole commit span. Run
    /// from the repo root with libpython on `LD_LIBRARY_PATH`:
    /// `PYTHONPATH=$PWD uv run cargo test -p simulator --lib
    /// tensorfold_round_commit_local -- --include-ignored --nocapture`; set
    /// `TENSORFOLD_CAPTURED_ROUNDS` to also sum every captured round.
    #[test]
    #[ignore = "needs profile.db GB10 rows and the venv interpreter"]
    fn gb10_rows_price_captured_commits() {
        use super::super::tensorfold_dflash2_draft_local::tests::{
            captured_rounds_from_env, round_of, CAPTURED_ROUNDS,
        };
        let bridge = crate::timing::PerfApiBridge::new().expect("perf_api bridge");
        let w = TensorfoldRoundCommitLocalWorklet::build(
            "commit".into(),
            TensorfoldRoundCommitLocalWorklet::resolve_config(&config(false)),
            &bridge,
        )
        .expect("commit rows on GB10");
        let mut builder = CostTreeBuilder::new();
        let root = w.compile(&mut builder);
        let tree = builder.finish(root);
        let flat = tree.flatten();
        let mut buf = vec![LeafMetrics::MISS; tree.n_slots()];
        let mut price = |round: &VerifyRound| {
            let mut ev = Evaluator::new(&mut buf);
            w.eval(round, &mut ev);
            assert_eq!(ev.filled(), tree.n_slots());
            let ms = f64::from(CostTree::aggregate(&flat, &buf, &mut Vec::new()).m.time_ms);
            (ms, buf.clone())
        };
        for c in CAPTURED_ROUNDS {
            let round = round_of(c.streams);
            let (ms, _) = price(&round);
            eprintln!(
                "commit n{} S={} A={}: predicted {ms:.3} ms; captured modeled kernels {:.3} ms \
                 ({:+.1}%), span {:.3} ms ({:+.1}%)",
                c.n,
                round.num_streams(),
                round.gdn_replay_input().num_tokens,
                c.commit_modeled_ms,
                100.0 * (ms / c.commit_modeled_ms - 1.0),
                c.commit_span_ms,
                100.0 * (ms / c.commit_span_ms - 1.0),
            );
            assert!(ms > 0.0 && ms.is_finite());
        }
        let mut sums = BTreeMap::<u32, [f64; 4]>::new();
        // Each slot's share of the predicted total (its `Scale`s applied).
        let mut slot_ms = BTreeMap::<(u32, String), f64>::new();
        for (n, round, measured) in captured_rounds_from_env() {
            let e = sums.entry(n).or_default();
            let (ms, leaves) = price(&round);
            e[0] += ms;
            for (slot, desc) in tree.slots.iter().enumerate() {
                let mut alone = vec![LeafMetrics::ZERO; leaves.len()];
                alone[slot] = leaves[slot];
                *slot_ms.entry((n, desc.name.clone())).or_default() += f64::from(
                    CostTree::aggregate(&flat, &alone, &mut Vec::new())
                        .m
                        .time_ms,
                );
            }
            e[1] += measured[2];
            e[2] += measured[3];
            e[3] += 1.0;
        }
        for (n, [ms, modeled, span, rounds]) in sums {
            eprintln!(
                "commit n{n} all {rounds} rounds: predicted {ms:.1} ms; modeled kernels \
                 {modeled:.1} ms ({:+.1}%), span {span:.1} ms ({:+.1}%)",
                100.0 * (ms / modeled - 1.0),
                100.0 * (ms / span - 1.0),
            );
        }
        for ((n, slot), ms) in slot_ms {
            eprintln!("commit n{n} slot {slot}: {ms:.1} ms");
        }
    }
}
