//! TensorFold Qwen3.8-27B's output head on GB10
//! (`qwen35_tensorfold_nvfp4_gdn_dense_dflash2`): the final residual add +
//! RMSNorm after the last layer, the NVFP4 `lm_head`, and the sampling top-k
//! (sources under `tensorfold-0.6.6/src/tensorfold/`).
//!
//! **Boundary.** The final norm adds the last layer's MLP output (`pending`,
//! from `tensorfold_nvfp4_mlp_local`) into the residual and norms it with
//! `w.norm`. Everything after the logits up to the host draw is here; the
//! `.cpu()` reads and the host draw are not kernels (host time is fitted
//! outside the CostTree, decision D2). The DFlash2 taps (`cat` of the five
//! tapped layers' rows) are the drafter lane's (W4), not the head's.
//!
//! - **Verify** ([`TensorfoldHeadVerifyLocalWorklet`],
//!   `qwen3_5/cuda/forward.py` `multi_tree_forward` tail, then `cuda/sampling.py`
//!   `sample_streams`), W = Σ window rows:
//!   1. `residual_rms_norm` over W (`glue.add_rmsnorm(x, pending, w.norm)`);
//!   2. `nvfp4_quant` (k = hidden) and `tensorfold_lane` `lm_head`
//!      (n = vocab) over **all** W rows (`full_logits=True` on GB10);
//!   3. sampling: `torch.cat` of every stream's bf16 logits rows into one
//!      `[W, vocab]` block — a launch only when the top-k group has two or more
//!      streams; one stream's rows are a contiguous slice — then the
//!      bf16 → fp32 `.float()` cast of the `[W, vocab]` block, then
//!      `torch.topk(k = top_k + MARGIN, sorted=False)` over W rows
//!      (`logits_topk` / `tensorfold_torch`). The model assumes every stream
//!      samples with the same `top_k` (one group), as in the captures.
//! - **Prefill** ([`TensorfoldHeadPrefillLocalWorklet`], `qwen3_5/cuda/prefill.py`
//!   and `multi.py` → `decode_tp.first_token`):
//!   1. `residual_rms_norm` with m = 1 for **every piece** of the step: the
//!      last row of a one-piece step's last chunk (`prefill_chunk(last=True)`)
//!      or of each piece of a batched step (`prefill_rows`), whether or not it
//!      ends its prompt;
//!   2. for each prompt the step **finishes**: `nvfp4_quant` + `tensorfold_lane`
//!      `lm_head` at m = 1, the `.float()` cast of the one logits row, and
//!      `logits_topk` over one row (`sample_rows`: no `cat`).
//!
//! **Slots for lane tf-copies.** The `cat` and the cast are torch byte copies
//! still being profiled. Each is an `Option` slot (absent by default) built from
//! its `elementwise` backends once tf-copies merges:
//! - `sample_cat` (`CatArrayBatchedCopy_vectorized`, one launch): reads and
//!   writes `W × vocab × 2` bytes (bf16); its `elementwise` config is
//!   `vocab·2 → vocab·2` bytes per token over `num_tokens = W`.
//! - `sample_cast` (`unrolled_elementwise_kernel`, one launch per verify round
//!   and per finishing prompt): reads `rows × vocab × 2`, writes
//!   `rows × vocab × 4` bytes; config `vocab·2 → vocab·4` bytes per token.
//!
//! Capture check (`kernel_inventory_n8.json`): per verify forward the head is
//! 1 of the 129 `_add_rmsnorm`, 1 of the 129 `quant4_kernel` and 1 of the 337
//! `lane_kernel`; `tensorfold.sampling.d2h` is one `CatArrayBatchedCopy`, one
//! `unrolled_elementwise_kernel` and one `torch.topk` sequence
//! (`computeBlockDigitCounts` ×4, `gatherTopK`, …) per round. `prefill.step`
//! holds 9 m = 1 `lane_kernel` with 9 extra `quant4_kernel` and 9 top-k
//! sequences (one per finishing prompt).

use crate::arch::{Qwen35LinearPrecision, Qwen35ModelCfg};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, LogitsTopkKernel,
    LogitsTopkKernelConfig, LogitsTopkKernelInput, Nvfp4QuantKernel, Nvfp4QuantKernelConfig,
    Nvfp4QuantKernelInput, ResidualRmsNormKernel, ResidualRmsNormKernelConfig,
    ResidualRmsNormKernelInput, SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge};

use super::tensorfold_common::{
    atomic, elementwise, gemm, push_sum, PrefillForward, VerifyRound, TENSORFOLD_LANE,
    TENSORFOLD_TORCH,
};
use super::tensorfold_nvfp4_mlp_local::{add_rms_norm, nvfp4_quant};

/// Candidates `sample_rows` / `sample_streams` read past the request's `top_k`
/// (`engine/exact_sampling.py` `MARGIN`).
pub const TENSORFOLD_SAMPLING_MARGIN: u32 = 8;

const BF16_BYTES: u32 = 2;
const FP32_BYTES: u32 = 4;

/// Model facts plus the run's sampling width; shared by both head forms.
#[derive(Clone, Debug)]
pub struct TensorfoldHeadLocalWorkletConfig {
    pub hidden: u32,
    pub vocab_size: u32,
    /// `torch.topk`'s k: the request `top_k` plus [`TENSORFOLD_SAMPLING_MARGIN`]
    /// (28 for the captured `top_k = 20`).
    pub sample_candidates: u32,
    pub gpu_name: String,
    /// `elementwise` backends of the sampling row `cat` (tf-copies,
    /// `tensorfold_torch_cat`); `None` drops the slot.
    pub sample_cat_backends: Option<Vec<&'static str>>,
    /// `elementwise` backends of the bf16 → fp32 logits cast (tf-copies,
    /// `tensorfold_torch_cast`); `None` drops the slot.
    pub sample_cast_backends: Option<Vec<&'static str>>,
}

impl TensorfoldHeadLocalWorkletConfig {
    /// From the checkpoint config; its `lm_head` must be NVFP4. The copy slots
    /// start absent.
    pub fn from_model(model: &Qwen35ModelCfg, gpu_name: &str, sample_top_k: u32) -> Self {
        assert_eq!(
            model.precisions.lm_head,
            Qwen35LinearPrecision::Nvfp4,
            "the TensorFold head worklet models an NVFP4 lm_head"
        );
        Self {
            hidden: model.hidden,
            vocab_size: model.vocab_size,
            sample_candidates: (sample_top_k + TENSORFOLD_SAMPLING_MARGIN).min(model.vocab_size),
            gpu_name: gpu_name.to_string(),
            sample_cat_backends: None,
            sample_cast_backends: None,
        }
    }

    fn final_norm(&self) -> ResidualRmsNormKernelConfig {
        add_rms_norm(&self.gpu_name, self.hidden)
    }

    fn lm_head_quant(&self) -> Nvfp4QuantKernelConfig {
        nvfp4_quant(&self.gpu_name, self.hidden)
    }

    fn lm_head(&self) -> SingleGemmKernelConfig {
        gemm(
            TENSORFOLD_LANE,
            &self.gpu_name,
            self.vocab_size,
            self.hidden,
            DType::Nvfp4E2m1,
        )
    }

    fn sample_cat(&self) -> Option<ElementwiseKernelConfig> {
        let row = self.vocab_size * BF16_BYTES;
        self.sample_cat_backends
            .as_ref()
            .map(|b| elementwise(b, &self.gpu_name, row, row))
    }

    fn sample_cast(&self) -> Option<ElementwiseKernelConfig> {
        self.sample_cast_backends.as_ref().map(|b| {
            elementwise(
                b,
                &self.gpu_name,
                self.vocab_size * BF16_BYTES,
                self.vocab_size * FP32_BYTES,
            )
        })
    }

    fn sample_topk(&self) -> LogitsTopkKernelConfig {
        LogitsTopkKernelConfig {
            backends: vec![TENSORFOLD_TORCH],
            gpu_name: self.gpu_name.clone(),
            num_columns: self.vocab_size.into(),
            top_k: self.sample_candidates,
            dtype: DType::Fp32,
        }
    }
}

fn build_option(
    prefix: &str,
    suffix: &str,
    config: Option<ElementwiseKernelConfig>,
    bridge: &PerfApiBridge,
) -> Result<Option<Op<ElementwiseKernel>>, BuildError> {
    config
        .map(|c| atomic(prefix, suffix, c, ElementwiseKernel::build, bridge))
        .transpose()
}

// ─── verify ──────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct TensorfoldHeadVerifyLocalWorkletResolved {
    pub raw_cfg: TensorfoldHeadLocalWorkletConfig,
    pub final_norm: ResidualRmsNormKernelConfig,
    pub lm_head_quant: Nvfp4QuantKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub sample_cat: Option<ElementwiseKernelConfig>,
    pub sample_cast: Option<ElementwiseKernelConfig>,
    pub sample_topk: LogitsTopkKernelConfig,
}

/// One verify round: W rows, and how many streams share the top-k group.
pub type TensorfoldHeadVerifyLocalWorkletInput = VerifyRound;

/// The verify forward's head and the round's sampling launches.
pub struct TensorfoldHeadVerifyLocalWorklet {
    pub name: String,
    final_norm: Op<ResidualRmsNormKernel>,
    lm_head_quant: Op<Nvfp4QuantKernel>,
    lm_head: Op<SingleGemmKernel>,
    sample_cat: Option<Op<ElementwiseKernel>>,
    sample_cast: Option<Op<ElementwiseKernel>>,
    sample_topk: Op<LogitsTopkKernel>,
    resolved: TensorfoldHeadVerifyLocalWorkletResolved,
}

impl TensorfoldHeadVerifyLocalWorklet {
    pub fn resolve_config(
        cfg: &TensorfoldHeadLocalWorkletConfig,
    ) -> TensorfoldHeadVerifyLocalWorkletResolved {
        TensorfoldHeadVerifyLocalWorkletResolved {
            final_norm: cfg.final_norm(),
            lm_head_quant: cfg.lm_head_quant(),
            lm_head: cfg.lm_head(),
            sample_cat: cfg.sample_cat(),
            sample_cast: cfg.sample_cast(),
            sample_topk: cfg.sample_topk(),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: TensorfoldHeadVerifyLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let n = name.as_str();
        Ok(Self {
            final_norm: atomic(
                n,
                "final_norm",
                r.final_norm,
                ResidualRmsNormKernel::build,
                bridge,
            )?,
            lm_head_quant: atomic(
                n,
                "lm_head_quant",
                r.lm_head_quant,
                Nvfp4QuantKernel::build,
                bridge,
            )?,
            lm_head: atomic(n, "lm_head", r.lm_head, SingleGemmKernel::build, bridge)?,
            sample_cat: build_option(n, "sample_cat", r.sample_cat, bridge)?,
            sample_cast: build_option(n, "sample_cast", r.sample_cast, bridge)?,
            sample_topk: atomic(
                n,
                "sample_topk",
                r.sample_topk,
                LogitsTopkKernel::build,
                bridge,
            )?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        let mut children = vec![
            self.final_norm.compile(builder),
            self.lm_head_quant.compile(builder),
            self.lm_head.compile(builder),
        ];
        if let Some(op) = &self.sample_cat {
            children.push(op.compile(builder));
        }
        if let Some(op) = &self.sample_cast {
            children.push(op.compile(builder));
        }
        children.push(self.sample_topk.compile(builder));
        CostNode::Labeled {
            label: format!(
                "{} (TensorfoldHeadVerifyLocalWorklet) [local (1 GPU); H={}, vocab={}, \
                 top-k {}; NVFP4 lane lm_head over every verify row]",
                self.name, cfg.hidden, cfg.vocab_size, cfg.sample_candidates
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    pub fn eval(&self, round: &TensorfoldHeadVerifyLocalWorkletInput, ev: &mut Evaluator) {
        let w = VerifyHeadWork::of(round);
        push_sum(&self.final_norm, &w.norm, ev);
        push_sum(&self.lm_head_quant, &w.quant, ev);
        push_sum(&self.lm_head, &w.gemm, ev);
        if let Some(op) = &self.sample_cat {
            push_sum(op, &w.cat, ev);
        }
        if let Some(op) = &self.sample_cast {
            push_sum(op, &w.cast, ev);
        }
        push_sum(&self.sample_topk, &w.topk, ev);
    }
}

/// Each head slot's launches for one verify round.
struct VerifyHeadWork {
    norm: Vec<ResidualRmsNormKernelInput>,
    quant: Vec<Nvfp4QuantKernelInput>,
    gemm: Vec<SingleGemmKernelInput>,
    cat: Vec<ElementwiseKernelInput>,
    cast: Vec<ElementwiseKernelInput>,
    topk: Vec<LogitsTopkKernelInput>,
}

impl VerifyHeadWork {
    fn of(round: &VerifyRound) -> Self {
        let rows = round.total_rows();
        let once = |m: u32| (m > 0).then_some(m).into_iter().collect::<Vec<u32>>();
        let all = once(rows);
        Self {
            norm: all
                .iter()
                .map(|&m| ResidualRmsNormKernelInput { m })
                .collect(),
            quant: all
                .iter()
                .map(|&num_tokens| Nvfp4QuantKernelInput { num_tokens })
                .collect(),
            gemm: all.iter().map(|&m| SingleGemmKernelInput { m }).collect(),
            // One stream's rows are already a contiguous slice: no cat.
            cat: once(if round.num_streams() > 1 { rows } else { 0 })
                .into_iter()
                .map(|num_tokens| ElementwiseKernelInput { num_tokens })
                .collect(),
            cast: all
                .iter()
                .map(|&num_tokens| ElementwiseKernelInput { num_tokens })
                .collect(),
            topk: all
                .iter()
                .map(|&num_rows| LogitsTopkKernelInput { num_rows })
                .collect(),
        }
    }
}

// ─── prefill ─────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct TensorfoldHeadPrefillLocalWorkletResolved {
    pub raw_cfg: TensorfoldHeadLocalWorkletConfig,
    pub final_norm: ResidualRmsNormKernelConfig,
    pub lm_head_quant: Nvfp4QuantKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub sample_cast: Option<ElementwiseKernelConfig>,
    pub sample_topk: LogitsTopkKernelConfig,
}

/// One prefill step: a final-norm row per piece, a first token per finishing
/// prompt.
pub type TensorfoldHeadPrefillLocalWorkletInput = PrefillForward;

/// The prefill step's last-row norms and first-token heads.
pub struct TensorfoldHeadPrefillLocalWorklet {
    pub name: String,
    final_norm: Op<ResidualRmsNormKernel>,
    lm_head_quant: Op<Nvfp4QuantKernel>,
    lm_head: Op<SingleGemmKernel>,
    sample_cast: Option<Op<ElementwiseKernel>>,
    sample_topk: Op<LogitsTopkKernel>,
    resolved: TensorfoldHeadPrefillLocalWorkletResolved,
}

impl TensorfoldHeadPrefillLocalWorklet {
    pub fn resolve_config(
        cfg: &TensorfoldHeadLocalWorkletConfig,
    ) -> TensorfoldHeadPrefillLocalWorkletResolved {
        TensorfoldHeadPrefillLocalWorkletResolved {
            final_norm: cfg.final_norm(),
            lm_head_quant: cfg.lm_head_quant(),
            lm_head: cfg.lm_head(),
            sample_cast: cfg.sample_cast(),
            sample_topk: cfg.sample_topk(),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: TensorfoldHeadPrefillLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let n = name.as_str();
        Ok(Self {
            final_norm: atomic(
                n,
                "final_norm",
                r.final_norm,
                ResidualRmsNormKernel::build,
                bridge,
            )?,
            lm_head_quant: atomic(
                n,
                "lm_head_quant",
                r.lm_head_quant,
                Nvfp4QuantKernel::build,
                bridge,
            )?,
            lm_head: atomic(n, "lm_head", r.lm_head, SingleGemmKernel::build, bridge)?,
            sample_cast: build_option(n, "sample_cast", r.sample_cast, bridge)?,
            sample_topk: atomic(
                n,
                "sample_topk",
                r.sample_topk,
                LogitsTopkKernel::build,
                bridge,
            )?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        let mut children = vec![
            self.final_norm.compile(builder),
            self.lm_head_quant.compile(builder),
            self.lm_head.compile(builder),
        ];
        if let Some(op) = &self.sample_cast {
            children.push(op.compile(builder));
        }
        children.push(self.sample_topk.compile(builder));
        CostNode::Labeled {
            label: format!(
                "{} (TensorfoldHeadPrefillLocalWorklet) [local (1 GPU); H={}, vocab={}, \
                 top-k {}; m=1 norm per piece, m=1 lm_head per finishing prompt]",
                self.name, cfg.hidden, cfg.vocab_size, cfg.sample_candidates
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    pub fn eval(&self, step: &TensorfoldHeadPrefillLocalWorkletInput, ev: &mut Evaluator) {
        let w = PrefillHeadWork::of(step);
        push_sum(&self.final_norm, &w.norm, ev);
        push_sum(&self.lm_head_quant, &w.quant, ev);
        push_sum(&self.lm_head, &w.gemm, ev);
        if let Some(op) = &self.sample_cast {
            push_sum(op, &w.cast, ev);
        }
        push_sum(&self.sample_topk, &w.topk, ev);
    }
}

/// Each prefill head slot's launches: m = 1 each.
struct PrefillHeadWork {
    norm: Vec<ResidualRmsNormKernelInput>,
    quant: Vec<Nvfp4QuantKernelInput>,
    gemm: Vec<SingleGemmKernelInput>,
    cast: Vec<ElementwiseKernelInput>,
    topk: Vec<LogitsTopkKernelInput>,
}

impl PrefillHeadWork {
    fn of(step: &PrefillForward) -> Self {
        let pieces = step.pieces.len();
        let firsts = step.finishing_prompts() as usize;
        Self {
            norm: vec![ResidualRmsNormKernelInput { m: 1 }; pieces],
            quant: vec![Nvfp4QuantKernelInput { num_tokens: 1 }; firsts],
            gemm: vec![SingleGemmKernelInput { m: 1 }; firsts],
            cast: vec![ElementwiseKernelInput { num_tokens: 1 }; firsts],
            topk: vec![LogitsTopkKernelInput { num_rows: 1 }; firsts],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::{CostTree, LeafMetrics};
    use crate::worklet::tensorfold_common::{PrefillPiece, VerifyStream, WindowMode};
    use std::path::Path;

    fn qwen38() -> Qwen35ModelCfg {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("model/config/qwen3_8_27b_nvfp4.json");
        Qwen35ModelCfg::from_json(&path).unwrap()
    }

    fn cfg() -> TensorfoldHeadLocalWorkletConfig {
        TensorfoldHeadLocalWorkletConfig::from_model(&qwen38(), "NVIDIA GB10", 20)
    }

    fn round(rows: &[u32]) -> VerifyRound {
        VerifyRound {
            streams: rows
                .iter()
                .map(|&rows| VerifyStream {
                    committed_len: 1000,
                    rows,
                    accepted: 0,
                    tree_slots: 0,
                    mode: WindowMode::Copy,
                })
                .collect(),
        }
    }

    fn step(pieces: &[(u32, u32, bool)]) -> PrefillForward {
        PrefillForward {
            pieces: pieces
                .iter()
                .map(|&(prefix_len, rows, finishes_prompt)| PrefillPiece {
                    prefix_len,
                    rows,
                    finishes_prompt,
                })
                .collect(),
        }
    }

    fn names(tree: &CostTree) -> Vec<(&str, &str)> {
        tree.slots
            .iter()
            .map(|s| (s.name.as_str(), s.kind.as_str()))
            .collect()
    }

    fn compiled_verify(cfg: &TensorfoldHeadLocalWorkletConfig) -> CostTree {
        let bridge = PerfApiBridge::structure_only();
        let w = TensorfoldHeadVerifyLocalWorklet::build(
            "head".into(),
            TensorfoldHeadVerifyLocalWorklet::resolve_config(cfg),
            &bridge,
        )
        .unwrap();
        let mut b = CostTreeBuilder::new();
        let root = w.compile(&mut b);
        b.finish(root)
    }

    #[test]
    fn shapes_follow_the_checkpoint_and_sampling() {
        let r = TensorfoldHeadVerifyLocalWorklet::resolve_config(&cfg());
        assert_eq!((r.lm_head.n.get(), r.lm_head.k.get()), (248320, 5120));
        assert_eq!(r.lm_head.backends, [TENSORFOLD_LANE]);
        assert_eq!(r.lm_head_quant.hidden_size.get(), 5120);
        assert_eq!(
            (r.sample_topk.num_columns.get(), r.sample_topk.top_k),
            (248320, 28)
        );
        assert_eq!(r.sample_topk.dtype, DType::Fp32);
        assert!(r.sample_cat.is_none() && r.sample_cast.is_none());
    }

    #[test]
    fn verify_leaves_without_and_with_the_copy_slots() {
        assert_eq!(
            names(&compiled_verify(&cfg())),
            [
                ("head.final_norm", "residual_rms_norm"),
                ("head.lm_head_quant", "nvfp4_quant"),
                ("head.lm_head", "single_gemm"),
                ("head.sample_topk", "logits_topk"),
            ]
        );
        let mut with = cfg();
        with.sample_cat_backends = Some(vec!["tensorfold_torch_cat"]);
        with.sample_cast_backends = Some(vec!["tensorfold_torch_cast"]);
        let tree = compiled_verify(&with);
        assert_eq!(
            names(&tree)[3..],
            [
                ("head.sample_cat", "elementwise"),
                ("head.sample_cast", "elementwise"),
                ("head.sample_topk", "logits_topk"),
            ]
        );
        let bytes = |slot: usize| {
            let c = &tree.slots[slot].kernel_config;
            (
                c["input_bytes_per_token"]["value"].clone(),
                c["output_bytes_per_token"]["value"].clone(),
            )
        };
        assert_eq!(bytes(3), (496640.into(), 496640.into()));
        assert_eq!(bytes(4), (496640.into(), 993280.into()));
    }

    #[test]
    fn verify_samples_every_row_and_cats_only_several_streams() {
        let w = VerifyHeadWork::of(&round(&[16, 5, 1]));
        assert_eq!(w.gemm.iter().map(|i| i.m).collect::<Vec<_>>(), [22]);
        assert_eq!(w.topk.iter().map(|i| i.num_rows).collect::<Vec<_>>(), [22]);
        assert_eq!(w.cat.iter().map(|i| i.num_tokens).collect::<Vec<_>>(), [22]);
        assert_eq!(w.cast.len() + w.norm.len() + w.quant.len(), 3);
        let one = VerifyHeadWork::of(&round(&[16]));
        assert!(one.cat.is_empty());
        assert_eq!(one.cast.len(), 1);
        let empty = VerifyHeadWork::of(&VerifyRound::default());
        assert!(empty.norm.is_empty() && empty.topk.is_empty() && empty.cat.is_empty());
    }

    #[test]
    fn prefill_norms_every_piece_and_heads_finishing_prompts() {
        let w = PrefillHeadWork::of(&step(&[(0, 600, false), (2048, 424, true), (0, 9, true)]));
        assert_eq!(w.norm.len(), 3);
        assert_eq!(
            (w.quant.len(), w.gemm.len(), w.cast.len(), w.topk.len()),
            (2, 2, 2, 2)
        );
        assert!(w.gemm.iter().all(|i| i.m == 1));
        // One piece over three chunks: only the last chunk's last row is normed.
        let w = PrefillHeadWork::of(&step(&[(0, 10_000, false)]));
        assert_eq!((w.norm.len(), w.gemm.len()), (1, 0));
    }

    #[test]
    fn empty_inputs_push_zero_into_every_slot() {
        let bridge = PerfApiBridge::structure_only();
        let mut with = cfg();
        with.sample_cat_backends = Some(vec!["tensorfold_torch_cat"]);
        with.sample_cast_backends = Some(vec!["tensorfold_torch_cast"]);
        let v = TensorfoldHeadVerifyLocalWorklet::build(
            "head".into(),
            TensorfoldHeadVerifyLocalWorklet::resolve_config(&with),
            &bridge,
        )
        .unwrap();
        let p = TensorfoldHeadPrefillLocalWorklet::build(
            "head".into(),
            TensorfoldHeadPrefillLocalWorklet::resolve_config(&with),
            &bridge,
        )
        .unwrap();
        let mut b = CostTreeBuilder::new();
        let root = CostNode::Sum(vec![v.compile(&mut b), p.compile(&mut b)]);
        let tree = b.finish(root);
        assert_eq!(tree.slots.len(), 6 + 5);
        let mut buf = vec![LeafMetrics::MISS; tree.slots.len()];
        let mut ev = Evaluator::new(&mut buf);
        v.eval(&VerifyRound::default(), &mut ev);
        p.eval(&PrefillForward::default(), &mut ev);
        assert_eq!(ev.filled(), tree.slots.len());
        assert!(buf.iter().all(|leaf| leaf.m.time_ms == 0.0));
    }

    /// Against the GB10 rows in `profile.db`. Run from the repo root:
    /// `PYTHONPATH=$PWD LD_LIBRARY_PATH=<libpython dir> uv run cargo test -p simulator --lib
    /// tensorfold_head -- --include-ignored --nocapture`.
    #[test]
    #[ignore = "needs profile.db GB10 rows and the venv interpreter"]
    fn gb10_rows_price_the_head() {
        let bridge = PerfApiBridge::new().expect("perf_api bridge");
        let v = TensorfoldHeadVerifyLocalWorklet::build(
            "head".into(),
            TensorfoldHeadVerifyLocalWorklet::resolve_config(&cfg()),
            &bridge,
        )
        .expect("verify head rows on GB10");
        let p = TensorfoldHeadPrefillLocalWorklet::build(
            "head".into(),
            TensorfoldHeadPrefillLocalWorklet::resolve_config(&cfg()),
            &bridge,
        )
        .expect("prefill head rows on GB10");
        let price = |slots: usize, fill: &dyn Fn(&mut Evaluator)| -> Vec<f32> {
            let mut buf = vec![LeafMetrics::MISS; slots];
            let mut ev = Evaluator::new(&mut buf);
            fill(&mut ev);
            assert_eq!(ev.filled(), slots);
            buf.iter().map(|leaf| leaf.m.time_ms).collect()
        };
        // W = 1, then 8 streams at the n8 capture's W buckets (64..127).
        for w in [1, 68, 76, 84, 92, 100, 108, 116, 124] {
            let streams = w.min(8);
            let rows: Vec<u32> = (0..streams)
                .map(|s| w / streams + u32::from(s < w % streams))
                .collect();
            let r = round(&rows);
            let ms = price(4, &|ev| v.eval(&r, ev));
            eprintln!(
                "verify W={:3}: norm {:.4} q4 {:.4} lm_head {:.4} topk {:.4} | {:.4} ms",
                r.total_rows(),
                ms[0],
                ms[1],
                ms[2],
                ms[3],
                ms.iter().sum::<f32>()
            );
            assert!(ms.iter().all(|t| *t > 0.0 && t.is_finite()));
        }
        let s = step(&[(0, 600, true), (2048, 424, true), (100, 9, false)]);
        let ms = price(4, &|ev| p.eval(&s, ev));
        eprintln!(
            "prefill 3 pieces / 2 finishing: norm {:.4} q4 {:.4} lm_head {:.4} topk {:.4} | {:.4} ms",
            ms[0],
            ms[1],
            ms[2],
            ms[3],
            ms.iter().sum::<f32>()
        );
        assert!(ms.iter().all(|t| *t > 0.0 && t.is_finite()));
    }
}
