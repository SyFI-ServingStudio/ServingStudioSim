//! DFlash2 candidate selector — the draft's output head.
//!
//! After the six draft layers, DFlash2 does not sample one token per position
//! independently. It scores *edges* between adjacent positions' candidates: a
//! rank-`selector_rank` projection of the hidden state gates a bilinear form
//! between a predecessor and a successor codebook, both indexed by token id.
//! That turns the per-position top-k lists into a scored lattice, which is why
//! the section exists at all and why it carries two vocabulary-sized gathers.
//!
//! `compute_candidates` runs first (the LM head plus a top-k over the vocabulary,
//! producing the `selector_top_k` candidate ids per position); the selector then
//! scores the edges between consecutive positions.
//!
//! The vocabulary top-k is two selections, not one. `LogitsProcessor::
//! get_top_k_tokens` never all-gathers the logits: each rank selects `top_k`
//! from its own vocabulary shard, the ranks all-gather `top_k` values and ids,
//! and a second selection takes the global `top_k` out of the `top_k * tp_size`
//! gathered candidates. Both calls are `logits_processor._topk`, so both are
//! `logits_topk` leaves — `candidate_topk` at the shard width and
//! `candidate_topk_merge` at the gathered width. The all-gather between them is
//! `ncclDevKernel_AllGather_RING_LL` in the capture and is not priced here.
//!
//! Measured against `logs/20260920_0_glm53_dflash2_phase0` (B200, tp=4): nothing
//! in this section reached the top-12 of the `draft` phase by busy time. The
//! codebook gather shows up as `vectorized_gather_kernel` at 216 launches. The
//! top-k was the section's worst error all the same: as an `elementwise`
//! placeholder it swept the vocabulary shard's bytes and predicted 0.388 ms
//! against a measured 0.030 ms, because a radix select does not read a row the
//! way an elementwise pass does.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, LogitsTopkKernel,
    LogitsTopkKernelConfig, LogitsTopkKernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, LeafMetrics, PerfApiBridge, Probe,
    SlotInput,
};

const HIDDEN_DIM: u32 = 6144;
const VOCAB_SIZE: u32 = 154880;
/// `dflash_config.selector_rank`.
const SELECTOR_RANK: u32 = 256;
/// `dflash_config.selector_top_k`.
const SELECTOR_TOP_K: u32 = 16;

#[cfg(test)]
const SOURCE_ORDER: [&str; 6] = [
    "lm_head",
    "candidate_topk",
    "candidate_topk_merge",
    "hidden_projection",
    "codebook_gather",
    "edge_scores",
];

#[derive(Clone, Debug)]
pub struct Dflash2SelectorLocalWorkletConfig {
    pub gemm_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
    pub topk_backends: Vec<&'static str>,
    pub tp_size: u16,
    pub gpu_name: String,
    pub hidden_dim: Dim,
    pub vocab_size: Dim,
    pub selector_rank: u32,
    pub selector_top_k: u32,
    pub dtype: DType,
    pub gemm_dtype: DType,
}

#[derive(Clone, Debug)]
pub struct Dflash2SelectorLocalWorkletResolved {
    pub raw_cfg: Dflash2SelectorLocalWorkletConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub candidate_topk: LogitsTopkKernelConfig,
    pub candidate_topk_merge: LogitsTopkKernelConfig,
    pub hidden_projection: SingleGemmKernelConfig,
    pub codebook_gather: ElementwiseKernelConfig,
    pub edge_scores: SingleGemmKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct Dflash2SelectorLocalWorkletInput {
    /// Rows the selector scores: one per drafted position per request, i.e.
    /// `requests * draft_tokens`. The bonus token is not drafted, so it does
    /// not reach this head.
    pub scored_rows: u32,
}

pub struct Dflash2SelectorLocalWorklet {
    pub name: String,
    pub lm_head: Op<SingleGemmKernel>,
    pub candidate_topk: Op<LogitsTopkKernel>,
    pub candidate_topk_merge: Op<LogitsTopkKernel>,
    pub hidden_projection: Op<SingleGemmKernel>,
    pub codebook_gather: Op<ElementwiseKernel>,
    pub edge_scores: Op<SingleGemmKernel>,
    resolved: Dflash2SelectorLocalWorkletResolved,
}

impl Dflash2SelectorLocalWorklet {
    pub fn resolve_config(
        cfg: &Dflash2SelectorLocalWorkletConfig,
    ) -> Dflash2SelectorLocalWorkletResolved {
        validate_config(cfg)
            .unwrap_or_else(|reason| panic!("invalid Dflash2SelectorLocalWorkletConfig: {reason}"));

        let vocab_per_rank =
            cfg.vocab_size.clone() / Dim::param("vocab_tp", u32::from(cfg.tp_size));
        let dtype_bytes = cfg.dtype.size_bytes();
        // The merge selection runs over what the all-gather produced: every
        // rank's `top_k` candidates side by side, so `top_k` out of
        // `top_k * tp_size`.
        let gathered_candidates = checked_product(
            "candidate_topk_merge.num_columns",
            &[cfg.selector_top_k, u32::from(cfg.tp_size)],
        )
        .expect("validated DFlash2 gathered candidate width must fit u32");
        // Two codebook rows per candidate edge: one predecessor, one successor,
        // each `selector_rank` wide. The gather is indirect, so the vocabulary
        // row count is an identity of the access pattern, not of the bytes.
        let gather_bytes = checked_product(
            "codebook_gather.bytes_per_token",
            &[2, cfg.selector_top_k, cfg.selector_rank, dtype_bytes],
        )
        .expect("validated DFlash2 codebook gather byte rate must fit u32");

        Dflash2SelectorLocalWorkletResolved {
            lm_head: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: vocab_per_rank.clone(),
                k: cfg.hidden_dim.clone(),
                dtype: cfg.gemm_dtype,
            },
            // The selection reads this rank's own vocabulary shard. `lm_head`
            // emits it in `gemm_dtype`, and the top-k templates on the value
            // type it is handed, so the two dtypes are the same one.
            candidate_topk: LogitsTopkKernelConfig {
                backends: cfg.topk_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_columns: vocab_per_rank.clone(),
                top_k: cfg.selector_top_k,
                dtype: cfg.gemm_dtype,
            },
            candidate_topk_merge: LogitsTopkKernelConfig {
                backends: cfg.topk_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_columns: gathered_candidates.into(),
                top_k: cfg.selector_top_k,
                dtype: cfg.gemm_dtype,
            },
            hidden_projection: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                // `hidden_projection` is a ReplicatedLinear: rank is not sharded.
                n: cfg.selector_rank.into(),
                k: cfg.hidden_dim.clone(),
                dtype: cfg.gemm_dtype,
            },
            codebook_gather: ElementwiseKernelConfig {
                backends: cfg.elementwise_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: gather_bytes.into(),
                output_bytes_per_token: gather_bytes.into(),
            },
            // `einsum("blpr,blcr->blpc")` is, per scored row, `top_k` batches
            // of `[top_k, rank] x [rank, top_k]`. It is priced as one GEMM of
            // the same operand widths over `rows * top_k` rows: identical
            // multiply-accumulate count and identical `n`/`k`, differing only in
            // how the work is batched. `batched_gemm`'s backend identities are
            // deliberately frozen to GLM's Q-absorption and V-up layouts, and
            // this section sits below the measured top-12 of a phase that is
            // itself 9.5% of an iteration, so inventing a generic batched
            // backend would buy nothing it could not also distort.
            edge_scores: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.selector_top_k.into(),
                k: cfg.selector_rank.into(),
                dtype: cfg.gemm_dtype,
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: Dflash2SelectorLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        Ok(Self {
            lm_head: build_atomic(
                &name,
                "lm_head",
                resolved.lm_head.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            candidate_topk: build_atomic(
                &name,
                "candidate_topk",
                resolved.candidate_topk.clone(),
                LogitsTopkKernel::build,
                bridge,
            )?,
            candidate_topk_merge: build_atomic(
                &name,
                "candidate_topk_merge",
                resolved.candidate_topk_merge.clone(),
                LogitsTopkKernel::build,
                bridge,
            )?,
            hidden_projection: build_atomic(
                &name,
                "hidden_projection",
                resolved.hidden_projection.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            codebook_gather: build_atomic(
                &name,
                "codebook_gather",
                resolved.codebook_gather.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            edge_scores: build_atomic(
                &name,
                "edge_scores",
                resolved.edge_scores.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: worklet_label(&self.name, &self.resolved.raw_cfg),
            child: Box::new(CostNode::Sum(vec![
                self.lm_head.compile(builder),
                self.candidate_topk.compile(builder),
                self.candidate_topk_merge.compile(builder),
                self.hidden_projection.compile(builder),
                self.codebook_gather.compile(builder),
                self.edge_scores.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &Dflash2SelectorLocalWorkletInput, ev: &mut Evaluator) {
        let rows = input.scored_rows;
        let zero = rows == 0;
        let streamed = ElementwiseKernelInput { num_tokens: rows };
        // Both selections run over the same scored rows; only the matrix width
        // differs, and that is config, not input.
        let selected = LogitsTopkKernelInput { num_rows: rows };

        eval_atomic_or_zero(&self.lm_head, SingleGemmKernelInput { m: rows }, zero, ev);
        eval_atomic_or_zero(&self.candidate_topk, selected.clone(), zero, ev);
        eval_atomic_or_zero(&self.candidate_topk_merge, selected, zero, ev);
        eval_atomic_or_zero(
            &self.hidden_projection,
            SingleGemmKernelInput { m: rows },
            zero,
            ev,
        );
        eval_atomic_or_zero(&self.codebook_gather, streamed, zero, ev);
        // `rows * top_k` because the batch axis is folded into the row count.
        eval_atomic_or_zero(
            &self.edge_scores,
            SingleGemmKernelInput {
                m: rows.saturating_mul(self.resolved.raw_cfg.selector_top_k),
            },
            zero,
            ev,
        );
    }
}

fn validate_config(cfg: &Dflash2SelectorLocalWorkletConfig) -> Result<(), String> {
    for (name, actual, required) in [
        ("hidden_dim", cfg.hidden_dim.get(), HIDDEN_DIM),
        ("vocab_size", cfg.vocab_size.get(), VOCAB_SIZE),
        ("selector_rank", cfg.selector_rank, SELECTOR_RANK),
        ("selector_top_k", cfg.selector_top_k, SELECTOR_TOP_K),
    ] {
        if actual != required {
            return Err(format!("{name} must be {required}, got {actual}"));
        }
    }
    if cfg.tp_size == 0 || cfg.vocab_size.get() % u32::from(cfg.tp_size) != 0 {
        return Err(format!(
            "vocab_size {} must be divisible by positive tp_size {}",
            cfg.vocab_size, cfg.tp_size
        ));
    }
    Ok(())
}

fn checked_product(name: &str, factors: &[u32]) -> Result<u32, String> {
    factors.iter().try_fold(1_u32, |value, &factor| {
        value
            .checked_mul(factor)
            .ok_or_else(|| format!("{name} overflows u32"))
    })
}

fn worklet_label(name: &str, cfg: &Dflash2SelectorLocalWorkletConfig) -> String {
    format!(
        "{name} (Dflash2SelectorLocalWorklet) \
         [rank-local; tp={}; vocab/rank={}; rank={}; top_k={}]",
        cfg.tp_size,
        cfg.vocab_size.get() / u32::from(cfg.tp_size),
        cfg.selector_rank,
        cfg.selector_top_k
    )
}

fn build_atomic<K, C, F>(
    parent: &str,
    suffix: &str,
    config: C,
    build: F,
    bridge: &PerfApiBridge,
) -> Result<Op<K>, BuildError>
where
    K: Probe,
    F: FnOnce(String, C, &PerfApiBridge) -> Result<K, BuildError>,
{
    let name = format!("{parent}.{suffix}");
    Ok(Op::new(
        name.clone(),
        Arc::new(build(name, config, bridge)?),
    ))
}

fn eval_atomic_or_zero<K>(op: &Op<K>, input: K::Input, zero: bool, ev: &mut Evaluator)
where
    K: Probe,
    K::Input: Clone + Into<SlotInput>,
{
    let metrics = if zero {
        LeafMetrics::ZERO
    } else {
        op.kernel.eval(&input)
    };
    ev.push(metrics, || input.clone().into());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Dflash2SelectorLocalWorkletConfig {
        Dflash2SelectorLocalWorkletConfig {
            gemm_backends: vec!["torch"],
            elementwise_backends: vec!["triton"],
            topk_backends: vec!["flashinfer", "torch"],
            tp_size: 4,
            gpu_name: "NVIDIA B200".to_string(),
            hidden_dim: Dim::param("hidden_dim", HIDDEN_DIM),
            vocab_size: Dim::param("vocab_size", VOCAB_SIZE),
            selector_rank: SELECTOR_RANK,
            selector_top_k: SELECTOR_TOP_K,
            dtype: DType::Bf16,
            gemm_dtype: DType::Bf16,
        }
    }

    #[test]
    fn source_order_runs_the_head_before_the_edges_it_scores() {
        assert_eq!(
            SOURCE_ORDER,
            [
                "lm_head",
                "candidate_topk",
                "candidate_topk_merge",
                "hidden_projection",
                "codebook_gather",
                "edge_scores",
            ]
        );
    }

    #[test]
    fn the_lm_head_shards_the_vocabulary_but_the_rank_projection_does_not() {
        let resolved = Dflash2SelectorLocalWorklet::resolve_config(&cfg());
        assert_eq!(resolved.lm_head.n.get(), VOCAB_SIZE / 4);
        // `hidden_projection` is a ReplicatedLinear onto the selector rank.
        assert_eq!(resolved.hidden_projection.n.get(), SELECTOR_RANK);
        assert_eq!(resolved.hidden_projection.k.get(), HIDDEN_DIM);
    }

    #[test]
    fn edge_scoring_keeps_the_bilinear_operand_widths() {
        let resolved = Dflash2SelectorLocalWorklet::resolve_config(&cfg());
        assert_eq!(resolved.edge_scores.n.get(), SELECTOR_TOP_K);
        assert_eq!(resolved.edge_scores.k.get(), SELECTOR_RANK);
    }

    #[test]
    fn a_vocabulary_that_does_not_shard_is_rejected() {
        let mut bad = cfg();
        bad.tp_size = 3;
        assert!(validate_config(&bad).is_err());
    }
}
