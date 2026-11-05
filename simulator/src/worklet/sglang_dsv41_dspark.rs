//! The DSpark draft of DeepSeek-V4.1-Flash as SGLang v0.5.23 runs it under
//! TP4 on SM12x (`--speculative-algorithm DSPARK --speculative-dspark-block-size
//! 6`; `srt/models/deepseek_v4_dspark.py`, `srt/speculative/dspark_components/`).
//!
//! Per speculative iteration (server log 2026-11-05: `gamma=6,
//! verify_num_draft_tokens=7, query_token_num=6, sample_from_anchor=True`):
//!
//! - **KV inject** (`dspark_kv_inject.py`): each draft stage projects the target
//!   hidden states of its target layer (37, 38, 39) into its own window-only
//!   KV (`DSparkAttention.kv_proj_only`), for every row the target forward ran
//!   (prefill chunks and verify rows alike).
//! - **Draft forward** (decode iterations only): one block-parallel pass of
//!   `gamma` rows per decoding request through the stages
//!   (`DraftTPExecution.TARGET_LAYOUT`: the target's TP4 layout). Each stage is
//!   a DeepSeek-V4 decoder layer with window-only attention
//!   (`compress_ratio == 0`, SWA 128) and an MoE of `dspark_n_routed_experts`
//!   128 experts, top-3, plus the shared expert. Draft decode CUDA graphs are
//!   captured at the target's batch sizes with 6 rows per request.
//! - **Base logits**: the final hc collapse, norm and the shared `lm_head` over
//!   the `gamma` draft rows (vocab-parallel, no gather).
//! - **Markov chain** (`draft_markov_block`, one step per draft position): the
//!   rank-256 Markov correction of the previous token (`markov_w1` row,
//!   `markov_w2` GEMM on this rank's vocab shard: bf16, TP-sharded by default),
//!   `_markov_sample_partial`, the `NcclVocabGather` all-gather of every rank's
//!   partial, `_markov_sample_finish`, and the token broadcast.
//!
//! The stages share one geometry, so they compile once under `Scale { 3 }`.

use super::deepseek_v41_common::{all_gather_as_all_reduce_bytes, eval_or_zero, placeholder};
use super::sglang_dsv41_tp::{
    op, SglangDsv41KernelPath, SglangDsv41LayerConfig, SglangDsv41LayerInput,
    SglangDsv41LayerWorklet,
};
use crate::common::Fabric;
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    AllReduceKernel, AllReduceKernelConfig, AllReduceKernelInput, ElementwiseKernel,
    ElementwiseKernelInput, SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge};

/// Draft geometry beyond the target layer config.
#[derive(Clone, Debug)]
pub struct DsparkDraftConfig {
    /// One draft stage's layer config (window-only attention, draft experts,
    /// draft rows per request as its `verify_width`).
    pub stage: SglangDsv41LayerConfig,
    /// Draft stages (`len(dspark_target_layer_ids)`).
    pub num_stages: u32,
    /// Draft positions per request (`gamma`).
    pub gamma: u32,
    /// `dspark_markov_rank`.
    pub markov_rank: u32,
    pub vocab_size: u32,
}

pub struct DsparkDraftWorklet {
    pub name: String,
    cfg: DsparkDraftConfig,
    kv_inject_proj: Op<SingleGemmKernel>,
    kv_inject_store: Op<ElementwiseKernel>,
    embed: Op<ElementwiseKernel>,
    stage: SglangDsv41LayerWorklet,
    collapse: Op<ElementwiseKernel>,
    norm: Op<ElementwiseKernel>,
    lm_head: Op<SingleGemmKernel>,
    markov_w1: Op<ElementwiseKernel>,
    markov_w2: Op<SingleGemmKernel>,
    markov_partial: Op<ElementwiseKernel>,
    markov_gather: Op<AllReduceKernel>,
    markov_finish: Op<ElementwiseKernel>,
    token_broadcast: Op<AllReduceKernel>,
}

/// Bytes one rank contributes to the Markov partial all-gather per request:
/// a (value, index) pair (`_markov_sample_partial` reduces its vocab shard).
const MARKOV_PARTIAL_BYTES: u64 = 16;
/// The drafted token id broadcast per request.
const TOKEN_BYTES: u64 = 8;

impl DsparkDraftWorklet {
    pub fn build(
        name: String,
        cfg: DsparkDraftConfig,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let s = &cfg.stage;
        let p: SglangDsv41KernelPath = s.path;
        let gpu = s.gpu_name.as_str();
        let tp = s.tp_size;
        let hidden = s.hidden_size;
        let vocab_per_rank = cfg.vocab_size.div_ceil(tp);
        let ew = |i: u32, o: u32| placeholder(&[p.elementwise()], gpu, i, o);
        let gemm = |backend: &'static str, n: u32, k: u32, dtype: DType| SingleGemmKernelConfig {
            backends: vec![backend],
            gpu_name: s.gpu_name.clone(),
            n: n.into(),
            k: k.into(),
            dtype,
        };
        let comm = || AllReduceKernelConfig {
            backends: vec![p.comm()],
            gpu_name: s.gpu_name.clone(),
            num_gpus: tp,
            fabric: Fabric::Pcie,
        };
        let n = &name;
        let stage = SglangDsv41LayerWorklet::build(format!("{name}.stage"), s.clone(), bridge)?;
        Ok(Self {
            // `kv_proj_only`: target hidden (bf16) to the stage's 512-wide KV
            // latent, FP8 block-32 like every draft linear.
            kv_inject_proj: op(
                n,
                "kv_inject.kv_proj",
                gemm(p.fp8_gemm(), s.head_dim, hidden, DType::Fp8E4m3),
                bridge,
            )?,
            // KV norm + RoPE + the 584-byte V4 SWA row.
            kv_inject_store: op(n, "kv_inject.store", ew(s.head_dim * 2, 584), bridge)?,
            // Draft block token ids (anchor + mask) to hc-replicated rows.
            embed: op(n, "embed", ew(8, hidden * 2 * s.hc_mult), bridge)?,
            stage,
            collapse: op(
                n,
                "hc_collapse",
                ew(hidden * 2 * s.hc_mult, hidden * 2),
                bridge,
            )?,
            norm: op(n, "final_rmsnorm", ew(hidden * 2, hidden * 2), bridge)?,
            lm_head: op(
                n,
                "lm_head",
                gemm(p.bf16_gemm(), vocab_per_rank, hidden, DType::Bf16),
                bridge,
            )?,
            // `markov_w1` embedding row of the previous token.
            markov_w1: op(n, "markov.w1", ew(8, cfg.markov_rank * 2), bridge)?,
            markov_w2: op(
                n,
                "markov.w2",
                gemm(p.bf16_gemm(), vocab_per_rank, cfg.markov_rank, DType::Bf16),
                bridge,
            )?,
            // Base + Markov logits on this rank's shard, reduced to a partial.
            markov_partial: op(
                n,
                "markov.sample_partial",
                ew(vocab_per_rank * 4, MARKOV_PARTIAL_BYTES as u32),
                bridge,
            )?,
            markov_gather: op(n, "markov.vocab_all_gather", comm(), bridge)?,
            markov_finish: op(
                n,
                "markov.sample_finish",
                ew(MARKOV_PARTIAL_BYTES as u32 * tp, TOKEN_BYTES as u32),
                bridge,
            )?,
            token_broadcast: op(n, "markov.token_broadcast", comm(), bridge)?,
            cfg,
            name,
        })
    }

    /// Mint order here is the eval order in [`Self::eval`].
    pub fn compile(&self, b: &mut CostTreeBuilder) -> CostNode {
        let inject = CostNode::Scale {
            n: self.cfg.num_stages,
            child: Box::new(CostNode::Sum(vec![
                self.kv_inject_proj.compile(b),
                self.kv_inject_store.compile(b),
            ])),
        };
        let embed = self.embed.compile(b);
        let stages = CostNode::Scale {
            n: self.cfg.num_stages,
            child: Box::new(self.stage.compile(b)),
        };
        let logits = CostNode::Sum(vec![
            self.collapse.compile(b),
            self.norm.compile(b),
            self.lm_head.compile(b),
        ]);
        let chain = CostNode::Scale {
            n: self.cfg.gamma,
            child: Box::new(CostNode::Sum(vec![
                self.markov_w1.compile(b),
                self.markov_w2.compile(b),
                self.markov_partial.compile(b),
                self.markov_gather.compile(b),
                self.markov_finish.compile(b),
                self.token_broadcast.compile(b),
            ])),
        };
        CostNode::Labeled {
            label: format!(
                "{} (DsparkDraftWorklet) [{} stages x (window-only attention + {} experts top-{}); gamma={}; markov rank {}]",
                self.name,
                self.cfg.num_stages,
                self.cfg.stage.num_experts,
                self.cfg.stage.top_k,
                self.cfg.gamma,
                self.cfg.markov_rank
            ),
            child: Box::new(CostNode::Sum(vec![inject, embed, stages, logits, chain])),
        }
    }

    /// `target_rows`: rows the target forward ran this iteration (KV inject).
    /// `draft`: the draft forward's batch (`None` when no request decodes:
    /// SGLang drafts only from a verify).
    pub fn eval(
        &self,
        target_rows: u32,
        draft: Option<&SglangDsv41LayerInput>,
        ev: &mut Evaluator,
    ) {
        let inject_zero = target_rows == 0;
        eval_or_zero(
            &self.kv_inject_proj,
            SingleGemmKernelInput { m: target_rows },
            inject_zero,
            ev,
        );
        eval_or_zero(
            &self.kv_inject_store,
            ElementwiseKernelInput {
                num_tokens: target_rows,
            },
            inject_zero,
            ev,
        );
        let empty = SglangDsv41LayerInput::default();
        let input = draft.unwrap_or(&empty);
        let rows = input.rows;
        let requests = input.decode_kv_lens.len() as u32;
        let zero = draft.is_none() || rows == 0;
        eval_or_zero(
            &self.embed,
            ElementwiseKernelInput { num_tokens: rows },
            zero,
            ev,
        );
        self.stage.eval(input, ev);
        let ew = ElementwiseKernelInput { num_tokens: rows };
        eval_or_zero(&self.collapse, ew.clone(), zero, ev);
        eval_or_zero(&self.norm, ew, zero, ev);
        eval_or_zero(&self.lm_head, SingleGemmKernelInput { m: rows }, zero, ev);
        let per_request = ElementwiseKernelInput {
            num_tokens: requests,
        };
        let tp = u64::from(self.cfg.stage.tp_size);
        eval_or_zero(&self.markov_w1, per_request.clone(), zero, ev);
        eval_or_zero(
            &self.markov_w2,
            SingleGemmKernelInput { m: requests },
            zero,
            ev,
        );
        eval_or_zero(&self.markov_partial, per_request.clone(), zero, ev);
        eval_or_zero(
            &self.markov_gather,
            AllReduceKernelInput {
                message_size_bytes: all_gather_as_all_reduce_bytes(
                    u64::from(requests) * MARKOV_PARTIAL_BYTES * tp,
                ),
            },
            zero,
            ev,
        );
        eval_or_zero(&self.markov_finish, per_request, zero, ev);
        eval_or_zero(
            &self.token_broadcast,
            AllReduceKernelInput {
                message_size_bytes: u64::from(requests) * TOKEN_BYTES,
            },
            zero,
            ev,
        );
    }
}
