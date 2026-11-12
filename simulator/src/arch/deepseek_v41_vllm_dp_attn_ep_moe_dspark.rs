//! DeepSeek-V4.1-Flash with its DSpark proposer under vLLM DP4 attention +
//! EP4 experts on B200 (`--data-parallel-size 4 --enable-expert-parallel`,
//! `--speculative-config {"method":"dspark","num_speculative_tokens":5}`;
//! capture `logs/20261106_0_dsv41_dspark_capture/profile_dp4ep4_*`,
//! `phase1_split.md` §1.3, §3, §6.2).
//!
//! The DP sibling of [`super::deepseek_v41_vllm_dspark`]: the DP4 target
//! ([`DeepseekV41VllmDpModel`]) at verify width 6, then per rank the sample
//! phase and the proposer. Every rank is its own engine core, so the sampler,
//! the context KV, the embedding, the head and the Markov loop are rank-local
//! (TP1: full-vocab lm_head, no logits gather, no embedding all-reduce). The
//! draft layers are DP layer bodies: the draft MoE is EP-sharded like the
//! target's (32 local of 128 experts), behind the same MXFP8 all-gatherv /
//! bf16 reduce-scatterv exchange (§3 row 15: "AllGather x3 + ReduceScatter
//! x3, replaces the draft TP all-reduces"):
//!
//! ```text
//! target (DP4/EP4, verify width 6)
//! Max_r [sample, context KV, embedding]
//! draft layer 0: Max_r [attention, FFN up to the dispatch] -> exchange -> Max_r [add]
//! draft layers 1..2 (Scale{2}), the same
//! Max_r [head, fp32 cast, 5 Markov steps]
//! ```
//!
//! The rank-local stretches between collectives are maxed separately (sum of
//! maxes), as the target's head is: a slight over-estimate when ranks differ.
//!
//! Draft shape (`dispatch_cg_and_sync_dp`, `v1/worker/gpu/dp_utils.py`): the
//! draft is one FULL graph keyed by request count, every rank padded to the
//! busiest rank's graph size; past the largest graph every rank runs eager on
//! its own requests. A rank with nothing scheduled runs the one-request dummy.

use crate::arch::contract::{SpeculativeArchInput, SpeculativeUnifiedModel};
use crate::arch::deepseek_v41_dspark_plumbing::{
    PlumbingShape, StepPlumbing, StepPlumbingConfig,
};
use crate::arch::deepseek_v41_vllm::{
    kv_cache_layout, prefill_pairs, BoundedReplay, DeepseekV41CudaGraphs, DeepseekV41ModelCfg,
    DeepseekV41Step, DeepseekV41VllmParallel, LayerInput, MAX_NUM_SEQS,
};
use crate::arch::deepseek_v41_vllm_dp_attn_ep_moe::{
    self as dp_arch, dp_layer_config, labeled_max, normalize_dp_steps, resolve_layer,
    DeepseekV41DpLayerResolved, DeepseekV41VllmDpConfigs, DeepseekV41VllmDpModel,
    DeepseekV41VllmDpParallel, DeepseekV41VllmDpResolved, DpLayerBody, DpLayerInput, DpStep,
    DP_SIZE,
};
use crate::arch::deepseek_v41_vllm_dspark::{
    atomic_elementwise, check_draft, context_kv_config, draft_layer_configs, logits_placeholders,
    markov_config, widen_replay_scatter, DsparkCfg, LogitsPlaceholders, SampleStage,
};
use crate::arch::kv_layout::KvCacheLayout;
use crate::op::spec::{DsparkMarkovStepConfig, DsparkMarkovStepInput, DsparkMarkovStepOp};
use crate::op::Op;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::kernels::{ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::deepseek_v41_common::eval_or_zero;
use crate::worklet::{
    DeepseekV41AttentionTpWorkletInput, DeepseekV41DsparkContextKvLocalWorklet,
    DeepseekV41DsparkContextKvLocalWorkletConfig, DeepseekV41DsparkContextKvLocalWorkletInput,
    DeepseekV41DsparkContextKvLocalWorkletResolved, DeepseekV41HeadTpWorklet,
    DeepseekV41HeadTpWorkletConfig, DeepseekV41HeadTpWorkletInput,
    DeepseekV41HeadTpWorkletResolved, DeepseekV41PrologueTpWorklet,
};

pub struct DeepseekV41DpDsparkConfigs {
    pub target: DeepseekV41VllmDpConfigs,
    pub draft_tokens: u32,
    pub num_draft_layers: u32,
    pub context_kv: DeepseekV41DsparkContextKvLocalWorkletConfig,
    /// Draft layer 0 and the shared body of the rest, as DP layer bodies.
    pub draft_first: dp_arch::DeepseekV41DpLayerConfig,
    pub draft_rest: dp_arch::DeepseekV41DpLayerConfig,
    /// The target's full-vocab head (TP1, no gather).
    pub head: DeepseekV41HeadTpWorkletConfig,
    pub markov: DsparkMarkovStepConfig,
    pub vocab_size: u32,
    pub kv_cache_layout: KvCacheLayout,
    /// `--cudagraph-capture-sizes` per rank (verify/draft FULL graphs).
    pub cudagraph_capture_sizes: Vec<u32>,
    pub plumbing: StepPlumbingConfig,
}

pub struct DeepseekV41DpDsparkResolved {
    target: DeepseekV41VllmDpResolved,
    draft_tokens: u32,
    num_draft_layers: u32,
    context_kv: DeepseekV41DsparkContextKvLocalWorkletResolved,
    embedding: ElementwiseKernelConfig,
    draft_first: DeepseekV41DpLayerResolved,
    draft_rest: DeepseekV41DpLayerResolved,
    head: DeepseekV41HeadTpWorkletResolved,
    markov: DsparkMarkovStepConfig,
    logits: LogitsPlaceholders,
    kv_cache_layout: KvCacheLayout,
    cudagraph_capture_sizes: Vec<u32>,
    plumbing: StepPlumbingConfig,
}

/// `target` is lane B's DP expansion; `draft_demand` covers the draft MoE.
pub fn build_configs(
    model: &DeepseekV41ModelCfg,
    dspark: &DsparkCfg,
    parallel: &DeepseekV41VllmDpParallel,
    mut target: DeepseekV41VllmDpConfigs,
    draft_demand: &ExpertDemand,
    draft_tokens: u32,
) -> std::result::Result<DeepseekV41DpDsparkConfigs, BuildError> {
    check_draft(dspark, draft_demand, draft_tokens)?;
    widen_replay_scatter(target.replay.as_mut(), model, dspark);
    // The rank-local recipe, as the DP target's: TP1, experts over the group.
    let local = DeepseekV41VllmParallel {
        tp_size: 1,
        ep_size: DP_SIZE,
        gpu_name: parallel.gpu_name.clone(),
        serialize_streams: false,
        decoder_swa_bounded_replay: false,
        max_model_len: parallel.max_model_len,
        max_num_batched_tokens: parallel.max_num_batched_tokens,
        cudagraph_capture_sizes: parallel.cudagraph_capture_sizes.clone(),
    };
    let (first, rest) = draft_layer_configs(model, dspark, &local, draft_demand);
    // The draft exchange's all-gatherv per-rank cap, the target's rule: a
    // rank's rows are its scheduled tokens or the graph it pads to.
    let graphs = DeepseekV41CudaGraphs::new(
        &parallel.cudagraph_capture_sizes,
        MAX_NUM_SEQS,
        &[draft_tokens + 1],
    );
    let max_rank_tokens = parallel.max_num_batched_tokens.max(graphs.max_tokens());
    Ok(DeepseekV41DpDsparkConfigs {
        draft_tokens,
        num_draft_layers: dspark.num_draft_layers,
        context_kv: context_kv_config(model, dspark, &first.attention),
        draft_first: dp_layer_config(first, max_rank_tokens),
        draft_rest: dp_layer_config(rest, max_rank_tokens),
        head: target.head.clone(),
        markov: markov_config(model, dspark, &parallel.gpu_name),
        vocab_size: model.vocab_size,
        kv_cache_layout: kv_cache_layout(model, 1, dspark.num_draft_layers),
        cudagraph_capture_sizes: parallel.cudagraph_capture_sizes.clone(),
        // Each rank runs its own engine core at TP1.
        plumbing: StepPlumbingConfig::new(
            model,
            dspark.num_draft_layers,
            draft_tokens,
            1,
            &parallel.gpu_name,
            parallel.max_model_len,
            parallel.max_num_batched_tokens,
        ),
        target,
    })
}

pub fn resolve_configs(configs: DeepseekV41DpDsparkConfigs) -> DeepseekV41DpDsparkResolved {
    let prologue = DeepseekV41PrologueTpWorklet::resolve_config(&configs.target.prologue);
    assert!(
        prologue.embedding_all_reduce.is_none(),
        "a TP1 rank embeds without an all-reduce"
    );
    DeepseekV41DpDsparkResolved {
        draft_tokens: configs.draft_tokens,
        num_draft_layers: configs.num_draft_layers,
        context_kv: DeepseekV41DsparkContextKvLocalWorklet::resolve_config(&configs.context_kv),
        embedding: prologue.embedding,
        draft_first: resolve_layer(&configs.draft_first),
        draft_rest: resolve_layer(&configs.draft_rest),
        head: DeepseekV41HeadTpWorklet::resolve_config(&configs.head),
        logits: logits_placeholders(configs.vocab_size, &configs.markov.gpu_name),
        markov: configs.markov,
        kv_cache_layout: configs.kv_cache_layout,
        cudagraph_capture_sizes: configs.cudagraph_capture_sizes,
        plumbing: configs.plumbing,
        target: dp_arch::resolve_configs(&configs.target),
    }
}

/// One rank's sample phase and the draft's rank-local launches. Each rank
/// owns its own copy of every leaf (one slot set per rank under `Max_r`).
struct RankEdges {
    sample: SampleStage,
    context_kv: DeepseekV41DsparkContextKvLocalWorklet,
    embedding: Op<ElementwiseKernel>,
    head: DeepseekV41HeadTpWorklet,
    markov: DsparkMarkovStepOp,
    plumbing: StepPlumbing,
}

/// One rank's draft call.
#[derive(Clone, Debug)]
struct RankDraft {
    /// Verified rows sampled on this rank (0 on an idle rank).
    logits_rows: u32,
    /// Decode requests resampled on this rank.
    decode_requests: u32,
    /// Target rows this rank scheduled (the context-KV rows).
    context_tokens: u32,
    /// Requests the draft graph runs (padded).
    requests: u32,
    layer: LayerInput,
    plumbing: PlumbingShape,
}

/// DeepSeek-V4.1-Flash + DSpark, DP4 attention + EP4 experts.
pub struct DeepseekV41VllmDpDsparkModel {
    name: String,
    target: DeepseekV41VllmDpModel,
    ranks: Vec<RankEdges>,
    draft_first: DpLayerBody,
    draft_rest: DpLayerBody,
    draft_tokens: u32,
    num_draft_layers: u32,
    kv_cache_layout: KvCacheLayout,
    /// Graphs at the verify width (decode query length `draft_tokens + 1`).
    graphs: DeepseekV41CudaGraphs,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

pub fn build(
    name: String,
    resolved: DeepseekV41DpDsparkResolved,
    bridge: &PerfApiBridge,
) -> std::result::Result<DeepseekV41VllmDpDsparkModel, BuildError> {
    let dspark = format!("{name}.dspark");
    let ranks = (0..DP_SIZE)
        .map(|_| -> std::result::Result<_, BuildError> {
            Ok(RankEdges {
                sample: SampleStage::build(&format!("{name}.sample"), &resolved.logits, bridge)?,
                context_kv: DeepseekV41DsparkContextKvLocalWorklet::build(
                    format!("{dspark}.context_kv"),
                    resolved.context_kv.clone(),
                    bridge,
                )?,
                embedding: atomic_elementwise(
                    format!("{dspark}.embedding"),
                    resolved.embedding.clone(),
                    bridge,
                )?,
                head: DeepseekV41HeadTpWorklet::build(
                    format!("{dspark}.head"),
                    resolved.head.clone(),
                    bridge,
                )?,
                markov: DsparkMarkovStepOp::build(
                    format!("{dspark}.markov"),
                    resolved.markov.clone(),
                    bridge,
                )?,
                plumbing: StepPlumbing::build(
                    &format!("{name}.plumbing"),
                    &resolved.plumbing,
                    bridge,
                )?,
            })
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut model = DeepseekV41VllmDpDsparkModel {
        draft_first: DpLayerBody::build(&dspark, resolved.draft_first, bridge)?,
        draft_rest: DpLayerBody::build(&dspark, resolved.draft_rest, bridge)?,
        target: dp_arch::build(name.clone(), resolved.target, bridge)?,
        ranks,
        graphs: DeepseekV41CudaGraphs::new(
            &resolved.cudagraph_capture_sizes,
            MAX_NUM_SEQS,
            &[resolved.draft_tokens + 1],
        ),
        draft_tokens: resolved.draft_tokens,
        num_draft_layers: resolved.num_draft_layers,
        kv_cache_layout: resolved.kv_cache_layout,
        cost_flat: Vec::new(),
        n_slots: 0,
        name,
    };
    let tree = model.cost_tree();
    model.cost_flat = tree.flatten();
    model.n_slots = tree.n_slots();
    Ok(model)
}

impl DeepseekV41VllmDpDsparkModel {
    pub fn cost_tree(&self) -> CostTree {
        let mut b = CostTreeBuilder::new();
        let mut sections = self.target.compile_sections(&mut b);
        let dspark = format!("{}.dspark", self.name);
        let pre = self
            .ranks
            .iter()
            .map(|rank| {
                CostNode::Sum(vec![
                    rank.sample.compile(&mut b),
                    rank.context_kv.compile(&mut b),
                    rank.embedding.compile(&mut b),
                ])
            })
            .collect();
        sections.push(labeled_max(
            format!(
                "{}.sample [Max over DP ranks: rejection sampler -> DSpark context KV -> \
                 embedding]",
                self.name
            ),
            pre,
        ));
        let mut draft = vec![self.draft_first.compile(None, &mut b)];
        draft.push(CostNode::Labeled {
            label: format!(
                "{dspark}.layers [{} identical SWA-only layers]",
                self.num_draft_layers - 1
            ),
            child: Box::new(CostNode::Scale {
                n: self.num_draft_layers - 1,
                child: Box::new(self.draft_rest.compile(None, &mut b)),
            }),
        });
        let post = self
            .ranks
            .iter()
            .map(|rank| {
                CostNode::Sum(vec![
                    rank.head.compile(&mut b),
                    CostNode::Scale {
                        n: self.draft_tokens,
                        child: Box::new(rank.markov.compile(&mut b)),
                    },
                ])
            })
            .collect();
        draft.push(labeled_max(
            format!(
                "{dspark}.head [Max over DP ranks: full-vocab head -> {} Markov steps]",
                self.draft_tokens
            ),
            post,
        ));
        sections.push(CostNode::Labeled {
            label: format!(
                "{dspark} [DSpark proposer; draft_tokens={}; {} draft layers, EP-sharded MoE]",
                self.draft_tokens, self.num_draft_layers
            ),
            child: Box::new(CostNode::Sum(draft)),
        });
        let plumbing = self
            .ranks
            .iter()
            .map(|rank| rank.plumbing.compile(&mut b))
            .collect();
        sections.push(labeled_max(
            format!(
                "{}.plumbing [Max over DP ranks: step metadata, input prep, sampler and draft \
                 bookkeeping]",
                self.name
            ),
            plumbing,
        ));
        b.finish(CostNode::Labeled {
            label: format!(
                "{} (DeepseekV41VllmDpDsparkModel) [DP{DP_SIZE} attention at TP1, EP{DP_SIZE} \
                 experts; verify_width={}]",
                self.name,
                self.draft_tokens + 1
            ),
            child: Box::new(CostNode::Sum(sections)),
        })
    }

    fn normalize(
        &self,
        input: &SpeculativeArchInput,
    ) -> std::result::Result<(DpStep, Vec<RankDraft>, DpLayerInput), String> {
        normalize_dp_dspark_input(
            &self.graphs,
            input,
            self.draft_tokens,
            self.target.max_model_len(),
            self.target.replay_rule(),
        )
    }

    fn eval_checked(&self, input: &SpeculativeArchInput, ev: &mut Evaluator) {
        let (target, ranks, draft) = self.normalize(input).unwrap_or_else(|reason| {
            panic!("invalid DeepseekV41VllmDpDsparkModel input: {reason}")
        });
        self.target.eval_step(&target, ev);
        for (edges, rank) in self.ranks.iter().zip(&ranks) {
            edges
                .sample
                .eval(rank.logits_rows, rank.decode_requests, ev);
            edges.context_kv.eval(
                &DeepseekV41DsparkContextKvLocalWorkletInput {
                    context_tokens: rank.context_tokens,
                },
                ev,
            );
            let rows = rank.layer.rows;
            eval_or_zero(
                &edges.embedding,
                ElementwiseKernelInput { num_tokens: rows },
                rows == 0,
                ev,
            );
        }
        self.draft_first.eval(&draft, None, ev);
        self.draft_rest.eval(&draft, None, ev);
        for (edges, rank) in self.ranks.iter().zip(&ranks) {
            let rows = rank.layer.rows;
            edges.head.eval(
                &DeepseekV41HeadTpWorkletInput {
                    num_tokens: rows,
                    mhc_post_rows: rows,
                    logits_rows: rows,
                },
                ev,
            );
            edges.markov.eval(
                &DsparkMarkovStepInput {
                    num_requests: rank.requests,
                },
                ev,
            );
        }
        for (edges, rank) in self.ranks.iter().zip(&ranks) {
            edges.plumbing.eval(&rank.plumbing, ev);
        }
        assert_eq!(
            ev.filled(),
            self.n_slots,
            "eval must fill every compiled slot"
        );
    }
}

/// Split one DP speculative step (one group per rank) into the target's DP
/// step, each rank's sample/draft shapes, and the draft layers' DP batch.
fn normalize_dp_dspark_input(
    graphs: &DeepseekV41CudaGraphs,
    input: &SpeculativeArchInput,
    draft_tokens: u32,
    max_model_len: u32,
    replay: Option<&BoundedReplay>,
) -> std::result::Result<(DpStep, Vec<RankDraft>, DpLayerInput), String> {
    if input.draft_tokens != draft_tokens {
        return Err(format!(
            "input draft_tokens {} must match model draft_tokens {draft_tokens}",
            input.draft_tokens
        ));
    }
    if input.groups.len() != DP_SIZE as usize {
        return Err(format!(
            "DP{DP_SIZE} attention takes one group per rank, got {}",
            input.groups.len()
        ));
    }
    let verify_width = draft_tokens + 1;
    let mut steps = Vec::with_capacity(input.groups.len());
    let mut logits_rows = Vec::with_capacity(input.groups.len());
    for (rank, group) in input.groups.iter().enumerate() {
        let prefill = prefill_pairs(&group.prefill_chunk_pairs, max_model_len)
            .map_err(|reason| format!("rank {rank}: {reason}"))?;
        let mut decode = Vec::with_capacity(group.decode_requests.len());
        for (index, request) in group.decode_requests.iter().enumerate() {
            request
                .check_verify_row(draft_tokens, max_model_len)
                .map_err(|reason| format!("rank {rank} decode request {index}: {reason}"))?;
            decode.push((request.query_len, request.kv_len));
        }
        let prefill_tokens: u32 = prefill.iter().map(|&(q, _)| q).sum();
        let decode_tokens: u32 = decode.iter().map(|&(q, _)| q).sum();
        if group.prefill_tokens != prefill_tokens
            || group.decode_tokens != decode_tokens
            || group.batch_tokens != prefill_tokens + decode_tokens
        {
            return Err(format!("rank {rank} token accounting is inconsistent"));
        }
        logits_rows.push(prefill.len() as u32 + decode_tokens);
        steps.push(DeepseekV41Step { prefill, decode });
    }
    let target = normalize_dp_steps(graphs, &steps, verify_width, logits_rows.clone(), replay);

    // Every scheduled request drafts a block; an idle rank runs the dummy's
    // one request (a one-token context). Ranks agree on the draft graph.
    let live: Vec<u32> = steps.iter().map(|s| s.request_count().max(1)).collect();
    let busiest = live.iter().copied().max().unwrap_or(1);
    let graph = graphs.full_decode_requests(busiest, verify_width);
    let ranks: Vec<RankDraft> = steps
        .iter()
        .zip(&live)
        .zip(&logits_rows)
        .zip(&target.full.ranks)
        .map(|(((step, &live), &logits), target_rank)| {
            let requests = graph.unwrap_or(live);
            let rows = requests * draft_tokens;
            let contexts: Vec<u32> = if step.request_count() == 0 {
                vec![1]
            } else {
                step.prefill
                    .iter()
                    .map(|&(_, context)| context)
                    .chain(step.decode.iter().map(|&(_, kv)| kv))
                    .collect()
            };
            // Each draft row attends its request's window plus the block before it.
            let decode_kv_lens = contexts
                .iter()
                .flat_map(|&context| {
                    (0..draft_tokens).map(move |offset| {
                        context
                            .saturating_add(offset)
                            .min(max_model_len.saturating_sub(1))
                    })
                })
                .collect();
            let plumbing = PlumbingShape {
                padded_tokens: target_rank.rows,
                requests: step.request_count(),
                contexts: step
                    .prefill
                    .iter()
                    .map(|&(_, context)| context)
                    .chain(step.decode.iter().map(|&(_, kv)| kv))
                    .collect(),
                logits_rows: logits,
                draft_requests: requests,
                draft_rows: rows,
            };
            RankDraft {
                plumbing,
                logits_rows: logits,
                decode_requests: step.decode.len() as u32,
                context_tokens: if step.request_count() == 0 {
                    1
                } else {
                    step.prefill
                        .iter()
                        .chain(&step.decode)
                        .map(|&(q, _)| q)
                        .sum()
                },
                requests,
                layer: LayerInput {
                    rows,
                    // `_prepare_dflash_inputs_kernel` marks each request's query
                    // rows non-padding, the idle rank's dummy request included
                    // (its draft is no `make_dummy` batch); graph slots past
                    // them are padding.
                    scheduled_rows: live * draft_tokens,
                    attention: DeepseekV41AttentionTpWorkletInput {
                        num_tokens: rows,
                        prefill_query_context_pairs: Vec::new(),
                        decode_kv_lens,
                        aux_stream_live: graph.is_some(),
                        full_graph: graph.is_some(),
                        // Draft layers have no `mhc_stream` (`dspark.py:128-138`).
                        mhc_overlap: false,
                        write_kv_only: false,
                    },
                },
            }
        })
        .collect();
    let draft = DpLayerInput::new(ranks.iter().map(|rank| rank.layer.clone()).collect());
    Ok((target, ranks, draft))
}

impl SpeculativeUnifiedModel for DeepseekV41VllmDpDsparkModel {
    fn check_input(&self, batch: &SpeculativeArchInput) -> std::result::Result<(), String> {
        self.normalize(batch).map(|_| ())
    }

    fn eval_speculative_iter(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut evaluator = Evaluator::new(slots);
        self.eval_checked(batch, &mut evaluator);
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }

    fn eval_speculative_iter_with_inputs(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut evaluator = Evaluator::with_inputs(slots, inputs);
        self.eval_checked(batch, &mut evaluator);
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }

    fn cost_log_manifest(&self) -> CostManifest {
        self.cost_tree().manifest()
    }

    /// One DP rank's cache: the target's TP1 groups plus three 128-token
    /// draft SWA caches.
    fn kv_cache_layout(&self) -> KvCacheLayout {
        self.kv_cache_layout.clone()
    }

    fn max_model_len(&self) -> u32 {
        self.target.max_model_len()
    }

    fn gpus_per_replica(&self) -> u16 {
        DP_SIZE as u16
    }

    fn num_attn_dp_groups(&self) -> u16 {
        DP_SIZE as u16
    }

    fn num_attn_shards(&self) -> u16 {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::deepseek_v41_vllm::GraphMode;
    use std::path::Path;

    use crate::arch::config::ModelSpec;
    use crate::arch::contract::{SpeculativeArchGroupInput, SpeculativeDecodeInput};
    use crate::arch::deepseek_v41_vllm::vllm_default_cudagraph_capture_sizes;

    const CONFIG: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/model/config/deepseek_v41_flash.json"
    );

    fn model_cfg() -> DeepseekV41ModelCfg {
        let spec = ModelSpec {
            model_config: CONFIG.into(),
            num_layers: None,
            sim_num_layers: None,
            fp8: true,
        };
        DeepseekV41ModelCfg::from_json(Path::new(CONFIG), &spec).unwrap()
    }

    fn parallel() -> DeepseekV41VllmDpParallel {
        DeepseekV41VllmDpParallel {
            gpu_name: "NVIDIA B200".into(),
            max_model_len: 131_072,
            decoder_swa_bounded_replay: true,
            max_num_batched_tokens: 2048,
            cudagraph_capture_sizes: vllm_default_cudagraph_capture_sizes(2048),
        }
    }

    fn uniform(experts: usize, layers: usize) -> ExpertDemand {
        ExpertDemand::Popularity {
            layerwise_global_ppm: vec![vec![1_000_000 / experts as u32; experts]; layers],
        }
    }

    fn configs() -> DeepseekV41DpDsparkConfigs {
        let model = model_cfg();
        let dspark = DsparkCfg::from_json(Path::new(CONFIG)).unwrap();
        let target = dp_arch::build_configs(&model, &parallel(), &uniform(384, 40)).unwrap();
        build_configs(&model, &dspark, &parallel(), target, &uniform(128, 1), 5).unwrap()
    }

    #[test]
    fn draft_layers_are_tp1_with_the_ep_sharded_draft_moe() {
        let configs = configs();
        for layer in [&configs.draft_first, &configs.draft_rest] {
            assert_eq!(layer.attention.tp_size, 1);
            assert_eq!(layer.attention.layer.compress_ratio, 0);
            assert_eq!(
                (
                    layer.ffn.ffn.num_experts.get(),
                    layer.ffn.ffn.top_k,
                    layer.ffn.ffn.ep_size
                ),
                (128, 3, 4)
            );
        }
        assert_eq!(configs.draft_first.layers, vec![40]);
        assert_eq!(configs.draft_rest.layers, vec![41, 42]);
        // The full-vocab head, no gather.
        let head = DeepseekV41HeadTpWorklet::resolve_config(&configs.head);
        assert_eq!(head.lm_head.n.get(), 129_280);
    }

    /// The DP4 DSpark capture's per-rank pool: 493,892 blocks of the TP1
    /// stride logged 270.92x at 131,072 tokens (35,510,374 tokens).
    #[test]
    fn kv_cache_layout_reproduces_the_dp4_dspark_pool() {
        use crate::worker::kv::{KvAccounting, KvAccountingOptions};
        let layout = configs().kv_cache_layout;
        let block = layout.engine.fixed_device_bytes;
        let options = KvAccountingOptions {
            max_in_flight_tokens: Some(2 * 2048),
            ..Default::default()
        };
        let accounting = KvAccounting::resolve(&layout, options, 493_892 * block).unwrap();
        let concurrency = accounting.concurrency_at(131_072);
        assert!(
            (concurrency / 270.92 - 1.0).abs() < 0.005,
            "{concurrency} vs 270.92"
        );
    }

    fn group(decode: &[(u32, u32)]) -> SpeculativeArchGroupInput {
        let decode_tokens = decode.iter().map(|&(_, q)| q).sum();
        SpeculativeArchGroupInput {
            batch_tokens: decode_tokens,
            prefill_tokens: 0,
            decode_tokens,
            prefill_chunk_pairs: Vec::new(),
            decode_requests: decode
                .iter()
                .map(|&(kv_len, query_len)| SpeculativeDecodeInput { kv_len, query_len })
                .collect(),
            total_kv_len: 0,
        }
    }

    #[test]
    fn ranks_pad_the_draft_to_the_busiest_rank_and_idle_ranks_run_the_dummy() {
        let input = SpeculativeArchInput {
            draft_tokens: 5,
            groups: vec![
                group(&[(1000, 6); 5]),
                group(&[(2000, 6)]),
                group(&[]),
                group(&[(3000, 6); 2]),
            ],
            tokens_per_source_rank: Vec::new(),
        };
        let (target, ranks, draft) = normalize_dp_dspark_input(
            &DeepseekV41CudaGraphs::new(&parallel().cudagraph_capture_sizes, MAX_NUM_SEQS, &[6]),
            &input,
            5,
            131_072,
            None,
        )
        .unwrap();
        // The target: the idle rank's dummy is one verify-width request, so
        // the step stays a FULL graph at the busiest rank's six requests.
        assert_eq!(target.mode, GraphMode::Full);
        assert!(target.full.ranks.iter().all(|rank| rank.rows == 36));
        // The draft: every rank at the six-request graph, 30 rows.
        assert!(ranks
            .iter()
            .all(|rank| rank.requests == 6 && rank.layer.rows == 30));
        assert_eq!(draft.exchange.collective_tokens, vec![30; 4]);
        assert_eq!(
            ranks
                .iter()
                .map(|rank| rank.logits_rows)
                .collect::<Vec<_>>(),
            vec![30, 6, 0, 12]
        );
        assert_eq!(ranks[2].layer.attention.decode_kv_lens.len(), 5);
    }
}
