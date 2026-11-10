//! DeepSeek-V4.1-Flash with DSpark speculative decoding as SGLang v0.5.23
//! serves it on SM12x (4x RTX PRO 6000; `--speculative-algorithm DSPARK
//! --speculative-dspark-block-size 6`).
//!
//! One speculative iteration is the target forward ([`DeepseekV41SglangModel`]
//! built at the verify width: `draft_tokens + 1` rows per decoding request,
//! padded to the target-verify CUDA-graph batch) followed by the DSpark draft
//! ([`DsparkDraftWorklet`]): KV inject for every target row, then, when any
//! request decodes, one block-parallel draft forward of `draft_tokens` rows
//! per request and the per-position Markov sampling chain.
//!
//! Scheduling budget: SGLang's separate-prefill scheduler caps only extend
//! (prefill) tokens by `--chunked-prefill-size`; the verify and draft rows run
//! in decode iterations outside that budget, and a prefill chunk reserves no
//! draft slots. So [`SpeculativeUnifiedModel::drafting_slots_per_request`] is
//! 0 (MTP-like), unlike vLLM's DFlash.
//!
//! Not modeled: drafting the first block of a request in the iteration that
//! finishes its prefill (the input does not say which chunk is final; the
//! first draft is billed with the next verify), and DSpark's ragged verify
//! (`SGLANG_RAGGED_VERIFY_MODE`; verify always runs the full window).

use anyhow::Result;

use crate::arch::contract::{SpeculativeArchInput, SpeculativeUnifiedModel};
use crate::arch::deepseek_v41_sglang::{
    self as target, decode_graph_batch, layer_config, DeepseekV41SglangModel,
    DeepseekV41SglangSettings,
};
use crate::arch::deepseek_v41_vllm::DeepseekV41ModelCfg;
use crate::arch::kv_layout::KvCacheLayout;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::sglang_dsv41_dspark::{DsparkDraftConfig, DsparkDraftWorklet};
use crate::worklet::sglang_dsv41_tp::{SglangDsv41LayerInput, SglangDsv41LayerType};

pub const ARCH_KIND: &str = "deepseek_v41_sglang_dspark";

/// DSpark draft geometry from the checkpoint (`dspark_*` fields).
#[derive(Clone, Debug)]
pub struct DsparkSpec {
    /// `len(dspark_target_layer_ids)`.
    pub num_stages: u32,
    /// `dspark_n_routed_experts`.
    pub num_experts: u32,
    /// `dspark_num_experts_per_tok`.
    pub top_k: u32,
    /// `dspark_markov_rank`.
    pub markov_rank: u32,
}

pub struct DeepseekV41SglangDsparkModel {
    name: String,
    target: DeepseekV41SglangModel,
    draft: DsparkDraftWorklet,
    draft_tokens: u32,
    max_model_len: u32,
    kv_cache_layout: KvCacheLayout,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

/// The draft forward's batch and the target's, from one speculative input.
pub(crate) struct DsparkBatch {
    pub target: SglangDsv41LayerInput,
    pub logits_rows: u32,
    pub draft: Option<SglangDsv41LayerInput>,
}

#[allow(clippy::too_many_arguments)]
pub fn build(
    name: String,
    model: &DeepseekV41ModelCfg,
    settings: &DeepseekV41SglangSettings,
    demand: &ExpertDemand,
    dspark: &DsparkSpec,
    draft_demand: &ExpertDemand,
    draft_tokens: u32,
    bridge: &PerfApiBridge,
) -> std::result::Result<DeepseekV41SglangDsparkModel, BuildError> {
    let verify = DeepseekV41SglangSettings {
        verify_width: draft_tokens + 1,
        ..settings.clone()
    };
    let target_model = target::build(format!("{name}.target"), model, &verify, demand, bridge)?;
    // Window-only attention, no compressor / indexer / Engram, the draft
    // experts; `draft_tokens` rows per request.
    let draft_settings = DeepseekV41SglangSettings {
        verify_width: draft_tokens,
        ..settings.clone()
    };
    let mut stage = layer_config(
        model,
        &draft_settings,
        SglangDsv41LayerType {
            compress_ratio: 0,
            kv_source: false,
            index_source: false,
            engram: false,
        },
        draft_demand,
        0,
    );
    stage.num_experts = dspark.num_experts;
    stage.top_k = dspark.top_k;
    let draft = DsparkDraftWorklet::build(
        format!("{name}.dspark"),
        DsparkDraftConfig {
            stage,
            num_stages: dspark.num_stages,
            gamma: draft_tokens,
            markov_rank: dspark.markov_rank,
            vocab_size: model.vocab_size,
        },
        bridge,
    )?;
    let mut out = DeepseekV41SglangDsparkModel {
        name,
        target: target_model,
        draft,
        draft_tokens,
        max_model_len: settings.max_model_len,
        kv_cache_layout: target::kv_cache_layout_with_draft(model, u64::from(dspark.num_stages)),
        cost_flat: Vec::new(),
        n_slots: 0,
    };
    let tree = out.cost_tree();
    out.cost_flat = tree.flatten();
    out.n_slots = tree.n_slots();
    Ok(out)
}

/// The target verify forward and the draft forward of one iteration.
pub(crate) fn normalize_speculative_input(
    input: &SpeculativeArchInput,
    draft_tokens: u32,
    max_model_len: u32,
) -> std::result::Result<DsparkBatch, String> {
    if input.draft_tokens != draft_tokens {
        return Err(format!(
            "input draft_tokens {} must match model draft_tokens {draft_tokens}",
            input.draft_tokens
        ));
    }
    let [group] = input.groups.as_slice() else {
        return Err(format!(
            "TP4 attention takes exactly one group, got {}",
            input.groups.len()
        ));
    };
    let verify_width = draft_tokens + 1;
    let mut prefill = Vec::with_capacity(group.prefill_chunk_pairs.len());
    let mut prefill_tokens = 0_u32;
    for (i, &(prefix, append)) in group.prefill_chunk_pairs.iter().enumerate() {
        let context = prefix
            .checked_add(append)
            .ok_or("prefill context overflows u32")?;
        if append == 0 || context > max_model_len {
            return Err(format!(
                "prefill request {i} ({prefix}, {append}) must append 1..=max_model_len {max_model_len}"
            ));
        }
        prefill_tokens += append;
        prefill.push((append, context));
    }
    if group.prefill_tokens != prefill_tokens {
        return Err(format!(
            "prefill_tokens {} must equal the append sum {prefill_tokens}",
            group.prefill_tokens
        ));
    }
    let mut decode_kv_lens = Vec::with_capacity(group.decode_requests.len());
    let mut decode_tokens = 0_u32;
    for (i, d) in group.decode_requests.iter().enumerate() {
        if d.query_len == 0 || d.query_len > verify_width || d.kv_len < d.query_len {
            return Err(format!(
                "decode request {i}: query_len {} must be 1..={verify_width} and <= kv_len {}",
                d.query_len, d.kv_len
            ));
        }
        if d.kv_len > max_model_len {
            return Err(format!(
                "decode request {i}: kv_len {} exceeds max_model_len {max_model_len}",
                d.kv_len
            ));
        }
        decode_tokens += d.query_len;
        decode_kv_lens.push(d.kv_len - d.query_len);
    }
    if group.decode_tokens != decode_tokens || group.batch_tokens != prefill_tokens + decode_tokens
    {
        return Err("group token accounting is inconsistent".into());
    }
    let decodes = decode_kv_lens.len() as u32;
    let decode_only = prefill_tokens == 0 && decodes > 0;
    // Verify runs the full window (clipped requests still run the captured
    // graph width), padded to the target-verify CUDA-graph batch.
    let rows = if decode_only {
        decode_graph_batch(decodes) * verify_width
    } else {
        prefill_tokens + decodes * verify_width
    };
    let draft = (decodes > 0).then(|| SglangDsv41LayerInput {
        rows: decode_graph_batch(decodes) * draft_tokens,
        decode_only: true,
        prefill: Vec::new(),
        decode_kv_lens: decode_kv_lens.clone(),
        decode_rows_per_request: draft_tokens,
        source_rows: None,
    });
    Ok(DsparkBatch {
        target: SglangDsv41LayerInput {
            rows,
            decode_only,
            prefill,
            decode_kv_lens,
            decode_rows_per_request: verify_width,
            source_rows: None,
        },
        // One logit row per prefill request, every verify row of a decode.
        logits_rows: group.prefill_chunk_pairs.len() as u32 + decodes * verify_width,
        draft,
    })
}

impl DeepseekV41SglangDsparkModel {
    pub fn cost_tree(&self) -> CostTree {
        let mut b = CostTreeBuilder::new();
        let mut sections = self.target.compile_target(&mut b);
        sections.push(self.draft.compile(&mut b));
        let root = CostNode::Labeled {
            label: format!(
                "{} (DeepseekV41SglangDsparkModel) [SGLang SM12x; TP4/EP4; DSpark gamma={}, verify {} rows/request; {} unique target bodies]",
                self.name,
                self.draft_tokens,
                self.draft_tokens + 1,
                self.target.unique_bodies()
            ),
            child: Box::new(CostNode::Sum(sections)),
        };
        b.finish(root)
    }

    fn eval_into(&self, input: &SpeculativeArchInput, ev: &mut Evaluator) {
        let batch = normalize_speculative_input(input, self.draft_tokens, self.max_model_len)
            .unwrap_or_else(|reason| panic!("invalid {ARCH_KIND} input: {reason}"));
        let injected_rows = self
            .target
            .eval_target(&batch.target, batch.logits_rows, ev);
        self.draft.eval(injected_rows, batch.draft.as_ref(), ev);
    }
}

impl SpeculativeUnifiedModel for DeepseekV41SglangDsparkModel {
    fn check_input(&self, batch: &SpeculativeArchInput) -> Result<(), String> {
        normalize_speculative_input(batch, self.draft_tokens, self.max_model_len).map(|_| ())
    }

    fn eval_speculative_iter(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut ev = Evaluator::new(slots);
        self.eval_into(batch, &mut ev);
        assert_eq!(
            ev.filled(),
            self.n_slots,
            "eval must fill every compiled slot"
        );
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
        let mut ev = Evaluator::with_inputs(slots, inputs);
        self.eval_into(batch, &mut ev);
        assert_eq!(
            ev.filled(),
            self.n_slots,
            "eval must fill every compiled slot"
        );
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }

    fn cost_log_manifest(&self) -> CostManifest {
        self.cost_tree().manifest()
    }

    fn kv_cache_layout(&self) -> KvCacheLayout {
        self.kv_cache_layout.clone()
    }

    fn max_model_len(&self) -> u32 {
        self.max_model_len
    }

    fn gpus_per_replica(&self) -> u16 {
        target::TP_SIZE as u16
    }

    fn num_attn_shards(&self) -> u16 {
        target::TP_SIZE as u16
    }

    /// SGLang budgets only extend tokens by `--chunked-prefill-size`; verify
    /// and draft rows run in decode iterations and reserve no prefill slots.
    fn drafting_slots_per_request(&self) -> u32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::contract::{SpeculativeArchGroupInput, SpeculativeDecodeInput};

    fn decode_batch(kv: &[u32], draft_tokens: u32) -> SpeculativeArchInput {
        let w = draft_tokens + 1;
        SpeculativeArchInput {
            draft_tokens,
            groups: vec![SpeculativeArchGroupInput {
                batch_tokens: kv.len() as u32 * w,
                prefill_tokens: 0,
                decode_tokens: kv.len() as u32 * w,
                prefill_chunk_pairs: vec![],
                decode_requests: kv
                    .iter()
                    .map(|&k| SpeculativeDecodeInput {
                        kv_len: k + w,
                        query_len: w,
                    })
                    .collect(),
                total_kv_len: kv.iter().sum(),
            }],
            tokens_per_source_rank: vec![],
        }
    }

    #[test]
    fn verify_runs_seven_rows_per_request_and_the_draft_six() {
        let batch = normalize_speculative_input(&decode_batch(&[1000, 3000, 5000], 6), 6, 1 << 20)
            .expect("valid");
        // 3 requests pad to the bs-3 graph: 21 verify rows, 18 draft rows.
        assert_eq!(batch.target.rows, 21);
        assert_eq!(batch.target.decode_rows_per_request, 7);
        assert_eq!(batch.target.decode_kv_lens, vec![1000, 3000, 5000]);
        assert_eq!(batch.logits_rows, 21);
        let draft = batch.draft.expect("decodes draft");
        assert_eq!((draft.rows, draft.decode_rows_per_request), (18, 6));
        // 9 requests pad to the bs-10 graph.
        let nine = normalize_speculative_input(&decode_batch(&[100; 9], 6), 6, 1 << 20).unwrap();
        assert_eq!((nine.target.rows, nine.draft.unwrap().rows), (70, 60));
    }

    #[test]
    fn a_prefill_only_iteration_runs_no_draft_forward() {
        let input = SpeculativeArchInput {
            draft_tokens: 6,
            groups: vec![SpeculativeArchGroupInput {
                batch_tokens: 2048,
                prefill_tokens: 2048,
                decode_tokens: 0,
                prefill_chunk_pairs: vec![(10_000, 2048)],
                decode_requests: vec![],
                total_kv_len: 10_000,
            }],
            tokens_per_source_rank: vec![],
        };
        let batch = normalize_speculative_input(&input, 6, 1 << 20).unwrap();
        assert!(batch.draft.is_none());
        assert_eq!(batch.target.rows, 2048);
        assert_eq!(batch.logits_rows, 1);
    }

    #[test]
    fn the_draft_width_must_match() {
        assert!(normalize_speculative_input(&decode_batch(&[10], 5), 6, 1 << 20).is_err());
    }
}
