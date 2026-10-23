//! `llama3_vllm_neuron` — the original Llama 3.1 8B on stock vLLM Neuron,
//! TP4 over the four LNC2 cores of one Trainium2 chip.
//!
//! The engine runs every layer, its collectives and greedy sampling inside one
//! compiled executable, and by default (`composition: whole_forward`) the arch
//! costs it as that one measured leaf (`neuron_llama_forward`, slot
//! `<name>.forward`): its time already spans all ranks. The engine compiles one
//! prefill graph of the whole context and one decode graph per configured
//! bucket (`decode_buckets`, an explicit inventory, not vLLM Neuron's default
//! one). This arch lowers each iteration's logical batch onto those graphs: a
//! prefill must be one prefix-free prompt alone, and a decode-only batch pads
//! to the smallest bucket that holds it. The stock scheduler that only forms
//! such batches is the L6 pairing (`deployment/unified.rs`,
//! `ensure_stock_neuron_worker`).
//!
//! Two experimental compositions cost the same executables in finer slots:
//! `model_head_regions` sums the stock operations compiled as two executables
//! split at the `LlamaModel` return (`neuron_llama_region`: model, then head),
//! and `layer_segments` cuts the unchanged executable's native timeline at the
//! sublayer-closing collectives (`neuron_llama_segment`: embedding, a 32-layer
//! fold of attention and MLP blocks, then head).
//!
//! With `prefill_segment_size = S` the engine compiles one S-token prefill
//! graph (`max_num_batched_tokens = S`) instead of the whole-context one. A
//! prompt then runs as one call per S-token chunk, each attending the chunk's
//! cached prefix; both phases cost through `neuron_llama_segmented_forward`,
//! still in the one `forward` slot.

use std::sync::Arc;

use anyhow::{ensure, Result};

use crate::arch::config::VllmNeuronComposition;
use crate::arch::contract::{ArchGroupInput, IterwiseUnifiedModel, UnifiedArchInput};
use crate::arch::kv_layout::{gqa_kv_bytes_per_token, KvCacheLayout, KvCachePart};
use crate::arch::model_cfg::ModelCfg;
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::engine::KernelSpec;
use crate::timing::kernels::neuron_llama_forward::{
    resident_capacity, STOCK_BLOCK_SIZE, STOCK_KV_BLOCKS,
};
use crate::timing::kernels::{
    NeuronLlamaForwardKernel, NeuronLlamaForwardKernelConfig, NeuronLlamaForwardKernelInput,
    NeuronLlamaForwardPhase, NeuronLlamaForwardSpec, NeuronLlamaRegion, NeuronLlamaRegionKernel,
    NeuronLlamaRegionKernelConfig, NeuronLlamaRegionKernelInput, NeuronLlamaSegment,
    NeuronLlamaSegmentKernel, NeuronLlamaSegmentKernelConfig, NeuronLlamaSegmentKernelInput,
    NeuronLlamaSegmentedForwardKernel, NeuronLlamaSegmentedForwardKernelConfig,
    NeuronLlamaSegmentedForwardKernelInput, NeuronLlamaSegmentedForwardSpec,
};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Dim, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};

#[derive(Clone, Debug)]
pub struct VllmNeuronParallel {
    pub gpu_name: String,
    pub max_model_len: u32,
    pub decode_buckets: Vec<u32>,
    pub tp_size: u16,
    pub composition: VllmNeuronComposition,
    /// Compiled prefill segment S; `None` keeps the whole-context prefill graph.
    pub prefill_segment_size: Option<u32>,
    /// The compiled prefill inventory (`num_batched_tokens_buckets`). A prompt
    /// pads to the smallest bucket that holds it. Empty means `[max_model_len]`.
    pub prefill_buckets: Vec<u32>,
}

/// The whole-forward kernel config (the runtime identity every composition
/// shares), the selected composition, the segmented-prefill config when the
/// prefill is segmented (it then replaces `forward`), and the KV one token adds
/// over the four ranks (32 layers x 8 KV heads x (K + V) x 128 x BF16).
pub struct Llama3VllmNeuronConfigs {
    pub forward: NeuronLlamaForwardKernelConfig,
    pub composition: VllmNeuronComposition,
    pub segmented: Option<NeuronLlamaSegmentedForwardKernelConfig>,
    pub kv_bytes_per_token: u64,
}

/// Atomic ops pass their kernel configs straight through.
pub type Llama3VllmNeuronResolved = Llama3VllmNeuronConfigs;

/// The measured executables one iteration runs, in execution order.
pub enum VllmNeuronLeaves {
    WholeForward(Op<NeuronLlamaForwardKernel>),
    /// One call per prefill segment or decode step, in the same `forward` slot.
    SegmentedForward(Op<NeuronLlamaSegmentedForwardKernel>),
    ModelHeadRegions {
        model: Op<NeuronLlamaRegionKernel>,
        head: Op<NeuronLlamaRegionKernel>,
    },
    LayerSegments {
        embedding: Op<NeuronLlamaSegmentKernel>,
        attention: Op<NeuronLlamaSegmentKernel>,
        mlp: Op<NeuronLlamaSegmentKernel>,
        head: Op<NeuronLlamaSegmentKernel>,
    },
}

/// The validated model's layer count; `validate_model` pins it.
const NUM_LAYERS: u32 = 32;

pub struct Llama3VllmNeuronModel {
    pub name: String,
    pub max_model_len: u32,
    pub decode_buckets: Vec<u32>,
    /// The prefill inventory, ascending; the last bucket is the context.
    pub prefill_buckets: Vec<u32>,
    pub tp_size: u16,
    pub leaves: VllmNeuronLeaves,
    /// Segment geometry when prefill is segmented: (S, largest measured prior).
    segmented: Option<(u32, u32)>,
    kv_bytes_per_token: u64,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

/// The measured executable is this exact model: a reduced-depth or same-shape
/// model in another dtype is a different program and cannot borrow its time.
pub fn validate_model(model: &ModelCfg) -> Result<()> {
    ensure!(
        (
            model.hidden.get(),
            model.intermediate.get(),
            model.num_qo_heads.get(),
            model.num_kv_heads.get(),
            model.head_dim.get(),
            model.vocab.get(),
            model.num_layers
        ) == (4096, 14336, 32, 8, 128, 128256, 32)
            && model.dtype == DType::Bf16
            && model.kv_dtype == DType::Bf16,
        "vLLM Neuron requires the original full 32-layer BF16 Llama3.1-8B model"
    );
    Ok(())
}

/// The checkpoint math the measured executable computes, beyond the dims
/// `ModelCfg` carries: Llama 3.1 RoPE scaling, RMSNorm epsilon, unbiased SiLU.
pub fn validate_checkpoint_math(raw: &serde_json::Value) -> Result<()> {
    let rope = &raw["rope_scaling"];
    ensure!(
        raw["model_type"].as_str() == Some("llama")
            && raw["hidden_act"].as_str() == Some("silu")
            && raw["rms_norm_eps"].as_f64() == Some(1e-5)
            && raw["rope_theta"].as_f64() == Some(500000.0)
            && rope["rope_type"].as_str() == Some("llama3")
            && rope["factor"].as_f64() == Some(8.0)
            && rope["low_freq_factor"].as_f64() == Some(1.0)
            && rope["high_freq_factor"].as_f64() == Some(4.0)
            && rope["original_max_position_embeddings"].as_u64() == Some(8192)
            && !raw["attention_bias"].as_bool().unwrap_or(false)
            && !raw["mlp_bias"].as_bool().unwrap_or(false),
        "vLLM Neuron requires the measured Llama3.1 RoPE, RMSNorm and unbiased SiLU math"
    );
    Ok(())
}

pub fn build_configs(model: &ModelCfg, parallel: &VllmNeuronParallel) -> Llama3VllmNeuronConfigs {
    let forward = NeuronLlamaForwardKernelConfig {
        backends: vec!["vllm_neuron"],
        gpu_name: parallel.gpu_name.clone(),
        max_model_len: Dim::param("max_model_len", parallel.max_model_len),
        kv_blocks: Dim::param("kv_blocks", STOCK_KV_BLOCKS),
        block_size: Dim::param("block_size", STOCK_BLOCK_SIZE),
        tp_size: Dim::param("tp_size", u32::from(parallel.tp_size)),
        dtype: model.dtype,
        decode_buckets: parallel.decode_buckets.clone(),
        // `[C]` is the default inventory; store it as empty so both spellings
        // build the same kernel config.
        prefill_buckets: if parallel.prefill_buckets == [parallel.max_model_len] {
            Vec::new()
        } else {
            parallel.prefill_buckets.clone()
        },
    };
    let segmented = parallel
        .prefill_segment_size
        .map(|segment| segmented_config(&forward, segment));
    Llama3VllmNeuronConfigs {
        forward,
        composition: parallel.composition,
        segmented,
        kv_bytes_per_token: gqa_kv_bytes_per_token(
            u64::from(model.num_layers),
            u64::from(model.num_kv_heads.get()),
            u64::from(model.head_dim.get()),
            u64::from(model.head_dim.get()),
            u64::from(model.kv_dtype.size_bytes()),
            u64::from(parallel.tp_size),
        ),
    }
}

pub fn resolve_configs(cfgs: &Llama3VllmNeuronConfigs) -> Llama3VllmNeuronResolved {
    Llama3VllmNeuronConfigs {
        forward: cfgs.forward.clone(),
        composition: cfgs.composition,
        segmented: cfgs.segmented.clone(),
        kv_bytes_per_token: cfgs.kv_bytes_per_token,
    }
}

/// The segmented forward shares the whole forward's runtime identity and
/// decode inventory; only its prefill graph and context list differ.
pub fn segmented_config(
    forward: &NeuronLlamaForwardKernelConfig,
    segment: u32,
) -> NeuronLlamaSegmentedForwardKernelConfig {
    NeuronLlamaSegmentedForwardKernelConfig {
        backends: vec!["vllm_neuron_segmented"],
        gpu_name: forward.gpu_name.clone(),
        max_model_len: forward.max_model_len.clone(),
        kv_blocks: forward.kv_blocks.clone(),
        block_size: forward.block_size.clone(),
        tp_size: forward.tp_size.clone(),
        dtype: forward.dtype,
        segment_size: Dim::param("prefill_segment_size", segment),
        decode_buckets: forward.decode_buckets.clone(),
    }
}

/// A region shares the whole forward's compiled inventory and runtime identity.
pub fn region_config(
    forward: &NeuronLlamaForwardKernelConfig,
    region: NeuronLlamaRegion,
) -> NeuronLlamaRegionKernelConfig {
    NeuronLlamaRegionKernelConfig {
        backends: vec!["vllm_neuron_fx_regions"],
        gpu_name: forward.gpu_name.clone(),
        region,
        max_model_len: forward.max_model_len.clone(),
        kv_blocks: forward.kv_blocks.clone(),
        block_size: forward.block_size.clone(),
        tp_size: forward.tp_size.clone(),
        dtype: forward.dtype,
        decode_buckets: forward.decode_buckets.clone(),
    }
}

/// A segment shares the whole forward's compiled inventory and runtime identity.
pub fn segment_config(
    forward: &NeuronLlamaForwardKernelConfig,
    segment: NeuronLlamaSegment,
) -> NeuronLlamaSegmentKernelConfig {
    NeuronLlamaSegmentKernelConfig {
        backends: vec!["vllm_neuron_collective_segments"],
        gpu_name: forward.gpu_name.clone(),
        segment,
        max_model_len: forward.max_model_len.clone(),
        kv_blocks: forward.kv_blocks.clone(),
        block_size: forward.block_size.clone(),
        tp_size: forward.tp_size.clone(),
        dtype: forward.dtype,
        decode_buckets: forward.decode_buckets.clone(),
    }
}

/// Validate the executable the selected composition actually costs: the
/// segmented leaf when prefill is segmented (only under `whole_forward`), else
/// the whole forward, whose rules the region and segment kinds share.
pub fn validate_selected(
    forward: &NeuronLlamaForwardKernelConfig,
    composition: VllmNeuronComposition,
    segmented: Option<&NeuronLlamaSegmentedForwardKernelConfig>,
) -> Result<()> {
    // Region, segment and segmented-prefill rows exist for the whole-context
    // prefill graph only; smaller prefill graphs are whole-forward rows.
    ensure!(
        *forward.prefill_inventory() == [forward.max_model_len.get()]
            || (composition == VllmNeuronComposition::WholeForward && segmented.is_none()),
        "a prefill inventory other than [max_model_len] requires composition whole_forward \
         and no prefill_segment_size"
    );
    match segmented {
        None => NeuronLlamaForwardSpec::validate_config(forward),
        Some(cfg) => {
            ensure!(
                composition == VllmNeuronComposition::WholeForward,
                "segmented prefill (prefill_segment_size) requires composition whole_forward"
            );
            NeuronLlamaSegmentedForwardSpec::validate_config(cfg)
        }
    }
}

fn region_op(
    name: &str,
    forward: &NeuronLlamaForwardKernelConfig,
    region: NeuronLlamaRegion,
    bridge: &PerfApiBridge,
) -> Result<Op<NeuronLlamaRegionKernel>, BuildError> {
    let slot = format!("{name}.{}", region.as_str());
    let kernel =
        NeuronLlamaRegionKernel::build(slot.clone(), region_config(forward, region), bridge)?;
    Ok(Op::new(slot, Arc::new(kernel)))
}

fn segment_op(
    name: &str,
    forward: &NeuronLlamaForwardKernelConfig,
    segment: NeuronLlamaSegment,
    bridge: &PerfApiBridge,
) -> Result<Op<NeuronLlamaSegmentKernel>, BuildError> {
    let slot = format!("{name}.{}", segment.as_str());
    let kernel =
        NeuronLlamaSegmentKernel::build(slot.clone(), segment_config(forward, segment), bridge)?;
    Ok(Op::new(slot, Arc::new(kernel)))
}

pub fn build(
    name: String,
    resolved: Llama3VllmNeuronResolved,
    bridge: &PerfApiBridge,
) -> Result<Llama3VllmNeuronModel, BuildError> {
    validate_selected(
        &resolved.forward,
        resolved.composition,
        resolved.segmented.as_ref(),
    )
    .map_err(|e| BuildError::FitFailed {
        kind: if resolved.segmented.is_some() {
            NeuronLlamaSegmentedForwardSpec::KIND
        } else {
            NeuronLlamaForwardSpec::KIND
        },
        reason: e.to_string(),
    })?;
    let segmented = resolved
        .segmented
        .as_ref()
        .map(|cfg| (cfg.segment_size.get(), cfg.largest_prior()));
    let leaves = match (resolved.composition, resolved.segmented) {
        (_, Some(cfg)) => {
            let slot = format!("{name}.forward");
            VllmNeuronLeaves::SegmentedForward(Op::new(
                slot.clone(),
                Arc::new(NeuronLlamaSegmentedForwardKernel::build(slot, cfg, bridge)?),
            ))
        }
        (VllmNeuronComposition::WholeForward, None) => {
            let slot = format!("{name}.forward");
            VllmNeuronLeaves::WholeForward(Op::new(
                slot.clone(),
                Arc::new(NeuronLlamaForwardKernel::build(
                    slot,
                    resolved.forward.clone(),
                    bridge,
                )?),
            ))
        }
        (VllmNeuronComposition::ModelHeadRegions, None) => VllmNeuronLeaves::ModelHeadRegions {
            model: region_op(&name, &resolved.forward, NeuronLlamaRegion::Model, bridge)?,
            head: region_op(&name, &resolved.forward, NeuronLlamaRegion::Head, bridge)?,
        },
        (VllmNeuronComposition::LayerSegments, None) => {
            let op = |segment| segment_op(&name, &resolved.forward, segment, bridge);
            VllmNeuronLeaves::LayerSegments {
                embedding: op(NeuronLlamaSegment::Embedding)?,
                attention: op(NeuronLlamaSegment::AttentionBlock)?,
                mlp: op(NeuronLlamaSegment::MlpBlock)?,
                head: op(NeuronLlamaSegment::Head)?,
            }
        }
    };
    let mut model = Llama3VllmNeuronModel {
        max_model_len: resolved.forward.max_model_len.get(),
        decode_buckets: resolved.forward.decode_buckets.clone(),
        prefill_buckets: resolved.forward.prefill_inventory().into_owned(),
        tp_size: resolved.forward.tp_size.get() as u16,
        leaves,
        segmented,
        kv_bytes_per_token: resolved.kv_bytes_per_token,
        name,
        cost_flat: Vec::new(),
        n_slots: 0,
    };
    let tree = model.cost_tree();
    model.cost_flat = tree.flatten();
    model.n_slots = tree.n_slots();
    tracing::info!(
        "[build] cost tree ({} leaf slots):\n{}",
        tree.n_slots(),
        tree.describe()
    );
    Ok(model)
}

/// Lower one iteration's logical batch onto the compiled executable it runs.
/// Only the graph identity enters the leaf input; request histories stay in
/// the ordinary request log (paged KV contents are runtime data, not a graph).
/// A prompt of `q` tokens runs the smallest compiled prefill graph that holds
/// it, as the stock scheduler pads it (`NeuronScheduler._calculate_padded_count`,
/// `vllm_neuron/vllm/core/scheduler.py`).
fn lower_input(
    batch: &UnifiedArchInput,
    context: u32,
    prefill_buckets: &[u32],
    buckets: &[u32],
) -> Result<NeuronLlamaForwardKernelInput, String> {
    if batch.groups.len() != 1 || !batch.tokens_per_source_rank.is_empty() {
        return Err("vLLM Neuron requires exactly one group and no source-rank routing".into());
    }
    let g = &batch.groups[0];
    if let [(prefix, q)] = g.prefill_chunk_pairs.as_slice() {
        if *prefix == 0
            && (1..=context).contains(q)
            && g.decode_kv_lens.is_empty()
            && g.decode_tokens == 0
            && g.prefill_tokens == *q
            && g.batch_tokens == *q
        {
            let token_bucket = prefill_buckets
                .iter()
                .copied()
                .find(|&bucket| bucket >= *q)
                .ok_or_else(|| {
                    "vLLM Neuron prompt exceeds the largest prefill bucket".to_string()
                })?;
            return Ok(NeuronLlamaForwardKernelInput {
                phase: NeuronLlamaForwardPhase::Prefill,
                token_bucket,
            });
        }
        return Err(
            "vLLM Neuron prefill requires one prefix-free prompt within the context and no decodes"
                .into(),
        );
    }
    Ok(NeuronLlamaForwardKernelInput {
        phase: NeuronLlamaForwardPhase::Decode,
        token_bucket: lower_decode(g, context, buckets)?,
    })
}

/// The compiled decode bucket a decode-only group pads to.
fn lower_decode(g: &ArchGroupInput, context: u32, buckets: &[u32]) -> Result<u32, String> {
    if !g.prefill_chunk_pairs.is_empty() || g.prefill_tokens != 0 || g.decode_kv_lens.is_empty() {
        return Err("vLLM Neuron requires one prefill or a nonempty decode-only batch".into());
    }
    let count = g.decode_kv_lens.len();
    if count > resident_capacity(context) as usize
        || g.decode_tokens as usize != count
        || g.batch_tokens as usize != count
        || g.decode_kv_lens.iter().any(|&kv| kv == 0 || kv > context)
    {
        return Err(
            "vLLM Neuron decode exceeds the KV pool or the context, or has inconsistent token counts"
                .into(),
        );
    }
    buckets
        .iter()
        .copied()
        .find(|&bucket| bucket as usize >= count)
        .ok_or_else(|| "vLLM Neuron decode exceeds the largest configured bucket".into())
}

/// Segmented prefill: one request's chunk of at most S tokens, whose cached
/// prefix is block-aligned and within the measured prior range, runs the
/// S-token graph at that prior. Decode lowers as in the whole-context engine.
fn lower_segmented_input(
    batch: &UnifiedArchInput,
    context: u32,
    buckets: &[u32],
    (segment, largest_prior): (u32, u32),
) -> Result<NeuronLlamaSegmentedForwardKernelInput, String> {
    if batch.groups.len() != 1 || !batch.tokens_per_source_rank.is_empty() {
        return Err("vLLM Neuron requires exactly one group and no source-rank routing".into());
    }
    let g = &batch.groups[0];
    if let [(prefix, q)] = g.prefill_chunk_pairs.as_slice() {
        if (1..=segment).contains(q)
            && prefix % STOCK_BLOCK_SIZE == 0
            && *prefix <= largest_prior
            && prefix + q <= context
            && g.decode_kv_lens.is_empty()
            && g.decode_tokens == 0
            && g.prefill_tokens == *q
            && g.batch_tokens == *q
        {
            return Ok(NeuronLlamaSegmentedForwardKernelInput {
                phase: NeuronLlamaForwardPhase::Prefill,
                token_bucket: segment,
                prior_tokens: *prefix,
            });
        }
        return Err(format!(
            "vLLM Neuron segmented prefill requires one chunk of at most {segment} tokens, a \
             {STOCK_BLOCK_SIZE}-aligned prefix within the measured {largest_prior}-token prior \
             range, and no decodes"
        ));
    }
    Ok(NeuronLlamaSegmentedForwardKernelInput {
        phase: NeuronLlamaForwardPhase::Decode,
        token_bucket: lower_decode(g, context, buckets)?,
        prior_tokens: 0,
    })
}

impl Llama3VllmNeuronModel {
    pub fn cost_tree(&self) -> CostTree {
        let mut b = CostTreeBuilder::new();
        let (detail, child) = match &self.leaves {
            VllmNeuronLeaves::WholeForward(forward) => ("full 32 layers", forward.compile(&mut b)),
            VllmNeuronLeaves::SegmentedForward(forward) => {
                ("full 32 layers, segmented prefill", forward.compile(&mut b))
            }
            VllmNeuronLeaves::ModelHeadRegions { model, head } => (
                "model then head regions",
                CostNode::Sum(vec![model.compile(&mut b), head.compile(&mut b)]),
            ),
            VllmNeuronLeaves::LayerSegments {
                embedding,
                attention,
                mlp,
                head,
            } => {
                let embedding = embedding.compile(&mut b);
                let layer = CostNode::Labeled {
                    label: "layer".to_string(),
                    child: Box::new(CostNode::Scale {
                        n: NUM_LAYERS,
                        child: Box::new(CostNode::Sum(vec![
                            attention.compile(&mut b),
                            mlp.compile(&mut b),
                        ])),
                    }),
                };
                let head = head.compile(&mut b);
                (
                    "collective-delimited embedding, 32 layers and head",
                    CostNode::Sum(vec![embedding, layer, head]),
                )
            }
        };
        b.finish(CostNode::Labeled {
            label: format!("{} [stock vLLM Neuron; {detail}, TP4/LNC2]", self.name),
            child: Box::new(child),
        })
    }

    fn eval_into(&self, batch: &UnifiedArchInput, ev: &mut Evaluator) {
        if let (VllmNeuronLeaves::SegmentedForward(forward), Some(geometry)) =
            (&self.leaves, self.segmented)
        {
            let input =
                lower_segmented_input(batch, self.max_model_len, &self.decode_buckets, geometry)
                    .expect("unsupported vLLM Neuron segmented iteration");
            forward.eval(&input, ev);
            return;
        }
        let input = lower_input(
            batch,
            self.max_model_len,
            &self.prefill_buckets,
            &self.decode_buckets,
        )
        .expect("unsupported vLLM Neuron iteration");
        match &self.leaves {
            VllmNeuronLeaves::WholeForward(forward) => forward.eval(&input, ev),
            VllmNeuronLeaves::SegmentedForward(_) => {
                unreachable!("segmented leaf without segment geometry")
            }
            VllmNeuronLeaves::ModelHeadRegions { model, head } => {
                let region_input = NeuronLlamaRegionKernelInput {
                    phase: input.phase,
                    token_bucket: input.token_bucket,
                };
                model.eval(&region_input, ev);
                head.eval(&region_input, ev);
            }
            VllmNeuronLeaves::LayerSegments {
                embedding,
                attention,
                mlp,
                head,
            } => {
                let segment_input = NeuronLlamaSegmentKernelInput {
                    phase: input.phase,
                    token_bucket: input.token_bucket,
                };
                for op in [embedding, attention, mlp, head] {
                    op.eval(&segment_input, ev);
                }
            }
        }
    }
}

impl IterwiseUnifiedModel for Llama3VllmNeuronModel {
    fn check_input(&self, batch: &UnifiedArchInput) -> Result<(), String> {
        match self.segmented {
            Some(geometry) => {
                lower_segmented_input(batch, self.max_model_len, &self.decode_buckets, geometry)
                    .map(|_| ())
            }
            None => lower_input(
                batch,
                self.max_model_len,
                &self.prefill_buckets,
                &self.decode_buckets,
            )
            .map(|_| ()),
        }
    }

    fn kv_cache_layout(&self) -> KvCacheLayout {
        KvCacheLayout::new().with_part(KvCachePart::per_token("gqa", self.kv_bytes_per_token))
    }

    fn gpus_per_replica(&self) -> u16 {
        self.tp_size
    }

    fn logs_decode_kv_lens(&self) -> bool {
        true
    }

    fn cost_log_manifest(&self) -> CostManifest {
        self.cost_tree().manifest()
    }

    fn eval_iter(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut ev = Evaluator::new(slots);
        self.eval_into(batch, &mut ev);
        debug_assert_eq!(ev.filled(), self.n_slots);
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }

    fn eval_iter_with_inputs(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut ev = Evaluator::with_inputs(slots, inputs);
        self.eval_into(batch, &mut ev);
        debug_assert_eq!(ev.filled(), self.n_slots);
        debug_assert_eq!(inputs.len(), self.n_slots);
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(n: usize, kv: u32) -> UnifiedArchInput {
        UnifiedArchInput {
            groups: vec![ArchGroupInput {
                batch_tokens: n as u32,
                decode_tokens: n as u32,
                decode_kv_lens: vec![kv; n],
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn bucket(n: usize, context: u32, buckets: &[u32]) -> Result<u32, String> {
        lower_input(&decode(n, context), context, &[context], buckets)
            .map(|input| input.token_bucket)
    }

    #[test]
    fn decode_pads_to_the_smallest_configured_bucket_within_the_pool() {
        for (n, expected) in [(1, 1), (2, 16), (9, 16), (16, 16)] {
            assert_eq!(bucket(n, 512, &[1, 16]), Ok(expected));
        }
        // C4096 compiles only buckets 16 and 32: one request pads to 16.
        for (n, expected) in [(1, 16), (16, 16), (17, 32), (32, 32)] {
            assert_eq!(bucket(n, 4096, &[16, 32]), Ok(expected));
        }
        assert!(bucket(33, 4096, &[16, 32]).is_err());
        // 217024 slots hold 423 full C512 requests, not 424.
        assert_eq!(bucket(423, 512, &[1, 512]), Ok(512));
        assert!(bucket(424, 512, &[1, 512]).is_err());
        for (n, kv) in [(0, 512), (17, 512), (1, 0), (1, 513)] {
            assert!(lower_input(&decode(n, kv), 512, &[512], &[1, 16]).is_err());
        }
        let mut bad = decode(1, 512);
        bad.groups[0].batch_tokens = 2;
        assert!(lower_input(&bad, 512, &[512], &[1, 16]).is_err());
        bad = decode(1, 512);
        bad.tokens_per_source_rank.push(1);
        assert!(lower_input(&bad, 512, &[512], &[1, 16]).is_err());
    }

    #[test]
    fn prefill_is_one_prefix_free_prompt_on_the_context_graph() {
        let mut b = UnifiedArchInput {
            groups: vec![ArchGroupInput {
                prefill_chunk_pairs: vec![(0, 504)],
                prefill_tokens: 504,
                batch_tokens: 504,
                ..Default::default()
            }],
            ..Default::default()
        };
        let input = lower_input(&b, 512, &[512], &[1, 16]).unwrap();
        assert_eq!(input.phase, NeuronLlamaForwardPhase::Prefill);
        assert_eq!(input.token_bucket, 512);
        b.groups[0].prefill_chunk_pairs[0].0 = 1;
        assert!(lower_input(&b, 512, &[512], &[1, 16]).is_err());
        b.groups[0].prefill_chunk_pairs[0] = (0, 513);
        assert!(lower_input(&b, 512, &[512], &[1, 16]).is_err());
        b.groups[0].prefill_chunk_pairs[0] = (0, 504);
        b.groups[0].decode_kv_lens.push(10);
        assert!(lower_input(&b, 512, &[512], &[1, 16]).is_err());
        b.groups[0].decode_kv_lens.clear();
        b.groups[0].prefill_chunk_pairs.push((0, 504));
        assert!(lower_input(&b, 512, &[512], &[1, 16]).is_err());
    }

    #[test]
    fn prefill_pads_to_the_smallest_compiled_prefill_graph() {
        let prompt = |q: u32| UnifiedArchInput {
            groups: vec![ArchGroupInput {
                prefill_chunk_pairs: vec![(0, q)],
                prefill_tokens: q,
                batch_tokens: q,
                ..Default::default()
            }],
            ..Default::default()
        };
        let inventory = [128, 256, 384, 512];
        for (q, bucket) in [(1, 128), (128, 128), (129, 256), (300, 384), (512, 512)] {
            let input = lower_input(&prompt(q), 512, &inventory, &[1, 16]).unwrap();
            assert_eq!(
                (input.phase, input.token_bucket),
                (NeuronLlamaForwardPhase::Prefill, bucket)
            );
        }
        assert!(lower_input(&prompt(513), 512, &inventory, &[1, 16]).is_err());
        // Smaller prefill graphs are whole-forward rows only.
        let mut parallel = VllmNeuronParallel {
            gpu_name: "AWS Trainium2 LNC2".into(),
            max_model_len: 512,
            decode_buckets: vec![1, 16],
            tp_size: 4,
            composition: VllmNeuronComposition::WholeForward,
            prefill_segment_size: None,
            prefill_buckets: inventory.to_vec(),
        };
        let cfgs = build_configs(&ModelCfg::llama3_8b(), &parallel);
        validate_selected(&cfgs.forward, cfgs.composition, cfgs.segmented.as_ref()).unwrap();
        for composition in [
            VllmNeuronComposition::ModelHeadRegions,
            VllmNeuronComposition::LayerSegments,
        ] {
            parallel.composition = composition;
            let cfgs = build_configs(&ModelCfg::llama3_8b(), &parallel);
            assert!(
                validate_selected(&cfgs.forward, cfgs.composition, cfgs.segmented.as_ref())
                    .is_err()
            );
        }
        parallel.composition = VllmNeuronComposition::WholeForward;
        parallel.max_model_len = 16384;
        parallel.prefill_buckets = vec![2048, 16384];
        parallel.prefill_segment_size = Some(2048);
        let cfgs = build_configs(&ModelCfg::llama3_8b(), &parallel);
        assert!(
            validate_selected(&cfgs.forward, cfgs.composition, cfgs.segmented.as_ref()).is_err()
        );
        parallel.max_model_len = 512;
        parallel.prefill_buckets = inventory.to_vec();
        parallel.prefill_segment_size = None;
        // `[C]` spells the default inventory and builds the default config.
        parallel.prefill_buckets = vec![512];
        let spelled = build_configs(&ModelCfg::llama3_8b(), &parallel).forward;
        parallel.prefill_buckets = vec![];
        assert_eq!(
            spelled,
            build_configs(&ModelCfg::llama3_8b(), &parallel).forward
        );
        validate_selected(&spelled, parallel.composition, None).unwrap();
    }

    #[test]
    fn configs_carry_the_stock_identity_and_reject_other_models() {
        let mut model = ModelCfg::llama3_8b();
        let parallel = VllmNeuronParallel {
            gpu_name: "AWS Trainium2 LNC2".into(),
            max_model_len: 512,
            decode_buckets: vec![1, 16],
            tp_size: 4,
            composition: VllmNeuronComposition::WholeForward,
            prefill_segment_size: None,
            prefill_buckets: vec![],
        };
        let resolved = resolve_configs(&build_configs(&model, &parallel));
        assert_eq!(resolved.kv_bytes_per_token, 131_072);
        let cfg = resolved.forward;
        NeuronLlamaForwardSpec::validate_config(&cfg).unwrap();
        assert_eq!(cfg.kv_blocks, 6782);
        assert_eq!(cfg.block_size, 32);
        assert_eq!(cfg.gpu_name, parallel.gpu_name);
        assert_eq!(cfg.backends, ["vllm_neuron"]);
        validate_model(&model).unwrap();
        model.num_layers = 1;
        assert!(validate_model(&model).is_err());
        model.num_layers = 32;
        model.kv_dtype = DType::Fp16;
        assert!(validate_model(&model).is_err());
    }

    #[test]
    fn checkpoint_math_rejects_the_same_shape_with_other_rope_or_norm() {
        let raw: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/model/config/llama3_8b.json"
            ))
            .unwrap(),
        )
        .unwrap();
        validate_checkpoint_math(&raw).unwrap();
        for (key, value) in [
            ("rope_theta", serde_json::json!(10000.0)),
            ("rms_norm_eps", serde_json::json!(1e-6)),
            ("mlp_bias", serde_json::json!(true)),
        ] {
            let mut changed = raw.clone();
            changed[key] = value;
            assert!(validate_checkpoint_math(&changed).is_err(), "{key}");
        }
    }

    #[test]
    fn regions_share_the_whole_forward_inventory() {
        use crate::timing::kernels::NeuronLlamaRegionSpec;
        let parallel = VllmNeuronParallel {
            gpu_name: "AWS Trainium2 LNC2".into(),
            max_model_len: 512,
            decode_buckets: vec![1, 16],
            tp_size: 4,
            composition: VllmNeuronComposition::ModelHeadRegions,
            prefill_segment_size: None,
            prefill_buckets: vec![],
        };
        let resolved = resolve_configs(&build_configs(&ModelCfg::llama3_8b(), &parallel));
        assert_eq!(
            resolved.composition,
            VllmNeuronComposition::ModelHeadRegions
        );
        for region in [NeuronLlamaRegion::Model, NeuronLlamaRegion::Head] {
            let cfg = region_config(&resolved.forward, region);
            NeuronLlamaRegionSpec::validate_config(&cfg).unwrap();
            assert_eq!(cfg.region, region);
            assert_eq!(cfg.decode_buckets, resolved.forward.decode_buckets);
            assert_eq!(cfg.backends, ["vllm_neuron_fx_regions"]);
        }
    }
    #[test]
    fn segments_share_the_whole_forward_inventory() {
        use crate::timing::kernels::NeuronLlamaSegmentSpec;
        let parallel = VllmNeuronParallel {
            gpu_name: "AWS Trainium2 LNC2".into(),
            max_model_len: 512,
            decode_buckets: vec![1, 16],
            tp_size: 4,
            composition: VllmNeuronComposition::LayerSegments,
            prefill_segment_size: None,
            prefill_buckets: vec![],
        };
        let resolved = resolve_configs(&build_configs(&ModelCfg::llama3_8b(), &parallel));
        for segment in [
            NeuronLlamaSegment::Embedding,
            NeuronLlamaSegment::AttentionBlock,
            NeuronLlamaSegment::MlpBlock,
            NeuronLlamaSegment::Head,
        ] {
            let cfg = segment_config(&resolved.forward, segment);
            NeuronLlamaSegmentSpec::validate_config(&cfg).unwrap();
            assert_eq!(cfg.segment, segment);
            assert_eq!(cfg.backends, ["vllm_neuron_collective_segments"]);
        }
        assert_eq!(ModelCfg::llama3_8b().num_layers, NUM_LAYERS);
    }
    fn segmented_prefill(prefix: u32, q: u32) -> UnifiedArchInput {
        UnifiedArchInput {
            groups: vec![ArchGroupInput {
                prefill_chunk_pairs: vec![(prefix, q)],
                prefill_tokens: q,
                batch_tokens: q,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn segmented_prefill_lowers_each_chunk_to_the_segment_graph_at_its_prior() {
        // C16384/S2048: the measured priors are 0..=14336.
        let geometry = (2048, 14336);
        let buckets = [1, 2, 4, 8, 16];
        for (prefix, q) in [
            (0, 2048),
            (2048, 2048),
            (14336, 2040),
            (0, 1),
            (4096 + 32, 100),
        ] {
            let input =
                lower_segmented_input(&segmented_prefill(prefix, q), 16384, &buckets, geometry)
                    .unwrap();
            assert_eq!(input.phase, NeuronLlamaForwardPhase::Prefill);
            assert_eq!((input.token_bucket, input.prior_tokens), (2048, prefix));
        }
        for (prefix, q) in [
            (0, 2049),       // chunk above the segment
            (16, 2048),      // prefix off the 32-token page grid
            (14336 + 32, 8), // prior beyond the measured range
            (0, 0),
        ] {
            assert!(lower_segmented_input(
                &segmented_prefill(prefix, q),
                16384,
                &buckets,
                geometry
            )
            .is_err());
        }
        let mut mixed = segmented_prefill(0, 2048);
        mixed.groups[0].decode_kv_lens.push(10);
        assert!(lower_segmented_input(&mixed, 16384, &buckets, geometry).is_err());
        let mut two = segmented_prefill(0, 1024);
        two.groups[0].prefill_chunk_pairs.push((0, 1024));
        two.groups[0].prefill_tokens = 2048;
        two.groups[0].batch_tokens = 2048;
        assert!(lower_segmented_input(&two, 16384, &buckets, geometry).is_err());
        // Decode lowers as in the whole-context engine; 13 residents fit C16384.
        for (n, bucket) in [(1, 1), (3, 4), (13, 16)] {
            let input =
                lower_segmented_input(&decode(n, 16384), 16384, &buckets, geometry).unwrap();
            assert_eq!(input.phase, NeuronLlamaForwardPhase::Decode);
            assert_eq!((input.token_bucket, input.prior_tokens), (bucket, 0));
        }
        assert!(lower_segmented_input(&decode(14, 16384), 16384, &buckets, geometry).is_err());
        assert!(lower_segmented_input(&decode(1, 16385), 16384, &buckets, geometry).is_err());
    }

    #[test]
    fn segmented_prefill_requires_whole_forward_and_a_segmented_context() {
        use crate::timing::kernels::NeuronLlamaSegmentedForwardSpec;
        let mut parallel = VllmNeuronParallel {
            gpu_name: "AWS Trainium2 LNC2".into(),
            max_model_len: 16384,
            decode_buckets: vec![1, 2, 4, 8, 16],
            tp_size: 4,
            composition: VllmNeuronComposition::WholeForward,
            prefill_segment_size: Some(2048),
            prefill_buckets: vec![],
        };
        let cfgs = build_configs(&ModelCfg::llama3_8b(), &parallel);
        let segmented = cfgs.segmented.as_ref().unwrap();
        assert_eq!(segmented.backends, ["vllm_neuron_segmented"]);
        assert_eq!(segmented.segment_size, 2048);
        assert_eq!(segmented.decode_buckets, parallel.decode_buckets);
        NeuronLlamaSegmentedForwardSpec::validate_config(segmented).unwrap();
        // C16384 is outside the unsegmented ladder; only the segmented leaf is checked.
        assert!(NeuronLlamaForwardSpec::validate_config(&cfgs.forward).is_err());
        validate_selected(&cfgs.forward, cfgs.composition, cfgs.segmented.as_ref()).unwrap();
        for composition in [
            VllmNeuronComposition::ModelHeadRegions,
            VllmNeuronComposition::LayerSegments,
        ] {
            parallel.composition = composition;
            let cfgs = build_configs(&ModelCfg::llama3_8b(), &parallel);
            assert!(
                validate_selected(&cfgs.forward, cfgs.composition, cfgs.segmented.as_ref())
                    .is_err()
            );
        }
        parallel.composition = VllmNeuronComposition::WholeForward;
        parallel.prefill_segment_size = None;
        let cfgs = build_configs(&ModelCfg::llama3_8b(), &parallel);
        assert!(cfgs.segmented.is_none());
        assert!(
            validate_selected(&cfgs.forward, cfgs.composition, cfgs.segmented.as_ref()).is_err(),
            "unsegmented C16384 has no whole-context profile"
        );
    }

    #[test]
    fn selector_defaults_to_the_whole_forward() {
        use crate::arch::config::IterArchSel;
        let parse = |extra: &str| -> IterArchSel {
            serde_json::from_str(&format!(
                r#"{{"type":"llama3_vllm_neuron","model_config":"m.json","fp8":false{extra}}}"#
            ))
            .unwrap()
        };
        for (extra, expected) in [
            ("", VllmNeuronComposition::WholeForward),
            (
                r#","composition":"model_head_regions""#,
                VllmNeuronComposition::ModelHeadRegions,
            ),
            (
                r#","composition":"layer_segments""#,
                VllmNeuronComposition::LayerSegments,
            ),
        ] {
            let IterArchSel::Llama3VllmNeuron {
                composition,
                prefill_segment_size,
                ..
            } = parse(extra)
            else {
                panic!("wrong selector")
            };
            assert_eq!(composition, expected);
            assert_eq!(prefill_segment_size, None);
        }
        let IterArchSel::Llama3VllmNeuron {
            prefill_segment_size,
            ..
        } = parse(r#","prefill_segment_size":2048"#)
        else {
            panic!("wrong selector")
        };
        assert_eq!(prefill_segment_size, Some(2048));
    }
}
