//! `llama3_neuron_ops` — the original Llama 3.1 8B on vLLM Neuron, TP4 over
//! the four LNC2 cores of one Trainium2 chip, costed per launch unit. It
//! composes the leaves of `doc/pr/llama31-8b-trn2/README.md` ("Kernels": the
//! standalone NKI per-op units for prefill, the TP4 decode-forward leaves for
//! decode, the head and the NeuronLink collectives) as the engine launches
//! them.
//!
//! ```text
//! Sum(
//!   layer: Scale{32}(Sum(attn_block     (AttnBlockNeuronTpWorklet),
//!                        mlp_block      (MlpBlockNeuronTpWorklet),
//!                        glue           (neuron_glue),
//!                        decoder_layer  (neuron_decoder_layer; `layer` only))),
//!   head            (neuron_head),
//!   head_all_gather (neuron_collective: the vocab-sharded logits),
//!   decode_fixed    (neuron_decode_fixed; `layer` only))
//! ```
//!
//! Prefill is always per op: the four prefill units, the glue, the two
//! sequence-parallel collectives of each block and the head add up to the
//! engine's step within 1.3% (26.43 against 26.09 ms at T512). Decode has two
//! granularities (`decode_granularity`). `layer` (the default) charges the
//! decode step as `decode_fixed + 32 x decoder_layer`. Both terms come from
//! one reduced-depth TP4 decode forward on the whole chip, at 8 and 32 layers.
//! That graph holds the block all-reduces, the embedding, the head and the
//! engine's sampler. The engine's decode step is one graph whose units overlap
//! and fuse, so its standalone rows do not add up. Under `layer` the
//! worklets mint no decode compute slots and charge their decode all-reduce
//! zero, and the head and its all-gather are zero in decode (all are in the
//! two leaves). `per_op` charges the standalone decode units instead
//! (attention, MLP, glue, head, the host-timed collectives): the attribution
//! view, 12-17% high on the kernel path, +24% (B1) and -23% (B16) on the torch
//! fallback at C512 (doc/pr/llama31-8b-trn2/README.md, "Check 1").
//!
//! The graphs are static, so `lower_variant` first lowers the iteration to a
//! compiled variant. A prefill must be one prefix-free prompt alone; it pads
//! to the smallest configured prefill token bucket that holds it
//! (`num_batched_tokens_buckets`; the largest is the context). A decode-only
//! batch pads to the smallest configured decode bucket (`num_seqs_buckets`).
//! Every leaf then reads the variant's row: the kernels by `(phase, bucket)`,
//! the collectives by the padded activation's bytes. The L6 pairing is the
//! stock scheduler (`deployment/unified.rs`, `ensure_stock_neuron_worker`).
//! The engine's own profile of one iteration is the gate (doc/pr/llama31-8b-trn2/README.md,
//! "Check 1").

use std::sync::Arc;

use anyhow::{ensure, Result};

use crate::arch::config::{DecodeAttention, DecodeGranularity};
use crate::arch::contract::{ArchGroupInput, IterwiseUnifiedModel, UnifiedArchInput};
use crate::arch::kv_layout::{gqa_kv_bytes_per_token, KvCacheLayout, KvCachePart};
use crate::arch::model_cfg::ModelCfg;
use crate::common::Fabric;
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::neuron_nki::{
    largest_decode_bucket, resident_capacity, STOCK_CONTEXTS,
};
use crate::timing::kernels::{
    KernelSpec, NeuronAttentionDecodeSpec, NeuronCollectiveKernel, NeuronCollectiveKernelConfig,
    NeuronCollectiveKernelInput, NeuronCollectiveOp, NeuronCollectiveSpec, NeuronDecodeFixedKernel,
    NeuronDecodeFixedSpec, NeuronDecoderLayerKernel, NeuronDecoderLayerSpec, NeuronGlueKernel,
    NeuronGlueSpec, NeuronHeadKernel, NeuronHeadSpec, NeuronNkiAttentionCteSpec,
    NeuronNkiKernelConfig, NeuronNkiKernelInput, NeuronNkiMlpSpec, NeuronNkiOutputProjectionSpec,
    NeuronNkiQkvSpec, NeuronPhase, NEURON_COLLECTIVE_BACKEND, NEURON_TORCH_BACKEND, NKI_BACKEND,
};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::{
    AttnBlockNeuronTpWorklet, AttnBlockNeuronTpWorkletConfig, AttnBlockNeuronTpWorkletInput,
    AttnBlockNeuronTpWorkletResolved, MlpBlockNeuronTpWorklet, MlpBlockNeuronTpWorkletConfig,
    MlpBlockNeuronTpWorkletInput, MlpBlockNeuronTpWorkletResolved,
};

/// The selector's parameters (`config.rs`, `Llama3NeuronOps`): the stock
/// target, the compiled inventory, the decode attention path and the decode
/// leaf granularity.
#[derive(Clone, Debug)]
pub struct NeuronOpsParallel {
    pub gpu_name: String,
    pub max_model_len: u32,
    pub prefill_buckets: Vec<u32>,
    pub decode_buckets: Vec<u32>,
    pub tp_size: u16,
    /// Which path the decode attention rows come from (see
    /// `neuron_attention_decode.rs`); the layer leaf's backend follows it.
    pub decode_attention: DecodeAttention,
    /// The decode leaf: the whole layer, or the standalone units added up.
    pub decode_granularity: DecodeGranularity,
}

impl DecodeAttention {
    /// The backend of the decode attention leaf's rows.
    pub fn backend(self) -> &'static str {
        match self {
            DecodeAttention::NkiKernel => NKI_BACKEND,
            DecodeAttention::TorchFallback => NEURON_TORCH_BACKEND,
        }
    }
}

/// Raw worklet and leaf configs; the two blocks derive their per-rank shapes
/// in `resolve_configs`. `decoder_layer` and `decode_fixed` are present
/// under `decode_granularity: layer` only; the glue and the head then carry
/// the prefill inventory alone.
pub struct Llama3NeuronOpsConfigs {
    pub attn_block: AttnBlockNeuronTpWorkletConfig,
    pub mlp_block: MlpBlockNeuronTpWorkletConfig,
    pub glue: NeuronNkiKernelConfig,
    pub decoder_layer: Option<NeuronNkiKernelConfig>,
    pub decode_fixed: Option<NeuronNkiKernelConfig>,
    pub head: NeuronNkiKernelConfig,
    pub head_all_gather: NeuronCollectiveKernelConfig,
    pub num_layers: u32,
    pub max_model_len: u32,
    pub prefill_buckets: Vec<u32>,
    pub decode_buckets: Vec<u32>,
    pub tp_size: u16,
    /// One sampled row of logits, the full vocab in the model dtype: the
    /// all-gather's buffer per sampled token.
    pub logits_bytes_per_row: u64,
    pub kv_bytes_per_token: u64,
}

/// Post-resolve aggregate; the glue, layer, fixed-term, head and head
/// all-gather leaves carry their config straight through.
pub struct Llama3NeuronOpsResolved {
    pub attn_block: AttnBlockNeuronTpWorkletResolved,
    pub mlp_block: MlpBlockNeuronTpWorkletResolved,
    pub glue: NeuronNkiKernelConfig,
    pub decoder_layer: Option<NeuronNkiKernelConfig>,
    pub decode_fixed: Option<NeuronNkiKernelConfig>,
    pub head: NeuronNkiKernelConfig,
    pub head_all_gather: NeuronCollectiveKernelConfig,
    pub num_layers: u32,
    pub max_model_len: u32,
    pub prefill_buckets: Vec<u32>,
    pub decode_buckets: Vec<u32>,
    pub tp_size: u16,
    pub logits_bytes_per_row: u64,
    pub kv_bytes_per_token: u64,
}

/// The built forward: two worklets and three leaves, plus the two decode
/// leaves of the default `layer` granularity.
pub struct Llama3NeuronOpsModel {
    pub name: String,
    pub attn_block: AttnBlockNeuronTpWorklet,
    pub mlp_block: MlpBlockNeuronTpWorklet,
    pub glue: Op<NeuronGlueKernel>,
    /// The decode layer leaf (`decode_granularity: layer`); under `per_op`
    /// the worklets and the glue leaf carry the decode compute.
    pub decoder_layer: Option<Op<NeuronDecoderLayerKernel>>,
    /// The decode step's fixed term (`layer` only): embedding and its
    /// all-reduce, RoPE, final norm, lm_head and the sampler's cross-rank
    /// top-k, measured with the layer leaf in one TP4 forward.
    pub decode_fixed: Option<Op<NeuronDecodeFixedKernel>>,
    pub head: Op<NeuronHeadKernel>,
    pub head_all_gather: Op<NeuronCollectiveKernel>,
    pub num_layers: u32,
    pub max_model_len: u32,
    pub prefill_buckets: Vec<u32>,
    pub decode_buckets: Vec<u32>,
    pub tp_size: u16,
    logits_bytes_per_row: u64,
    kv_bytes_per_token: u64,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

/// The checkpoint the leaves were measured for: the original full 32-layer
/// BF16 Llama 3.1 8B.
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

/// The checkpoint math the measured units compute, beyond the dims `ModelCfg`
/// carries: Llama 3.1 RoPE scaling, RMSNorm epsilon, unbiased SiLU.
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

/// The compiled inventory the engine can be started with: the stock context
/// ladder, ascending prefill token buckets ending at the context, and
/// ascending decode buckets within the KV pool. The leaf rules (power-of-two
/// decode buckets, the TP4/BF16 identity, the ladder cap) are checked at
/// `build`, where every leaf validates its own config.
pub fn validate_parallel(parallel: &NeuronOpsParallel) -> Result<()> {
    ensure!(
        STOCK_CONTEXTS.contains(&parallel.max_model_len),
        "llama3_neuron_ops: max_model_len must be one of {STOCK_CONTEXTS:?}"
    );
    ensure!(
        parallel.prefill_buckets.windows(2).all(|p| p[0] < p[1])
            && parallel.prefill_buckets.last() == Some(&parallel.max_model_len),
        "llama3_neuron_ops: prefill_buckets must be ascending and end at max_model_len \
         (the engine's max_num_batched_tokens)"
    );
    // The largest bucket the engine compiles for the context: the smallest
    // power of two holding every KV-resident request (26 residents at C8192
    // pad to a bucket of 32), the same cap the leaves apply.
    ensure!(
        !parallel.decode_buckets.is_empty()
            && parallel.decode_buckets.windows(2).all(|p| p[0] < p[1])
            && parallel.decode_buckets.last().copied().unwrap_or(0)
                <= largest_decode_bucket(parallel.max_model_len),
        "llama3_neuron_ops: decode_buckets must be ascending and within the largest bucket \
         the context's KV-resident capacity pads to"
    );
    Ok(())
}

/// Forward the checkpoint and the selector's parameters into the raw worklet
/// and leaf configs. `decode_granularity` decides which leaves carry decode.
pub fn build_configs(model: &ModelCfg, parallel: &NeuronOpsParallel) -> Llama3NeuronOpsConfigs {
    let nki_backends = vec![NKI_BACKEND];
    let collective_backends = vec![NEURON_COLLECTIVE_BACKEND];
    let per_op = parallel.decode_granularity == DecodeGranularity::PerOp;
    // Under the layer leaf the per-op decode compute (attention, MLP, glue)
    // has no slot, so those leaves carry the prefill inventory only.
    let per_op_decode_buckets: &[u32] = if per_op {
        &parallel.decode_buckets
    } else {
        &[]
    };
    let stock = |backends: &[&'static str], prefill: &[u32], decode: &[u32]| {
        NeuronNkiKernelConfig::stock(
            &parallel.gpu_name,
            backends,
            model.dtype,
            parallel.tp_size,
            parallel.max_model_len,
            prefill,
            decode,
        )
    };
    Llama3NeuronOpsConfigs {
        attn_block: AttnBlockNeuronTpWorkletConfig {
            hidden: model.hidden.clone(),
            num_qo_heads: model.num_qo_heads.clone(),
            num_kv_heads: model.num_kv_heads.clone(),
            dtype: model.dtype,
            tp_size: parallel.tp_size,
            max_model_len: parallel.max_model_len,
            prefill_buckets: parallel.prefill_buckets.clone(),
            decode_buckets: parallel.decode_buckets.clone(),
            gpu_name: parallel.gpu_name.clone(),
            nki_backends: nki_backends.clone(),
            attention_decode_backends: vec![parallel.decode_attention.backend()],
            collective_backends: collective_backends.clone(),
            collective_fabric: Fabric::NeuronLink,
            decode_ops_in_block: per_op,
        },
        mlp_block: MlpBlockNeuronTpWorkletConfig {
            hidden: model.hidden.clone(),
            intermediate: model.intermediate.clone(),
            dtype: model.dtype,
            tp_size: parallel.tp_size,
            max_model_len: parallel.max_model_len,
            prefill_buckets: parallel.prefill_buckets.clone(),
            decode_buckets: parallel.decode_buckets.clone(),
            gpu_name: parallel.gpu_name.clone(),
            nki_backends: nki_backends.clone(),
            collective_backends: collective_backends.clone(),
            collective_fabric: Fabric::NeuronLink,
            decode_ops_in_block: per_op,
        },
        glue: stock(
            &nki_backends,
            &parallel.prefill_buckets,
            per_op_decode_buckets,
        ),
        decoder_layer: (!per_op).then(|| {
            stock(
                &[parallel.decode_attention.backend()],
                &[],
                &parallel.decode_buckets,
            )
        }),
        decode_fixed: (!per_op).then(|| {
            stock(
                &[parallel.decode_attention.backend()],
                &[],
                &parallel.decode_buckets,
            )
        }),
        head: stock(
            &nki_backends,
            &parallel.prefill_buckets,
            per_op_decode_buckets,
        ),
        head_all_gather: NeuronCollectiveKernelConfig::tp_group(
            &parallel.gpu_name,
            &collective_backends,
            Fabric::NeuronLink,
            parallel.tp_size,
            NeuronCollectiveOp::AllGather,
        ),
        num_layers: model.num_layers,
        max_model_len: parallel.max_model_len,
        prefill_buckets: parallel.prefill_buckets.clone(),
        decode_buckets: parallel.decode_buckets.clone(),
        tp_size: parallel.tp_size,
        logits_bytes_per_row: u64::from(model.vocab.get()) * u64::from(model.dtype.size_bytes()),
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

/// Resolve the two blocks' per-rank shapes; the leaves pass through.
pub fn resolve_configs(cfgs: &Llama3NeuronOpsConfigs) -> Llama3NeuronOpsResolved {
    Llama3NeuronOpsResolved {
        attn_block: AttnBlockNeuronTpWorklet::resolve_config(&cfgs.attn_block),
        mlp_block: MlpBlockNeuronTpWorklet::resolve_config(&cfgs.mlp_block),
        glue: cfgs.glue.clone(),
        decoder_layer: cfgs.decoder_layer.clone(),
        decode_fixed: cfgs.decode_fixed.clone(),
        head: cfgs.head.clone(),
        head_all_gather: cfgs.head_all_gather.clone(),
        num_layers: cfgs.num_layers,
        max_model_len: cfgs.max_model_len,
        prefill_buckets: cfgs.prefill_buckets.clone(),
        decode_buckets: cfgs.decode_buckets.clone(),
        tp_size: cfgs.tp_size,
        logits_bytes_per_row: cfgs.logits_bytes_per_row,
        kv_bytes_per_token: cfgs.kv_bytes_per_token,
    }
}

/// `S::validate_config`, as the build failure of that kind.
fn validate_leaf<S: KernelSpec>(config: &S::Config) -> Result<(), BuildError> {
    S::validate_config(config).map_err(|e| BuildError::FitFailed {
        kind: S::KIND,
        reason: e.to_string(),
    })
}

/// Every leaf config against its kind's rules (`Kernel::build` does not call
/// `validate_config`): the inventory per phase, the power-of-two decode
/// buckets, the TP4/BF16 identity and the stock pool.
fn validate_leaves(resolved: &Llama3NeuronOpsResolved) -> Result<(), BuildError> {
    validate_leaf::<NeuronNkiQkvSpec>(&resolved.attn_block.qkv)?;
    validate_leaf::<NeuronNkiAttentionCteSpec>(&resolved.attn_block.attention_cte)?;
    validate_leaf::<NeuronNkiOutputProjectionSpec>(&resolved.attn_block.output_projection)?;
    if let Some(attention_decode) = &resolved.attn_block.attention_decode {
        validate_leaf::<NeuronAttentionDecodeSpec>(attention_decode)?;
    }
    validate_leaf::<NeuronNkiMlpSpec>(&resolved.mlp_block.mlp)?;
    validate_leaf::<NeuronGlueSpec>(&resolved.glue)?;
    if let Some(decoder_layer) = &resolved.decoder_layer {
        validate_leaf::<NeuronDecoderLayerSpec>(decoder_layer)?;
    }
    if let Some(decode_fixed) = &resolved.decode_fixed {
        validate_leaf::<NeuronDecodeFixedSpec>(decode_fixed)?;
    }
    validate_leaf::<NeuronHeadSpec>(&resolved.head)?;
    for collective in [
        &resolved.attn_block.all_gather,
        &resolved.attn_block.reduce_scatter,
        &resolved.attn_block.all_reduce,
        &resolved.mlp_block.all_gather,
        &resolved.mlp_block.reduce_scatter,
        &resolved.mlp_block.all_reduce,
        &resolved.head_all_gather,
    ] {
        validate_leaf::<NeuronCollectiveSpec>(collective)?;
    }
    Ok(())
}

/// Validate every leaf config, build the worklets and leaves, and compile the
/// cost tree once.
pub fn build(
    name: String,
    resolved: Llama3NeuronOpsResolved,
    bridge: &PerfApiBridge,
) -> Result<Llama3NeuronOpsModel, BuildError> {
    validate_leaves(&resolved)?;
    let attn_block =
        AttnBlockNeuronTpWorklet::build(format!("{name}.attn_block"), resolved.attn_block, bridge)?;
    let mlp_block =
        MlpBlockNeuronTpWorklet::build(format!("{name}.mlp_block"), resolved.mlp_block, bridge)?;
    let glue_name = format!("{name}.glue");
    let glue = Op::new(
        glue_name.clone(),
        Arc::new(NeuronGlueKernel::build(glue_name, resolved.glue, bridge)?),
    );
    let decoder_layer = match resolved.decoder_layer {
        Some(cfg) => {
            let layer_name = format!("{name}.decoder_layer");
            Some(Op::new(
                layer_name.clone(),
                Arc::new(NeuronDecoderLayerKernel::build(layer_name, cfg, bridge)?),
            ))
        }
        None => None,
    };
    let decode_fixed = match resolved.decode_fixed {
        Some(cfg) => {
            let fixed_name = format!("{name}.decode_fixed");
            Some(Op::new(
                fixed_name.clone(),
                Arc::new(NeuronDecodeFixedKernel::build(fixed_name, cfg, bridge)?),
            ))
        }
        None => None,
    };
    let head_name = format!("{name}.head");
    let head = Op::new(
        head_name.clone(),
        Arc::new(NeuronHeadKernel::build(head_name, resolved.head, bridge)?),
    );
    let gather_name = format!("{name}.head_all_gather");
    let head_all_gather = Op::new(
        gather_name.clone(),
        Arc::new(NeuronCollectiveKernel::build(
            gather_name,
            resolved.head_all_gather,
            bridge,
        )?),
    );
    let mut model = Llama3NeuronOpsModel {
        name,
        attn_block,
        mlp_block,
        glue,
        decoder_layer,
        decode_fixed,
        head,
        head_all_gather,
        num_layers: resolved.num_layers,
        max_model_len: resolved.max_model_len,
        prefill_buckets: resolved.prefill_buckets,
        decode_buckets: resolved.decode_buckets,
        tp_size: resolved.tp_size,
        logits_bytes_per_row: resolved.logits_bytes_per_row,
        kv_bytes_per_token: resolved.kv_bytes_per_token,
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

/// The compiled variant `(phase, bucket)` one iteration's logical batch runs:
/// a prefill must be one prefix-free prompt alone within the context and pads
/// to the smallest configured prefill token bucket that holds it; a decode-only
/// batch pads to the smallest configured decode bucket (`lower_decode`). Only
/// the graph identity enters the leaf input; request histories stay in the
/// ordinary request log (paged KV contents are runtime data, not a graph).
pub fn lower_variant(
    batch: &UnifiedArchInput,
    context: u32,
    prefill_buckets: &[u32],
    decode_buckets: &[u32],
) -> Result<(NeuronPhase, u32), String> {
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
            let bucket = prefill_buckets
                .iter()
                .copied()
                .find(|&bucket| bucket >= *q)
                .ok_or("vLLM Neuron prefill exceeds the largest configured token bucket")?;
            return Ok((NeuronPhase::Prefill, bucket));
        }
        return Err(
            "vLLM Neuron prefill requires one prefix-free prompt within the context and no decodes"
                .into(),
        );
    }
    Ok((
        NeuronPhase::Decode,
        lower_decode(g, context, decode_buckets)?,
    ))
}

/// The decode bucket a decode-only batch pads to: every request KV-resident
/// within the context, consistent token counts, the smallest configured
/// bucket holding the batch.
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

impl Llama3NeuronOpsModel {
    /// The per-iteration tree of the module doc: 15 leaf slots under `layer`,
    /// 14 under `per_op`.
    pub fn cost_tree(&self) -> CostTree {
        let mut b = CostTreeBuilder::new();
        let mut per_layer = vec![
            self.attn_block.compile(&mut b),
            self.mlp_block.compile(&mut b),
            self.glue.compile(&mut b),
        ];
        if let Some(decoder_layer) = &self.decoder_layer {
            per_layer.push(decoder_layer.compile(&mut b));
        }
        let layer = CostNode::Labeled {
            label: "layer".to_string(),
            child: Box::new(CostNode::Scale {
                n: self.num_layers,
                child: Box::new(CostNode::Sum(per_layer)),
            }),
        };
        let head = self.head.compile(&mut b);
        let head_all_gather = self.head_all_gather.compile(&mut b);
        let mut step = vec![layer, head, head_all_gather];
        if let Some(decode_fixed) = &self.decode_fixed {
            step.push(decode_fixed.compile(&mut b));
        }
        b.finish(CostNode::Labeled {
            label: format!(
                "{} [vLLM Neuron per launch unit; {} layers, TP{}/LNC2, context {}]",
                self.name, self.num_layers, self.tp_size, self.max_model_len
            ),
            child: Box::new(CostNode::Sum(step)),
        })
    }

    fn lower(&self, batch: &UnifiedArchInput) -> Result<NeuronNkiKernelInput, String> {
        lower_variant(
            batch,
            self.max_model_len,
            &self.prefill_buckets,
            &self.decode_buckets,
        )
        .map(|(phase, token_bucket)| NeuronNkiKernelInput {
            phase,
            token_bucket,
        })
    }

    fn eval_into(&self, batch: &UnifiedArchInput, ev: &mut Evaluator) {
        let variant = self
            .lower(batch)
            .expect("unsupported vLLM Neuron iteration");
        self.attn_block.eval(
            &AttnBlockNeuronTpWorkletInput {
                phase: variant.phase,
                token_bucket: variant.token_bucket,
            },
            ev,
        );
        self.mlp_block.eval(
            &MlpBlockNeuronTpWorkletInput {
                phase: variant.phase,
                token_bucket: variant.token_bucket,
            },
            ev,
        );
        let decode = variant.phase == NeuronPhase::Decode;
        // Under the layer leaf the glue has no decode rows: the layer carries
        // the norms and residuals.
        self.glue
            .eval_if(&variant, !decode || self.decoder_layer.is_none(), ev);
        if let Some(decoder_layer) = &self.decoder_layer {
            decoder_layer.eval_if(&variant, decode, ev);
        }
        // Under the layer leaf the decode head, sampling and logits gather are
        // in the measured TP4 forward's fixed term.
        let per_op_decode = self.decode_fixed.is_none();
        self.head.eval_if(&variant, !decode || per_op_decode, ev);
        // The head samples one row per request: the prompt's last token in
        // prefill, every request of the batch bucket in decode.
        let sampled = match variant.phase {
            NeuronPhase::Prefill => 1,
            NeuronPhase::Decode => u64::from(variant.token_bucket),
        };
        self.head_all_gather.eval_if(
            &NeuronCollectiveKernelInput {
                message_size_bytes: sampled * self.logits_bytes_per_row,
            },
            !decode || per_op_decode,
            ev,
        );
        if let Some(decode_fixed) = &self.decode_fixed {
            decode_fixed.eval_if(&variant, decode, ev);
        }
    }
}

impl IterwiseUnifiedModel for Llama3NeuronOpsModel {
    fn check_input(&self, batch: &UnifiedArchInput) -> Result<(), String> {
        self.lower(batch).map(|_| ())
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

    fn parallel() -> NeuronOpsParallel {
        NeuronOpsParallel {
            gpu_name: "AWS Trainium2 LNC2".to_string(),
            max_model_len: 512,
            prefill_buckets: vec![512],
            decode_buckets: vec![1, 16],
            tp_size: 4,
            decode_attention: DecodeAttention::NkiKernel,
            decode_granularity: DecodeGranularity::PerOp,
        }
    }

    fn layer_parallel() -> NeuronOpsParallel {
        NeuronOpsParallel {
            decode_granularity: DecodeGranularity::Layer,
            ..parallel()
        }
    }

    fn prefill(q: u32) -> UnifiedArchInput {
        UnifiedArchInput {
            groups: vec![ArchGroupInput {
                batch_tokens: q,
                prefill_tokens: q,
                prefill_chunk_pairs: vec![(0, q)],
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn decode(n: u32, kv: u32) -> UnifiedArchInput {
        UnifiedArchInput {
            groups: vec![ArchGroupInput {
                batch_tokens: n,
                decode_tokens: n,
                decode_kv_lens: vec![kv; n as usize],
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn build_configs_threads_the_inventory_and_the_leaf_backends() {
        let cfgs = build_configs(&ModelCfg::llama3_8b(), &parallel());
        assert!(cfgs.decoder_layer.is_none(), "per_op has no layer leaf");
        assert!(cfgs.attn_block.decode_ops_in_block && cfgs.mlp_block.decode_ops_in_block);
        assert_eq!(cfgs.head.decode_buckets, vec![1, 16]);
        assert_eq!(cfgs.attn_block.attention_decode_backends, vec![NKI_BACKEND]);
        let mut torch = parallel();
        torch.decode_attention = DecodeAttention::TorchFallback;
        let torch = build_configs(&ModelCfg::llama3_8b(), &torch);
        assert_eq!(
            torch.attn_block.attention_decode_backends,
            vec![NEURON_TORCH_BACKEND]
        );
        // the layer leaf: decode compute leaves the blocks, the glue and the
        // head; both layer leaves follow the attention path
        let mut layer = layer_parallel();
        layer.decode_attention = DecodeAttention::TorchFallback;
        let layer = build_configs(&ModelCfg::llama3_8b(), &layer);
        let leaf = layer.decoder_layer.as_ref().expect("layer leaf");
        assert_eq!(leaf.backends, vec![NEURON_TORCH_BACKEND]);
        assert!(leaf.prefill_buckets.is_empty());
        assert_eq!(leaf.decode_buckets, vec![1, 16]);
        assert!(!layer.attn_block.decode_ops_in_block && !layer.mlp_block.decode_ops_in_block);
        assert!(layer.glue.decode_buckets.is_empty());
        // the decode head is inside the measured forward's fixed term
        assert!(layer.head.decode_buckets.is_empty());
        let fixed = layer.decode_fixed.as_ref().expect("fixed term");
        assert_eq!(fixed.backends, vec![NEURON_TORCH_BACKEND]);
        assert_eq!(fixed.decode_buckets, vec![1, 16]);
        // the full vocab in BF16, per sampled row
        assert_eq!(cfgs.logits_bytes_per_row, 128256 * 2);
        // 32 layers x 8 KV heads x (K + V) x 128 x BF16, summed over the four
        // ranks; L6 divides by the shards of the replica.
        assert_eq!(cfgs.kv_bytes_per_token, 32 * 8 * 2 * 128 * 2);
        let resolved = resolve_configs(&cfgs);
        assert_eq!(resolved.attn_block.num_qo_heads_per_rank, 8);
        assert_eq!(resolved.mlp_block.intermediate_per_rank, 3584);
    }

    #[test]
    fn prefill_pads_to_the_smallest_token_bucket_and_decode_to_its_batch_bucket() {
        let lower = |batch: &UnifiedArchInput| lower_variant(batch, 512, &[128, 512], &[1, 16]);
        assert_eq!(lower(&prefill(100)), Ok((NeuronPhase::Prefill, 128)));
        assert_eq!(lower(&prefill(129)), Ok((NeuronPhase::Prefill, 512)));
        assert!(lower(&prefill(513)).is_err());
        assert_eq!(lower(&decode(1, 505)), Ok((NeuronPhase::Decode, 1)));
        assert_eq!(lower(&decode(9, 505)), Ok((NeuronPhase::Decode, 16)));
        assert!(lower(&decode(17, 505)).is_err());
    }

    #[test]
    fn parallel_inventory_rules() {
        validate_parallel(&parallel()).unwrap();
        let mut p = parallel();
        p.prefill_buckets = vec![128, 256];
        assert!(
            validate_parallel(&p).is_err(),
            "largest prefill bucket must be the context"
        );
        let mut p = parallel();
        p.decode_buckets = vec![16, 1];
        assert!(validate_parallel(&p).is_err());
        let mut p = parallel();
        p.max_model_len = 1000;
        assert!(validate_parallel(&p).is_err());
        // the largest bucket is the one the resident capacity pads to, not the
        // capacity itself: 26 residents at C8192 compile a bucket of 32
        let mut p = parallel();
        p.max_model_len = 8192;
        p.prefill_buckets = vec![8192];
        p.decode_buckets = vec![1, 32];
        validate_parallel(&p).unwrap();
        p.decode_buckets = vec![1, 64];
        assert!(validate_parallel(&p).is_err());
    }

    #[test]
    fn build_validates_every_leaf_and_mints_fourteen_slots() {
        let bridge = PerfApiBridge::structure_only();
        let model = build(
            "m".to_string(),
            resolve_configs(&build_configs(&ModelCfg::llama3_8b(), &parallel())),
            &bridge,
        )
        .unwrap();
        assert_eq!(model.cost_tree().n_slots(), 14);
        assert!(model.decoder_layer.is_none());
        assert!(model.attn_block.attention_decode.is_some());

        // the layer granularity swaps the attention_decode slot for the layer
        // leaf and adds the step's fixed term: 15 slots
        let layer = build(
            "m".to_string(),
            resolve_configs(&build_configs(&ModelCfg::llama3_8b(), &layer_parallel())),
            &bridge,
        )
        .unwrap();
        assert_eq!(layer.cost_tree().n_slots(), 15);
        assert!(layer.decoder_layer.is_some() && layer.decode_fixed.is_some());
        assert!(layer.attn_block.attention_decode.is_none());
        let manifest = layer.cost_tree().describe();
        assert!(manifest.contains("decoder_layer") && manifest.contains("decode_fixed"));
        assert!(!manifest.contains("attention_decode"));

        // a decode bucket the engine cannot compile is the leaf's rule, not
        // the inventory's: build reports it instead of panicking in the grid
        let mut p = parallel();
        p.decode_buckets = vec![3];
        validate_parallel(&p).unwrap();
        let err = build(
            "m".to_string(),
            resolve_configs(&build_configs(&ModelCfg::llama3_8b(), &p)),
            &bridge,
        )
        .err()
        .expect("non-power-of-two decode bucket");
        assert!(
            matches!(err, BuildError::FitFailed { kind, .. } if kind == NeuronAttentionDecodeSpec::KIND),
            "{err:?}"
        );
    }
}
