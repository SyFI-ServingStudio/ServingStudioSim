//! `llama3_vllm_neuron` — the original Llama 3.1 8B on stock vLLM Neuron,
//! TP4 over the four LNC2 cores of one Trainium2 chip.
//!
//! The engine runs every layer, its collectives and greedy sampling inside one
//! compiled executable, and the arch costs it as that one measured leaf
//! (`neuron_llama_forward`, slot `<name>.forward`): its time already spans all
//! ranks. The engine compiles one prefill graph of the whole context and one
//! decode graph per configured bucket (`decode_buckets`, an explicit inventory,
//! not vLLM Neuron's default one). This arch lowers each iteration's logical
//! batch onto those graphs: a prefill must be one prefix-free prompt alone, and
//! a decode-only batch pads to the smallest bucket that holds it. The stock
//! scheduler that only forms such batches is the L6 pairing
//! (`deployment/unified.rs`, `ensure_stock_neuron_worker`).

use std::sync::Arc;

use anyhow::{ensure, Result};

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
    NeuronLlamaForwardPhase, NeuronLlamaForwardSpec,
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
}

/// The one measured leaf's kernel config, plus the KV one token adds over the
/// four ranks (32 layers x 8 KV heads x (K + V) x 128 x BF16).
pub struct Llama3VllmNeuronConfigs {
    pub forward: NeuronLlamaForwardKernelConfig,
    pub kv_bytes_per_token: u64,
}

/// An atomic op passes its kernel config straight through.
pub type Llama3VllmNeuronResolved = Llama3VllmNeuronConfigs;

pub struct Llama3VllmNeuronModel {
    pub name: String,
    pub max_model_len: u32,
    pub decode_buckets: Vec<u32>,
    pub tp_size: u16,
    pub forward: Op<NeuronLlamaForwardKernel>,
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
    Llama3VllmNeuronConfigs {
        forward: NeuronLlamaForwardKernelConfig {
            backends: vec!["vllm_neuron"],
            gpu_name: parallel.gpu_name.clone(),
            max_model_len: Dim::param("max_model_len", parallel.max_model_len),
            kv_blocks: Dim::param("kv_blocks", STOCK_KV_BLOCKS),
            block_size: Dim::param("block_size", STOCK_BLOCK_SIZE),
            tp_size: Dim::param("tp_size", u32::from(parallel.tp_size)),
            dtype: model.dtype,
            decode_buckets: parallel.decode_buckets.clone(),
        },
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
        kv_bytes_per_token: cfgs.kv_bytes_per_token,
    }
}

pub fn build(
    name: String,
    resolved: Llama3VllmNeuronResolved,
    bridge: &PerfApiBridge,
) -> Result<Llama3VllmNeuronModel, BuildError> {
    let forward = resolved.forward;
    NeuronLlamaForwardSpec::validate_config(&forward).map_err(|e| BuildError::FitFailed {
        kind: NeuronLlamaForwardSpec::KIND,
        reason: e.to_string(),
    })?;
    let tp_size = forward.tp_size.get() as u16;
    let slot = format!("{name}.forward");
    let mut model = Llama3VllmNeuronModel {
        max_model_len: forward.max_model_len.get(),
        decode_buckets: forward.decode_buckets.clone(),
        tp_size,
        forward: Op::new(
            slot.clone(),
            Arc::new(NeuronLlamaForwardKernel::build(slot, forward, bridge)?),
        ),
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
fn lower_input(
    batch: &UnifiedArchInput,
    context: u32,
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
            return Ok(NeuronLlamaForwardKernelInput {
                phase: NeuronLlamaForwardPhase::Prefill,
                token_bucket: context,
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

impl Llama3VllmNeuronModel {
    pub fn cost_tree(&self) -> CostTree {
        let mut b = CostTreeBuilder::new();
        let forward = self.forward.compile(&mut b);
        b.finish(CostNode::Labeled {
            label: format!(
                "{} [stock vLLM Neuron; full 32 layers, TP4/LNC2]",
                self.name
            ),
            child: Box::new(forward),
        })
    }

    fn eval_into(&self, batch: &UnifiedArchInput, ev: &mut Evaluator) {
        let input = lower_input(batch, self.max_model_len, &self.decode_buckets)
            .expect("unsupported vLLM Neuron iteration");
        self.forward.eval(&input, ev);
    }
}

impl IterwiseUnifiedModel for Llama3VllmNeuronModel {
    fn check_input(&self, batch: &UnifiedArchInput) -> Result<(), String> {
        lower_input(batch, self.max_model_len, &self.decode_buckets).map(|_| ())
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
        lower_input(&decode(n, context), context, buckets).map(|input| input.token_bucket)
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
            assert!(lower_input(&decode(n, kv), 512, &[1, 16]).is_err());
        }
        let mut bad = decode(1, 512);
        bad.groups[0].batch_tokens = 2;
        assert!(lower_input(&bad, 512, &[1, 16]).is_err());
        bad = decode(1, 512);
        bad.tokens_per_source_rank.push(1);
        assert!(lower_input(&bad, 512, &[1, 16]).is_err());
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
        let input = lower_input(&b, 512, &[1, 16]).unwrap();
        assert_eq!(input.phase, NeuronLlamaForwardPhase::Prefill);
        assert_eq!(input.token_bucket, 512);
        b.groups[0].prefill_chunk_pairs[0].0 = 1;
        assert!(lower_input(&b, 512, &[1, 16]).is_err());
        b.groups[0].prefill_chunk_pairs[0] = (0, 513);
        assert!(lower_input(&b, 512, &[1, 16]).is_err());
        b.groups[0].prefill_chunk_pairs[0] = (0, 504);
        b.groups[0].decode_kv_lens.push(10);
        assert!(lower_input(&b, 512, &[1, 16]).is_err());
        b.groups[0].decode_kv_lens.clear();
        b.groups[0].prefill_chunk_pairs.push((0, 504));
        assert!(lower_input(&b, 512, &[1, 16]).is_err());
    }

    #[test]
    fn configs_carry_the_stock_identity_and_reject_other_models() {
        let mut model = ModelCfg::llama3_8b();
        let parallel = VllmNeuronParallel {
            gpu_name: "AWS Trainium2 LNC2".into(),
            max_model_len: 512,
            decode_buckets: vec![1, 16],
            tp_size: 4,
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
}
