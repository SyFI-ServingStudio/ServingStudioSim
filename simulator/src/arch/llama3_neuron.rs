//! Llama 3.1 8B composed from measured Trainium2 LNC2 invocations.
//! One request, no prefix reuse, KV capacity512. A complete compiled decoder
//! is scaled across homogeneous layers; this is not NxDI whole-model fusion.

use std::sync::Arc;

use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::arch::model_cfg::ModelCfg;
use crate::op::Op;
use crate::timing::kernels::{
    NeuronEmbeddingKernel, NeuronEmbeddingKernelConfig, NeuronEmbeddingKernelInput,
    NeuronLlamaDecoderPhase, RmsNormKernel, RmsNormKernelConfig, RmsNormKernelInput,
    SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Dim, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::{
    NeuronLlamaDecoderLocalWorklet, NeuronLlamaDecoderLocalWorkletConfig,
    NeuronLlamaDecoderLocalWorkletInput, NeuronLlamaDecoderLocalWorkletResolved,
};

#[derive(Clone, Debug)]
pub struct NeuronParallel {
    pub gpu_name: String,
    pub kv_capacity: u32,
}

pub struct Llama3NeuronConfigs {
    pub decoder: NeuronLlamaDecoderLocalWorkletConfig,
    pub embed: NeuronEmbeddingKernelConfig,
    pub final_norm: RmsNormKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub num_layers: u32,
}

pub struct Llama3NeuronResolved {
    pub decoder: NeuronLlamaDecoderLocalWorkletResolved,
    pub embed: NeuronEmbeddingKernelConfig,
    pub final_norm: RmsNormKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub num_layers: u32,
}

pub struct Llama3NeuronModel {
    pub name: String,
    pub num_layers: u32,
    pub kv_capacity: u32,
    pub total_kv_bytes_per_token: Dim,
    pub decoder: NeuronLlamaDecoderLocalWorklet,
    pub embed: Op<NeuronEmbeddingKernel>,
    pub final_norm: Op<RmsNormKernel>,
    pub lm_head: Op<SingleGemmKernel>,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

pub fn build_configs(model: &ModelCfg, parallel: &NeuronParallel) -> Llama3NeuronConfigs {
    let gpu = &parallel.gpu_name;
    Llama3NeuronConfigs {
        decoder: NeuronLlamaDecoderLocalWorkletConfig {
            hidden: model.hidden.clone(),
            intermediate: model.intermediate.clone(),
            q_heads: model.num_qo_heads.clone(),
            kv_heads: model.num_kv_heads.clone(),
            head_dim: model.head_dim.clone(),
            kv_capacity: Dim::param("kv_capacity", parallel.kv_capacity),
            dtype: model.dtype,
            gpu_name: gpu.clone(),
            backends: vec!["nxdi_compiler"],
        },
        embed: NeuronEmbeddingKernelConfig {
            backends: vec!["neuron_torch"],
            gpu_name: gpu.clone(),
            hidden: model.hidden.clone(),
            vocab: model.vocab.clone(),
            dtype: model.dtype,
        },
        final_norm: RmsNormKernelConfig {
            backends: vec!["neuron_torch_rms"],
            gpu_name: gpu.clone(),
            hidden: model.hidden.clone(),
            dtype: model.dtype,
        },
        lm_head: SingleGemmKernelConfig {
            backends: vec!["neuron_nki_qkv"],
            gpu_name: gpu.clone(),
            n: model.vocab.clone(),
            k: model.hidden.clone(),
            dtype: model.dtype,
        },
        num_layers: model.num_layers,
    }
}

pub fn resolve_configs(cfgs: &Llama3NeuronConfigs) -> Llama3NeuronResolved {
    Llama3NeuronResolved {
        decoder: NeuronLlamaDecoderLocalWorklet::resolve_config(&cfgs.decoder),
        embed: cfgs.embed.clone(),
        final_norm: cfgs.final_norm.clone(),
        lm_head: cfgs.lm_head.clone(),
        num_layers: cfgs.num_layers,
    }
}

pub fn build(
    name: String,
    resolved: Llama3NeuronResolved,
    bridge: &PerfApiBridge,
) -> Result<Llama3NeuronModel, BuildError> {
    let kv = &resolved.decoder.raw_cfg;
    let total_kv_bytes_per_token = 2
        * kv.kv_heads.clone()
        * kv.head_dim.clone()
        * Dim::param("kv_bytes", kv.dtype.size_bytes())
        * Dim::param("num_layers", resolved.num_layers);
    let kv_capacity = kv.kv_capacity.get();
    let embed_name = format!("{name}.embedding");
    let norm_name = format!("{name}.final_norm");
    let head_name = format!("{name}.lm_head");
    let mut model = Llama3NeuronModel {
        embed: Op::new(
            embed_name.clone(),
            Arc::new(NeuronEmbeddingKernel::build(
                embed_name,
                resolved.embed,
                bridge,
            )?),
        ),
        decoder: NeuronLlamaDecoderLocalWorklet::build(
            format!("{name}.decoder"),
            resolved.decoder,
            bridge,
        )?,
        final_norm: Op::new(
            norm_name.clone(),
            Arc::new(RmsNormKernel::build(
                norm_name,
                resolved.final_norm,
                bridge,
            )?),
        ),
        lm_head: Op::new(
            head_name.clone(),
            Arc::new(SingleGemmKernel::build(
                head_name,
                resolved.lm_head,
                bridge,
            )?),
        ),
        name,
        num_layers: resolved.num_layers,
        kv_capacity,
        total_kv_bytes_per_token,
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

/// Reject unsupported shapes before evaluating any cache. A decode KV length
/// includes the current token; the executable's physical allocation stays fixed.
fn validate_batch(
    batch: &UnifiedArchInput,
    capacity: u32,
) -> Result<(NeuronLlamaDecoderPhase, u32), String> {
    if batch.groups.len() != 1 {
        return Err("Trainium2 Llama requires exactly one local group".into());
    }
    let g = &batch.groups[0];
    if g.request_count() != 1 {
        return Err("Trainium2 Llama initially supports exactly one request per iteration".into());
    }
    if let Some(&(prefix, q)) = g.prefill_chunk_pairs.first() {
        if prefix != 0
            || !(1..=128).contains(&q)
            || g.batch_tokens != q
            || g.prefill_tokens != q
            || g.decode_tokens != 0
        {
            return Err(
                "Trainium2 prefill requires a prefix-free request with 1..128 query tokens".into(),
            );
        }
        Ok((NeuronLlamaDecoderPhase::Prefill, q))
    } else {
        let kv = g.decode_kv_lens[0];
        if kv == 0
            || kv > capacity
            || g.batch_tokens != 1
            || g.decode_tokens != 1
            || g.prefill_tokens != 0
        {
            return Err(format!(
                "Trainium2 decode requires one token and KV length1..{capacity}"
            ));
        }
        Ok((NeuronLlamaDecoderPhase::Decode, 1))
    }
}

impl Llama3NeuronModel {
    pub fn cost_tree(&self) -> CostTree {
        let mut b = CostTreeBuilder::new();
        let embed = self.embed.compile(&mut b);
        let layers = CostNode::Labeled {
            label: "layer".into(),
            child: Box::new(CostNode::Scale {
                n: self.num_layers,
                child: Box::new(self.decoder.compile(&mut b)),
            }),
        };
        let norm = self.final_norm.compile(&mut b);
        let head = self.lm_head.compile(&mut b);
        b.finish(CostNode::Labeled {
            label: format!(
                "{} [Trainium2 LNC2; {} separately compiled layers]",
                self.name, self.num_layers
            ),
            child: Box::new(CostNode::Sum(vec![embed, layers, norm, head])),
        })
    }

    fn eval_into(&self, batch: &UnifiedArchInput, ev: &mut Evaluator) {
        let (phase, q_tokens) =
            validate_batch(batch, self.kv_capacity).expect("unsupported Trainium2 iteration");
        self.embed.eval(
            &NeuronEmbeddingKernelInput {
                num_tokens: q_tokens,
            },
            ev,
        );
        self.decoder
            .eval(&NeuronLlamaDecoderLocalWorkletInput { phase, q_tokens }, ev);
        self.final_norm
            .eval(&RmsNormKernelInput { m: q_tokens }, ev);
        self.lm_head.eval(&SingleGemmKernelInput { m: 1 }, ev);
    }
}

impl IterwiseUnifiedModel for Llama3NeuronModel {
    fn check_input(&self, batch: &UnifiedArchInput) -> Result<(), String> {
        validate_batch(batch, self.kv_capacity).map(|_| ())
    }
    fn total_kv_bytes_per_token(&self) -> u64 {
        self.total_kv_bytes_per_token.get() as u64
    }
    fn gpus_per_replica(&self) -> u16 {
        1
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
        let result = CostTree::aggregate(&self.cost_flat, slots, scratch);
        debug_assert_eq!(inputs.len(), self.n_slots);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::contract::ArchGroupInput;

    fn prefill(prefix: u32, q: u32) -> UnifiedArchInput {
        UnifiedArchInput {
            groups: vec![ArchGroupInput {
                batch_tokens: q,
                prefill_tokens: q,
                prefill_chunk_pairs: vec![(prefix, q)],
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn refuses_unmeasured_batch_and_prefix_semantics() {
        assert!(validate_batch(&prefill(0, 128), 512).is_ok());
        assert!(validate_batch(&prefill(0, 129), 512).is_err());
        assert!(validate_batch(&prefill(1, 16), 512).is_err());
        let mut mixed = prefill(0, 16);
        mixed.groups[0].decode_kv_lens.push(127);
        assert!(validate_batch(&mixed, 512).is_err());
    }

    #[test]
    fn decode_cannot_exceed_physical_cache() {
        let mut batch = UnifiedArchInput {
            groups: vec![ArchGroupInput {
                batch_tokens: 1,
                decode_tokens: 1,
                decode_kv_lens: vec![512],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(validate_batch(&batch, 512).is_ok());
        batch.groups[0].decode_kv_lens[0] = 513;
        assert!(validate_batch(&batch, 512).is_err());
    }
}
