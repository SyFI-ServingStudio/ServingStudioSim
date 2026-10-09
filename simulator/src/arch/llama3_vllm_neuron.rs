//! Stock vLLM Neuron's complete TP4/LNC2 Llama3.1-8B forward.
//!
//! All layers, communication and greedy sampling are inside one measured
//! executable. Its time already spans all ranks; work estimates are per rank.
//! The configured bucket inventory is explicit, not the stock default inventory.
//!
//! The experimental `model_head_regions` composition costs the same stock
//! operations compiled as two executables split at the `LlamaModel` return:
//! the model region (embedding, layers, final norm, KV updates) then the head
//! region (vocabulary projection, logit collectives and sampling). Its rows are
//! accepted only when the split reproduces unsplit stock logits and the region
//! sum stays within 5% of the whole-forward time.

use crate::arch::config::VllmNeuronComposition;
use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::arch::model_cfg::ModelCfg;
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::engine::KernelSpec;
use crate::timing::kernels::neuron_llama_forward::NeuronLlamaForwardSpec;
use crate::timing::kernels::{
    NeuronLlamaForwardKernel, NeuronLlamaForwardKernelConfig, NeuronLlamaForwardKernelInput,
    NeuronLlamaForwardPhase, NeuronLlamaRegion, NeuronLlamaRegionKernel,
    NeuronLlamaRegionKernelConfig, NeuronLlamaRegionKernelInput,
};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Dim, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};
use anyhow::{Result, ensure};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct VllmNeuronParallel {
    pub gpu_name: String,
    pub max_model_len: u32,
    pub decode_buckets: Vec<u32>,
    pub tp_size: u16,
    pub composition: VllmNeuronComposition,
}

pub struct Llama3VllmNeuronConfigs {
    pub forward: NeuronLlamaForwardKernelConfig,
    pub composition: VllmNeuronComposition,
}

pub struct Llama3VllmNeuronResolved {
    pub forward: NeuronLlamaForwardKernelConfig,
    pub composition: VllmNeuronComposition,
}

/// The measured executables one iteration runs, in execution order.
pub enum VllmNeuronLeaves {
    WholeForward(Op<NeuronLlamaForwardKernel>),
    ModelHeadRegions {
        model: Op<NeuronLlamaRegionKernel>,
        head: Op<NeuronLlamaRegionKernel>,
    },
}

pub struct Llama3VllmNeuronModel {
    pub name: String,
    pub max_model_len: u32,
    pub decode_buckets: Vec<u32>,
    pub leaves: VllmNeuronLeaves,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

/// Fixed executable identity: reduced-depth and same-shape other dtypes are
/// different programs and cannot borrow this whole-forward timing.
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

pub fn build_configs(model: &ModelCfg, parallel: &VllmNeuronParallel) -> Llama3VllmNeuronConfigs {
    validate_model(model).expect("unsupported vLLM Neuron model");
    Llama3VllmNeuronConfigs {
        forward: NeuronLlamaForwardKernelConfig {
            backends: vec!["vllm_neuron"],
            gpu_name: parallel.gpu_name.clone(),
            max_model_len: Dim::param("max_model_len", parallel.max_model_len),
            kv_blocks: Dim::param("kv_blocks", 6782),
            block_size: Dim::param("block_size", 32),
            tp_size: Dim::param("tp_size", u32::from(parallel.tp_size)),
            dtype: model.dtype,
            decode_buckets: parallel.decode_buckets.clone(),
        },
        composition: parallel.composition,
    }
}

pub fn resolve_configs(cfgs: &Llama3VllmNeuronConfigs) -> Llama3VllmNeuronResolved {
    Llama3VllmNeuronResolved {
        forward: cfgs.forward.clone(),
        composition: cfgs.composition,
    }
}

pub fn validate_config(cfg: &NeuronLlamaForwardKernelConfig) -> Result<()> {
    NeuronLlamaForwardSpec::validate_config(cfg)?;
    ensure!(
        cfg.decode_buckets.first() == Some(&1),
        "vLLM Neuron's configured decode inventory must start with bucket 1"
    );
    Ok(())
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

fn region_op(
    name: &str,
    forward: &NeuronLlamaForwardKernelConfig,
    region: NeuronLlamaRegion,
    bridge: &PerfApiBridge,
) -> Result<Op<NeuronLlamaRegionKernel>, BuildError> {
    let slot = format!("{name}.{}", region.as_str());
    let kernel = NeuronLlamaRegionKernel::build(slot.clone(), region_config(forward, region), bridge)?;
    Ok(Op::new(slot, Arc::new(kernel)))
}

pub fn build(
    name: String,
    resolved: Llama3VllmNeuronResolved,
    bridge: &PerfApiBridge,
) -> Result<Llama3VllmNeuronModel, BuildError> {
    validate_config(&resolved.forward).map_err(|e| BuildError::FitFailed {
        kind: "neuron_llama_forward",
        reason: e.to_string(),
    })?;
    let leaves = match resolved.composition {
        VllmNeuronComposition::WholeForward => {
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
        VllmNeuronComposition::ModelHeadRegions => VllmNeuronLeaves::ModelHeadRegions {
            model: region_op(&name, &resolved.forward, NeuronLlamaRegion::Model, bridge)?,
            head: region_op(&name, &resolved.forward, NeuronLlamaRegion::Head, bridge)?,
        },
    };
    let mut model = Llama3VllmNeuronModel {
        name,
        max_model_len: resolved.forward.max_model_len.get(),
        decode_buckets: resolved.forward.decode_buckets.clone(),
        leaves,
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

/// Lower logical request geometry into the exact configured executable. Actual
/// KV histories remain in the ordinary request log; only the compiled bucket
/// goes into SlotInput. Paged KV contents are runtime data, not another graph.
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
    if !g.prefill_chunk_pairs.is_empty() || g.prefill_tokens != 0 || g.decode_kv_lens.is_empty() {
        return Err("vLLM Neuron requires one prefill or a nonempty decode-only batch".into());
    }
    let count = g.decode_kv_lens.len();
    let capacity = (6782 * 32 / context) as usize;
    if count > capacity
        || count > *buckets.last().unwrap_or(&0) as usize
        || g.decode_tokens as usize != count
        || g.batch_tokens as usize != count
        || g.decode_kv_lens.iter().any(|&kv| kv == 0 || kv > context)
    {
        return Err("vLLM Neuron decode exceeds the configured inventory/cache capacity or has inconsistent token counts".into());
    }
    let bucket = *buckets.iter().find(|&&b| b as usize >= count).unwrap();
    Ok(NeuronLlamaForwardKernelInput {
        phase: NeuronLlamaForwardPhase::Decode,
        token_bucket: bucket,
    })
}

impl Llama3VllmNeuronModel {
    pub fn cost_tree(&self) -> CostTree {
        let mut b = CostTreeBuilder::new();
        let (detail, child) = match &self.leaves {
            VllmNeuronLeaves::WholeForward(forward) => ("full 32 layers", forward.compile(&mut b)),
            VllmNeuronLeaves::ModelHeadRegions { model, head } => (
                "model then head regions",
                CostNode::Sum(vec![model.compile(&mut b), head.compile(&mut b)]),
            ),
        };
        b.finish(CostNode::Labeled {
            label: format!("{} [stock vLLM Neuron; {detail}, TP4/LNC2]", self.name),
            child: Box::new(child),
        })
    }
    fn eval_into(&self, batch: &UnifiedArchInput, ev: &mut Evaluator) {
        let input = lower_input(batch, self.max_model_len, &self.decode_buckets)
            .expect("unsupported vLLM Neuron iteration");
        match &self.leaves {
            VllmNeuronLeaves::WholeForward(forward) => forward.eval(&input, ev),
            VllmNeuronLeaves::ModelHeadRegions { model, head } => {
                let region_input = NeuronLlamaRegionKernelInput {
                    phase: input.phase,
                    token_bucket: input.token_bucket,
                };
                model.eval(&region_input, ev);
                head.eval(&region_input, ev);
            }
        }
    }
}

impl IterwiseUnifiedModel for Llama3VllmNeuronModel {
    fn check_input(&self, batch: &UnifiedArchInput) -> Result<(), String> {
        lower_input(batch, self.max_model_len, &self.decode_buckets).map(|_| ())
    }
    fn total_kv_bytes_per_token(&self) -> u64 {
        2 * 8 * 128 * 2 * 32
    }
    fn gpus_per_replica(&self) -> u16 {
        4
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
    use crate::arch::contract::ArchGroupInput;
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
    #[test]
    fn configured_ceiling_and_pool_boundary() {
        for (n, bucket) in [(1, 1), (2, 16), (9, 16), (16, 16)] {
            assert_eq!(
                lower_input(&decode(n, 512), 512, &[1, 16])
                    .unwrap()
                    .token_bucket,
                bucket
            );
        }
        for (n, kv) in [(0, 512), (17, 512), (1, 0), (1, 513)] {
            assert!(lower_input(&decode(n, kv), 512, &[1, 16]).is_err());
        }
        assert!(lower_input(&decode(423, 512), 512, &[1, 512]).is_ok());
        assert!(lower_input(&decode(424, 512), 512, &[1, 512]).is_err());
        let mut bad = decode(1, 512);
        bad.groups[0].batch_tokens = 2;
        assert!(lower_input(&bad, 512, &[1, 16]).is_err());
        bad = decode(1, 512);
        bad.tokens_per_source_rank.push(1);
        assert!(lower_input(&bad, 512, &[1, 16]).is_err());
    }
    #[test]
    fn prefill_is_single_prefix_free_compiled_bucket() {
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
    fn fixed_model_and_compiler_inventory() {
        let mut model = ModelCfg::llama3_8b();
        let parallel = VllmNeuronParallel {
            gpu_name: "AWS Trainium2 LNC2".into(),
            max_model_len: 512,
            decode_buckets: vec![1, 16],
            tp_size: 4,
            composition: VllmNeuronComposition::WholeForward,
        };
        let cfg = resolve_configs(&build_configs(&model, &parallel)).forward;
        assert!(validate_config(&cfg).is_ok());
        assert_eq!(cfg.kv_blocks, 6782);
        assert_eq!(cfg.block_size, 32);
        assert_eq!(cfg.tp_size, 4);
        assert_eq!(cfg.gpu_name, parallel.gpu_name);
        assert_eq!(cfg.backends, ["vllm_neuron"]);
        let mut bad = cfg.clone();
        bad.decode_buckets = vec![16];
        assert!(validate_config(&bad).is_err());
        bad = cfg;
        bad.tp_size = Dim::param("tp_size", 1);
        assert!(validate_config(&bad).is_err());
        model.num_layers = 1;
        assert!(validate_model(&model).is_err());
        model.num_layers = 32;
        model.kv_dtype = DType::Fp16;
        assert!(validate_model(&model).is_err());
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
        };
        let resolved = resolve_configs(&build_configs(&ModelCfg::llama3_8b(), &parallel));
        assert_eq!(resolved.composition, VllmNeuronComposition::ModelHeadRegions);
        for region in [NeuronLlamaRegion::Model, NeuronLlamaRegion::Head] {
            let cfg = region_config(&resolved.forward, region);
            NeuronLlamaRegionSpec::validate_config(&cfg).unwrap();
            assert_eq!(cfg.region, region);
            assert_eq!(cfg.decode_buckets, resolved.forward.decode_buckets);
            assert_eq!(cfg.backends, ["vllm_neuron_fx_regions"]);
        }
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
        ] {
            let IterArchSel::Llama3VllmNeuron { composition, .. } = parse(extra) else {
                panic!("wrong selector")
            };
            assert_eq!(composition, expected);
        }
    }
}
