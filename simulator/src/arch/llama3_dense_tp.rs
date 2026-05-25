//! `llama3_dense_tp` — L4 model_arch for Llama3-8B dense under Megatron tensor
//! parallelism (`tp_size` ranks).
//!
//! Wires the two `TP` worklets (attn-block / mlp-block) per layer, plus an
//! embedding placeholder + final-norm + lm_head, into an iter-wise unified model.
//! Mirrors `llama3_dense`'s build shape (`build_configs` / `resolve_configs` /
//! `build`) and `IterwiseUnifiedModel`, but each layer is two sync sections
//! (attn-block all-reduce, then mlp-block all-reduce) instead of three local
//! worklets.
//!
//! Deviations from L4 design.md for this v1 TP vertical (see plan):
//!   - only TP is sharded (head-split lives inside the attn-block worklet); HP
//!     (`num_hp_groups`) stays 1, EP/MoE deferred;
//!   - embed / final_norm / lm_head are kept **replicated (full shapes)** — no
//!     vocab-parallel split yet;
//!   - one worklet instance per type, reused across `num_layers` via the
//!     `Scale{num_layers}` fold (not per-layer `build`);
//!   - `tp_size == 1` degenerates to full shapes + no collective, so this arch
//!     at tp=1 sums the same leaves as `llama3_dense` (a validation convenience;
//!     deployment still picks `llama3_dense` at tp=1).

use std::sync::Arc;

use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::arch::model_cfg::{ModelCfg, ParallelCfg};
use crate::common::Fabric;
use crate::op::Op;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, RmsNormKernel,
    RmsNormKernelConfig, RmsNormKernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge,
};
use crate::worklet::{
    AttnBlockTpWorklet, AttnBlockTpWorkletConfig, AttnBlockTpWorkletInput,
    AttnBlockTpWorkletResolved, MlpBlockTpWorklet, MlpBlockTpWorkletConfig, MlpBlockTpWorkletInput,
    MlpBlockTpWorkletResolved,
};

const NORM_BACKENDS: &[&str] = &["flashinfer"];
const GEMM_BACKENDS: &[&str] = &["torch"];
const ACT_BACKENDS: &[&str] = &["triton"];
// See llama3_dense: FlashInfer impls registered under fa2/fa3, not "flashinfer".
const ATTN_BACKENDS: &[&str] = &["fa2", "fa3"];
const ALLREDUCE_BACKENDS: &[&str] = &["nccl"];
// v1 TP fabric: single-node NVLink. Multi-node fabrics deferred.
const TP_FABRIC: Fabric = Fabric::Nvlink;

/// Raw worklet/op configs (TP degree baked into the two TP worklets).
pub struct Llama3DenseTpConfigs {
    pub attn_block: AttnBlockTpWorkletConfig,
    pub mlp_block: MlpBlockTpWorkletConfig,
    pub embed: ElementwiseKernelConfig,
    pub final_norm: RmsNormKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub num_layers: u32,
    pub tp_size: u16,
}

/// Post-resolve aggregate; atomic ops (embed / final_norm / lm_head) carry their
/// kernel config straight through (replicated, no partition).
pub struct Llama3DenseTpResolved {
    pub attn_block: AttnBlockTpWorkletResolved,
    pub mlp_block: MlpBlockTpWorkletResolved,
    pub embed: ElementwiseKernelConfig,
    pub final_norm: RmsNormKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub num_layers: u32,
    pub tp_size: u16,
}

pub struct Llama3DenseTpModel {
    pub name: String,
    pub num_layers: u32,
    pub tp_size: u16,
    pub kv_bytes_per_token: u64,
    pub attn_block: AttnBlockTpWorklet,
    pub mlp_block: MlpBlockTpWorklet,
    pub embed: Op<ElementwiseKernel>,
    pub final_norm: Op<RmsNormKernel>,
    pub lm_head: Op<SingleGemmKernel>,
    /// CostTree structure compiled once at build (flattened) + its slot count,
    /// so per-iter `eval_iter` only evals leaves + aggregates.
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

pub fn build_configs(model: &ModelCfg, parallel: &ParallelCfg) -> Llama3DenseTpConfigs {
    let gpu = &parallel.gpu_name;
    let dtype_bytes = model.dtype.size_bytes();
    Llama3DenseTpConfigs {
        attn_block: AttnBlockTpWorkletConfig {
            hidden: model.hidden,
            num_qo_heads: model.num_qo_heads,
            num_kv_heads: model.num_kv_heads,
            head_dim: model.head_dim,
            dtype: model.dtype,
            kv_dtype: model.kv_dtype,
            tp_size: parallel.tp_size,
            allreduce_fabric: TP_FABRIC,
            gpu_name: gpu.clone(),
            norm_backends: NORM_BACKENDS.to_vec(),
            gemm_backends: GEMM_BACKENDS.to_vec(),
            attn_backends: ATTN_BACKENDS.to_vec(),
            allreduce_backends: ALLREDUCE_BACKENDS.to_vec(),
        },
        mlp_block: MlpBlockTpWorkletConfig {
            hidden: model.hidden,
            intermediate: model.intermediate,
            dtype: model.dtype,
            tp_size: parallel.tp_size,
            allreduce_fabric: TP_FABRIC,
            gpu_name: gpu.clone(),
            norm_backends: NORM_BACKENDS.to_vec(),
            gemm_backends: GEMM_BACKENDS.to_vec(),
            act_backends: ACT_BACKENDS.to_vec(),
            allreduce_backends: ALLREDUCE_BACKENDS.to_vec(),
        },
        // Embedding gather placeholder: read one hidden-wide row, write one out
        // (replicated; hidden NOT sharded).
        embed: ElementwiseKernelConfig {
            backends: ACT_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            input_bytes_per_token: model.hidden * dtype_bytes,
            output_bytes_per_token: model.hidden * dtype_bytes,
        },
        final_norm: RmsNormKernelConfig {
            backends: NORM_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            hidden: model.hidden,
            dtype: model.dtype,
        },
        // Replicated full lm_head for v1 (vocab-parallel split deferred).
        lm_head: SingleGemmKernelConfig {
            backends: GEMM_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            n: model.vocab,
            k: model.hidden,
            dtype: model.dtype,
        },
        num_layers: model.num_layers,
        tp_size: parallel.tp_size,
    }
}

/// Per-GPU KV-cache footprint one token occupies across the whole model. Under
/// TP the KV heads are head-sharded across ranks, so each GPU sizes its KvPool
/// from the **per-rank** KV-head count: `2` (K and V) × `num_kv_heads/tp` ×
/// `head_dim` × `kv_dtype` bytes × `num_layers`, read off the resolved attn
/// config's already-divided per-rank head count.
fn kv_bytes_per_token(resolved: &Llama3DenseTpResolved) -> u64 {
    let attn = &resolved.attn_block.attn; // per-rank FlashInferAttentionConfig
    2 * attn.num_kv_heads as u64
        * attn.head_dim as u64
        * attn.kv_dtype.size_bytes() as u64
        * resolved.num_layers as u64
}

pub fn resolve_configs(cfgs: &Llama3DenseTpConfigs) -> Llama3DenseTpResolved {
    Llama3DenseTpResolved {
        attn_block: AttnBlockTpWorklet::resolve_config(&cfgs.attn_block),
        mlp_block: MlpBlockTpWorklet::resolve_config(&cfgs.mlp_block),
        embed: cfgs.embed.clone(),
        final_norm: cfgs.final_norm.clone(),
        lm_head: cfgs.lm_head.clone(),
        num_layers: cfgs.num_layers,
        tp_size: cfgs.tp_size,
    }
}

pub fn build(
    model_name: String,
    resolved: Llama3DenseTpResolved,
    bridge: &PerfApiBridge,
) -> Result<Llama3DenseTpModel, BuildError> {
    let num_layers = resolved.num_layers;
    let tp_size = resolved.tp_size;
    let kv_bytes_per_token = kv_bytes_per_token(&resolved);

    let embed_name = format!("{model_name}.embedding");
    let final_norm_name = format!("{model_name}.final_norm");
    let lm_head_name = format!("{model_name}.lm_head");

    let embed = Op::new(
        embed_name.clone(),
        Arc::new(ElementwiseKernel::build(embed_name, resolved.embed, bridge)?),
    );

    let attn_block = AttnBlockTpWorklet::build(
        format!("{model_name}.attn_block"),
        resolved.attn_block,
        bridge,
    )?;

    let mlp_block = MlpBlockTpWorklet::build(
        format!("{model_name}.mlp_block"),
        resolved.mlp_block,
        bridge,
    )?;

    let final_norm = Op::new(
        final_norm_name.clone(),
        Arc::new(RmsNormKernel::build(final_norm_name, resolved.final_norm, bridge)?),
    );

    let lm_head = Op::new(
        lm_head_name.clone(),
        Arc::new(SingleGemmKernel::build(lm_head_name, resolved.lm_head, bridge)?),
    );

    let mut model = Llama3DenseTpModel {
        attn_block,
        mlp_block,
        embed,
        final_norm,
        lm_head,
        name: model_name,
        num_layers,
        tp_size,
        kv_bytes_per_token,
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

impl Llama3DenseTpModel {
    /// Compile the per-iteration cost *structure* once:
    /// `Sum( embed, Scale{num_layers}( Sum(attn_block, mlp_block) ),
    /// final_norm, lm_head )`. The `Scale` folds the homogeneous layers — the
    /// per-layer leaves are minted once, matching the `eval_iter` fold.
    pub fn cost_tree(&self) -> CostTree {
        let mut b = CostTreeBuilder::new();
        let embed = self.embed.compile(&mut b);
        let layer = CostNode::Scale {
            n: self.num_layers,
            child: Box::new(CostNode::Sum(vec![
                self.attn_block.compile(&mut b),
                self.mlp_block.compile(&mut b),
            ])),
        };
        let final_norm = self.final_norm.compile(&mut b);
        let lm_head = self.lm_head.compile(&mut b);
        let root = CostNode::Labeled {
            label: format!(
                "{} [dense TP (tp={}), {} layers]",
                self.name, self.tp_size, self.num_layers
            ),
            child: Box::new(CostNode::Sum(vec![embed, layer, final_norm, lm_head])),
        };
        b.finish(root)
    }

    /// CostTree eval: stream this iteration's per-leaf [`LeafMetrics`] into `buf`
    /// in the exact order [`cost_tree`](Self::cost_tree) minted slots (embed, then
    /// ONE layer's attn_block/mlp_block — the `Scale{num_layers}` fold multiplies
    /// it — then final_norm, lm_head).
    fn eval_buf(&self, batch: &UnifiedArchInput, buf: &mut [LeafMetrics]) {
        let g = &batch.groups[0];
        let m = g.batch_tokens;
        let n = buf.len();
        let mut ev = Evaluator::new(buf);
        self.embed
            .eval(&ElementwiseKernelInput { num_tokens: m }, &mut ev);
        self.attn_block.eval(
            &AttnBlockTpWorkletInput {
                batch_tokens: m,
                prefill_chunk_pairs: g.prefill_chunk_pairs.clone(),
                decode_kv_lens: g.decode_kv_lens.clone(),
            },
            &mut ev,
        );
        self.mlp_block
            .eval(&MlpBlockTpWorkletInput { batch_tokens: m }, &mut ev);
        self.final_norm.eval(&RmsNormKernelInput { m }, &mut ev);
        self.lm_head.eval(&SingleGemmKernelInput { m }, &mut ev);
        debug_assert_eq!(ev.filled(), n, "eval cursor must fill every slot");
    }
}

impl IterwiseUnifiedModel for Llama3DenseTpModel {
    fn kv_bytes_per_token(&self) -> u64 {
        self.kv_bytes_per_token
    }

    /// One replica spans the `tp_size` ranks of the TP group (no HP/EP nesting in
    /// v1, so the extent is exactly the TP degree).
    fn gpus_per_replica(&self) -> u16 {
        self.tp_size
    }

    fn cost_log_manifest(&self) -> CostManifest {
        self.cost_tree().manifest()
    }

    fn eval_iter(&self, batch: &UnifiedArchInput, slots: &mut Vec<LeafMetrics>) -> LeafMetrics {
        assert_eq!(
            batch.groups.len(),
            1,
            "Llama3 dense TP has exactly one HP group (HP deferred)"
        );
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        self.eval_buf(batch, slots);
        CostTree::aggregate(&self.cost_flat, slots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_configs_threads_tp_into_worklets() {
        let model = ModelCfg::llama3_8b();
        let parallel = ParallelCfg::new(4, 1, 1, "H100");
        let cfgs = build_configs(&model, &parallel);
        assert_eq!(cfgs.tp_size, 4);
        assert_eq!(cfgs.attn_block.tp_size, 4);
        assert_eq!(cfgs.mlp_block.tp_size, 4);
        // lm_head replicated full (vocab-parallel deferred).
        assert_eq!(cfgs.lm_head.n, 128256);
        assert_eq!(cfgs.lm_head.k, 4096);
    }

    #[test]
    fn resolve_shards_heads_and_intermediate() {
        let cfgs = build_configs(&ModelCfg::llama3_8b(), &ParallelCfg::new(4, 1, 1, "H100"));
        let r = resolve_configs(&cfgs);
        // per-rank: qo 32/4=8, kv 8/4=2; fused qkv (8+2·2)·128 = 1536.
        assert_eq!(r.attn_block.qkv.n, 1536);
        assert_eq!(r.attn_block.qkv.k, 4096); // hidden NOT sharded
        // per-rank intermediate 14336/4=3584; up_gate n = 2·3584.
        assert_eq!(r.mlp_block.up_gate.n, 2 * 3584);
        assert_eq!(r.mlp_block.down.k, 3584);
    }

    #[test]
    fn kv_bytes_per_token_is_per_rank() {
        let cfgs = build_configs(&ModelCfg::llama3_8b(), &ParallelCfg::new(4, 1, 1, "H100"));
        let r = resolve_configs(&cfgs);
        // per-rank kv heads = 8/4 = 2; 2·2·128·2·32 = 32768 bytes/token/GPU.
        assert_eq!(kv_bytes_per_token(&r), 2 * 2 * 128 * 2 * 32);
    }

    #[test]
    fn tp1_kv_bytes_matches_full_model() {
        // At tp=1 the per-rank footprint equals the full dense value (8 kv heads).
        let cfgs = build_configs(&ModelCfg::llama3_8b(), &ParallelCfg::new(1, 1, 1, "H100"));
        let r = resolve_configs(&cfgs);
        assert_eq!(kv_bytes_per_token(&r), 2 * 8 * 128 * 2 * 32);
        assert_eq!(r.tp_size, 1);
    }
}
