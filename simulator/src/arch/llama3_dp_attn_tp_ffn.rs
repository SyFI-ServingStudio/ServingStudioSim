//! `llama3_dp_attn_tp_ffn` — L4 model_arch for Llama3-8B with **split attention /
//! FFN parallelism**: attention is sharded over `attn_tp_size` head-parallel ranks
//! and replicated as `num_dp_groups` independent data-parallel shards; the dense
//! FFN is sharded over a larger `ffn_tp_size` TP group. DeepSeek-V3-style DP
//! attention, applied to a dense decoder.
//!
//! The DP degree is derived, not configured: `num_dp_groups = ffn_tp_size /
//! attn_tp_size` (e.g. attn_tp=4, ffn_tp=8 → 2 DP shards of 4 head-parallel ranks
//! each, all 8 GPUs forming the FFN TP group). One replica spans `ffn_tp_size`
//! GPUs.
//!
//! Cost shape (iter-wise, mirrors `llama3_dense_tp` but with a DP fan-out on
//! attention):
//!   - per layer: `Sum( Max{1.0}( attn_block × num_dp_groups ), mlp_block )` — the
//!     `Max` is the L4 §3.3 DP fan-out (independent shards run concurrently; the
//!     sync wallclock is the slowest shard, `overlap = 1.0`), and the FFN sees the
//!     pooled token total (L4 §3.5, TP collective is intra-group symmetric);
//!   - `attn_block` is fed `attn_tp_size`, `mlp_block` is fed `ffn_tp_size`; the
//!     two TP worklets are reused unchanged (each takes one `tp_size`);
//!   - embed / final_norm / lm_head stay replicated (full shapes) over the pooled
//!     token total — a v1 simplification, not per-rank.
//!
//! v1 deviation: the attention→FFN **resharding** (an all-to-all moving the
//! (dp×attn_tp) attention output layout into the ffn_tp layout) is NOT modeled —
//! deferred. The per-layer cost is attention fan-out + FFN only.

use std::sync::Arc;

use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::arch::model_cfg::ModelCfg;
use crate::common::Fabric;
use crate::op::Op;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, RmsNormKernel,
    RmsNormKernelConfig, RmsNormKernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
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
// v1 fabric: single-node NVLink for both the attn-TP and ffn-TP all-reduces.
const TP_FABRIC: Fabric = Fabric::Nvlink;

/// Raw worklet/op configs. `attn_block` bakes `attn_tp_size`, `mlp_block` bakes
/// `ffn_tp_size`; the DP degree is carried separately for the cost fan-out.
pub struct Llama3DpAttnTpFfnConfigs {
    pub attn_block: AttnBlockTpWorkletConfig,
    pub mlp_block: MlpBlockTpWorkletConfig,
    pub embed: ElementwiseKernelConfig,
    pub final_norm: RmsNormKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub num_layers: u32,
    pub attn_tp_size: u16,
    pub ffn_tp_size: u16,
    pub num_dp_groups: u16,
}

/// Post-resolve aggregate; atomic ops (embed / final_norm / lm_head) carry their
/// kernel config straight through (replicated, no partition).
pub struct Llama3DpAttnTpFfnResolved {
    pub attn_block: AttnBlockTpWorkletResolved,
    pub mlp_block: MlpBlockTpWorkletResolved,
    pub embed: ElementwiseKernelConfig,
    pub final_norm: RmsNormKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub num_layers: u32,
    pub attn_tp_size: u16,
    pub ffn_tp_size: u16,
    pub num_dp_groups: u16,
}

pub struct Llama3DpAttnTpFfnModel {
    pub name: String,
    pub num_layers: u32,
    pub attn_tp_size: u16,
    pub ffn_tp_size: u16,
    pub num_dp_groups: u16,
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

/// This arch's numeric parallel input: the attention TP degree (`attn_tp_size`,
/// heads sharded) and the FFN TP degree (`ffn_tp_size`, hidden/intermediate
/// sharded), plus the `gpu_name` every kernel lookup keys on (L4 §3.8). The DP
/// degree is derived (`ffn_tp_size / attn_tp_size`). Per new-interface-design §13:
/// each arch owns the numeric parallel struct it needs.
#[derive(Clone, Debug)]
pub struct DpAttnTpFfnParallel {
    pub attn_tp_size: u16,
    pub ffn_tp_size: u16,
    pub gpu_name: String,
}

pub fn build_configs(model: &ModelCfg, parallel: &DpAttnTpFfnParallel) -> Llama3DpAttnTpFfnConfigs {
    let gpu = &parallel.gpu_name;
    let dtype_bytes = model.dtype.size_bytes();
    assert!(
        parallel.attn_tp_size > 0 && parallel.ffn_tp_size > 0,
        "attn_tp_size / ffn_tp_size must be non-zero"
    );
    assert!(
        parallel.ffn_tp_size % parallel.attn_tp_size == 0,
        "ffn_tp_size {} must be a multiple of attn_tp_size {} (DP groups = ffn_tp/attn_tp)",
        parallel.ffn_tp_size,
        parallel.attn_tp_size
    );
    let num_dp_groups = parallel.ffn_tp_size / parallel.attn_tp_size;
    Llama3DpAttnTpFfnConfigs {
        attn_block: AttnBlockTpWorkletConfig {
            hidden: model.hidden,
            num_qo_heads: model.num_qo_heads,
            num_kv_heads: model.num_kv_heads,
            head_dim: model.head_dim,
            dtype: model.dtype,
            kv_dtype: model.kv_dtype,
            tp_size: parallel.attn_tp_size,
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
            tp_size: parallel.ffn_tp_size,
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
        attn_tp_size: parallel.attn_tp_size,
        ffn_tp_size: parallel.ffn_tp_size,
        num_dp_groups,
    }
}

/// Per-GPU KV-cache footprint one token occupies across the whole model. Under DP
/// attention the KV heads are head-sharded across the `attn_tp_size` ranks of a DP
/// shard, so each attention GPU sizes its KvPool from the **per-rank** KV-head
/// count, read off the resolved attn config's already-divided value: `2` (K and V)
/// × `num_kv_heads/attn_tp` × `head_dim` × `kv_dtype` bytes × `num_layers`.
fn kv_bytes_per_token(resolved: &Llama3DpAttnTpFfnResolved) -> u64 {
    let attn = &resolved.attn_block.attn; // per-rank FlashInferAttentionConfig
    2 * attn.num_kv_heads as u64
        * attn.head_dim as u64
        * attn.kv_dtype.size_bytes() as u64
        * resolved.num_layers as u64
}

pub fn resolve_configs(cfgs: &Llama3DpAttnTpFfnConfigs) -> Llama3DpAttnTpFfnResolved {
    Llama3DpAttnTpFfnResolved {
        attn_block: AttnBlockTpWorklet::resolve_config(&cfgs.attn_block),
        mlp_block: MlpBlockTpWorklet::resolve_config(&cfgs.mlp_block),
        embed: cfgs.embed.clone(),
        final_norm: cfgs.final_norm.clone(),
        lm_head: cfgs.lm_head.clone(),
        num_layers: cfgs.num_layers,
        attn_tp_size: cfgs.attn_tp_size,
        ffn_tp_size: cfgs.ffn_tp_size,
        num_dp_groups: cfgs.num_dp_groups,
    }
}

pub fn build(
    model_name: String,
    resolved: Llama3DpAttnTpFfnResolved,
    bridge: &PerfApiBridge,
) -> Result<Llama3DpAttnTpFfnModel, BuildError> {
    let num_layers = resolved.num_layers;
    let attn_tp_size = resolved.attn_tp_size;
    let ffn_tp_size = resolved.ffn_tp_size;
    let num_dp_groups = resolved.num_dp_groups;
    let kv_bytes_per_token = kv_bytes_per_token(&resolved);

    let embed_name = format!("{model_name}.embedding");
    let final_norm_name = format!("{model_name}.final_norm");
    let lm_head_name = format!("{model_name}.lm_head");

    let embed = Op::new(
        embed_name.clone(),
        Arc::new(ElementwiseKernel::build(
            embed_name,
            resolved.embed,
            bridge,
        )?),
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
        Arc::new(RmsNormKernel::build(
            final_norm_name,
            resolved.final_norm,
            bridge,
        )?),
    );

    let lm_head = Op::new(
        lm_head_name.clone(),
        Arc::new(SingleGemmKernel::build(
            lm_head_name,
            resolved.lm_head,
            bridge,
        )?),
    );

    let mut model = Llama3DpAttnTpFfnModel {
        attn_block,
        mlp_block,
        embed,
        final_norm,
        lm_head,
        name: model_name,
        num_layers,
        attn_tp_size,
        ffn_tp_size,
        num_dp_groups,
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

impl Llama3DpAttnTpFfnModel {
    /// Compile the per-iteration cost *structure* once:
    /// `Sum( embed, Scale{num_layers}( Sum( Max{1.0}(attn_block × num_dp_groups),
    /// mlp_block ) ), final_norm, lm_head )`. The `Max` is the DP fan-out — one
    /// attn_block subtree (own slots) per DP shard; the `Scale` folds the
    /// homogeneous layers, so the per-layer leaves are minted once.
    pub fn cost_tree(&self) -> CostTree {
        let mut b = CostTreeBuilder::new();
        let embed = self.embed.compile(&mut b);
        // One attn_block subtree per DP shard — each `compile` mints its own slots,
        // so the `eval_into` pass fills them per group in the same order.
        let attn_groups: Vec<CostNode> = (0..self.num_dp_groups)
            .map(|_| self.attn_block.compile(&mut b))
            .collect();
        let attn_fanout = CostNode::Max {
            overlap: 1.0,
            children: attn_groups,
        };
        // Tag the homogeneous fold with its repeat-unit noun ("layer") so a
        // downstream consumer (the Perfetto trace) names the `Scale` repeats
        // `layer 0..n` semantically instead of assuming `Scale == layer`.
        let layer = CostNode::Labeled {
            label: "layer".to_string(),
            child: Box::new(CostNode::Scale {
                n: self.num_layers,
                child: Box::new(CostNode::Sum(vec![
                    attn_fanout,
                    self.mlp_block.compile(&mut b),
                ])),
            }),
        };
        let final_norm = self.final_norm.compile(&mut b);
        let lm_head = self.lm_head.compile(&mut b);
        let root = CostNode::Labeled {
            label: format!(
                "{} [DP attn (attn_tp={}, dp={}) + FFN tp={}, {} layers]",
                self.name, self.attn_tp_size, self.num_dp_groups, self.ffn_tp_size, self.num_layers
            ),
            child: Box::new(CostNode::Sum(vec![embed, layer, final_norm, lm_head])),
        };
        b.finish(root)
    }

    /// CostTree eval: stream this iteration's per-leaf [`LeafMetrics`] through `ev`
    /// in the exact order [`cost_tree`](Self::cost_tree) minted slots: embed (pooled
    /// tokens), then ONE layer's `num_dp_groups` attn_block evals (one per DP shard,
    /// each with that shard's batch) followed by the FFN (pooled tokens) — the
    /// `Scale{num_layers}` fold multiplies it — then final_norm, lm_head (pooled).
    fn eval_into(&self, batch: &UnifiedArchInput, ev: &mut Evaluator) {
        let m_total: u32 = batch.groups.iter().map(|g| g.batch_tokens).sum();
        self.embed.eval(
            &ElementwiseKernelInput {
                num_tokens: m_total,
            },
            ev,
        );
        for g in &batch.groups {
            self.attn_block.eval(
                &AttnBlockTpWorkletInput {
                    batch_tokens: g.batch_tokens,
                    prefill_chunk_pairs: g.prefill_chunk_pairs.clone(),
                    decode_kv_lens: g.decode_kv_lens.clone(),
                },
                ev,
            );
        }
        self.mlp_block.eval(
            &MlpBlockTpWorkletInput {
                batch_tokens: m_total,
            },
            ev,
        );
        self.final_norm.eval(&RmsNormKernelInput { m: m_total }, ev);
        self.lm_head.eval(&SingleGemmKernelInput { m: m_total }, ev);
    }
}

impl IterwiseUnifiedModel for Llama3DpAttnTpFfnModel {
    fn kv_bytes_per_token(&self) -> u64 {
        self.kv_bytes_per_token
    }

    /// One replica spans the FFN TP group — `ffn_tp_size` GPUs, with the DP
    /// attention shards (`attn_tp_size` ranks each) nested inside it.
    fn gpus_per_replica(&self) -> u16 {
        self.ffn_tp_size
    }

    fn num_attn_dp_groups(&self) -> u16 {
        self.num_dp_groups
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
        assert_eq!(
            batch.groups.len(),
            self.num_dp_groups as usize,
            "DP-attn arch expects one group per DP shard (num_dp_groups)"
        );
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut ev = Evaluator::new(slots);
        self.eval_into(batch, &mut ev);
        debug_assert_eq!(
            ev.filled(),
            self.n_slots,
            "eval cursor must fill every slot"
        );
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }

    fn eval_iter_with_inputs(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        assert_eq!(
            batch.groups.len(),
            self.num_dp_groups as usize,
            "DP-attn arch expects one group per DP shard (num_dp_groups)"
        );
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut ev = Evaluator::with_inputs(slots, inputs);
        self.eval_into(batch, &mut ev);
        debug_assert_eq!(
            ev.filled(),
            self.n_slots,
            "eval cursor must fill every slot"
        );
        let agg = CostTree::aggregate(&self.cost_flat, slots, scratch);
        debug_assert_eq!(inputs.len(), self.n_slots, "slot_input must align to slots");
        agg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parallel(attn_tp: u16, ffn_tp: u16) -> DpAttnTpFfnParallel {
        DpAttnTpFfnParallel {
            attn_tp_size: attn_tp,
            ffn_tp_size: ffn_tp,
            gpu_name: "H100".to_string(),
        }
    }

    #[test]
    fn build_configs_threads_split_tp_and_derives_dp() {
        let cfgs = build_configs(&ModelCfg::llama3_8b(), &parallel(4, 8));
        assert_eq!(cfgs.attn_tp_size, 4);
        assert_eq!(cfgs.ffn_tp_size, 8);
        // dp = ffn_tp / attn_tp.
        assert_eq!(cfgs.num_dp_groups, 2);
        // attn worklet sees attn_tp, mlp worklet sees ffn_tp.
        assert_eq!(cfgs.attn_block.tp_size, 4);
        assert_eq!(cfgs.mlp_block.tp_size, 8);
        // lm_head replicated full (vocab-parallel deferred).
        assert_eq!(cfgs.lm_head.n, 128256);
        assert_eq!(cfgs.lm_head.k, 4096);
    }

    #[test]
    fn resolve_shards_heads_by_attn_tp_and_intermediate_by_ffn_tp() {
        let r = resolve_configs(&build_configs(&ModelCfg::llama3_8b(), &parallel(4, 8)));
        // attention per-rank under attn_tp=4: qo 32/4=8, kv 8/4=2 → fused (8+2·2)·128 = 1536.
        assert_eq!(r.attn_block.qkv.n, 1536);
        assert_eq!(r.attn_block.qkv.k, 4096); // hidden NOT sharded
                                              // FFN per-rank under ffn_tp=8: intermediate 14336/8=1792; up_gate n = 2·1792.
        assert_eq!(r.mlp_block.up_gate.n, 2 * 1792);
        assert_eq!(r.mlp_block.down.k, 1792);
    }

    #[test]
    fn kv_bytes_per_token_uses_attn_rank() {
        let r = resolve_configs(&build_configs(&ModelCfg::llama3_8b(), &parallel(4, 8)));
        // per-attn-rank kv heads = 8/4 = 2; 2·2·128·2·32 = 32768 bytes/token/GPU.
        assert_eq!(kv_bytes_per_token(&r), 2 * 2 * 128 * 2 * 32);
    }

    #[test]
    fn attn_tp_equal_ffn_tp_is_single_dp_group() {
        // attn_tp == ffn_tp → no DP replication (one group, like plain TP).
        let cfgs = build_configs(&ModelCfg::llama3_8b(), &parallel(8, 8));
        assert_eq!(cfgs.num_dp_groups, 1);
    }

    #[test]
    #[should_panic(expected = "must be a multiple of")]
    fn ffn_tp_not_multiple_of_attn_tp_panics() {
        // ffn_tp=6, attn_tp=4 → 6 % 4 != 0.
        let _ = build_configs(&ModelCfg::llama3_8b(), &parallel(4, 6));
    }
}
