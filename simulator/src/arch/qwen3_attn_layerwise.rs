//! `qwen3_attn_layerwise` — L4 attn-side model_arch for AFD (attention-FFN
//! disaggregation) of a Qwen3-MoE decoder. This is the attn pool's half of the
//! `qwen3_moe_dp_attn_ep_ffn` split: it owns ONLY the per-layer attention
//! kernel(s) — the moesim-faithful cut (`ref/moesim-rs`). Everything else
//! (input_norm / qkv / o_proj / tp_allreduce / router / MoE / embed / lm_head)
//! lives on the ffn side (`qwen3_ffn_moe_layerwise`).
//!
//! **One model instance = one DP shard.** A DP shard is `attn_tp_size` head-
//! parallel ranks with its own KV cache and its own request stream — the natural
//! attn-worker unit. Data parallelism (the unified arch's
//! `num_dp_groups = ep_size / attn_tp_size`) is the attn POOL running that many
//! independent workers (its `replicas`), NOT a fan-out inside one model; the pool
//! synchronizes the shards per layer before the ffn dispatch. So the cost here is
//! a single attention:
//!
//! ```text
//! attn   // this shard's tokens (prefill chunks + decode kv-lens)
//! ```
//!
//! The attention kernel uses the standard head-parallel split
//! (`num_qo_heads / attn_tp_size`, `num_kv_heads / attn_tp_size`) via the shared
//! `AttnBlockTpWorklet`. This arch is self-contained — it does NOT depend on the
//! unified `qwen3_moe_dp_attn_ep_ffn` arch (the AFD sides are independent).
//!
//! Comm: this side emits `attn_to_ffn_bytes_per_token` (the attention output,
//! `q_dim · bpe`) per token to the ffn side; the ffn side emits the QKV
//! projection back (`(q_dim + 2·kv_dim) · bpe`).

use crate::arch::contract::{AttnArchInput, AttnLayerwiseModel};
use crate::arch::moe_model_cfg::MoeModelCfg;
use crate::common::Fabric;
use crate::op::attention::{FlashInferAttentionInput, FlashInferAttentionOp};
use crate::timing::{
    BuildError, CostTree, CostTreeBuilder, Evaluator, FlatCostNode, LeafMetrics, PerfApiBridge,
};
use crate::worklet::{AttnBlockTpWorklet, AttnBlockTpWorkletConfig, AttnBlockTpWorkletResolved};

// Backend / fabric policy for this arch's attention block. Local copy — the AFD
// archs are self-contained (no shared arch-level config).
const NORM_BACKENDS: &[&str] = &["flashinfer"];
const GEMM_BACKENDS: &[&str] = &["torch"];
// See llama3_dense: FlashInfer impls registered under fa2/fa3, not "flashinfer".
const ATTN_BACKENDS: &[&str] = &["fa2", "fa3"];
const ALLREDUCE_BACKENDS: &[&str] = &["nccl"];
const TP_FABRIC: Fabric = Fabric::Nvlink;

/// Build this arch's attention-block config from the model dims + `attn_tp_size`.
/// Only the attention sub-kernel of the resolved block is used by this arch; the
/// norm/gemm/allreduce fields feed `AttnBlockTpWorklet`'s head-split resolve and
/// are otherwise unused here.
fn attn_block_config(
    model: &MoeModelCfg,
    attn_tp_size: u16,
    gpu_name: &str,
) -> AttnBlockTpWorkletConfig {
    AttnBlockTpWorkletConfig {
        hidden: model.hidden,
        num_qo_heads: model.num_qo_heads,
        num_kv_heads: model.num_kv_heads,
        head_dim: model.head_dim,
        dtype: model.dtype,
        kv_dtype: model.kv_dtype,
        tp_size: attn_tp_size,
        allreduce_fabric: TP_FABRIC,
        gpu_name: gpu_name.to_string(),
        norm_backends: NORM_BACKENDS.to_vec(),
        gemm_backends: GEMM_BACKENDS.to_vec(),
        attn_backends: ATTN_BACKENDS.to_vec(),
        allreduce_backends: ALLREDUCE_BACKENDS.to_vec(),
    }
}

/// Attn-side numeric parallel input. `attn_tp_size` shards the attention heads
/// across the ranks of one DP shard (the model instance = one shard, spanning
/// `attn_tp_size` GPUs). The DP-shard count lives on the pool (`replicas`), not
/// here.
#[derive(Clone, Debug)]
pub struct Qwen3AttnParallel {
    pub attn_tp_size: u16,
    pub gpu_name: String,
}

/// Raw config: the (shared) attn-block config + scalars the cost/comm path needs.
pub struct Qwen3AttnLayerwiseConfigs {
    pub attn_block: AttnBlockTpWorkletConfig,
    pub num_layers: u32,
    pub attn_tp_size: u16,
    pub total_kv_bytes_per_token: u64,
    pub attn_to_ffn_bytes_per_token: u64,
}

/// Post-resolve aggregate: the attn-block partition (only `.attn` is built) +
/// scalars carried through unchanged.
pub struct Qwen3AttnLayerwiseResolved {
    pub attn_block: AttnBlockTpWorkletResolved,
    pub num_layers: u32,
    pub attn_tp_size: u16,
    pub total_kv_bytes_per_token: u64,
    pub attn_to_ffn_bytes_per_token: u64,
}

pub struct Qwen3AttnLayerwiseModel {
    pub name: String,
    pub num_layers: u32,
    pub attn_tp_size: u16,
    pub total_kv_bytes_per_token: u64,
    pub attn_to_ffn_bytes_per_token: u64,
    pub attn: FlashInferAttentionOp,
    /// This shard's per-layer attention cost tree (a single `attn`), compiled once
    /// + flattened. Layer-homogeneous — every layer sees the same batch within an
    /// iteration, so one compiled tree serves all layers.
    attn_flat: Vec<FlatCostNode>,
    attn_n_slots: usize,
}

pub fn build_configs(
    model: &MoeModelCfg,
    parallel: &Qwen3AttnParallel,
) -> Qwen3AttnLayerwiseConfigs {
    assert!(parallel.attn_tp_size > 0, "attn_tp_size must be non-zero");
    let dtype_bytes = model.dtype.size_bytes() as u64;
    Qwen3AttnLayerwiseConfigs {
        attn_block: attn_block_config(model, parallel.attn_tp_size, &parallel.gpu_name),
        num_layers: model.num_layers,
        attn_tp_size: parallel.attn_tp_size,
        // Total KV bytes per token: 2 (k+v) × kv_heads × head_dim × kv_dtype × layers.
        // FULL (un-sharded) wire size — same definition as the iter-wise arch.
        total_kv_bytes_per_token: 2
            * model.num_kv_heads as u64
            * model.head_dim as u64
            * model.kv_dtype.size_bytes() as u64
            * model.num_layers as u64,
        // attn → ffn handoff: the attention output, q_dim · bpe per token.
        attn_to_ffn_bytes_per_token: model.num_qo_heads as u64
            * model.head_dim as u64
            * dtype_bytes,
    }
}

pub fn resolve_configs(cfgs: &Qwen3AttnLayerwiseConfigs) -> Qwen3AttnLayerwiseResolved {
    Qwen3AttnLayerwiseResolved {
        attn_block: AttnBlockTpWorklet::resolve_config(&cfgs.attn_block),
        num_layers: cfgs.num_layers,
        attn_tp_size: cfgs.attn_tp_size,
        total_kv_bytes_per_token: cfgs.total_kv_bytes_per_token,
        attn_to_ffn_bytes_per_token: cfgs.attn_to_ffn_bytes_per_token,
    }
}

pub fn build(
    model_name: String,
    resolved: Qwen3AttnLayerwiseResolved,
    bridge: &PerfApiBridge,
) -> Result<Qwen3AttnLayerwiseModel, BuildError> {
    let attn = FlashInferAttentionOp::build(
        format!("{model_name}.attn"),
        resolved.attn_block.attn.clone(),
        bridge,
    )?;
    let mut model = Qwen3AttnLayerwiseModel {
        name: model_name,
        num_layers: resolved.num_layers,
        attn_tp_size: resolved.attn_tp_size,
        total_kv_bytes_per_token: resolved.total_kv_bytes_per_token,
        attn_to_ffn_bytes_per_token: resolved.attn_to_ffn_bytes_per_token,
        attn,
        attn_flat: Vec::new(),
        attn_n_slots: 0,
    };
    let tree = model.attn_cost_tree();
    model.attn_flat = tree.flatten();
    model.attn_n_slots = tree.n_slots();
    // Build-time cost-tree printout, mirroring the iter-wise archs' `[build] cost
    // tree` line. The attn side has a single per-layer tree (attention only).
    tracing::info!(
        "[build] {} cost tree [attn] ({} leaf slots):\n{}",
        model.name,
        tree.n_slots(),
        tree.describe()
    );
    Ok(model)
}

impl Qwen3AttnLayerwiseModel {
    /// Per-layer attention cost STRUCTURE for one DP shard: a single `attn` subtree
    /// over this shard's tokens. `compile` mints its slots; `attn_cost` fills them
    /// in the same order.
    fn attn_cost_tree(&self) -> CostTree {
        let mut b = CostTreeBuilder::new();
        let root = self.attn.compile(&mut b);
        b.finish(root)
    }
}

impl AttnLayerwiseModel for Qwen3AttnLayerwiseModel {
    fn num_layers(&self) -> u32 {
        self.num_layers
    }

    /// One attn worker = one DP shard, spanning `attn_tp_size` head-parallel ranks.
    /// (`num_attn_dp_groups` stays the trait default of 1; the DP-shard count is a
    /// pool concern, not the model's.)
    fn gpus_per_replica(&self) -> u16 {
        self.attn_tp_size
    }

    fn total_kv_bytes_per_token(&self) -> u64 {
        self.total_kv_bytes_per_token
    }

    fn attn_to_ffn_bytes_per_token(&self) -> u64 {
        self.attn_to_ffn_bytes_per_token
    }

    fn attn_cost(
        &self,
        _layer_idx: usize,
        batch: &AttnArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        assert_eq!(
            batch.groups.len(),
            1,
            "AFD attn worker computes one DP shard — expects exactly one group"
        );
        slots.clear();
        slots.resize(self.attn_n_slots, LeafMetrics::ZERO);
        let mut ev = Evaluator::new(slots);
        // This shard's attention. `layer_idx` does not change the cost (every layer
        // sees the same batch within an iteration).
        let g = &batch.groups[0];
        self.attn.eval(
            &FlashInferAttentionInput {
                prefill_chunk_pairs: g.prefill_chunk_pairs.clone(),
                decode_kv_lens: g.decode_kv_lens.clone(),
            },
            &mut ev,
        );
        debug_assert_eq!(
            ev.filled(),
            self.attn_n_slots,
            "eval cursor must fill every slot"
        );
        CostTree::aggregate(&self.attn_flat, slots, scratch)
    }
}
