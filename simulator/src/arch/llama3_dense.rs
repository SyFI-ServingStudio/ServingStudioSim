//! `llama3_dense` — L4 model_arch for Llama3-8B dense, local single GPU.
//!
//! Wires the three `Local` worklets (pre-attn / attn / post-attn) per layer,
//! plus an embedding placeholder + final-norm + lm_head, into an iter-wise
//! unified model. Follows the L4 build shape (`build_configs` /
//! `resolve_configs` / `build`) and exposes `IterwiseUnifiedModel`. Dry-run
//! coverage is no longer a separate traversal: `build` against a dry-run
//! [`PerfApiBridge`] tallies missing specs per kernel (see `Kernel::init`).
//!
//! Deviations from L4 design.md for this v1 dense vertical (see plan):
//!   - all worklets are `Local` (tp/ep/hp = 1, no collective);
//!   - one worklet instance per type, reused across `num_layers` in
//!     `eval_iter` with a per-layer `with_label` (not per-layer `build`);
//!   - embedding modeled as an `ElementwiseKernel` gather placeholder.

use std::sync::Arc;

use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::arch::model_cfg::{ModelCfg, ParallelCfg};
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
    AttnLocalWorklet, AttnLocalWorkletConfig, AttnLocalWorkletInput, AttnLocalWorkletResolved,
    PostAttnLocalWorklet, PostAttnLocalWorkletConfig, PostAttnLocalWorkletInput,
    PostAttnLocalWorkletResolved, PreAttnLocalWorklet, PreAttnLocalWorkletConfig,
    PreAttnLocalWorkletInput, PreAttnLocalWorkletResolved,
};

const NORM_BACKENDS: &[&str] = &["flashinfer"];
const GEMM_BACKENDS: &[&str] = &["torch"];
const ACT_BACKENDS: &[&str] = &["triton"];
// Attention backends are FlashInfer *implementations* (the profiler registers
// `flashinfer_attn_{prefill,decode,rect}` under `fa2`/`fa3`/`trt`/`cudnn`), NOT a
// backend literally named "flashinfer". List the two FlashAttention paths; the
// kernel engine picks whichever the GPU/profile.db actually has.
const ATTN_BACKENDS: &[&str] = &["fa2", "fa3"];

/// Raw worklet/op configs (parallelism-agnostic except for the baked gpu_name).
pub struct Llama3DenseConfigs {
    pub pre_attn: PreAttnLocalWorkletConfig,
    pub attn: AttnLocalWorkletConfig,
    pub post_attn: PostAttnLocalWorkletConfig,
    pub embed: ElementwiseKernelConfig,
    pub final_norm: RmsNormKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub num_layers: u32,
}

/// Post-resolve aggregate; atomic ops (embed / final_norm / lm_head) carry their
/// kernel config straight through (no partition).
pub struct Llama3DenseResolved {
    pub pre_attn: PreAttnLocalWorkletResolved,
    pub attn: AttnLocalWorkletResolved,
    pub post_attn: PostAttnLocalWorkletResolved,
    pub embed: ElementwiseKernelConfig,
    pub final_norm: RmsNormKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub num_layers: u32,
}

pub struct Llama3DenseModel {
    pub name: String,
    pub num_layers: u32,
    pub kv_bytes_per_token: u64,
    pub pre_attn: PreAttnLocalWorklet,
    pub attn: AttnLocalWorklet,
    pub post_attn: PostAttnLocalWorklet,
    pub embed: Op<ElementwiseKernel>,
    pub final_norm: Op<RmsNormKernel>,
    pub lm_head: Op<SingleGemmKernel>,
    /// CostTree structure compiled once at build (flattened form) + its slot
    /// count, so the per-iter `eval_iter` path only evals leaves +
    /// aggregates — no per-tick recompile/`String` minting.
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

pub fn build_configs(model: &ModelCfg, parallel: &ParallelCfg) -> Llama3DenseConfigs {
    let gpu = &parallel.gpu_name;
    let dtype_bytes = model.dtype.size_bytes();
    Llama3DenseConfigs {
        pre_attn: PreAttnLocalWorkletConfig {
            hidden: model.hidden,
            num_qo_heads: model.num_qo_heads,
            num_kv_heads: model.num_kv_heads,
            head_dim: model.head_dim,
            dtype: model.dtype,
            gpu_name: gpu.clone(),
            norm_backends: NORM_BACKENDS.to_vec(),
            gemm_backends: GEMM_BACKENDS.to_vec(),
        },
        attn: AttnLocalWorkletConfig {
            num_qo_heads: model.num_qo_heads,
            num_kv_heads: model.num_kv_heads,
            head_dim: model.head_dim,
            q_dtype: model.dtype,
            kv_dtype: model.kv_dtype,
            o_dtype: model.dtype,
            gpu_name: gpu.clone(),
            backends: ATTN_BACKENDS.to_vec(),
        },
        post_attn: PostAttnLocalWorkletConfig {
            hidden: model.hidden,
            intermediate: model.intermediate,
            num_qo_heads: model.num_qo_heads,
            head_dim: model.head_dim,
            dtype: model.dtype,
            gpu_name: gpu.clone(),
            norm_backends: NORM_BACKENDS.to_vec(),
            gemm_backends: GEMM_BACKENDS.to_vec(),
            act_backends: ACT_BACKENDS.to_vec(),
        },
        // Embedding gather placeholder: read one hidden-wide row, write one out.
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
        lm_head: SingleGemmKernelConfig {
            backends: GEMM_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            n: model.vocab,
            k: model.hidden,
            dtype: model.dtype,
        },
        num_layers: model.num_layers,
    }
}

/// KV-cache bytes one token occupies across the whole model, for this dense-GQA
/// architecture: `2` (K and V) × `num_kv_heads` × `head_dim` × `kv_dtype` bytes ×
/// `num_layers`, read off the resolved attention config. Arch-specific (MLA's
/// compressed latent KV, cross-layer KV sharing, … would compute it differently),
/// so it lives in the model_arch, not on the parallelism-agnostic `ModelCfg`.
fn kv_bytes_per_token(resolved: &Llama3DenseResolved) -> u64 {
    let attn = &resolved.attn.attn;
    2 * attn.num_kv_heads as u64
        * attn.head_dim as u64
        * attn.kv_dtype.size_bytes() as u64
        * resolved.num_layers as u64
}

pub fn resolve_configs(cfgs: &Llama3DenseConfigs) -> Llama3DenseResolved {
    Llama3DenseResolved {
        pre_attn: PreAttnLocalWorklet::resolve_config(&cfgs.pre_attn),
        attn: AttnLocalWorklet::resolve_config(&cfgs.attn),
        post_attn: PostAttnLocalWorklet::resolve_config(&cfgs.post_attn),
        embed: cfgs.embed.clone(),
        final_norm: cfgs.final_norm.clone(),
        lm_head: cfgs.lm_head.clone(),
        num_layers: cfgs.num_layers,
    }
}

pub fn build(
    model_name: String,
    resolved: Llama3DenseResolved,
    bridge: &PerfApiBridge,
) -> Result<Llama3DenseModel, BuildError> {
    let num_layers = resolved.num_layers;
    let kv_bytes_per_token = kv_bytes_per_token(&resolved);

    let embed_name = format!("{model_name}.embedding");
    let final_norm_name = format!("{model_name}.final_norm");
    let lm_head_name = format!("{model_name}.lm_head");

    let embed = Op::new(
        embed_name.clone(),
        Arc::new(ElementwiseKernel::build(embed_name, resolved.embed, bridge)?),
    );

    let pre_attn = PreAttnLocalWorklet::build(
        format!("{model_name}.pre_attn"),
        resolved.pre_attn,
        bridge,
    )?;

    let attn = AttnLocalWorklet::build(format!("{model_name}.attn"), resolved.attn, bridge)?;

    let post_attn = PostAttnLocalWorklet::build(
        format!("{model_name}.post_attn"),
        resolved.post_attn,
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

    let mut model = Llama3DenseModel {
        pre_attn,
        attn,
        post_attn,
        embed,
        final_norm,
        lm_head,
        name: model_name,
        num_layers,
        kv_bytes_per_token,
        cost_flat: Vec::new(),
        n_slots: 0,
    };

    // Compile the CostTree structure once and cache its flattened form on the
    // model, so the per-iter metrics path skips recompiling / minting names.
    let tree = model.cost_tree();
    model.cost_flat = tree.flatten();
    model.n_slots = tree.n_slots();

    // Print the compiled cost-tree structure once after build: the per-slot
    // manifest with each leaf's kernel kind + config + the worklet/partition
    // labels (the shape render, folded in from the retired `Describe`).
    tracing::info!(
        "[build] cost tree ({} leaf slots):\n{}",
        tree.n_slots(),
        tree.describe()
    );

    Ok(model)
}

impl Llama3DenseModel {
    /// Compile the per-iteration cost *structure* once (CostTree, milestone 1):
    /// `Sum( embed, Scale{num_layers}( Sum(pre_attn, attn, post_attn) ),
    /// final_norm, lm_head )`. The `Scale` folds the homogeneous layers — the
    /// per-layer leaves are minted once (not `×num_layers`), matching the
    /// `eval_iter` fold. Structure only; per-iter eval lands later.
    pub fn cost_tree(&self) -> CostTree {
        let mut b = CostTreeBuilder::new();
        let embed = self.embed.compile(&mut b);
        let layer = CostNode::Scale {
            n: self.num_layers,
            child: Box::new(CostNode::Sum(vec![
                self.pre_attn.compile(&mut b),
                self.attn.compile(&mut b),
                self.post_attn.compile(&mut b),
            ])),
        };
        let final_norm = self.final_norm.compile(&mut b);
        let lm_head = self.lm_head.compile(&mut b);
        // Root carries the model header (the old `Describe` top line).
        let root = CostNode::Labeled {
            label: format!("{} [dense local, {} layers]", self.name, self.num_layers),
            child: Box::new(CostNode::Sum(vec![embed, layer, final_norm, lm_head])),
        };
        b.finish(root)
    }

    /// CostTree eval (milestone 2): stream this iteration's per-leaf [`Metrics4`]
    /// into `buf` in the exact order [`cost_tree`](Self::cost_tree) minted slots
    /// (embed, then ONE layer's pre/attn/post — the `Scale{num_layers}` fold
    /// multiplies it, INV-3 — then final_norm, lm_head). The cursor must end at
    /// `buf.len()`; otherwise the eval walk and the compiled slot list disagree.
    fn eval_buf(&self, batch: &UnifiedArchInput, buf: &mut [LeafMetrics]) {
        let g = &batch.groups[0];
        let m = g.batch_tokens;
        let n = buf.len();
        let mut ev = Evaluator::new(buf);
        self.embed
            .eval(&ElementwiseKernelInput { num_tokens: m }, &mut ev);
        self.pre_attn
            .eval(&PreAttnLocalWorkletInput { batch_tokens: m }, &mut ev);
        self.attn.eval(
            &AttnLocalWorkletInput {
                prefill_chunk_pairs: g.prefill_chunk_pairs.clone(),
                decode_kv_lens: g.decode_kv_lens.clone(),
            },
            &mut ev,
        );
        self.post_attn
            .eval(&PostAttnLocalWorkletInput { batch_tokens: m }, &mut ev);
        self.final_norm.eval(&RmsNormKernelInput { m }, &mut ev);
        self.lm_head.eval(&SingleGemmKernelInput { m }, &mut ev);
        debug_assert_eq!(ev.filled(), n, "eval cursor must fill every slot");
    }
}

impl IterwiseUnifiedModel for Llama3DenseModel {
    fn kv_bytes_per_token(&self) -> u64 {
        self.kv_bytes_per_token
    }

    /// The compiled CostTree's serializable manifest (slots + flat aggregation
    /// nodes). Recompiled once at logger setup (off the hot path), so a consumer
    /// can reproduce `total_time_ms` from a `cost_log` row's per-slot breakdown.
    fn cost_log_manifest(&self) -> CostManifest {
        self.cost_tree().manifest()
    }

    /// Per-iter cost via the cached compiled CostTree: fill `slots` with the
    /// per-leaf [`LeafMetrics`] for this iter (reusing the caller's `Vec`
    /// capacity), then roll up `cost_flat` (the `Scale` fold supplies the
    /// `×num_layers`). One eval pass feeds both the `cost_log` row and the clock.
    fn eval_iter(&self, batch: &UnifiedArchInput, slots: &mut Vec<LeafMetrics>) -> LeafMetrics {
        assert_eq!(
            batch.groups.len(),
            1,
            "Llama3 dense local has exactly one HP group"
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
    fn build_configs_threads_dims_and_gpu_name() {
        let model = ModelCfg::llama3_8b();
        let parallel = ParallelCfg::local("H100");
        let cfgs = build_configs(&model, &parallel);
        assert_eq!(cfgs.num_layers, 32);
        assert_eq!(cfgs.lm_head.n, 128256);
        assert_eq!(cfgs.lm_head.k, 4096);
        assert_eq!(cfgs.lm_head.gpu_name, "H100");
        assert_eq!(cfgs.pre_attn.gpu_name, "H100");
        assert_eq!(cfgs.embed.input_bytes_per_token, 4096 * 2);
    }

    #[test]
    fn resolve_configs_bakes_worklet_shapes() {
        let cfgs = build_configs(&ModelCfg::llama3_8b(), &ParallelCfg::local("H100"));
        let r = resolve_configs(&cfgs);
        assert_eq!(r.pre_attn.qkv.n, 6144);
        assert_eq!(r.post_attn.up_gate.n, 28672);
        assert_eq!(r.lm_head.n, 128256);
        assert_eq!(r.num_layers, 32);
    }
}
