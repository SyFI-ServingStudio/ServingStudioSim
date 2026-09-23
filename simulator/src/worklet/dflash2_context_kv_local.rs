//! DFlash2 context-KV precomputation worklet — the draft's prefill-shaped half.
//!
//! One `DFlashProposer` call runs two differently shaped pieces of work. This
//! worklet is the first: it turns the target's aux hidden states into every
//! draft layer's K/V and writes them straight into the paged cache. It runs
//! once per draft call over the *target* rows (the tokens the verify step
//! accepted), not over the draft's own query rows, which is why it is its own
//! worklet rather than part of the per-layer section.
//!
//! The launch sequence is fixed and short because `precompute_and_store_context_kv`
//! is written to avoid torch.compile and CUDA graphs entirely (the context shape
//! differs from the query shape, so neither applies). It hand-fuses across
//! layers: **one** GEMM produces all six layers' K and V, **one** grouped RMSNorm
//! normalizes every layer's K, and **one** RoPE call rotates them all. Only the
//! cache write is per-layer. The cost is therefore almost independent of
//! `num_draft_layers`, which is the opposite of an ordinary decoder stack.
//!
//! Measured against `logs/20260920_0_glm53_dflash2_phase0` (B200, tp=4): the
//! whole `draft` NVTX phase is 1.61 ms of a 16.90 ms decode iteration (9.5%),
//! and every leaf here sits in its tail — none reached the top-12 by busy time.
//!
//! The architecture owns nothing here: there is no collective, and `fc` is a
//! `ReplicatedLinear`, so it is *not* sharded and every rank runs it whole.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, KvCacheAppendKernel,
    KvCacheAppendKernelConfig, KvCacheAppendKernelInput, RmsNormKernel, RmsNormKernelConfig,
    RmsNormKernelInput, SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, LeafMetrics, PerfApiBridge, Probe,
    SlotInput,
};

const HIDDEN_DIM: u32 = 6144;
const HEAD_DIM: u32 = 128;
const NUM_KV_HEADS: u32 = 8;
const NUM_DRAFT_LAYERS: u32 = 6;
/// Target layers whose hidden states the draft consumes — `dflash_config.target_layer_ids`
/// is `[5, 19, 33, 47, 61, 75]`, so `fc` reads six concatenated hidden vectors.
const NUM_AUX_LAYERS: u32 = 6;

/// Raw DFlash2 context-KV identity for one rank-local compute section.
#[derive(Clone, Debug)]
pub struct Dflash2ContextKvLocalWorkletConfig {
    pub norm_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
    pub kv_cache_append_backends: Vec<&'static str>,
    pub tp_size: u16,
    pub gpu_name: String,
    pub hidden_dim: Dim,
    pub num_draft_layers: u32,
    pub num_aux_layers: u32,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    pub kv_cache_block_size: u32,
    pub kv_cache_layout: String,
    pub kv_scale_granularity: String,
    /// Base activation/norm dtype. GEMM leaves use `gemm_dtype`.
    pub dtype: DType,
    pub gemm_dtype: DType,
    pub kv_dtype: DType,
}

/// Pure resolved data with every atomic child config fully baked.
#[derive(Clone, Debug)]
pub struct Dflash2ContextKvLocalWorkletResolved {
    pub raw_cfg: Dflash2ContextKvLocalWorkletConfig,
    pub aux_hidden_fc: SingleGemmKernelConfig,
    pub hidden_norm: RmsNormKernelConfig,
    pub fused_kv_proj: SingleGemmKernelConfig,
    pub layer_major_transpose: ElementwiseKernelConfig,
    pub grouped_k_norm: RmsNormKernelConfig,
    pub fused_rope: ElementwiseKernelConfig,
    pub context_kv_append: KvCacheAppendKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct Dflash2ContextKvLocalWorkletInput {
    /// Target rows this draft call consumes — the tokens the verify step
    /// committed, summed over the batch. Not the draft's query rows.
    pub context_tokens: u32,
}

pub struct Dflash2ContextKvLocalWorklet {
    pub name: String,
    pub aux_hidden_fc: Op<SingleGemmKernel>,
    pub hidden_norm: Op<RmsNormKernel>,
    pub fused_kv_proj: Op<SingleGemmKernel>,
    pub layer_major_transpose: Op<ElementwiseKernel>,
    pub grouped_k_norm: Op<RmsNormKernel>,
    pub fused_rope: Op<ElementwiseKernel>,
    pub context_kv_append: Op<KvCacheAppendKernel>,
    resolved: Dflash2ContextKvLocalWorkletResolved,
}

impl Dflash2ContextKvLocalWorklet {
    /// Resolve the one supported DFlash2 context-KV identity without touching a
    /// bridge, GPU, cache, or `Arc`.
    pub fn resolve_config(
        cfg: &Dflash2ContextKvLocalWorkletConfig,
    ) -> Dflash2ContextKvLocalWorkletResolved {
        validate_config(cfg).unwrap_or_else(|reason| {
            panic!("invalid Dflash2ContextKvLocalWorkletConfig: {reason}")
        });

        let kv_heads_per_rank =
            cfg.num_kv_heads.clone() / Dim::param("kv_tp", u32::from(cfg.tp_size));
        let dtype_bytes = cfg.dtype.size_bytes();
        // One rank's K (or V) width for a single layer.
        let kv_width = checked_product(
            "fused_kv_proj.kv_width",
            &[kv_heads_per_rank.get(), cfg.head_dim.get()],
        )
        .expect("validated DFlash2 KV width must fit u32");
        // `_fused_kv_weight` is `[num_layers * 2 * kv_size, hidden_size]`: one
        // GEMM emits K and V for every draft layer at once.
        let fused_kv_n = checked_product("fused_kv_proj.n", &[cfg.num_draft_layers, 2, kv_width])
            .expect("validated DFlash2 fused KV width must fit u32");
        let fc_k = checked_product(
            "aux_hidden_fc.k",
            &[cfg.num_aux_layers, cfg.hidden_dim.get()],
        )
        .expect("validated DFlash2 aux-hidden width must fit u32");
        // `.permute(2, 1, 0, 3, 4).contiguous()` reads and writes the whole
        // fused K/V block once, separating K from V into layer-major order.
        let transpose_bytes = checked_product(
            "layer_major_transpose.bytes_per_token",
            &[fused_kv_n, dtype_bytes],
        )
        .expect("validated DFlash2 transpose byte rate must fit u32");
        // `ops.rotary_embedding` is in-place over K only, plus the gathered
        // cos/sin row. It is the hand-written vLLM kernel, not an inductor
        // fusion, so the byte rate is the honest identity here — the trap
        // documented in `vllm_mla_rope` (a fusion that rewrites a whole tensor
        // to change a slice of it) does not apply.
        let rope_input_bytes = checked_product(
            "fused_rope.input_bytes_per_token",
            &[2, kv_width, dtype_bytes],
        )
        .expect("validated DFlash2 RoPE input byte rate must fit u32");
        let rope_output_bytes = checked_product(
            "fused_rope.output_bytes_per_token",
            &[kv_width, dtype_bytes],
        )
        .expect("validated DFlash2 RoPE output byte rate must fit u32");

        Dflash2ContextKvLocalWorkletResolved {
            aux_hidden_fc: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.hidden_dim.clone(),
                k: fc_k.into(),
                dtype: cfg.gemm_dtype,
            },
            hidden_norm: RmsNormKernelConfig {
                backends: cfg.norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden_dim.clone(),
                dtype: cfg.dtype,
            },
            fused_kv_proj: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: fused_kv_n.into(),
                k: cfg.hidden_dim.clone(),
                dtype: cfg.gemm_dtype,
            },
            layer_major_transpose: ElementwiseKernelConfig {
                backends: cfg.elementwise_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: transpose_bytes.into(),
                output_bytes_per_token: transpose_bytes.into(),
            },
            grouped_k_norm: RmsNormKernelConfig {
                backends: cfg.norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                // The weight is selected per layer by the outermost index, so
                // one launch normalizes `[L, num_ctx, nkv, head_dim]` with
                // `head_dim` as the reduction width.
                hidden: cfg.head_dim.clone(),
                dtype: cfg.dtype,
            },
            fused_rope: ElementwiseKernelConfig {
                backends: cfg.elementwise_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: rope_input_bytes.into(),
                output_bytes_per_token: rope_output_bytes.into(),
            },
            context_kv_append: KvCacheAppendKernelConfig {
                backends: cfg.kv_cache_append_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_kv_heads: kv_heads_per_rank,
                head_dim: cfg.head_dim.clone(),
                block_size: cfg.kv_cache_block_size,
                input_dtype: cfg.dtype,
                kv_dtype: cfg.kv_dtype,
                cache_layout: cfg.kv_cache_layout.clone(),
                scale_granularity: cfg.kv_scale_granularity.clone(),
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: Dflash2ContextKvLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let aux_hidden_fc = build_atomic(
            &name,
            "aux_hidden_fc",
            resolved.aux_hidden_fc.clone(),
            SingleGemmKernel::build,
            bridge,
        )?;
        let hidden_norm = build_atomic(
            &name,
            "hidden_norm",
            resolved.hidden_norm.clone(),
            RmsNormKernel::build,
            bridge,
        )?;
        let fused_kv_proj = build_atomic(
            &name,
            "fused_kv_proj",
            resolved.fused_kv_proj.clone(),
            SingleGemmKernel::build,
            bridge,
        )?;
        let layer_major_transpose = build_atomic(
            &name,
            "layer_major_transpose",
            resolved.layer_major_transpose.clone(),
            ElementwiseKernel::build,
            bridge,
        )?;
        let grouped_k_norm = build_atomic(
            &name,
            "grouped_k_norm",
            resolved.grouped_k_norm.clone(),
            RmsNormKernel::build,
            bridge,
        )?;
        let fused_rope = build_atomic(
            &name,
            "fused_rope",
            resolved.fused_rope.clone(),
            ElementwiseKernel::build,
            bridge,
        )?;
        let context_kv_append = build_atomic(
            &name,
            "context_kv_append",
            resolved.context_kv_append.clone(),
            KvCacheAppendKernel::build,
            bridge,
        )?;

        Ok(Self {
            name,
            aux_hidden_fc,
            hidden_norm,
            fused_kv_proj,
            layer_major_transpose,
            grouped_k_norm,
            fused_rope,
            context_kv_append,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let layers = self.resolved.raw_cfg.num_draft_layers;
        CostNode::Labeled {
            label: worklet_label(&self.name, &self.resolved.raw_cfg),
            child: Box::new(CostNode::Sum(vec![
                self.aux_hidden_fc.compile(builder),
                self.hidden_norm.compile(builder),
                self.fused_kv_proj.compile(builder),
                self.layer_major_transpose.compile(builder),
                self.grouped_k_norm.compile(builder),
                self.fused_rope.compile(builder),
                // The only per-layer step: `do_kv_cache_update` runs once per
                // draft layer, each over the same `context_tokens` rows.
                CostNode::Labeled {
                    label: format!(
                        "{}.context_kv_append [per draft layer; {layers} layers]",
                        self.name
                    ),
                    child: Box::new(CostNode::Scale {
                        n: layers,
                        child: Box::new(self.context_kv_append.compile(builder)),
                    }),
                },
            ])),
        }
    }

    pub fn eval(&self, input: &Dflash2ContextKvLocalWorkletInput, ev: &mut Evaluator) {
        let tokens = input.context_tokens;
        let zero = tokens == 0;
        let layers = self.resolved.raw_cfg.num_draft_layers;
        let kv_heads_per_rank = self.resolved.context_kv_append.num_kv_heads.get();

        eval_atomic_or_zero(
            &self.aux_hidden_fc,
            SingleGemmKernelInput { m: tokens },
            zero,
            ev,
        );
        eval_atomic_or_zero(
            &self.hidden_norm,
            RmsNormKernelInput { m: tokens },
            zero,
            ev,
        );
        eval_atomic_or_zero(
            &self.fused_kv_proj,
            SingleGemmKernelInput { m: tokens },
            zero,
            ev,
        );
        eval_atomic_or_zero(
            &self.layer_major_transpose,
            ElementwiseKernelInput { num_tokens: tokens },
            zero,
            ev,
        );
        // Grouped K-norm walks `[L, num_ctx, nkv, head_dim]`; its row count is
        // the product of everything left of the reduction width.
        eval_atomic_or_zero(
            &self.grouped_k_norm,
            RmsNormKernelInput {
                m: tokens
                    .saturating_mul(layers)
                    .saturating_mul(kv_heads_per_rank),
            },
            zero,
            ev,
        );
        // RoPE sees `[L * num_ctx, kv_size]` as one flat batch.
        eval_atomic_or_zero(
            &self.fused_rope,
            ElementwiseKernelInput {
                num_tokens: tokens.saturating_mul(layers),
            },
            zero,
            ev,
        );
        // Billed `num_draft_layers` times by the folded node above, so it is
        // evaluated once at the width one layer actually writes.
        eval_atomic_or_zero(
            &self.context_kv_append,
            KvCacheAppendKernelInput { num_tokens: tokens },
            zero,
            ev,
        );
    }
}

fn validate_config(cfg: &Dflash2ContextKvLocalWorkletConfig) -> Result<(), String> {
    for (name, actual, required) in [
        ("hidden_dim", cfg.hidden_dim.get(), HIDDEN_DIM),
        ("head_dim", cfg.head_dim.get(), HEAD_DIM),
        ("num_kv_heads", cfg.num_kv_heads.get(), NUM_KV_HEADS),
        ("num_draft_layers", cfg.num_draft_layers, NUM_DRAFT_LAYERS),
        ("num_aux_layers", cfg.num_aux_layers, NUM_AUX_LAYERS),
    ] {
        if actual != required {
            return Err(format!("{name} must be {required}, got {actual}"));
        }
    }
    if cfg.dtype != DType::Bf16 {
        return Err(format!(
            "dtype must be {}, got {}",
            DType::Bf16.as_str(),
            cfg.dtype.as_str()
        ));
    }
    if cfg.tp_size == 0 || cfg.num_kv_heads.get() % u32::from(cfg.tp_size) != 0 {
        return Err(format!(
            "num_kv_heads {} must be divisible by positive tp_size {}",
            cfg.num_kv_heads, cfg.tp_size
        ));
    }
    Ok(())
}

fn checked_product(name: &str, factors: &[u32]) -> Result<u32, String> {
    factors.iter().try_fold(1_u32, |value, &factor| {
        value
            .checked_mul(factor)
            .ok_or_else(|| format!("{name} overflows u32"))
    })
}

fn worklet_label(name: &str, cfg: &Dflash2ContextKvLocalWorkletConfig) -> String {
    format!(
        "{name} (Dflash2ContextKvLocalWorklet) \
         [rank-local; tp={}; hidden={}; draft_layers={}; kv_heads/rank={}]",
        cfg.tp_size,
        cfg.hidden_dim,
        cfg.num_draft_layers,
        cfg.num_kv_heads.get() / u32::from(cfg.tp_size)
    )
}

fn build_atomic<K, C, F>(
    parent: &str,
    suffix: &str,
    config: C,
    build: F,
    bridge: &PerfApiBridge,
) -> Result<Op<K>, BuildError>
where
    K: Probe,
    F: FnOnce(String, C, &PerfApiBridge) -> Result<K, BuildError>,
{
    let name = format!("{parent}.{suffix}");
    Ok(Op::new(
        name.clone(),
        Arc::new(build(name, config, bridge)?),
    ))
}

fn eval_atomic_or_zero<K>(op: &Op<K>, input: K::Input, zero: bool, ev: &mut Evaluator)
where
    K: Probe,
    K::Input: Clone + Into<SlotInput>,
{
    let metrics = if zero {
        LeafMetrics::ZERO
    } else {
        op.kernel.eval(&input)
    };
    ev.push(metrics, || input.clone().into());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Dflash2ContextKvLocalWorkletConfig {
        Dflash2ContextKvLocalWorkletConfig {
            norm_backends: vec!["vllm_cuda"],
            gemm_backends: vec!["torch"],
            elementwise_backends: vec!["triton"],
            kv_cache_append_backends: vec!["vllm_cuda"],
            tp_size: 4,
            gpu_name: "NVIDIA B200".to_string(),
            hidden_dim: Dim::param("hidden_dim", HIDDEN_DIM),
            num_draft_layers: NUM_DRAFT_LAYERS,
            num_aux_layers: NUM_AUX_LAYERS,
            num_kv_heads: Dim::param("num_kv_heads", NUM_KV_HEADS),
            head_dim: Dim::param("head_dim", HEAD_DIM),
            kv_cache_block_size: 64,
            kv_cache_layout: "NHD".to_string(),
            kv_scale_granularity: "tensor".to_string(),
            dtype: DType::Bf16,
            gemm_dtype: DType::Bf16,
            kv_dtype: DType::Fp8E4m3,
        }
    }

    #[test]
    fn one_gemm_emits_every_draft_layer_k_and_v() {
        let resolved = Dflash2ContextKvLocalWorklet::resolve_config(&cfg());
        // 6 layers * 2 (K and V) * (8/4 kv heads * 128 head_dim) = 3072.
        assert_eq!(resolved.fused_kv_proj.n.get(), 3072);
        assert_eq!(resolved.fused_kv_proj.k.get(), HIDDEN_DIM);
    }

    #[test]
    fn the_fc_projection_is_replicated_not_sharded() {
        // `fc` is a `ReplicatedLinear`: every rank runs the whole GEMM, so tp
        // must not divide either dimension.
        let mut wide = cfg();
        wide.tp_size = 1;
        let narrow = Dflash2ContextKvLocalWorklet::resolve_config(&cfg());
        let whole = Dflash2ContextKvLocalWorklet::resolve_config(&wide);
        assert_eq!(narrow.aux_hidden_fc.n.get(), whole.aux_hidden_fc.n.get());
        assert_eq!(narrow.aux_hidden_fc.k.get(), whole.aux_hidden_fc.k.get());
    }

    #[test]
    fn kv_projection_shards_with_tp_but_the_layer_fold_does_not() {
        let mut single = cfg();
        single.tp_size = 1;
        let sharded = Dflash2ContextKvLocalWorklet::resolve_config(&cfg());
        let whole = Dflash2ContextKvLocalWorklet::resolve_config(&single);
        assert_eq!(
            whole.fused_kv_proj.n.get(),
            sharded.fused_kv_proj.n.get() * 4
        );
        assert_eq!(sharded.context_kv_append.num_kv_heads.get(), 2);
        assert_eq!(whole.context_kv_append.num_kv_heads.get(), NUM_KV_HEADS);
    }

    #[test]
    fn a_draft_layer_count_other_than_the_checkpoint_is_rejected() {
        let mut bad = cfg();
        bad.num_draft_layers = 8;
        assert!(validate_config(&bad).is_err());
    }
}
