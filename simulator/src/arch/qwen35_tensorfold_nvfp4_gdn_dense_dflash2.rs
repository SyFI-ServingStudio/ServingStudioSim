//! Qwen3.8-27B-NVFP4 served by TensorFold on one GB10 (DGX Spark), drafting
//! with DFlash2 (`MultiDecoder.round`, `tensorfold/families/qwen3_5/cuda/`).
//!
//! One case is one TensorFold round, four kernel phases in launch order:
//!
//! ```text
//! prefill_step    the round's prefill (`multi.py` `_fill`): one piece runs
//!                 `prefill_state` (a forward per even chunk of <= 4096 rows),
//!                 two or more one batched `prefill_rows` forward; then the
//!                 drafter's prompt taps over each piece's last <= window rows
//! draft           the DFlash2 pass over every TREE stream (16 rows each)
//! verify_forward  one batched forward over every stream's window, the
//!                 sampling top-k
//! commit          GDN state replay, KV/conv copies, verify-time tap capture,
//!                 drafter taps over the accepted rows
//! ```
//!
//! Each forward is `first_norm` (layer 0's `add_rmsnorm` with no residual,
//! `rms_norm` / tensorfold_triton), GDN layer 0 (no input norm of its own),
//! `Scale{gdn - 1}` GDN, `Scale{gqa}` gated GQA, `Scale{layers}` MLP (each
//! owns its post-attention norm), then the head (final norm, lm_head,
//! sampling). The CostTree holds kernels only: host time (tree build,
//! attention plan, sampling draw) is fitted outside it (design D2,
//! `tools/tensorfold_compare/`).
//!
//! **Case lowering.** A case is one `SpeculativeArchGroupInput`:
//!
//! - `prefill_chunk_pairs` `[start, rows]` become [`PrefillPiece`]s. The case
//!   does not say whether a piece reaches its prompt's end, which decides the
//!   m = 1 lm_head and sampling of the first token. TensorFold keeps a prefix
//!   state a few rows before every prompt end (`stops`), so a finishing piece
//!   is that short tail. This arch assumes a piece **finishes iff its rows are
//!   at most `prefill_finish_max_rows`** (default 16). On the captured
//!   `tf27b` batch it labels 227 of 230 pieces right; the misses are short
//!   whole prompts with no tail. The setting is a selector field, so a
//!   comparison can move it.
//! - `decode_requests` `[kv_len, rows]` + `decode_outcomes` become
//!   [`VerifyStream`]s with `committed_len = kv_len - rows`.
//!
//! **Copy slots.** The torch copies, casts, gathers and fills and the
//! drafter's `_group_sums` / `_dconv` / `_prep` (lane tf-copies) are wired
//! from one place, [`TensorfoldCopyBackends`]: [`TensorfoldCopyBackends::measured`]
//! is what the builder uses, `default()` leaves every such slot out (the
//! kernels-only A/B). Their counts and units follow tf-copies' wiring table
//! (`logs/20261023_8_arch_design/copies_wiring.md`). The arch owns the
//! forward-level glue no worklet holds: the verify and prefill embedding
//! gathers, the verify attention plan (`_paths`), the prefill position
//! arithmetic, the prefill tap capture and the prompt taps' copies. Not
//! modeled: the drafter's `head_ids[local_ids]` gather (no row), memcpys
//! outside the phase ranges and H2D/D2H copies. The prefill K/V writes are
//! DtoD memcpys priced by the `_foreach_copy_` proxy (0.69x the capture).
//!
//! **KV layout** ([`kv_cache_layout`]) is only read by the DES, which has no
//! TensorFold worker yet; timing-predict does not use it.

use std::sync::Arc;

use crate::arch::contract::{
    SpeculativeArchGroupInput, SpeculativeArchInput, SpeculativeUnifiedModel, SpeculativeWindowMode,
};
use crate::arch::kv_layout::{KvCacheLayout, KvCachePart};
use crate::arch::qwen35_dflash2_draft_cfg::Qwen35Dflash2DraftCfg;
use crate::arch::qwen35_model_cfg::{Qwen35Mixer, Qwen35ModelCfg};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, EmbeddingRowGatherKernel,
    EmbeddingRowGatherKernelConfig, EmbeddingRowGatherKernelInput, HeadMajorQkNormRopeKernel,
    HeadMajorQkNormRopeKernelConfig, HeadMajorQkNormRopeKernelInput, RmsNormKernel,
    RmsNormKernelConfig, RmsNormKernelInput, RowGroupSumsKernel, RowGroupSumsKernelConfig,
    RowGroupSumsKernelInput, SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::tensorfold_common::{
    elementwise, even_chunks, push_or_zero, push_sum, PrefillForward, PrefillPath, PrefillPiece,
    VerifyRound, VerifyStream, WindowMode, PREFILL_CHUNK_ROWS, TENSORFOLD_TRITON,
};
use crate::worklet::tensorfold_dflash2_draft_local::{
    compile_opt, eval_opt, optional, TensorfoldDflash2DraftLocalWorklet,
    TensorfoldDflash2DraftLocalWorkletConfig, TensorfoldDflash2DraftLocalWorkletInput,
    TensorfoldDflash2DraftLocalWorkletResolved, TENSORFOLD_MAX_ROWS,
};
use crate::worklet::tensorfold_head_local::{
    TensorfoldHeadLocalWorkletConfig, TensorfoldHeadPrefillLocalWorklet,
    TensorfoldHeadPrefillLocalWorkletResolved, TensorfoldHeadVerifyLocalWorklet,
    TensorfoldHeadVerifyLocalWorkletResolved,
};
use crate::worklet::tensorfold_nvfp4_mlp_local::{
    TensorfoldNvfp4MlpLocalWorkletConfig, TensorfoldNvfp4MlpPrefillLocalWorklet,
    TensorfoldNvfp4MlpPrefillLocalWorkletResolved, TensorfoldNvfp4MlpVerifyLocalWorklet,
    TensorfoldNvfp4MlpVerifyLocalWorkletResolved,
};
use crate::worklet::tensorfold_qwen35_gdn_prefill_local::{
    TensorfoldQwen35GdnPrefillLocalWorklet, TensorfoldQwen35GdnPrefillLocalWorkletResolved,
};
use crate::worklet::tensorfold_qwen35_gdn_verify_local::{
    TensorfoldQwen35GdnLocalWorkletConfig, TensorfoldQwen35GdnVerifyLocalWorklet,
    TensorfoldQwen35GdnVerifyLocalWorkletResolved,
};
use crate::worklet::tensorfold_round_commit_local::{
    TensorfoldRoundCommitLocalWorklet, TensorfoldRoundCommitLocalWorkletConfig,
    TensorfoldRoundCommitLocalWorkletResolved,
};
use crate::worklet::{
    TensorfoldQwen35GatedGqaConfig, TensorfoldQwen35GatedGqaPrefillCopies,
    TensorfoldQwen35GatedGqaPrefillLocalWorklet, TensorfoldQwen35GatedGqaPrefillLocalWorkletConfig,
    TensorfoldQwen35GatedGqaPrefillLocalWorkletResolved,
    TensorfoldQwen35GatedGqaVerifyLocalWorklet, TensorfoldQwen35GatedGqaVerifyLocalWorkletResolved,
};

const ARCH_KIND: &str = "qwen35_tensorfold_nvfp4_gdn_dense_dflash2";
/// TensorFold's activations, attention KV and conv windows are bf16.
const ACTIVATION_DTYPE: DType = DType::Bf16;
/// The widest single-stream copy chain `CopyIndex` proposes (`multi.py`
/// copy windows); trees and every multi-stream window are capped at
/// [`TENSORFOLD_MAX_ROWS`].
pub const TENSORFOLD_COPY_CHAIN_MAX_ROWS: u32 = 128;
/// Default for [`TensorfoldParallel::prefill_finish_max_rows`].
pub const DEFAULT_PREFILL_FINISH_MAX_ROWS: u32 = 16;

// ─── copy backends (tf-copies) ───────────────────────────────────────────────

/// Backends of every TensorFold copy / glue slot (lane tf-copies), one field
/// per backend family. `None` leaves every slot of that family out of the
/// tree; a field set here wires it into every worklet (and arch-level slot)
/// that has such a slot ([`Self::apply`]).
#[derive(Clone, Debug, Default)]
pub struct TensorfoldCopyBackends {
    /// `elementwise` tensorfold_torch_cat: contiguous `torch.cat`s (GDN conv
    /// windows, prefill chain outputs and attention outputs, sampling rows,
    /// drafter and commit cats).
    pub cat: Option<Vec<&'static str>>,
    /// `elementwise` tensorfold_torch_cast: `.float()` / int casts.
    pub cast: Option<Vec<&'static str>>,
    /// `elementwise`: the prefill K/V writes (device-to-device memcpy;
    /// priced by the `_foreach_copy_` proxy).
    pub kv_write: Option<Vec<&'static str>>,
    /// `elementwise` tensorfold_torch_fp32_add: the verify tap capture.
    pub fp32_add: Option<Vec<&'static str>>,
    /// `indexed_row_copy` tensorfold_torch: `index_select` gathers.
    pub gather: Option<Vec<&'static str>>,
    /// `elementwise` tensorfold_torch_foreach_copy: `_foreach_copy_`.
    pub foreach_copy: Option<Vec<&'static str>>,
    /// `elementwise` tensorfold_torch_strided_copy: strided copies (the
    /// drafter head's row reshape, the prefill taps' window trim).
    pub strided_copy: Option<Vec<&'static str>>,
    /// `elementwise` tensorfold_torch_cat_strided (the drafter head's spans).
    pub cat_strided: Option<Vec<&'static str>>,
    /// `elementwise` tensorfold_torch: small index launches (the drafter's
    /// `head_ids[local_ids]`; no row measured, so `measured()` leaves it out).
    pub index: Option<Vec<&'static str>>,
    /// `elementwise` tensorfold_torch: the prefill positions' arange / int
    /// adds.
    pub torch: Option<Vec<&'static str>>,
    /// `elementwise` tensorfold_triton_paths: the verify attention plan.
    pub paths: Option<Vec<&'static str>>,
    /// `embedding_row_gather` (`glue.embedding`: verify, prefill, drafter).
    pub embed: Option<Vec<&'static str>>,
    /// `elementwise` tensorfold_torch_rotary (drafter and taps cos/sin).
    pub rotary: Option<Vec<&'static str>>,
    /// `rms_norm` tensorfold_torch_eager (the taps' fp32 `_norm`).
    pub eager_norm: Option<Vec<&'static str>>,
    /// `_group_sums` ahead of a qmm.
    pub group_sums: Option<Vec<&'static str>>,
    /// The drafter's `_dconv_kernel`.
    pub dconv: Option<Vec<&'static str>>,
    /// The drafter's `_prep_kernel` (pass and taps).
    pub prep: Option<Vec<&'static str>>,
}

impl TensorfoldCopyBackends {
    /// Every family tf-copies measured on GB10 (943 rows).
    pub fn measured() -> Self {
        let b = |backend: &'static str| Some(vec![backend]);
        Self {
            cat: b("tensorfold_torch_cat"),
            cast: b("tensorfold_torch_cast"),
            kv_write: b("tensorfold_torch_foreach_copy"),
            fp32_add: b("tensorfold_torch_fp32_add"),
            gather: b("tensorfold_torch"),
            foreach_copy: b("tensorfold_torch_foreach_copy"),
            strided_copy: b("tensorfold_torch_strided_copy"),
            cat_strided: b("tensorfold_torch_cat_strided"),
            index: None,
            torch: b("tensorfold_torch"),
            paths: b("tensorfold_triton_paths"),
            embed: b("tensorfold_torch"),
            rotary: b("tensorfold_torch_rotary"),
            eager_norm: b("tensorfold_torch_eager"),
            group_sums: b("tensorfold_triton"),
            dconv: b("tensorfold_triton"),
            prep: b("tensorfold_triton"),
        }
    }

    /// Forward each family's backends into the worklet configs' pending
    /// slots.
    pub fn apply(&self, configs: &mut TensorfoldConfigs) {
        for gdn in [&mut configs.gdn_first, &mut configs.gdn] {
            gdn.cat_backends = self.cat.clone();
        }
        configs.gqa_prefill.copies = TensorfoldQwen35GatedGqaPrefillCopies {
            kv_write: self.kv_write.clone(),
            attention_out_cat: self.cat.clone(),
        };
        configs.head.sample_cat_backends = self.cat.clone();
        configs.head.sample_cast_backends = self.cast.clone();
        let d = &mut configs.draft_pass.pending;
        d.embed = self.embed.clone();
        d.rotary = self.rotary.clone();
        d.cat = self.cat.clone();
        d.cast = self.cast.clone();
        d.strided_copy = self.strided_copy.clone();
        d.cat_strided = self.cat_strided.clone();
        d.index = self.index.clone();
        d.group_sums = self.group_sums.clone();
        d.dconv = self.dconv.clone();
        d.prep = self.prep.clone();
        let c = &mut configs.commit.pending;
        c.fp32_add = self.fp32_add.clone();
        c.cat = self.cat.clone();
        c.cast = self.cast.clone();
        c.gather = self.gather.clone();
        c.foreach_copy = self.foreach_copy.clone();
        c.eager_norm = self.eager_norm.clone();
        c.rotary = self.rotary.clone();
        c.group_sums = self.group_sums.clone();
        c.prep = self.prep.clone();
        configs.copies = self.clone();
    }
}

// ─── configs ─────────────────────────────────────────────────────────────────

/// Deployment facts: one GB10, TensorFold's round scheduler.
#[derive(Clone, Debug)]
pub struct TensorfoldParallel {
    pub gpu_name: String,
    pub max_model_len: u32,
    /// The candidate budget (`speculative_iter` cases carry it); TensorFold's
    /// windows are ragged, so it does not fix a verify width.
    pub draft_tokens: u32,
    /// The requests' `top_k` (sampling selects `top_k + 8` candidates).
    pub sample_top_k: u32,
    /// A prefill piece of at most this many rows is taken to finish its
    /// prompt (see the module docs).
    pub prefill_finish_max_rows: u32,
}

#[derive(Clone, Debug)]
pub struct TensorfoldConfigs {
    pub model: Qwen35ModelCfg,
    pub draft: Qwen35Dflash2DraftCfg,
    pub parallel: TensorfoldParallel,
    /// GDN layers after layer 0, and gated-GQA layers.
    pub gdn_rest_layers: u32,
    pub gqa_layers: u32,
    /// Layer 0's `add_rmsnorm(x, None, w)`: no residual.
    pub first_norm: RmsNormKernelConfig,
    /// GDN layer 0 (`residual_input_norm = false`).
    pub gdn_first: TensorfoldQwen35GdnLocalWorkletConfig,
    pub gdn: TensorfoldQwen35GdnLocalWorkletConfig,
    pub gqa: TensorfoldQwen35GatedGqaConfig,
    pub gqa_prefill: TensorfoldQwen35GatedGqaPrefillLocalWorkletConfig,
    pub mlp: TensorfoldNvfp4MlpLocalWorkletConfig,
    pub head: TensorfoldHeadLocalWorkletConfig,
    pub draft_pass: TensorfoldDflash2DraftLocalWorkletConfig,
    pub commit: TensorfoldRoundCommitLocalWorkletConfig,
    /// The backends of the arch-level glue slots ([`TensorfoldGlueResolved`]).
    pub copies: TensorfoldCopyBackends,
}

fn fit_failed(reason: impl Into<String>) -> BuildError {
    BuildError::FitFailed {
        kind: ARCH_KIND,
        reason: reason.into(),
    }
}

fn gdn_config(
    model: &Qwen35ModelCfg,
    gpu: &str,
    residual_input_norm: bool,
) -> TensorfoldQwen35GdnLocalWorkletConfig {
    let p = &model.precisions;
    TensorfoldQwen35GdnLocalWorkletConfig {
        hidden: model.hidden,
        num_key_heads: model.gdn_key_heads,
        num_value_heads: model.gdn_value_heads,
        key_head_dim: model.gdn_key_head_dim,
        value_head_dim: model.gdn_value_head_dim,
        conv_kernel_size: model.gdn_conv_kernel,
        activation_dtype: ACTIVATION_DTYPE,
        ssm_state_dtype: model.gdn_state_dtype,
        proj_dtype: p.gdn_in_proj_qkvz.gemm_dtype(),
        ba_dtype: p.gdn_in_proj_ba.gemm_dtype(),
        gpu_name: gpu.to_string(),
        residual_input_norm,
        cat_backends: None,
    }
}

/// Every config with all copy slots absent; [`TensorfoldCopyBackends::apply`]
/// fills them.
pub fn build_configs(
    model: &Qwen35ModelCfg,
    draft: &Qwen35Dflash2DraftCfg,
    parallel: &TensorfoldParallel,
) -> Result<TensorfoldConfigs, BuildError> {
    let p = &model.precisions;
    if p.gdn_in_proj_qkvz != p.gdn_out_proj || p.gqa_qkv_proj != p.gqa_o_proj {
        // One GDN (GQA) worklet costs every projection of the layer at one
        // precision pair; the checkpoint must agree.
        return Err(fit_failed(format!(
            "TensorFold models one precision for GDN qkvz/out and GQA qkv/o, got {p:?}"
        )));
    }
    if model.mixers.first() != Some(&Qwen35Mixer::Gdn) {
        return Err(fit_failed(
            "layer 0 must be a GDN layer (its no-residual norm is priced by the arch)",
        ));
    }
    if parallel.draft_tokens == 0 {
        return Err(fit_failed("draft_tokens must be positive"));
    }
    let count = |mixer| model.mixers.iter().filter(|&&m| m == mixer).count() as u32;
    let gpu = parallel.gpu_name.as_str();
    let gqa = TensorfoldQwen35GatedGqaConfig::from_model(model, gpu);
    Ok(TensorfoldConfigs {
        model: model.clone(),
        draft: draft.clone(),
        parallel: parallel.clone(),
        gdn_rest_layers: count(Qwen35Mixer::Gdn) - 1,
        gqa_layers: count(Qwen35Mixer::Gqa),
        first_norm: RmsNormKernelConfig {
            backends: vec![TENSORFOLD_TRITON],
            gpu_name: gpu.to_string(),
            hidden: model.hidden.into(),
            dtype: ACTIVATION_DTYPE,
        },
        gdn_first: gdn_config(model, gpu, false),
        gdn: gdn_config(model, gpu, true),
        gqa_prefill: TensorfoldQwen35GatedGqaPrefillLocalWorkletConfig {
            gqa: gqa.clone(),
            copies: TensorfoldQwen35GatedGqaPrefillCopies::default(),
        },
        gqa,
        mlp: TensorfoldNvfp4MlpLocalWorkletConfig::from_model(model, gpu),
        head: TensorfoldHeadLocalWorkletConfig::from_model(model, gpu, parallel.sample_top_k),
        draft_pass: TensorfoldDflash2DraftLocalWorkletConfig::new(draft, model, gpu),
        commit: TensorfoldRoundCommitLocalWorkletConfig::new(draft, model, gpu),
        copies: TensorfoldCopyBackends::default(),
    })
}

/// The prefill taps' kernels: the commit's taps configs plus the
/// single-stream path's own cat and trim.
#[derive(Clone, Debug)]
pub struct TensorfoldPrefillTapsResolved {
    pub draft_layers: u32,
    /// Context rows the drafter keeps (`sliding_window - 1`).
    pub window: u32,
    pub fc_sums: Option<RowGroupSumsKernelConfig>,
    pub fc: SingleGemmKernelConfig,
    pub fc_norm: Option<RmsNormKernelConfig>,
    pub rotary: Option<ElementwiseKernelConfig>,
    pub kv_sums: Option<RowGroupSumsKernelConfig>,
    pub kv: SingleGemmKernelConfig,
    pub kv_prep: Option<HeadMajorQkNormRopeKernelConfig>,
    /// `cat((kc, k), dim=1)` of one layer's K (then V): the context rows and
    /// the new ones, one bf16 kv row a unit.
    pub kv_cat: Option<ElementwiseKernelConfig>,
    /// `kc[:, -window:].contiguous()` of K (then V) once the window is full.
    pub kv_trim: Option<ElementwiseKernelConfig>,
}

/// The forward-level glue no worklet holds (tf-copies' wiring table).
#[derive(Clone, Debug, Default)]
pub struct TensorfoldGlueResolved {
    /// Verify `glue.embedding` over W rows.
    pub verify_embed: Option<EmbeddingRowGatherKernelConfig>,
    /// Verify `_paths` (the tree-attention plan) over W rows: 4 / 516 bytes.
    pub verify_paths: Option<ElementwiseKernelConfig>,
    /// Prefill `glue.embedding`, once a forward.
    pub prefill_embed: Option<EmbeddingRowGatherKernelConfig>,
    /// Prefill positions: about three int launches a forward (0 / 4 bytes).
    pub prefill_positions: Option<ElementwiseKernelConfig>,
    /// Prefill tap capture `(x.float() + pending.float()).to(bf16)` at each
    /// tapped layer, over a tapped forward's rows.
    pub prefill_tap_add: Option<ElementwiseKernelConfig>,
    /// The `torch.cat(taps, -1)` of a tapped forward.
    pub prefill_tap_cat: Option<ElementwiseKernelConfig>,
    pub num_taps: u32,
}

#[derive(Clone, Debug)]
pub struct TensorfoldResolved {
    pub parallel: TensorfoldParallel,
    pub gdn_rest_layers: u32,
    pub gqa_layers: u32,
    pub num_layers: u32,
    pub first_norm: RmsNormKernelConfig,
    pub gdn_first_verify: TensorfoldQwen35GdnVerifyLocalWorkletResolved,
    pub gdn_verify: TensorfoldQwen35GdnVerifyLocalWorkletResolved,
    pub gqa_verify: TensorfoldQwen35GatedGqaVerifyLocalWorkletResolved,
    pub mlp_verify: TensorfoldNvfp4MlpVerifyLocalWorkletResolved,
    pub head_verify: TensorfoldHeadVerifyLocalWorkletResolved,
    pub gdn_first_prefill: TensorfoldQwen35GdnPrefillLocalWorkletResolved,
    pub gdn_prefill: TensorfoldQwen35GdnPrefillLocalWorkletResolved,
    pub gqa_prefill: TensorfoldQwen35GatedGqaPrefillLocalWorkletResolved,
    pub mlp_prefill: TensorfoldNvfp4MlpPrefillLocalWorkletResolved,
    pub head_prefill: TensorfoldHeadPrefillLocalWorkletResolved,
    pub prefill_taps: TensorfoldPrefillTapsResolved,
    pub glue: TensorfoldGlueResolved,
    pub draft_pass: TensorfoldDflash2DraftLocalWorkletResolved,
    pub commit: TensorfoldRoundCommitLocalWorkletResolved,
    pub kv_cache_layout: KvCacheLayout,
}

pub fn resolve_configs(cfgs: &TensorfoldConfigs) -> TensorfoldResolved {
    let commit = TensorfoldRoundCommitLocalWorklet::resolve_config(&cfgs.commit);
    let gpu = cfgs.parallel.gpu_name.as_str();
    let kv_row =
        cfgs.draft.num_key_value_heads * cfgs.draft.head_dim * ACTIVATION_DTYPE.size_bytes();
    let c = &cfgs.copies;
    let row_pass = |backends: &Option<Vec<&'static str>>| {
        backends
            .as_ref()
            .map(|b| elementwise(b, gpu, kv_row, kv_row))
    };
    let pass = |backends: &Option<Vec<&'static str>>, input: u32, output: u32| {
        backends
            .as_ref()
            .map(|b| elementwise(b, gpu, input, output))
    };
    let hidden_row = cfgs.model.hidden * ACTIVATION_DTYPE.size_bytes();
    let tap_row = cfgs.draft.tap_width() * ACTIVATION_DTYPE.size_bytes();
    let embed = c.embed.as_ref().map(|b| EmbeddingRowGatherKernelConfig {
        backends: b.clone(),
        gpu_name: gpu.to_string(),
        num_heads: 1,
        head_dim: cfgs.model.hidden,
        table_rows: u64::from(cfgs.model.vocab_size),
        residency: "device".to_string(),
        weight_dtype: ACTIVATION_DTYPE,
    });
    TensorfoldResolved {
        parallel: cfgs.parallel.clone(),
        gdn_rest_layers: cfgs.gdn_rest_layers,
        gqa_layers: cfgs.gqa_layers,
        num_layers: cfgs.model.num_layers(),
        first_norm: cfgs.first_norm.clone(),
        gdn_first_verify: TensorfoldQwen35GdnVerifyLocalWorklet::resolve_config(&cfgs.gdn_first),
        gdn_verify: TensorfoldQwen35GdnVerifyLocalWorklet::resolve_config(&cfgs.gdn),
        gqa_verify: TensorfoldQwen35GatedGqaVerifyLocalWorklet::resolve_config(&cfgs.gqa),
        mlp_verify: TensorfoldNvfp4MlpVerifyLocalWorklet::resolve_config(&cfgs.mlp),
        head_verify: TensorfoldHeadVerifyLocalWorklet::resolve_config(&cfgs.head),
        gdn_first_prefill: TensorfoldQwen35GdnPrefillLocalWorklet::resolve_config(&cfgs.gdn_first),
        gdn_prefill: TensorfoldQwen35GdnPrefillLocalWorklet::resolve_config(&cfgs.gdn),
        gqa_prefill: TensorfoldQwen35GatedGqaPrefillLocalWorklet::resolve_config(&cfgs.gqa_prefill),
        mlp_prefill: TensorfoldNvfp4MlpPrefillLocalWorklet::resolve_config(&cfgs.mlp),
        head_prefill: TensorfoldHeadPrefillLocalWorklet::resolve_config(&cfgs.head),
        prefill_taps: TensorfoldPrefillTapsResolved {
            draft_layers: cfgs.draft.num_hidden_layers,
            window: cfgs.draft.window(),
            fc_sums: commit.fc_sums.clone(),
            fc: commit.fc.clone(),
            fc_norm: commit.fc_norm.clone(),
            rotary: commit.taps_rotary.clone(),
            kv_sums: commit.kv_sums.clone(),
            kv: commit.kv.clone(),
            kv_prep: commit.kv_prep.clone(),
            kv_cat: row_pass(&c.cat),
            kv_trim: row_pass(&c.strided_copy),
        },
        glue: TensorfoldGlueResolved {
            verify_embed: embed.clone(),
            // int32 row in; the per-row plan entry out (tf-copies' 4 / 516).
            verify_paths: pass(&c.paths, 4, 516),
            prefill_embed: embed,
            prefill_positions: pass(&c.torch, 0, 4),
            // Two bf16 rows in, one out.
            prefill_tap_add: pass(&c.fp32_add, 2 * hidden_row, hidden_row),
            prefill_tap_cat: pass(&c.cat, tap_row, tap_row),
            num_taps: cfgs.draft.num_taps(),
        },
        draft_pass: TensorfoldDflash2DraftLocalWorklet::resolve_config(&cfgs.draft_pass),
        commit,
        kv_cache_layout: kv_cache_layout(&cfgs.model, &cfgs.draft),
    }
}

/// One request's resident state on the GB10, as TensorFold allocates it.
/// Only the DES reads this (no TensorFold worker exists yet); timing-predict
/// does not.
///
/// - `gqa_kv`: bf16 K and V of every gated-GQA layer, per token (Qwen3.8-27B:
///   16 x 4 heads x 256 x 2 x 2 B = 64 KiB).
/// - `gdn_state`: the fp32 recurrent state of every GDN layer, per request
///   (48 x 48 heads x 128 x 128 x 4 B).
/// - `gdn_conv`: the bf16 conv window (`conv_kernel - 1` rows of the q|k|v
///   channels) of every GDN layer, per request.
/// - `dflash2_kv`: the drafter's bf16 K and V over its kept window
///   (`sliding_window - 1` rows), every drafter layer.
pub fn kv_cache_layout(model: &Qwen35ModelCfg, draft: &Qwen35Dflash2DraftCfg) -> KvCacheLayout {
    let bf16 = u64::from(ACTIVATION_DTYPE.size_bytes());
    let count = |mixer| model.mixers.iter().filter(|&&m| m == mixer).count() as u64;
    let (gdn, gqa) = (count(Qwen35Mixer::Gdn), count(Qwen35Mixer::Gqa));
    let gqa_kv = gqa * u64::from(model.gqa_kv_heads) * u64::from(model.gqa_head_dim) * 2 * bf16;
    let gdn_state = gdn
        * u64::from(model.gdn_value_heads)
        * u64::from(model.gdn_key_head_dim)
        * u64::from(model.gdn_value_head_dim)
        * u64::from(model.gdn_state_dtype.size_bytes());
    let gdn_conv =
        gdn * u64::from(model.gdn_conv_kernel - 1) * u64::from(model.gdn_conv_channels()) * bf16;
    let draft_kv = u64::from(draft.num_hidden_layers)
        * u64::from(draft.num_key_value_heads)
        * u64::from(draft.head_dim)
        * 2
        * bf16;
    KvCacheLayout::new()
        .with_part(KvCachePart::per_token("gqa_kv", gqa_kv))
        .with_part(KvCachePart::per_request("gdn_state", gdn_state))
        .with_part(KvCachePart::per_request("gdn_conv", gdn_conv))
        .with_part(KvCachePart::sliding_window(
            "dflash2_kv",
            draft.window(),
            draft_kv,
        ))
}

// ─── the prefill taps ────────────────────────────────────────────────────────

/// One `DFlash2._add_taps` call of a prompt's prefill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillTapCall {
    /// Tapped rows this call projects (`n`).
    pub rows: u32,
    /// Drafter context rows the layer windows already hold (`n_old`): 0 when
    /// the prefill skipped ahead to the prompt's last window (`draft.skip`
    /// clears the windows), else the stream's context, taken as
    /// `min(window, prefix_len)`.
    pub context: u32,
}

/// `DFlash2._add_taps` (fast path) for one prompt's prefill: each call
/// projects one run of tapped rows into every drafter layer's window. Per
/// call: `_group_sums` + fc qmm, the fp32 `_norm`, the cos/sin rotary; per
/// drafter layer `_group_sums` + kv qmm, `_prep`, and for K then V a
/// `torch.cat` of the old window and the new rows (when a window exists) and
/// the window's `.contiguous()` trim (when it overflows). No `_append` (that
/// is the multi-stream commit path).
struct PrefillTaps {
    draft_layers: u32,
    window: u32,
    fc_sums: Option<Op<RowGroupSumsKernel>>,
    fc: Op<SingleGemmKernel>,
    fc_norm: Option<Op<RmsNormKernel>>,
    rotary: Option<Op<ElementwiseKernel>>,
    kv_sums: Option<Op<RowGroupSumsKernel>>,
    kv: Op<SingleGemmKernel>,
    kv_prep: Option<Op<HeadMajorQkNormRopeKernel>>,
    kv_cat: Option<Op<ElementwiseKernel>>,
    kv_trim: Option<Op<ElementwiseKernel>>,
}

impl PrefillTaps {
    fn build(
        name: &str,
        r: &TensorfoldPrefillTapsResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let pass = |suffix: &str, cfg: &Option<ElementwiseKernelConfig>| {
            optional(name, suffix, cfg.clone(), ElementwiseKernel::build, bridge)
        };
        let sums = |suffix: &str, cfg: &Option<RowGroupSumsKernelConfig>| {
            optional(name, suffix, cfg.clone(), RowGroupSumsKernel::build, bridge)
        };
        let gemm = |suffix: &str, cfg: &SingleGemmKernelConfig| {
            let full = format!("{name}.{suffix}");
            Ok::<_, BuildError>(Op::new(
                full.clone(),
                Arc::new(SingleGemmKernel::build(full, cfg.clone(), bridge)?),
            ))
        };
        Ok(Self {
            draft_layers: r.draft_layers,
            window: r.window,
            fc_sums: sums("fc_sums", &r.fc_sums)?,
            fc: gemm("fc", &r.fc)?,
            fc_norm: optional(
                name,
                "fc_norm",
                r.fc_norm.clone(),
                RmsNormKernel::build,
                bridge,
            )?,
            rotary: pass("rotary", &r.rotary)?,
            kv_sums: sums("kv_sums", &r.kv_sums)?,
            kv: gemm("kv", &r.kv)?,
            kv_prep: optional(
                name,
                "kv_prep",
                r.kv_prep.clone(),
                HeadMajorQkNormRopeKernel::build,
                bridge,
            )?,
            kv_cat: pass("kv_cat", &r.kv_cat)?,
            kv_trim: pass("kv_trim", &r.kv_trim)?,
        })
    }

    fn compile(&self, name: &str, builder: &mut CostTreeBuilder) -> CostNode {
        let mut children = Vec::new();
        children.extend(compile_opt(&self.fc_sums, builder));
        children.push(self.fc.compile(builder));
        children.extend(compile_opt(&self.fc_norm, builder));
        children.extend(compile_opt(&self.rotary, builder));
        let mut layer = Vec::new();
        layer.extend(compile_opt(&self.kv_sums, builder));
        layer.push(self.kv.compile(builder));
        layer.extend(compile_opt(&self.kv_prep, builder));
        for op in [&self.kv_cat, &self.kv_trim].into_iter().flatten() {
            // K, then V.
            layer.push(CostNode::Scale {
                n: 2,
                child: Box::new(op.compile(builder)),
            });
        }
        children.push(CostNode::Scale {
            n: self.draft_layers,
            child: Box::new(CostNode::Sum(layer)),
        });
        CostNode::Labeled {
            label: format!(
                "{name} [DFlash2 prompt taps; one add_taps per tapped run; window={}; \
                 draft layers={}]",
                self.window, self.draft_layers
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    /// The `add_taps` calls the step makes: each forward launch's rows that
    /// fall in its piece's last `window` rows (`prefill.py` `tap_from = stop -
    /// window`). A state-path chunk holds more than `window` rows, so a
    /// piece makes one call. Not modeled: a `keep_at` cut splitting a call in
    /// two.
    fn calls(&self, step: &PrefillForward) -> Vec<PrefillTapCall> {
        let call = |piece: &PrefillPiece, rows: u32| PrefillTapCall {
            rows,
            // Rows before `tap_from` make the prefill `skip` (clearing the
            // windows); otherwise the stream's earlier taps are its context.
            context: if piece.rows > self.window {
                0
            } else {
                piece.prefix_len.min(self.window)
            },
        };
        match step.path() {
            None => Vec::new(),
            Some(PrefillPath::State) => {
                let piece = step.pieces[0];
                let stop = piece.prefix_len + piece.rows;
                let tap_from = stop.saturating_sub(self.window).max(piece.prefix_len);
                even_chunks(piece.prefix_len, stop, PREFILL_CHUNK_ROWS)
                    .into_iter()
                    .filter(|&(_, b)| b > tap_from)
                    .map(|(a, b)| call(&piece, b - a.max(tap_from)))
                    .collect()
            }
            Some(PrefillPath::Rows) => step
                .pieces
                .iter()
                .map(|piece| call(piece, piece.rows.min(self.window)))
                .collect(),
        }
    }

    fn eval(&self, step: &PrefillForward, ev: &mut Evaluator) {
        let calls = self.calls(step);
        let rows = |f: &dyn Fn(u32) -> ElementwiseKernelInput| -> Vec<ElementwiseKernelInput> {
            calls.iter().map(|c| f(c.rows)).collect()
        };
        let ew = rows(&|num_tokens| ElementwiseKernelInput { num_tokens });
        let sums: Vec<_> = calls
            .iter()
            .map(|c| RowGroupSumsKernelInput { num_tokens: c.rows })
            .collect();
        let gemm: Vec<_> = calls
            .iter()
            .map(|c| SingleGemmKernelInput { m: c.rows })
            .collect();
        if let Some(op) = &self.fc_sums {
            push_sum(op, &sums, ev);
        }
        push_sum(&self.fc, &gemm, ev);
        if let Some(op) = &self.fc_norm {
            let norm: Vec<_> = calls
                .iter()
                .map(|c| RmsNormKernelInput { m: c.rows })
                .collect();
            push_sum(op, &norm, ev);
        }
        if let Some(op) = &self.rotary {
            push_sum(op, &ew, ev);
        }
        if let Some(op) = &self.kv_sums {
            push_sum(op, &sums, ev);
        }
        push_sum(&self.kv, &gemm, ev);
        if let Some(op) = &self.kv_prep {
            let prep: Vec<_> = calls
                .iter()
                .map(|c| HeadMajorQkNormRopeKernelInput { num_tokens: c.rows })
                .collect();
            push_sum(op, &prep, ev);
        }
        if let Some(op) = &self.kv_cat {
            let cats: Vec<_> = calls
                .iter()
                .filter(|c| c.context > 0)
                .map(|c| ElementwiseKernelInput {
                    num_tokens: c.context + c.rows,
                })
                .collect();
            push_sum(op, &cats, ev);
        }
        if let Some(op) = &self.kv_trim {
            let trims: Vec<_> = calls
                .iter()
                .filter(|c| c.context + c.rows > self.window)
                .map(|_| ElementwiseKernelInput {
                    num_tokens: self.window,
                })
                .collect();
            push_sum(op, &trims, ev);
        }
    }
}

// ─── forward-level glue ──────────────────────────────────────────────────────

/// The glue launches of the verify and prefill forwards that no worklet
/// holds ([`TensorfoldGlueResolved`]).
struct Glue {
    num_taps: u32,
    window: u32,
    verify_embed: Option<Op<EmbeddingRowGatherKernel>>,
    verify_paths: Option<Op<ElementwiseKernel>>,
    prefill_embed: Option<Op<EmbeddingRowGatherKernel>>,
    prefill_positions: Option<Op<ElementwiseKernel>>,
    prefill_tap_add: Option<Op<ElementwiseKernel>>,
    prefill_tap_cat: Option<Op<ElementwiseKernel>>,
}

/// Prefill position launches a forward (`arange` and int adds).
const PREFILL_POSITION_LAUNCHES: u32 = 3;

impl Glue {
    fn build(
        verify: &str,
        prefill: &str,
        r: &TensorfoldGlueResolved,
        window: u32,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let pass = |prefix: &str, suffix: &str, cfg: &Option<ElementwiseKernelConfig>| {
            optional(
                prefix,
                suffix,
                cfg.clone(),
                ElementwiseKernel::build,
                bridge,
            )
        };
        let embed = |prefix: &str| {
            optional(
                prefix,
                "embed",
                r.verify_embed.clone(),
                EmbeddingRowGatherKernel::build,
                bridge,
            )
        };
        Ok(Self {
            num_taps: r.num_taps,
            window,
            verify_embed: embed(verify)?,
            verify_paths: pass(verify, "paths", &r.verify_paths)?,
            prefill_embed: embed(prefill)?,
            prefill_positions: pass(prefill, "positions", &r.prefill_positions)?,
            prefill_tap_add: pass(prefill, "tap_add", &r.prefill_tap_add)?,
            prefill_tap_cat: pass(prefill, "tap_cat", &r.prefill_tap_cat)?,
        })
    }

    fn compile_prefill(&self, builder: &mut CostTreeBuilder) -> Vec<CostNode> {
        let mut nodes = Vec::new();
        nodes.extend(compile_opt(&self.prefill_embed, builder));
        if let Some(op) = &self.prefill_positions {
            nodes.push(CostNode::Scale {
                n: PREFILL_POSITION_LAUNCHES,
                child: Box::new(op.compile(builder)),
            });
        }
        if let Some(op) = &self.prefill_tap_add {
            nodes.push(CostNode::Scale {
                n: self.num_taps,
                child: Box::new(op.compile(builder)),
            });
        }
        nodes.extend(compile_opt(&self.prefill_tap_cat, builder));
        nodes
    }

    fn compile_verify(&self, builder: &mut CostTreeBuilder) -> Vec<CostNode> {
        let mut nodes = Vec::new();
        nodes.extend(compile_opt(&self.verify_embed, builder));
        nodes.extend(compile_opt(&self.verify_paths, builder));
        nodes
    }

    /// Rows of each forward that captures taps: a state-path chunk reaching
    /// the piece's last window, or the whole batched forward.
    fn tapped_forwards(&self, step: &PrefillForward) -> Vec<u32> {
        match step.path() {
            None => Vec::new(),
            Some(PrefillPath::State) => {
                let piece = step.pieces[0];
                let stop = piece.prefix_len + piece.rows;
                let tap_from = stop.saturating_sub(self.window).max(piece.prefix_len);
                even_chunks(piece.prefix_len, stop, PREFILL_CHUNK_ROWS)
                    .into_iter()
                    .filter(|&(_, b)| b > tap_from)
                    .map(|(a, b)| b - a)
                    .collect()
            }
            Some(PrefillPath::Rows) => vec![step.total_rows()],
        }
    }

    fn eval_prefill(&self, step: &PrefillForward, ev: &mut Evaluator) {
        let forwards = step.forward_rows();
        if let Some(op) = &self.prefill_embed {
            let inputs: Vec<_> = forwards
                .iter()
                .map(|&num_tokens| EmbeddingRowGatherKernelInput { num_tokens })
                .collect();
            push_sum(op, &inputs, ev);
        }
        let ew = |rows: &[u32]| -> Vec<ElementwiseKernelInput> {
            rows.iter()
                .map(|&num_tokens| ElementwiseKernelInput { num_tokens })
                .collect()
        };
        if let Some(op) = &self.prefill_positions {
            push_sum(op, &ew(&forwards), ev);
        }
        let tapped = self.tapped_forwards(step);
        for op in [&self.prefill_tap_add, &self.prefill_tap_cat]
            .into_iter()
            .flatten()
        {
            push_sum(op, &ew(&tapped), ev);
        }
    }

    fn eval_verify(&self, round: &VerifyRound, ev: &mut Evaluator) {
        let (rows, empty) = (round.total_rows(), round.is_empty());
        eval_opt(
            &self.verify_embed,
            EmbeddingRowGatherKernelInput { num_tokens: rows },
            empty,
            ev,
        );
        eval_opt(
            &self.verify_paths,
            ElementwiseKernelInput { num_tokens: rows },
            empty,
            ev,
        );
    }
}

// ─── the model ───────────────────────────────────────────────────────────────

/// The layers of one forward kind (verify or prefill), folded by type.
struct Forward<G, Q, M, H> {
    first_norm: Op<RmsNormKernel>,
    gdn_first: G,
    gdn: G,
    gqa: Q,
    mlp: M,
    head: H,
}

pub struct Qwen35TensorfoldNvfp4GdnDenseDflash2Model {
    name: String,
    parallel: TensorfoldParallel,
    gdn_rest_layers: u32,
    gqa_layers: u32,
    num_layers: u32,
    draft_window: u32,
    verify: Forward<
        TensorfoldQwen35GdnVerifyLocalWorklet,
        TensorfoldQwen35GatedGqaVerifyLocalWorklet,
        TensorfoldNvfp4MlpVerifyLocalWorklet,
        TensorfoldHeadVerifyLocalWorklet,
    >,
    prefill: Forward<
        TensorfoldQwen35GdnPrefillLocalWorklet,
        TensorfoldQwen35GatedGqaPrefillLocalWorklet,
        TensorfoldNvfp4MlpPrefillLocalWorklet,
        TensorfoldHeadPrefillLocalWorklet,
    >,
    prefill_taps: PrefillTaps,
    glue: Glue,
    draft: TensorfoldDflash2DraftLocalWorklet,
    commit: TensorfoldRoundCommitLocalWorklet,
    kv_cache_layout: KvCacheLayout,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

fn first_norm_op(
    name: String,
    cfg: &RmsNormKernelConfig,
    bridge: &PerfApiBridge,
) -> Result<Op<RmsNormKernel>, BuildError> {
    Ok(Op::new(
        name.clone(),
        Arc::new(RmsNormKernel::build(name, cfg.clone(), bridge)?),
    ))
}

pub fn build(
    name: String,
    r: TensorfoldResolved,
    bridge: &PerfApiBridge,
) -> Result<Qwen35TensorfoldNvfp4GdnDenseDflash2Model, BuildError> {
    let v = format!("{name}.verify");
    let p = format!("{name}.prefill");
    let verify = Forward {
        first_norm: first_norm_op(format!("{v}.first_norm"), &r.first_norm, bridge)?,
        gdn_first: TensorfoldQwen35GdnVerifyLocalWorklet::build(
            format!("{v}.gdn_first"),
            r.gdn_first_verify.clone(),
            bridge,
        )?,
        gdn: TensorfoldQwen35GdnVerifyLocalWorklet::build(
            format!("{v}.gdn"),
            r.gdn_verify.clone(),
            bridge,
        )?,
        gqa: TensorfoldQwen35GatedGqaVerifyLocalWorklet::build(
            format!("{v}.gqa"),
            r.gqa_verify.clone(),
            bridge,
        )?,
        mlp: TensorfoldNvfp4MlpVerifyLocalWorklet::build(
            format!("{v}.mlp"),
            r.mlp_verify.clone(),
            bridge,
        )?,
        head: TensorfoldHeadVerifyLocalWorklet::build(
            format!("{v}.head"),
            r.head_verify.clone(),
            bridge,
        )?,
    };
    let prefill = Forward {
        first_norm: first_norm_op(format!("{p}.first_norm"), &r.first_norm, bridge)?,
        gdn_first: TensorfoldQwen35GdnPrefillLocalWorklet::build(
            format!("{p}.gdn_first"),
            r.gdn_first_prefill.clone(),
            bridge,
        )?,
        gdn: TensorfoldQwen35GdnPrefillLocalWorklet::build(
            format!("{p}.gdn"),
            r.gdn_prefill.clone(),
            bridge,
        )?,
        gqa: TensorfoldQwen35GatedGqaPrefillLocalWorklet::build(
            format!("{p}.gqa"),
            r.gqa_prefill.clone(),
            bridge,
        )?,
        mlp: TensorfoldNvfp4MlpPrefillLocalWorklet::build(
            format!("{p}.mlp"),
            r.mlp_prefill.clone(),
            bridge,
        )?,
        head: TensorfoldHeadPrefillLocalWorklet::build(
            format!("{p}.head"),
            r.head_prefill.clone(),
            bridge,
        )?,
    };
    let prefill_taps = PrefillTaps::build(&format!("{p}.taps"), &r.prefill_taps, bridge)?;
    let glue = Glue::build(&v, &p, &r.glue, r.prefill_taps.window, bridge)?;
    let draft = TensorfoldDflash2DraftLocalWorklet::build(
        format!("{name}.draft"),
        r.draft_pass.clone(),
        bridge,
    )?;
    let commit = TensorfoldRoundCommitLocalWorklet::build(
        format!("{name}.commit"),
        r.commit.clone(),
        bridge,
    )?;
    let mut model = Qwen35TensorfoldNvfp4GdnDenseDflash2Model {
        name,
        gdn_rest_layers: r.gdn_rest_layers,
        gqa_layers: r.gqa_layers,
        num_layers: r.num_layers,
        draft_window: r.draft_pass.raw_cfg.window,
        parallel: r.parallel,
        verify,
        prefill,
        prefill_taps,
        glue,
        draft,
        commit,
        kv_cache_layout: r.kv_cache_layout,
        cost_flat: Vec::new(),
        n_slots: 0,
    };
    let tree = model.cost_tree();
    model.cost_flat = tree.flatten();
    model.n_slots = tree.n_slots();
    Ok(model)
}

/// One case's TensorFold round.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TensorfoldRound {
    pub prefill: PrefillForward,
    pub verify: VerifyRound,
}

fn window_mode(mode: SpeculativeWindowMode) -> WindowMode {
    match mode {
        SpeculativeWindowMode::Copy => WindowMode::Copy,
        SpeculativeWindowMode::Tree => WindowMode::Tree,
        SpeculativeWindowMode::One => WindowMode::One,
    }
}

impl Qwen35TensorfoldNvfp4GdnDenseDflash2Model {
    fn scaled(n: u32, child: CostNode) -> CostNode {
        CostNode::Scale {
            n,
            child: Box::new(child),
        }
    }

    fn cost_tree(&self) -> CostTree {
        let mut b = CostTreeBuilder::new();
        let (gdn, gqa, layers) = (self.gdn_rest_layers, self.gqa_layers, self.num_layers);
        let p = &self.prefill;
        let prefill = CostNode::Labeled {
            label: format!(
                "{}.prefill [prefill step: prefill_state chunks of <= {PREFILL_CHUNK_ROWS} rows \
                 or one prefill_rows forward; gdn 1+{gdn}, gqa {gqa}, mlp {layers}]",
                self.name
            ),
            child: Box::new(CostNode::Sum({
                let mut nodes = self.glue.compile_prefill(&mut b);
                nodes.extend([
                    p.first_norm.compile(&mut b),
                    p.gdn_first.compile(&mut b),
                    Self::scaled(gdn, p.gdn.compile(&mut b)),
                    Self::scaled(gqa, p.gqa.compile(&mut b)),
                    Self::scaled(layers, p.mlp.compile(&mut b)),
                    p.head.compile(&mut b),
                    self.prefill_taps
                        .compile(&format!("{}.prefill.taps", self.name), &mut b),
                ]);
                nodes
            })),
        };
        let draft = CostNode::Labeled {
            label: format!(
                "{}.draft [DFlash2 pass over TREE streams; {TENSORFOLD_MAX_ROWS} rows each]",
                self.name
            ),
            child: Box::new(self.draft.compile(&mut b)),
        };
        let v = &self.verify;
        let verify = CostNode::Labeled {
            label: format!(
                "{}.verify [one batched forward over every window; gdn 1+{gdn}, gqa {gqa}, \
                 mlp {layers}; sampling]",
                self.name
            ),
            child: Box::new(CostNode::Sum({
                let mut nodes = self.glue.compile_verify(&mut b);
                nodes.extend([
                    v.first_norm.compile(&mut b),
                    v.gdn_first.compile(&mut b),
                    Self::scaled(gdn, v.gdn.compile(&mut b)),
                    Self::scaled(gqa, v.gqa.compile(&mut b)),
                    Self::scaled(layers, v.mlp.compile(&mut b)),
                    v.head.compile(&mut b),
                ]);
                nodes
            })),
        };
        let commit = CostNode::Labeled {
            label: format!("{}.commit [replay, copies, taps]", self.name),
            child: Box::new(self.commit.compile(&mut b)),
        };
        b.finish(CostNode::Labeled {
            label: format!(
                "Qwen3.8-27B NVFP4 TensorFold + DFlash2 [{}; ragged windows; \
                 max_model_len={}; finish<= {} rows]",
                self.parallel.gpu_name,
                self.parallel.max_model_len,
                self.parallel.prefill_finish_max_rows
            ),
            child: Box::new(CostNode::Sum(vec![prefill, draft, verify, commit])),
        })
    }

    /// The case as a TensorFold round, after the checks
    /// [`SpeculativeUnifiedModel::check_input`] makes.
    pub fn lower(&self, input: &SpeculativeArchInput) -> Result<TensorfoldRound, String> {
        if input.draft_tokens != self.parallel.draft_tokens {
            return Err(format!(
                "draft_tokens {} differs from the model's {}",
                input.draft_tokens, self.parallel.draft_tokens
            ));
        }
        match input.groups.as_slice() {
            [] => Ok(TensorfoldRound::default()),
            [group] => self.lower_group(group),
            groups => Err(format!(
                "TensorFold runs one group on one GPU, got {}",
                groups.len()
            )),
        }
    }

    fn lower_group(&self, group: &SpeculativeArchGroupInput) -> Result<TensorfoldRound, String> {
        if group.decode_outcomes.len() != group.decode_requests.len() {
            return Err(format!(
                "TensorFold needs decode_outcomes for every decode request ({} outcomes, {} \
                 requests)",
                group.decode_outcomes.len(),
                group.decode_requests.len()
            ));
        }
        let finish = self.parallel.prefill_finish_max_rows;
        let prefill = PrefillForward {
            pieces: group
                .prefill_chunk_pairs
                .iter()
                .map(|&(prefix_len, rows)| PrefillPiece {
                    prefix_len,
                    rows,
                    finishes_prompt: rows <= finish,
                })
                .collect(),
        };
        prefill.check()?;
        let multi = group.decode_requests.len() > 1;
        let mut streams = Vec::with_capacity(group.decode_requests.len());
        for (index, (row, outcome)) in group
            .decode_requests
            .iter()
            .zip(&group.decode_outcomes)
            .enumerate()
        {
            outcome
                .check_against(row)
                .map_err(|e| format!("decode request {index}: {e}"))?;
            let mode = window_mode(outcome.mode);
            let cap = if multi || mode == WindowMode::Tree {
                TENSORFOLD_MAX_ROWS
            } else {
                TENSORFOLD_COPY_CHAIN_MAX_ROWS
            };
            if row.query_len > cap {
                return Err(format!(
                    "decode request {index}: {} rows exceed TensorFold's {cap}-row window \
                     ({mode:?}, {} streams)",
                    row.query_len,
                    group.decode_requests.len()
                ));
            }
            streams.push(VerifyStream {
                committed_len: row.kv_len - row.query_len,
                rows: row.query_len,
                accepted: outcome.accepted,
                tree_slots: outcome.tree_slots,
                mode,
            });
        }
        let verify = VerifyRound { streams };
        verify.check()?;
        Ok(TensorfoldRound { prefill, verify })
    }

    fn eval_into(&self, input: &SpeculativeArchInput, ev: &mut Evaluator) {
        let round = self.lower(input).unwrap_or_else(|reason| {
            panic!("invalid Qwen35TensorfoldNvfp4GdnDenseDflash2Model input: {reason}")
        });
        self.eval_round(&round, ev);
    }

    /// Stream every leaf in `cost_tree` order.
    pub(crate) fn eval_round(&self, round: &TensorfoldRound, ev: &mut Evaluator) {
        let step = &round.prefill;
        let p = &self.prefill;
        self.glue.eval_prefill(step, ev);
        let first: Vec<_> = step
            .forward_rows()
            .into_iter()
            .map(|m| RmsNormKernelInput { m })
            .collect();
        push_sum(&p.first_norm, &first, ev);
        p.gdn_first.eval(step, ev);
        p.gdn.eval(step, ev);
        p.gqa.eval(step, ev);
        p.mlp.eval(step, ev);
        p.head.eval(step, ev);
        self.prefill_taps.eval(step, ev);

        let verify = &round.verify;
        self.draft.eval(
            &TensorfoldDflash2DraftLocalWorkletInput::from_round(verify, self.draft_window),
            ev,
        );
        let v = &self.verify;
        self.glue.eval_verify(verify, ev);
        push_or_zero(
            &v.first_norm,
            RmsNormKernelInput {
                m: verify.total_rows(),
            },
            verify.is_empty(),
            ev,
        );
        v.gdn_first.eval(verify, ev);
        v.gdn.eval(verify, ev);
        v.gqa.eval(verify, ev);
        v.mlp.eval(verify, ev);
        v.head.eval(verify, ev);
        self.commit.eval(verify, ev);
    }

    fn fill<'a>(&self, input: &SpeculativeArchInput, mut ev: Evaluator<'a>) -> usize {
        self.eval_into(input, &mut ev);
        ev.filled()
    }
}

impl SpeculativeUnifiedModel for Qwen35TensorfoldNvfp4GdnDenseDflash2Model {
    fn check_input(&self, batch: &SpeculativeArchInput) -> Result<(), String> {
        self.lower(batch).map(|_| ())
    }

    fn kv_cache_layout(&self) -> KvCacheLayout {
        self.kv_cache_layout.clone()
    }

    fn max_model_len(&self) -> u32 {
        self.parallel.max_model_len
    }

    fn gpus_per_replica(&self) -> u16 {
        1
    }

    fn ragged_verify_windows(&self) -> bool {
        true
    }

    fn cost_log_manifest(&self) -> CostManifest {
        self.cost_tree().manifest()
    }

    fn eval_speculative_iter(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let filled = self.fill(batch, Evaluator::new(slots));
        assert_eq!(filled, self.n_slots, "eval must fill every compiled slot");
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }

    fn eval_speculative_iter_with_inputs(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let filled = self.fill(batch, Evaluator::with_inputs(slots, inputs));
        assert_eq!(filled, self.n_slots, "eval must fill every compiled slot");
        let total = CostTree::aggregate(&self.cost_flat, slots, scratch);
        assert_eq!(
            inputs.len(),
            self.n_slots,
            "slot inputs must align with compiled slots"
        );
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::contract::{SpeculativeDecodeInput, SpeculativeDecodeOutcome};
    use std::path::Path;

    const GB10: &str = "NVIDIA GB10";

    fn cfgs() -> (Qwen35ModelCfg, Qwen35Dflash2DraftCfg) {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        (
            Qwen35ModelCfg::from_json(&root.join("model/config/qwen3_8_27b_nvfp4.json")).unwrap(),
            Qwen35Dflash2DraftCfg::from_json(&root.join("model/config/qwen3_8_27b_dflash2.json"))
                .unwrap(),
        )
    }

    fn parallel() -> TensorfoldParallel {
        TensorfoldParallel {
            gpu_name: GB10.into(),
            max_model_len: 131072,
            draft_tokens: 15,
            sample_top_k: 20,
            prefill_finish_max_rows: DEFAULT_PREFILL_FINISH_MAX_ROWS,
        }
    }

    fn configs(copies: &TensorfoldCopyBackends) -> TensorfoldConfigs {
        let (model, draft) = cfgs();
        let mut c = build_configs(&model, &draft, &parallel()).unwrap();
        copies.apply(&mut c);
        c
    }

    fn model(copies: &TensorfoldCopyBackends) -> Qwen35TensorfoldNvfp4GdnDenseDflash2Model {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        build("m".into(), resolve_configs(&configs(copies)), &bridge).unwrap()
    }

    fn all_copies() -> TensorfoldCopyBackends {
        let b = || Some(vec!["test_backend"]);
        TensorfoldCopyBackends {
            cat: b(),
            cast: b(),
            kv_write: b(),
            fp32_add: b(),
            gather: b(),
            foreach_copy: b(),
            strided_copy: b(),
            cat_strided: b(),
            index: b(),
            torch: b(),
            paths: b(),
            embed: b(),
            rotary: b(),
            eager_norm: b(),
            group_sums: b(),
            dconv: b(),
            prep: b(),
        }
    }

    #[test]
    fn configs_follow_the_checkpoints() {
        let c = configs(&TensorfoldCopyBackends::default());
        assert_eq!((c.gdn_rest_layers, c.gqa_layers), (47, 16));
        assert!(!c.gdn_first.residual_input_norm && c.gdn.residual_input_norm);
        assert_eq!(c.gdn.proj_dtype, DType::Fp8E4m3);
        assert_eq!(c.gdn.ba_dtype, DType::Bf16);
        assert_eq!(c.first_norm.hidden.get(), 5120);
        assert_eq!(c.first_norm.backends, [TENSORFOLD_TRITON]);
        assert_eq!(c.head.sample_candidates, 28);
        assert!(c.gdn.cat_backends.is_none() && c.head.sample_cat_backends.is_none());
        let r = resolve_configs(&c);
        assert_eq!(
            (r.prefill_taps.fc.n.get(), r.prefill_taps.fc.k.get()),
            (5120, 25600)
        );
        assert_eq!(
            (r.prefill_taps.kv.n.get(), r.prefill_taps.kv.k.get()),
            (2048, 5120)
        );
        assert_eq!(
            (r.prefill_taps.window, r.prefill_taps.draft_layers),
            (2047, 5)
        );
        assert!(r.prefill_taps.kv_cat.is_none());
    }

    #[test]
    fn copy_backends_reach_every_pending_slot() {
        let c = configs(&all_copies());
        assert!(c.gdn_first.cat_backends.is_some() && c.gdn.cat_backends.is_some());
        assert!(c.gqa_prefill.copies.kv_write.is_some());
        assert!(c.head.sample_cast_backends.is_some());
        assert!(c.draft_pass.pending.dconv.is_some() && c.commit.pending.gather.is_some());
        let r = resolve_configs(&c);
        assert_eq!(
            r.prefill_taps.kv_cat.unwrap().input_bytes_per_token.get(),
            2048
        );
        // Every pending slot adds leaves.
        assert!(model(&all_copies()).n_slots > model(&Default::default()).n_slots);
        let m = model(&all_copies());
        let names: Vec<String> = m
            .cost_log_manifest()
            .slots
            .iter()
            .map(|s| format!("{} ({})", s.name, s.kind))
            .collect();
        for leaf in [
            "m.verify.embed (embedding_row_gather)",
            "m.verify.paths (elementwise)",
            "m.prefill.embed (embedding_row_gather)",
            "m.prefill.tap_add (elementwise)",
            "m.prefill.taps.kv_sums (row_group_sums)",
            "m.prefill.taps.kv_prep (head_major_qk_norm_rope)",
            "m.draft.dconv (dynamic_causal_conv)",
            "m.commit.kv_prep (head_major_qk_norm_rope)",
        ] {
            assert!(names.iter().any(|n| n == leaf), "{leaf} missing");
        }
    }

    #[test]
    fn kv_layout_parts() {
        let (m, d) = cfgs();
        let layout = kv_cache_layout(&m, &d);
        let bytes: Vec<(&str, u64)> = layout
            .parts
            .iter()
            .map(|p| (p.name, p.bytes_per_entry))
            .collect();
        assert_eq!(
            bytes,
            [
                ("gqa_kv", 65536),
                ("gdn_state", 48 * 48 * 128 * 128 * 4),
                ("gdn_conv", 48 * 3 * 10240 * 2),
                ("dflash2_kv", 5 * 8 * 128 * 2 * 2),
            ]
        );
    }

    fn input(
        prefill: &[(u32, u32)],
        decode: &[(u32, u32, u32, u32, SpeculativeWindowMode)],
    ) -> SpeculativeArchInput {
        SpeculativeArchInput {
            draft_tokens: 15,
            groups: vec![SpeculativeArchGroupInput {
                prefill_chunk_pairs: prefill.to_vec(),
                decode_requests: decode
                    .iter()
                    .map(|&(kv_len, query_len, ..)| SpeculativeDecodeInput { kv_len, query_len })
                    .collect(),
                decode_outcomes: decode
                    .iter()
                    .map(
                        |&(_, _, accepted, tree_slots, mode)| SpeculativeDecodeOutcome {
                            accepted,
                            tree_slots,
                            mode,
                        },
                    )
                    .collect(),
                ..Default::default()
            }],
            tokens_per_source_rank: Vec::new(),
        }
    }

    use SpeculativeWindowMode::{Copy, One, Tree};

    #[test]
    fn lowering_builds_the_round() {
        let m = model(&Default::default());
        let round = m
            .lower(&input(
                &[(0, 1019), (2048, 5)],
                &[
                    (1016, 16, 3, 4, Tree),
                    (50, 10, 9, 0, Copy),
                    (7, 1, 0, 0, One),
                ],
            ))
            .unwrap();
        let finishes: Vec<bool> = round
            .prefill
            .pieces
            .iter()
            .map(|p| p.finishes_prompt)
            .collect();
        assert_eq!(finishes, [false, true]);
        let s = &round.verify.streams;
        assert_eq!(
            (s[0].committed_len, s[0].rows, s[0].accepted),
            (1000, 16, 3)
        );
        assert_eq!((s[1].mode, s[2].mode), (WindowMode::Copy, WindowMode::One));
        assert_eq!(s[0].tree_slots, 4);
    }

    #[test]
    fn check_input_bounds_windows_and_requires_outcomes() {
        let m = model(&Default::default());
        // A single-stream copy chain may run to 128 rows.
        assert!(m
            .check_input(&input(&[], &[(300, 100, 50, 0, Copy)]))
            .is_ok());
        assert!(m
            .check_input(&input(&[], &[(300, 129, 50, 0, Copy)]))
            .unwrap_err()
            .contains("128"));
        // With two streams, or for a tree, 16.
        let two = input(&[], &[(300, 17, 5, 0, Copy), (40, 1, 0, 0, One)]);
        assert!(m.check_input(&two).unwrap_err().contains("16"));
        assert!(m
            .check_input(&input(&[], &[(300, 17, 5, 3, Tree)]))
            .is_err());
        let mut missing = input(&[], &[(300, 16, 5, 3, Tree)]);
        missing.groups[0].decode_outcomes.clear();
        assert!(m
            .check_input(&missing)
            .unwrap_err()
            .contains("decode_outcomes"));
        let mut two_groups = input(&[], &[]);
        two_groups.groups.push(Default::default());
        assert!(m.check_input(&two_groups).is_err());
        let mut depth = input(&[], &[]);
        depth.draft_tokens = 7;
        assert!(m.check_input(&depth).is_err());
    }

    #[test]
    fn prefill_taps_cover_each_pieces_last_window() {
        let m = model(&Default::default());
        let taps = &m.prefill_taps;
        let step = |pieces: &[(u32, u32)]| PrefillForward {
            pieces: pieces
                .iter()
                .map(|&(prefix_len, rows)| PrefillPiece {
                    prefix_len,
                    rows,
                    finishes_prompt: false,
                })
                .collect(),
        };
        let call = |rows, context| PrefillTapCall { rows, context };
        // One piece of 10000 rows: chunks of 3333/3333/3334; the last 2047 rows
        // fall in the last chunk, and the skip clears the windows.
        assert_eq!(taps.calls(&step(&[(0, 10000)])), [call(2047, 0)]);
        // 4100 rows: two chunks of 2050; the window lies in the last. A
        // window never spans two chunks (they hold > window rows each).
        assert_eq!(taps.calls(&step(&[(0, 4100)])), [call(2047, 0)]);
        // A short continuation keeps its stream's context.
        assert_eq!(taps.calls(&step(&[(100, 391)])), [call(391, 100)]);
        // Batched: one call a piece, its last window.
        assert_eq!(
            taps.calls(&step(&[(0, 3000), (5000, 5)])),
            [call(2047, 0), call(5, 2047)]
        );
        assert!(taps.calls(&PrefillForward::default()).is_empty());
    }

    #[test]
    fn tree_shape() {
        for copies in [TensorfoldCopyBackends::default(), all_copies()] {
            let m = model(&copies);
            let manifest = m.cost_log_manifest();
            assert_eq!(manifest.slots.len(), m.n_slots);
            let names: Vec<&str> = manifest.slots.iter().map(|s| s.name.as_str()).collect();
            for leaf in [
                "m.prefill.first_norm",
                "m.prefill.taps.fc",
                "m.prefill.taps.kv",
                "m.verify.first_norm",
                "m.verify.gdn.tree",
                "m.verify.gdn_first.tree",
                "m.commit.replay",
            ] {
                assert!(names.contains(&leaf), "{leaf} missing from {names:?}");
            }
            // Layer 0 has no input norm; the other GDN layers do.
            assert!(!names.contains(&"m.verify.gdn_first.input_norm"));
            assert!(names.contains(&"m.verify.gdn.input_norm"));
        }
    }
}
