//! One low-rank gated hyper-connection (HC) mix module on one GPU.
//!
//! The residual is `hc_count` streams of `hidden_size` (stream-outer). Each
//! module turns it into the next block's single-stream input (vLLM fork
//! `GatedResidual.mix` / `combine_and_mix`, `ops/hc.py`):
//!
//! 1. entry norm, one of
//!    - `hc_combine_norm`: the pending combine of the previous block's output
//!      fused with this module's per-stream Gemma RMSNorm (the usual case; the
//!      MTP layer's unit-injection combine launches the same kernel, and unit
//!      vs learned injection is within 3% above 1k rows, so no axis);
//!    - `grouped_rms_norm`: no combine is pending (the first module of the
//!      stack, and the module right after a standalone `hc_combine`, e.g. the
//!      one following a per-layer embedding addition);
//! 2. the down projection: merged low-rank down + injection GEMM, N =
//!    `hc_lowrank + hc_count` padded to 16 (`pad_size`), K = `hc_count *
//!    hidden_size`. A final mixer (`use_combine=False`) has no injection, so N
//!    = `hc_lowrank`;
//! 3. `scaled_silu` over the `hc_lowrank` columns;
//! 4. the up GEMM, N = `hc_count * hidden_size`, K = `hc_lowrank`;
//! 5. `hc_gate_mix`: the gated mean over the normalized streams.
//!
//! The GEMMs are BF16 `F.linear` (B200 never takes the SM90/SM103 skinny
//! low-latency plans). Plain `Sum`: every launch is on the main stream and
//! depends on the previous one.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    GroupedRmsNormKernel, GroupedRmsNormKernelConfig, GroupedRmsNormKernelInput,
    HcCombineNormKernel, HcGateMixKernel, MhcRmsNormKernelConfig, MhcRmsNormKernelInput,
    ScaledSiluKernel, ScaledSiluKernelConfig, ScaledSiluKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge};
use crate::worklet::glm53_common::{atomic, push_or_zero};

/// Row alignment the fork pads the merged down+inject projection to.
const DOWN_INJECT_ALIGN: u32 = 16;

/// What normalizes the residual streams at the module's entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HcMixEntry {
    /// A combine is pending: `hc_combine_norm`.
    CombineNorm,
    /// No combine is pending: `grouped_rms_norm`.
    GroupedNorm,
}

#[derive(Clone, Debug)]
pub struct HcMixLocalWorkletConfig {
    pub hidden_size: u32,
    pub hc_count: u32,
    pub hc_lowrank: u32,
    pub entry: HcMixEntry,
    /// `true` for a block module (the down GEMM also emits the next combine's
    /// injection); `false` for a final mixer.
    pub use_combine: bool,
    pub dtype: DType,
    pub gpu_name: String,
    /// The HC Triton kernels (norms, `scaled_silu`, `hc_gate_mix`).
    pub hc_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct HcMixLocalWorkletResolved {
    pub raw_cfg: HcMixLocalWorkletConfig,
    pub combine_norm: Option<MhcRmsNormKernelConfig>,
    pub grouped_norm: Option<GroupedRmsNormKernelConfig>,
    pub down: SingleGemmKernelConfig,
    pub silu: ScaledSiluKernelConfig,
    pub up: SingleGemmKernelConfig,
    pub gate_mix: MhcRmsNormKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct HcMixLocalWorkletInput {
    pub num_tokens: u32,
}

pub struct HcMixLocalWorklet {
    pub name: String,
    pub combine_norm: Option<Op<HcCombineNormKernel>>,
    pub grouped_norm: Option<Op<GroupedRmsNormKernel>>,
    pub down: Op<SingleGemmKernel>,
    pub silu: Op<ScaledSiluKernel>,
    pub up: Op<SingleGemmKernel>,
    pub gate_mix: Op<HcGateMixKernel>,
    resolved: HcMixLocalWorkletResolved,
}

/// Output columns of the down projection.
pub fn down_proj_width(hc_lowrank: u32, hc_count: u32, use_combine: bool) -> u32 {
    if use_combine {
        (hc_lowrank + hc_count).next_multiple_of(DOWN_INJECT_ALIGN)
    } else {
        hc_lowrank
    }
}

impl HcMixLocalWorklet {
    pub fn resolve_config(cfg: &HcMixLocalWorkletConfig) -> HcMixLocalWorkletResolved {
        assert!(
            cfg.hidden_size > 0 && cfg.hc_count > 0 && cfg.hc_lowrank > 0,
            "HC mix needs positive hidden_size, hc_count and hc_lowrank"
        );
        let streams = cfg.hidden_size * cfg.hc_count;
        let multi_stream = || MhcRmsNormKernelConfig {
            backends: cfg.hc_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            hidden_size: cfg.hidden_size.into(),
            hc_mult: cfg.hc_count,
            hidden_dtype: cfg.dtype,
        };
        let gemm = |n: u32, k: u32| SingleGemmKernelConfig {
            backends: cfg.gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n: n.into(),
            k: k.into(),
            dtype: cfg.dtype,
        };
        HcMixLocalWorkletResolved {
            combine_norm: (cfg.entry == HcMixEntry::CombineNorm).then(multi_stream),
            grouped_norm: (cfg.entry == HcMixEntry::GroupedNorm).then(|| {
                GroupedRmsNormKernelConfig {
                    backends: cfg.hc_backends.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    hidden_size: cfg.hidden_size.into(),
                    num_groups: cfg.hc_count,
                    hidden_dtype: cfg.dtype,
                }
            }),
            down: gemm(
                down_proj_width(cfg.hc_lowrank, cfg.hc_count, cfg.use_combine),
                streams,
            ),
            silu: ScaledSiluKernelConfig {
                backends: cfg.hc_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                width: cfg.hc_lowrank.into(),
                hidden_dtype: cfg.dtype,
            },
            up: gemm(streams, cfg.hc_lowrank),
            gate_mix: multi_stream(),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: HcMixLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let combine_norm = resolved
            .combine_norm
            .clone()
            .map(|c| atomic(&name, "combine_norm", c, HcCombineNormKernel::build, bridge))
            .transpose()?;
        let grouped_norm = resolved
            .grouped_norm
            .clone()
            .map(|c| {
                atomic(
                    &name,
                    "grouped_norm",
                    c,
                    GroupedRmsNormKernel::build,
                    bridge,
                )
            })
            .transpose()?;
        Ok(Self {
            combine_norm,
            grouped_norm,
            down: atomic(
                &name,
                "down",
                resolved.down.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            silu: atomic(
                &name,
                "silu",
                resolved.silu.clone(),
                ScaledSiluKernel::build,
                bridge,
            )?,
            up: atomic(
                &name,
                "up",
                resolved.up.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            gate_mix: atomic(
                &name,
                "gate_mix",
                resolved.gate_mix.clone(),
                HcGateMixKernel::build,
                bridge,
            )?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let mut children = Vec::with_capacity(5);
        if let Some(op) = &self.combine_norm {
            children.push(op.compile(builder));
        }
        if let Some(op) = &self.grouped_norm {
            children.push(op.compile(builder));
        }
        children.push(self.down.compile(builder));
        children.push(self.silu.compile(builder));
        children.push(self.up.compile(builder));
        children.push(self.gate_mix.compile(builder));
        let cfg = &self.resolved.raw_cfg;
        CostNode::Labeled {
            label: format!(
                "{} (HcMixLocalWorklet) [local; entry={:?}; streams={}x{}; lowrank={}; down_n={}]",
                self.name,
                cfg.entry,
                cfg.hc_count,
                cfg.hidden_size,
                cfg.hc_lowrank,
                self.resolved.down.n.get(),
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    pub fn eval(&self, input: &HcMixLocalWorkletInput, ev: &mut Evaluator) {
        let m = input.num_tokens;
        let zero = m == 0;
        if let Some(op) = &self.combine_norm {
            push_or_zero(op, MhcRmsNormKernelInput { num_tokens: m }, zero, ev);
        }
        if let Some(op) = &self.grouped_norm {
            push_or_zero(op, GroupedRmsNormKernelInput { num_tokens: m }, zero, ev);
        }
        push_or_zero(&self.down, SingleGemmKernelInput { m }, zero, ev);
        push_or_zero(
            &self.silu,
            ScaledSiluKernelInput { num_tokens: m },
            zero,
            ev,
        );
        push_or_zero(&self.up, SingleGemmKernelInput { m }, zero, ev);
        push_or_zero(
            &self.gate_mix,
            MhcRmsNormKernelInput { num_tokens: m },
            zero,
            ev,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(entry: HcMixEntry, use_combine: bool) -> HcMixLocalWorkletConfig {
        HcMixLocalWorkletConfig {
            hidden_size: 2560,
            hc_count: 4,
            hc_lowrank: 320,
            entry,
            use_combine,
            dtype: DType::Bf16,
            gpu_name: "NVIDIA B200".into(),
            hc_backends: vec!["vllm_triton"],
            gemm_backends: vec!["torch_linear_vllm"],
        }
    }

    /// The merged down+inject GEMM pads lowrank + hc_count to 16 rows; a final
    /// mixer has no injection and no pad. Both match the profiled rows.
    #[test]
    fn down_width_pads_the_injection_and_final_mixer_drops_it() {
        let block = HcMixLocalWorklet::resolve_config(&cfg(HcMixEntry::CombineNorm, true));
        assert_eq!((block.down.n.get(), block.down.k.get()), (336, 10240));
        assert_eq!((block.up.n.get(), block.up.k.get()), (10240, 320));
        assert_eq!(block.silu.width.get(), 320);
        let last = HcMixLocalWorklet::resolve_config(&cfg(HcMixEntry::CombineNorm, false));
        assert_eq!(last.down.n.get(), 320);
        assert_eq!(down_proj_width(320, 4, true), 336);
        assert_eq!(down_proj_width(12, 4, true), 16);
    }

    /// Exactly one entry norm exists, chosen by whether a combine is pending.
    #[test]
    fn entry_selects_exactly_one_norm_leaf() {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        for (entry, combine, grouped) in [
            (HcMixEntry::CombineNorm, true, false),
            (HcMixEntry::GroupedNorm, false, true),
        ] {
            let w = HcMixLocalWorklet::build(
                "hc".into(),
                HcMixLocalWorklet::resolve_config(&cfg(entry, true)),
                &bridge,
            )
            .unwrap();
            assert_eq!(w.combine_norm.is_some(), combine);
            assert_eq!(w.grouped_norm.is_some(), grouped);
            let mut b = CostTreeBuilder::new();
            let root = w.compile(&mut b);
            assert_eq!(b.finish(root).n_slots(), 5);
        }
    }
}
