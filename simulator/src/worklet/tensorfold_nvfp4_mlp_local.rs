//! One TensorFold Qwen3.8-27B layer's MLP section on GB10
//! (`qwen35_tensorfold_nvfp4_gdn_dense_dflash2`): the post-attention residual
//! add + RMSNorm, then the dense NVFP4 SwiGLU. Every one of the 64 layers runs
//! the same section, so the arch folds one worklet with `Scale{num_layers}`
//! (or one `Scale` per layer group it already has).
//!
//! **Boundary.** This worklet owns the layer's *post-attention*
//! `add_rmsnorm(x, r, post_norm)` — the mixer's output `r` added into the
//! residual stream, then normed. The mixer worklets (GDN, GQA) own the layer's
//! *input* `add_rmsnorm(x, pending, input_norm)`, which adds this worklet's
//! output (`pending`) of the layer before. The final norm after layer 63 is
//! the head's (`tensorfold_head_local`). Per verify forward that is
//! 64 (mixers) + 64 (here) + 1 (head) = 129 `_add_rmsnorm` launches.
//! The DFlash2 tap capture after the MLP (`(x.float() + pending.float())
//! .to(bf16)` at layers 5/19/33/47/61, and the `cat` of the five) is **not**
//! here: it belongs to the drafter lane (W4).
//!
//! Two forms, one struct each, sharing [`TensorfoldNvfp4MlpLocalWorkletConfig`]
//! (sources under `tensorfold-0.6.6/src/tensorfold/families/qwen3_5/cuda/`):
//!
//! - **Verify** (`forward.py` `multi_tree_forward` → `_mlp`), every leaf over
//!   the round's W = Σ window rows:
//!   `residual_rms_norm` → `nvfp4_quant` (k = hidden) → `tensorfold_lane`
//!   stacked gate|up (n = 2·intermediate) → stacked `_swiglu`
//!   (`silu_and_mul`, `glue.swiglu_stacked`) → `nvfp4_quant`
//!   (k = intermediate) → `tensorfold_lane` down. The two quants are the
//!   `quant4_kernel` launches NVFP4 `matmul` issues ahead of each lane GEMM.
//! - **Prefill** (`prefill.py` `_mlp` → `nvfp4/checkpoint.py` `mlp_prompt`),
//!   once per forward of the step ([`PrefillForward::forward_rows`]: each
//!   chunk of a one-piece step, or the one batched forward):
//!   `residual_rms_norm` → `nvfp4_quant` (k = hidden) → `gemm_swiglu_quant`
//!   (`gemm_gu_ck`: gate|up, SwiGLU and the NVFP4 rows for down in one launch)
//!   → `tensorfold_ck` down. The prefill norm is the bf16 `glue.add_rmsnorm`
//!   (`prefill_bf16`), the same Triton kernel as verify's.
//!
//! Capture check (`kernel_inventory_n8.json`, per verify forward): 64×2 MLP
//! `lane_kernel` of the 337, 128 of the 129 `quant4_kernel`, all 64 `_swiglu`;
//! per prefill forward: 64 `quant4_kernel` and 128 of the 336 `gemm_kernel`.

use crate::arch::{Qwen35LinearPrecision, Qwen35ModelCfg};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    GemmSwigluQuantKernel, GemmSwigluQuantKernelConfig, GemmSwigluQuantKernelInput,
    Nvfp4QuantKernel, Nvfp4QuantKernelConfig, Nvfp4QuantKernelInput, ResidualRmsNormKernel,
    ResidualRmsNormKernelConfig, ResidualRmsNormKernelInput, SiluAndMulKernel,
    SiluAndMulKernelConfig, SiluAndMulKernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge};

use super::tensorfold_common::{
    atomic, gemm, push_sum, PrefillForward, VerifyRound, TENSORFOLD_CK, TENSORFOLD_CUDA,
    TENSORFOLD_LANE, TENSORFOLD_TRITON,
};

/// Elements per NVFP4 scale group of TensorFold's activation quant.
pub const TENSORFOLD_NVFP4_GROUP_SIZE: u32 = 16;
/// `quant4_kernel`'s scale layout: `[k/64, ceil(m/64)·64, 4]` E4M3 k-slices.
pub const TENSORFOLD_NVFP4_SCALE_FORMAT: &str = "kslice_e4m3";

/// The NVFP4 activation quant ahead of a lane / ck GEMM reducing over `k`.
pub fn nvfp4_quant(gpu_name: &str, k: u32) -> Nvfp4QuantKernelConfig {
    Nvfp4QuantKernelConfig {
        backends: vec![TENSORFOLD_CUDA],
        gpu_name: gpu_name.to_string(),
        hidden_size: k.into(),
        group_size: TENSORFOLD_NVFP4_GROUP_SIZE,
        input_dtype: DType::Bf16,
        scale_format: TENSORFOLD_NVFP4_SCALE_FORMAT.to_string(),
    }
}

/// TensorFold's Triton `_add_rmsnorm` over bf16 rows of `hidden`.
pub fn add_rms_norm(gpu_name: &str, hidden: u32) -> ResidualRmsNormKernelConfig {
    ResidualRmsNormKernelConfig {
        backends: vec![TENSORFOLD_TRITON],
        gpu_name: gpu_name.to_string(),
        hidden: hidden.into(),
        dtype: DType::Bf16,
    }
}

/// Model facts for both MLP forms.
#[derive(Clone, Debug)]
pub struct TensorfoldNvfp4MlpLocalWorkletConfig {
    pub hidden: u32,
    pub intermediate: u32,
    pub gpu_name: String,
}

impl TensorfoldNvfp4MlpLocalWorkletConfig {
    /// From the checkpoint config; its MLP must be NVFP4.
    pub fn from_model(model: &Qwen35ModelCfg, gpu_name: &str) -> Self {
        assert_eq!(
            model.precisions.mlp,
            Qwen35LinearPrecision::Nvfp4,
            "the TensorFold MLP worklet models NVFP4 MLP weights"
        );
        Self {
            hidden: model.hidden,
            intermediate: model.intermediate,
            gpu_name: gpu_name.to_string(),
        }
    }
}

// ─── verify ──────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct TensorfoldNvfp4MlpVerifyLocalWorkletResolved {
    pub raw_cfg: TensorfoldNvfp4MlpLocalWorkletConfig,
    pub post_attn_norm: ResidualRmsNormKernelConfig,
    pub gate_up_quant: Nvfp4QuantKernelConfig,
    /// Stacked gate|up: n = 2·intermediate.
    pub gate_up: SingleGemmKernelConfig,
    pub swiglu: SiluAndMulKernelConfig,
    pub down_quant: Nvfp4QuantKernelConfig,
    pub down: SingleGemmKernelConfig,
}

/// One verify round; the leaves read its W = [`VerifyRound::total_rows`].
pub type TensorfoldNvfp4MlpVerifyLocalWorkletInput = VerifyRound;

/// One layer's MLP in the batched verify forward.
pub struct TensorfoldNvfp4MlpVerifyLocalWorklet {
    pub name: String,
    post_attn_norm: Op<ResidualRmsNormKernel>,
    gate_up_quant: Op<Nvfp4QuantKernel>,
    gate_up: Op<SingleGemmKernel>,
    swiglu: Op<SiluAndMulKernel>,
    down_quant: Op<Nvfp4QuantKernel>,
    down: Op<SingleGemmKernel>,
    resolved: TensorfoldNvfp4MlpVerifyLocalWorkletResolved,
}

impl TensorfoldNvfp4MlpVerifyLocalWorklet {
    pub fn resolve_config(
        cfg: &TensorfoldNvfp4MlpLocalWorkletConfig,
    ) -> TensorfoldNvfp4MlpVerifyLocalWorkletResolved {
        let gpu = cfg.gpu_name.as_str();
        TensorfoldNvfp4MlpVerifyLocalWorkletResolved {
            post_attn_norm: add_rms_norm(gpu, cfg.hidden),
            gate_up_quant: nvfp4_quant(gpu, cfg.hidden),
            gate_up: gemm(
                TENSORFOLD_LANE,
                gpu,
                2 * cfg.intermediate,
                cfg.hidden,
                DType::Nvfp4E2m1,
            ),
            swiglu: SiluAndMulKernelConfig {
                backends: vec![TENSORFOLD_TRITON],
                gpu_name: cfg.gpu_name.clone(),
                hidden_dim: cfg.intermediate.into(),
                dtype: DType::Bf16,
            },
            down_quant: nvfp4_quant(gpu, cfg.intermediate),
            down: gemm(
                TENSORFOLD_LANE,
                gpu,
                cfg.hidden,
                cfg.intermediate,
                DType::Nvfp4E2m1,
            ),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: TensorfoldNvfp4MlpVerifyLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let n = name.as_str();
        Ok(Self {
            post_attn_norm: atomic(
                n,
                "post_attn_norm",
                r.post_attn_norm,
                ResidualRmsNormKernel::build,
                bridge,
            )?,
            gate_up_quant: atomic(
                n,
                "gate_up_quant",
                r.gate_up_quant,
                Nvfp4QuantKernel::build,
                bridge,
            )?,
            gate_up: atomic(n, "gate_up", r.gate_up, SingleGemmKernel::build, bridge)?,
            swiglu: atomic(n, "swiglu", r.swiglu, SiluAndMulKernel::build, bridge)?,
            down_quant: atomic(
                n,
                "down_quant",
                r.down_quant,
                Nvfp4QuantKernel::build,
                bridge,
            )?,
            down: atomic(n, "down", r.down, SingleGemmKernel::build, bridge)?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        CostNode::Labeled {
            label: format!(
                "{} (TensorfoldNvfp4MlpVerifyLocalWorklet) [local (1 GPU); H={}, I={}; \
                 post-attn norm + NVFP4 lane SwiGLU over the verify rows]",
                self.name, cfg.hidden, cfg.intermediate
            ),
            child: Box::new(CostNode::Sum(vec![
                self.post_attn_norm.compile(builder),
                self.gate_up_quant.compile(builder),
                self.gate_up.compile(builder),
                self.swiglu.compile(builder),
                self.down_quant.compile(builder),
                self.down.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, round: &TensorfoldNvfp4MlpVerifyLocalWorkletInput, ev: &mut Evaluator) {
        let w = VerifyMlpWork::of(round);
        push_sum(&self.post_attn_norm, &w.norm, ev);
        push_sum(&self.gate_up_quant, &w.quant, ev);
        push_sum(&self.gate_up, &w.gemm, ev);
        push_sum(&self.swiglu, &w.swiglu, ev);
        push_sum(&self.down_quant, &w.quant, ev);
        push_sum(&self.down, &w.gemm, ev);
    }
}

/// Each verify slot's launches: one over W rows, none for an empty round.
struct VerifyMlpWork {
    norm: Vec<ResidualRmsNormKernelInput>,
    quant: Vec<Nvfp4QuantKernelInput>,
    gemm: Vec<SingleGemmKernelInput>,
    swiglu: Vec<SiluAndMulKernelInput>,
}

impl VerifyMlpWork {
    fn of(round: &VerifyRound) -> Self {
        let rows: Vec<u32> = Some(round.total_rows())
            .filter(|&w| w > 0)
            .into_iter()
            .collect();
        Self::over(&rows)
    }

    /// One launch of every slot per entry of `rows`.
    fn over(rows: &[u32]) -> Self {
        Self {
            norm: rows
                .iter()
                .map(|&m| ResidualRmsNormKernelInput { m })
                .collect(),
            quant: rows
                .iter()
                .map(|&num_tokens| Nvfp4QuantKernelInput { num_tokens })
                .collect(),
            gemm: rows.iter().map(|&m| SingleGemmKernelInput { m }).collect(),
            swiglu: rows
                .iter()
                .map(|&num_tokens| SiluAndMulKernelInput { num_tokens })
                .collect(),
        }
    }
}

// ─── prefill ─────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct TensorfoldNvfp4MlpPrefillLocalWorkletResolved {
    pub raw_cfg: TensorfoldNvfp4MlpLocalWorkletConfig,
    pub post_attn_norm: ResidualRmsNormKernelConfig,
    pub gate_up_quant: Nvfp4QuantKernelConfig,
    /// Fused gate|up + SwiGLU + NVFP4 quant: n = intermediate (each half).
    pub gate_up_swiglu: GemmSwigluQuantKernelConfig,
    pub down: SingleGemmKernelConfig,
}

/// One prefill step; the leaves launch once per entry of
/// [`PrefillForward::forward_rows`].
pub type TensorfoldNvfp4MlpPrefillLocalWorkletInput = PrefillForward;

/// One layer's MLP in a prefill step's forwards.
pub struct TensorfoldNvfp4MlpPrefillLocalWorklet {
    pub name: String,
    post_attn_norm: Op<ResidualRmsNormKernel>,
    gate_up_quant: Op<Nvfp4QuantKernel>,
    gate_up_swiglu: Op<GemmSwigluQuantKernel>,
    down: Op<SingleGemmKernel>,
    resolved: TensorfoldNvfp4MlpPrefillLocalWorkletResolved,
}

impl TensorfoldNvfp4MlpPrefillLocalWorklet {
    pub fn resolve_config(
        cfg: &TensorfoldNvfp4MlpLocalWorkletConfig,
    ) -> TensorfoldNvfp4MlpPrefillLocalWorkletResolved {
        let gpu = cfg.gpu_name.as_str();
        TensorfoldNvfp4MlpPrefillLocalWorkletResolved {
            post_attn_norm: add_rms_norm(gpu, cfg.hidden),
            gate_up_quant: nvfp4_quant(gpu, cfg.hidden),
            gate_up_swiglu: GemmSwigluQuantKernelConfig {
                backends: vec![TENSORFOLD_CK],
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.intermediate.into(),
                k: cfg.hidden.into(),
                dtype: DType::Nvfp4E2m1,
            },
            down: gemm(
                TENSORFOLD_CK,
                gpu,
                cfg.hidden,
                cfg.intermediate,
                DType::Nvfp4E2m1,
            ),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: TensorfoldNvfp4MlpPrefillLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let n = name.as_str();
        Ok(Self {
            post_attn_norm: atomic(
                n,
                "post_attn_norm",
                r.post_attn_norm,
                ResidualRmsNormKernel::build,
                bridge,
            )?,
            gate_up_quant: atomic(
                n,
                "gate_up_quant",
                r.gate_up_quant,
                Nvfp4QuantKernel::build,
                bridge,
            )?,
            gate_up_swiglu: atomic(
                n,
                "gate_up_swiglu",
                r.gate_up_swiglu,
                GemmSwigluQuantKernel::build,
                bridge,
            )?,
            down: atomic(n, "down", r.down, SingleGemmKernel::build, bridge)?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        CostNode::Labeled {
            label: format!(
                "{} (TensorfoldNvfp4MlpPrefillLocalWorklet) [local (1 GPU); H={}, I={}; \
                 post-attn norm + fused ck gate|up SwiGLU, ck down, per prefill forward]",
                self.name, cfg.hidden, cfg.intermediate
            ),
            child: Box::new(CostNode::Sum(vec![
                self.post_attn_norm.compile(builder),
                self.gate_up_quant.compile(builder),
                self.gate_up_swiglu.compile(builder),
                self.down.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, step: &TensorfoldNvfp4MlpPrefillLocalWorkletInput, ev: &mut Evaluator) {
        let w = PrefillMlpWork::of(step);
        push_sum(&self.post_attn_norm, &w.norm, ev);
        push_sum(&self.gate_up_quant, &w.quant, ev);
        push_sum(&self.gate_up_swiglu, &w.fused, ev);
        push_sum(&self.down, &w.down, ev);
    }
}

/// Each prefill slot's launches: one per forward of the step.
struct PrefillMlpWork {
    norm: Vec<ResidualRmsNormKernelInput>,
    quant: Vec<Nvfp4QuantKernelInput>,
    fused: Vec<GemmSwigluQuantKernelInput>,
    down: Vec<SingleGemmKernelInput>,
}

impl PrefillMlpWork {
    fn of(step: &PrefillForward) -> Self {
        let rows = step.forward_rows();
        let v = VerifyMlpWork::over(&rows);
        Self {
            norm: v.norm,
            quant: v.quant,
            fused: rows
                .iter()
                .map(|&m| GemmSwigluQuantKernelInput { m })
                .collect(),
            down: v.gemm,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::LeafMetrics;
    use crate::worklet::tensorfold_common::{PrefillPiece, VerifyStream, WindowMode};
    use std::path::Path;

    fn qwen38() -> Qwen35ModelCfg {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("model/config/qwen3_8_27b_nvfp4.json");
        Qwen35ModelCfg::from_json(&path).unwrap()
    }

    fn cfg() -> TensorfoldNvfp4MlpLocalWorkletConfig {
        TensorfoldNvfp4MlpLocalWorkletConfig::from_model(&qwen38(), "NVIDIA GB10")
    }

    fn round(rows: &[u32]) -> VerifyRound {
        VerifyRound {
            streams: rows
                .iter()
                .map(|&rows| VerifyStream {
                    committed_len: 1000,
                    rows,
                    accepted: 0,
                    tree_slots: 0,
                    mode: WindowMode::Copy,
                })
                .collect(),
        }
    }

    fn step(pieces: &[(u32, u32)]) -> PrefillForward {
        PrefillForward {
            pieces: pieces
                .iter()
                .map(|&(prefix_len, rows)| PrefillPiece {
                    prefix_len,
                    rows,
                    finishes_prompt: true,
                })
                .collect(),
        }
    }

    fn slot_names(tree: &crate::timing::CostTree) -> Vec<(&str, &str)> {
        tree.slots
            .iter()
            .map(|s| (s.name.as_str(), s.kind.as_str()))
            .collect()
    }

    #[test]
    fn shapes_follow_the_checkpoint() {
        let v = TensorfoldNvfp4MlpVerifyLocalWorklet::resolve_config(&cfg());
        assert_eq!((v.gate_up.n.get(), v.gate_up.k.get()), (34816, 5120));
        assert_eq!((v.down.n.get(), v.down.k.get()), (5120, 17408));
        assert_eq!(
            (
                v.gate_up_quant.hidden_size.get(),
                v.down_quant.hidden_size.get()
            ),
            (5120, 17408)
        );
        assert_eq!(v.swiglu.hidden_dim.get(), 17408);
        assert_eq!(v.gate_up.backends, [TENSORFOLD_LANE]);
        assert_eq!(v.down_quant.scale_format, "kslice_e4m3");
        let p = TensorfoldNvfp4MlpPrefillLocalWorklet::resolve_config(&cfg());
        assert_eq!(
            (p.gate_up_swiglu.n.get(), p.gate_up_swiglu.k.get()),
            (17408, 5120)
        );
        assert_eq!((p.down.n.get(), p.down.k.get()), (5120, 17408));
        assert_eq!(p.down.backends, [TENSORFOLD_CK]);
    }

    #[test]
    fn leaves_per_layer_match_the_capture() {
        let bridge = PerfApiBridge::structure_only();
        let v = TensorfoldNvfp4MlpVerifyLocalWorklet::build(
            "mlp".into(),
            TensorfoldNvfp4MlpVerifyLocalWorklet::resolve_config(&cfg()),
            &bridge,
        )
        .unwrap();
        let mut b = CostTreeBuilder::new();
        let root = v.compile(&mut b);
        let tree = b.finish(root);
        // One `_add_rmsnorm`, two `quant4_kernel`, two `lane_kernel`, one `_swiglu`.
        assert_eq!(
            slot_names(&tree),
            [
                ("mlp.post_attn_norm", "residual_rms_norm"),
                ("mlp.gate_up_quant", "nvfp4_quant"),
                ("mlp.gate_up", "single_gemm"),
                ("mlp.swiglu", "silu_and_mul"),
                ("mlp.down_quant", "nvfp4_quant"),
                ("mlp.down", "single_gemm"),
            ]
        );
        let p = TensorfoldNvfp4MlpPrefillLocalWorklet::build(
            "mlp".into(),
            TensorfoldNvfp4MlpPrefillLocalWorklet::resolve_config(&cfg()),
            &bridge,
        )
        .unwrap();
        let mut b = CostTreeBuilder::new();
        let root = p.compile(&mut b);
        let tree = b.finish(root);
        // One `_add_rmsnorm`, one `quant4_kernel`, two `gemm_kernel`.
        assert_eq!(
            slot_names(&tree),
            [
                ("mlp.post_attn_norm", "residual_rms_norm"),
                ("mlp.gate_up_quant", "nvfp4_quant"),
                ("mlp.gate_up_swiglu", "gemm_swiglu_quant"),
                ("mlp.down", "single_gemm"),
            ]
        );
    }

    #[test]
    fn verify_runs_one_launch_over_all_window_rows() {
        let w = VerifyMlpWork::of(&round(&[16, 5, 1]));
        assert_eq!(w.gemm.iter().map(|i| i.m).collect::<Vec<_>>(), [22]);
        assert_eq!(w.quant.len() + w.norm.len() + w.swiglu.len(), 3);
        let empty = VerifyMlpWork::of(&VerifyRound::default());
        assert!(empty.norm.is_empty() && empty.gemm.is_empty());
    }

    #[test]
    fn prefill_launches_once_per_forward() {
        // One piece: a forward per even chunk.
        let w = PrefillMlpWork::of(&step(&[(1000, 9000)]));
        assert_eq!(w.fused.iter().map(|i| i.m).collect::<Vec<_>>(), [3000; 3]);
        assert_eq!((w.norm.len(), w.quant.len(), w.down.len()), (3, 3, 3));
        // Several pieces: one forward over every row.
        let w = PrefillMlpWork::of(&step(&[(0, 600), (2048, 424)]));
        assert_eq!(w.down.iter().map(|i| i.m).collect::<Vec<_>>(), [1024]);
    }

    #[test]
    fn empty_inputs_push_zero_into_every_slot() {
        let bridge = PerfApiBridge::structure_only();
        let v = TensorfoldNvfp4MlpVerifyLocalWorklet::build(
            "mlp".into(),
            TensorfoldNvfp4MlpVerifyLocalWorklet::resolve_config(&cfg()),
            &bridge,
        )
        .unwrap();
        let p = TensorfoldNvfp4MlpPrefillLocalWorklet::build(
            "mlp".into(),
            TensorfoldNvfp4MlpPrefillLocalWorklet::resolve_config(&cfg()),
            &bridge,
        )
        .unwrap();
        let mut buf = vec![LeafMetrics::MISS; 10];
        let mut ev = Evaluator::new(&mut buf);
        v.eval(&VerifyRound::default(), &mut ev);
        p.eval(&PrefillForward::default(), &mut ev);
        assert_eq!(ev.filled(), 10);
        assert!(buf.iter().all(|leaf| leaf.m.time_ms == 0.0));
    }

    /// Against the GB10 rows in `profile.db`. Run from the repo root:
    /// `PYTHONPATH=$PWD LD_LIBRARY_PATH=<libpython dir> uv run cargo test -p simulator --lib
    /// tensorfold_nvfp4_mlp -- --include-ignored --nocapture`.
    #[test]
    #[ignore = "needs profile.db GB10 rows and the venv interpreter"]
    fn gb10_rows_price_one_layer() {
        let bridge = PerfApiBridge::new().expect("perf_api bridge");
        let v = TensorfoldNvfp4MlpVerifyLocalWorklet::build(
            "mlp".into(),
            TensorfoldNvfp4MlpVerifyLocalWorklet::resolve_config(&cfg()),
            &bridge,
        )
        .expect("verify MLP rows on GB10");
        let p = TensorfoldNvfp4MlpPrefillLocalWorklet::build(
            "mlp".into(),
            TensorfoldNvfp4MlpPrefillLocalWorklet::resolve_config(&cfg()),
            &bridge,
        )
        .expect("prefill MLP rows on GB10");
        let price = |slots: usize, fill: &dyn Fn(&mut Evaluator)| -> Vec<f32> {
            let mut buf = vec![LeafMetrics::MISS; slots];
            let mut ev = Evaluator::new(&mut buf);
            fill(&mut ev);
            assert_eq!(ev.filled(), slots);
            buf.iter().map(|leaf| leaf.m.time_ms).collect()
        };
        // W = 1, then 8 streams at the n8 capture's W buckets (64..127).
        for w in [1, 68, 76, 84, 92, 100, 108, 116, 124] {
            let streams = w.min(8);
            let rows: Vec<u32> = (0..streams)
                .map(|s| w / streams + u32::from(s < w % streams))
                .collect();
            let r = round(&rows);
            let ms = price(6, &|ev| v.eval(&r, ev));
            eprintln!(
                "verify W={:3}: norm {:.4} q4 {:.4} gate_up {:.4} swiglu {:.4} q4 {:.4} \
                 down {:.4} | layer {:.4} ms, x64 {:.3} ms",
                r.total_rows(),
                ms[0],
                ms[1],
                ms[2],
                ms[3],
                ms[4],
                ms[5],
                ms.iter().sum::<f32>(),
                64.0 * ms.iter().sum::<f32>()
            );
            assert!(ms.iter().all(|t| *t > 0.0 && t.is_finite()));
        }
        // Single-chunk forwards at the n8 capture's prefill row counts, a long
        // prompt in 7 chunks, and a batched 1024-row step.
        for pieces in [
            vec![(0, 5)],
            vec![(0, 148)],
            vec![(0, 464)],
            vec![(0, 997)],
            vec![(1109, 27_000)],
            vec![(0, 600), (2048, 424)],
        ] {
            let s = step(&pieces);
            let ms = price(4, &|ev| p.eval(&s, ev));
            eprintln!(
                "prefill rows {:?}: norm {:.4} q4 {:.4} gemm_swiglu {:.4} down {:.4} | \
                 layer {:.4} ms, x64 {:.3} ms",
                s.forward_rows(),
                ms[0],
                ms[1],
                ms[2],
                ms[3],
                ms.iter().sum::<f32>(),
                64.0 * ms.iter().sum::<f32>()
            );
            assert!(ms.iter().all(|t| *t > 0.0 && t.is_finite()));
        }
    }
}
