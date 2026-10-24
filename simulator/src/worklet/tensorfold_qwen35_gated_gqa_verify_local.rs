//! One Qwen3.8 gated full-attention layer of TensorFold's batched verify
//! forward (`forward.py` `multi_tree_forward`), on one GPU, from the layer's
//! input norm through the `o` projection.
//!
//! ```text
//! _add_rmsnorm        residual add + input RMSNorm (bf16 rows, 64-group sums)
//! quant8_kernel       static FP8 quant of the normed rows (k = hidden)
//! lane_kernel x3      FP8 q (with its gate half), k, v (`_mm_group`)
//! _attn_prep          q/k RMSNorm + partial RoPE (1-D positions: no mRoPE)
//! _shared/_tail/_merge  tree verify attention over every stream's window
//! _gate_mul           attention output x sigmoid(gate)
//! quant8_kernel       static FP8 quant of the gated rows (k = q heads x D)
//! lane_kernel         FP8 o
//! ```
//!
//! Every row-wise leaf runs over W = Σ window rows. The capture shows exactly
//! these 12 launches per GQA layer (`tf27b_n8.sqlite`, `tensorfold.engine.verify`).
//!
//! No KV write: verify reads the window's own keys and values from the
//! projection outputs, and only the committed rows are copied into the cache
//! at commit (`forward.py` `_commit`, a gather and one `_foreach_copy_` over
//! every layer; the commit worklet owns it). The attention plan (`_paths`,
//! host work) runs once a forward, outside this layer.

use crate::arch::qwen35_model_cfg::Qwen35ModelCfg;
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    AttnOutputGateKernel, AttnOutputGateKernelConfig, AttnOutputGateKernelInput,
    GatedQkRmsNormRopeKernel, GatedQkRmsNormRopeKernelConfig, GatedQkRmsNormRopeKernelInput,
    ResidualRmsNormKernel, ResidualRmsNormKernelConfig, ResidualRmsNormKernelInput,
    SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput, StaticFp8QuantKernel,
    StaticFp8QuantKernelConfig, StaticFp8QuantKernelInput, TreeVerifyAttentionKernel,
    TreeVerifyAttentionKernelConfig,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge};

use super::tensorfold_common::{
    atomic, gemm, push_or_zero, VerifyRound, TENSORFOLD_CUDA, TENSORFOLD_LANE, TENSORFOLD_TRITON,
};

/// TensorFold's activations, attention inputs and KV cache are bf16.
const ACTIVATION_DTYPE: DType = DType::Bf16;

/// One TensorFold gated-GQA layer's dimensions and linear precisions. Shared
/// by the verify and prefill worklets; [`Self::from_model`] reads them from
/// the checkpoint config.
#[derive(Clone, Debug)]
pub struct TensorfoldQwen35GatedGqaConfig {
    pub hidden: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    /// q (+gate), k and v.
    pub qkv_proj_dtype: DType,
    pub o_proj_dtype: DType,
    pub gpu_name: String,
}

impl TensorfoldQwen35GatedGqaConfig {
    pub fn from_model(model: &Qwen35ModelCfg, gpu_name: &str) -> Self {
        Self {
            hidden: model.hidden,
            num_q_heads: model.gqa_heads,
            num_kv_heads: model.gqa_kv_heads,
            head_dim: model.gqa_head_dim,
            rotary_dim: model.rotary_dim,
            qkv_proj_dtype: model.precisions.gqa_qkv_proj.gemm_dtype(),
            o_proj_dtype: model.precisions.gqa_o_proj.gemm_dtype(),
            gpu_name: gpu_name.to_string(),
        }
    }

    /// Query (and gate, and attention output) width: q heads x head_dim.
    pub fn q_width(&self) -> u32 {
        self.num_q_heads * self.head_dim
    }

    /// Key (and value) width: KV heads x head_dim.
    pub fn kv_width(&self) -> u32 {
        self.num_kv_heads * self.head_dim
    }

    pub(crate) fn input_norm(&self) -> ResidualRmsNormKernelConfig {
        ResidualRmsNormKernelConfig {
            backends: vec![TENSORFOLD_TRITON],
            gpu_name: self.gpu_name.clone(),
            hidden: self.hidden.into(),
            dtype: ACTIVATION_DTYPE,
        }
    }

    /// `quant8_kernel`: a plain (ungated) static FP8 quant of `k`-wide rows.
    pub(crate) fn quant8(&self, k: u32) -> StaticFp8QuantKernelConfig {
        StaticFp8QuantKernelConfig {
            backends: vec![TENSORFOLD_CUDA],
            gpu_name: self.gpu_name.clone(),
            hidden_dim: k.into(),
            gated: false,
            dtype: ACTIVATION_DTYPE,
        }
    }

    /// `_attn_prep`. TensorFold's text forwards pass one position a row, so
    /// the kernel runs its 1-D RoPE branch (`MROPE=False`) whatever the
    /// checkpoint's `mrope_section`.
    pub(crate) fn attn_prep(&self) -> GatedQkRmsNormRopeKernelConfig {
        GatedQkRmsNormRopeKernelConfig {
            backends: vec![TENSORFOLD_TRITON],
            gpu_name: self.gpu_name.clone(),
            num_q_heads: self.num_q_heads.into(),
            num_kv_heads: self.num_kv_heads.into(),
            head_dim: self.head_dim.into(),
            rotary_dim: self.rotary_dim.into(),
            mrope: false,
            dtype: ACTIVATION_DTYPE,
        }
    }

    /// `_gate_mul`.
    pub(crate) fn gate_mul(&self) -> AttnOutputGateKernelConfig {
        AttnOutputGateKernelConfig {
            backends: vec![TENSORFOLD_TRITON],
            gpu_name: self.gpu_name.clone(),
            num_heads: self.num_q_heads.into(),
            head_dim: self.head_dim.into(),
            dtype: ACTIVATION_DTYPE,
        }
    }

    /// The four projections at `backend`: q with its gate half (n = 2 x
    /// q_width), k, v (n = kv_width each), all k = hidden; o (n = hidden,
    /// k = q_width).
    pub(crate) fn projections(&self, backend: &'static str) -> [SingleGemmKernelConfig; 4] {
        let gpu = &self.gpu_name;
        [
            gemm(
                backend,
                gpu,
                2 * self.q_width(),
                self.hidden,
                self.qkv_proj_dtype,
            ),
            gemm(
                backend,
                gpu,
                self.kv_width(),
                self.hidden,
                self.qkv_proj_dtype,
            ),
            gemm(
                backend,
                gpu,
                self.kv_width(),
                self.hidden,
                self.qkv_proj_dtype,
            ),
            gemm(backend, gpu, self.hidden, self.q_width(), self.o_proj_dtype),
        ]
    }
}

#[derive(Clone, Debug)]
pub struct TensorfoldQwen35GatedGqaVerifyLocalWorkletResolved {
    pub raw_cfg: TensorfoldQwen35GatedGqaConfig,
    pub input_norm: ResidualRmsNormKernelConfig,
    pub qkv_quant: StaticFp8QuantKernelConfig,
    pub q_proj: SingleGemmKernelConfig,
    pub k_proj: SingleGemmKernelConfig,
    pub v_proj: SingleGemmKernelConfig,
    pub attn_prep: GatedQkRmsNormRopeKernelConfig,
    pub attention: TreeVerifyAttentionKernelConfig,
    pub gate_mul: AttnOutputGateKernelConfig,
    pub o_quant: StaticFp8QuantKernelConfig,
    pub o_proj: SingleGemmKernelConfig,
}

pub struct TensorfoldQwen35GatedGqaVerifyLocalWorklet {
    pub name: String,
    input_norm: Op<ResidualRmsNormKernel>,
    qkv_quant: Op<StaticFp8QuantKernel>,
    q_proj: Op<SingleGemmKernel>,
    k_proj: Op<SingleGemmKernel>,
    v_proj: Op<SingleGemmKernel>,
    attn_prep: Op<GatedQkRmsNormRopeKernel>,
    attention: Op<TreeVerifyAttentionKernel>,
    gate_mul: Op<AttnOutputGateKernel>,
    o_quant: Op<StaticFp8QuantKernel>,
    o_proj: Op<SingleGemmKernel>,
    resolved: TensorfoldQwen35GatedGqaVerifyLocalWorkletResolved,
}

impl TensorfoldQwen35GatedGqaVerifyLocalWorklet {
    pub fn resolve_config(
        cfg: &TensorfoldQwen35GatedGqaConfig,
    ) -> TensorfoldQwen35GatedGqaVerifyLocalWorkletResolved {
        let [q_proj, k_proj, v_proj, o_proj] = cfg.projections(TENSORFOLD_LANE);
        TensorfoldQwen35GatedGqaVerifyLocalWorkletResolved {
            input_norm: cfg.input_norm(),
            qkv_quant: cfg.quant8(cfg.hidden),
            q_proj,
            k_proj,
            v_proj,
            attn_prep: cfg.attn_prep(),
            attention: TreeVerifyAttentionKernelConfig {
                backends: vec![TENSORFOLD_TRITON],
                gpu_name: cfg.gpu_name.clone(),
                num_qo_heads: cfg.num_q_heads.into(),
                num_kv_heads: cfg.num_kv_heads.into(),
                head_dim: cfg.head_dim.into(),
                q_dtype: ACTIVATION_DTYPE,
                kv_dtype: ACTIVATION_DTYPE,
                o_dtype: ACTIVATION_DTYPE,
            },
            gate_mul: cfg.gate_mul(),
            o_quant: cfg.quant8(cfg.q_width()),
            o_proj,
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: TensorfoldQwen35GatedGqaVerifyLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let n = name.as_str();
        Ok(Self {
            input_norm: atomic(
                n,
                "input_norm",
                r.input_norm,
                ResidualRmsNormKernel::build,
                bridge,
            )?,
            qkv_quant: atomic(
                n,
                "qkv_quant",
                r.qkv_quant,
                StaticFp8QuantKernel::build,
                bridge,
            )?,
            q_proj: atomic(n, "q_proj", r.q_proj, SingleGemmKernel::build, bridge)?,
            k_proj: atomic(n, "k_proj", r.k_proj, SingleGemmKernel::build, bridge)?,
            v_proj: atomic(n, "v_proj", r.v_proj, SingleGemmKernel::build, bridge)?,
            attn_prep: atomic(
                n,
                "attn_prep",
                r.attn_prep,
                GatedQkRmsNormRopeKernel::build,
                bridge,
            )?,
            attention: atomic(
                n,
                "attention",
                r.attention,
                TreeVerifyAttentionKernel::build,
                bridge,
            )?,
            gate_mul: atomic(
                n,
                "gate_mul",
                r.gate_mul,
                AttnOutputGateKernel::build,
                bridge,
            )?,
            o_quant: atomic(n, "o_quant", r.o_quant, StaticFp8QuantKernel::build, bridge)?,
            o_proj: atomic(n, "o_proj", r.o_proj, SingleGemmKernel::build, bridge)?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        CostNode::Labeled {
            label: format!(
                "{} (TensorfoldQwen35GatedGqaVerifyLocalWorklet) [local (1 GPU); Hq={}, Hkv={}, D={}]",
                self.name, cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim
            ),
            child: Box::new(CostNode::Sum(vec![
                self.input_norm.compile(builder),
                self.qkv_quant.compile(builder),
                self.q_proj.compile(builder),
                self.k_proj.compile(builder),
                self.v_proj.compile(builder),
                self.attn_prep.compile(builder),
                self.attention.compile(builder),
                self.gate_mul.compile(builder),
                self.o_quant.compile(builder),
                self.o_proj.compile(builder),
            ])),
        }
    }

    /// Every leaf is one launch over the round's W rows; an empty round is
    /// zero throughout.
    pub fn eval(&self, round: &VerifyRound, ev: &mut Evaluator) {
        let rows = round.total_rows();
        let zero = round.is_empty();
        let gemm = SingleGemmKernelInput { m: rows };
        let quant = StaticFp8QuantKernelInput { num_tokens: rows };
        push_or_zero(
            &self.input_norm,
            ResidualRmsNormKernelInput { m: rows },
            zero,
            ev,
        );
        push_or_zero(&self.qkv_quant, quant.clone(), zero, ev);
        push_or_zero(&self.q_proj, gemm.clone(), zero, ev);
        push_or_zero(&self.k_proj, gemm.clone(), zero, ev);
        push_or_zero(&self.v_proj, gemm.clone(), zero, ev);
        push_or_zero(
            &self.attn_prep,
            GatedQkRmsNormRopeKernelInput { num_tokens: rows },
            zero,
            ev,
        );
        push_or_zero(&self.attention, round.attention_input(), zero, ev);
        push_or_zero(
            &self.gate_mul,
            AttnOutputGateKernelInput { num_tokens: rows },
            zero,
            ev,
        );
        push_or_zero(&self.o_quant, quant, zero, ev);
        push_or_zero(&self.o_proj, gemm, zero, ev);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::timing::cache::interp::CoverageFlags;
    use crate::timing::LeafMetrics;
    use crate::worklet::tensorfold_common::{VerifyStream, WindowMode};
    use std::path::Path;

    pub(crate) fn qwen38_gqa(gpu: &str) -> TensorfoldQwen35GatedGqaConfig {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("model/config/qwen3_8_27b_nvfp4.json");
        TensorfoldQwen35GatedGqaConfig::from_model(&Qwen35ModelCfg::from_json(&path).unwrap(), gpu)
    }

    fn build(bridge: &PerfApiBridge) -> TensorfoldQwen35GatedGqaVerifyLocalWorklet {
        let resolved =
            TensorfoldQwen35GatedGqaVerifyLocalWorklet::resolve_config(&qwen38_gqa("NVIDIA GB10"));
        TensorfoldQwen35GatedGqaVerifyLocalWorklet::build("m.gqa".into(), resolved, bridge)
            .expect("verify GQA worklet builds")
    }

    fn eval(
        worklet: &TensorfoldQwen35GatedGqaVerifyLocalWorklet,
        round: &VerifyRound,
    ) -> Vec<LeafMetrics> {
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        let n = builder.finish(root).n_slots();
        let mut buf = vec![LeafMetrics::MISS; n];
        let mut ev = Evaluator::new(&mut buf);
        worklet.eval(round, &mut ev);
        assert_eq!(ev.filled(), n);
        buf
    }

    /// A window as attention sees it; acceptance and tree shape do not
    /// reach this layer.
    fn stream(committed_len: u32, rows: u32) -> VerifyStream {
        VerifyStream {
            committed_len,
            rows,
            accepted: 0,
            tree_slots: 0,
            mode: WindowMode::Copy,
        }
    }

    #[test]
    fn shapes_are_the_profiled_gb10_rows() {
        let r = TensorfoldQwen35GatedGqaVerifyLocalWorklet::resolve_config(&qwen38_gqa("g"));
        let nk = |g: &SingleGemmKernelConfig| (g.n.get(), g.k.get(), g.dtype, g.backends[0]);
        assert_eq!(
            nk(&r.q_proj),
            (12288, 5120, DType::Fp8E4m3, TENSORFOLD_LANE)
        );
        assert_eq!(nk(&r.k_proj), (1024, 5120, DType::Fp8E4m3, TENSORFOLD_LANE));
        assert_eq!(nk(&r.v_proj), (1024, 5120, DType::Fp8E4m3, TENSORFOLD_LANE));
        assert_eq!(nk(&r.o_proj), (5120, 6144, DType::Fp8E4m3, TENSORFOLD_LANE));
        assert_eq!(r.input_norm.hidden.get(), 5120);
        assert_eq!(
            (r.qkv_quant.hidden_dim.get(), r.o_quant.hidden_dim.get()),
            (5120, 6144)
        );
        assert!(!r.qkv_quant.gated && !r.o_quant.gated);
        assert_eq!(r.attn_prep.rotary_dim.get(), 64);
        assert!(!r.attn_prep.mrope);
        assert_eq!(
            (r.gate_mul.num_heads.get(), r.gate_mul.head_dim.get()),
            (24, 256)
        );
        assert_eq!(r.attention.kv_dtype, DType::Bf16);
    }

    #[test]
    fn one_layer_is_ten_leaves_in_launch_order() {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let worklet = build(&bridge);
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        let tree = builder.finish(root);
        let leaves: Vec<_> = tree
            .slots
            .iter()
            .map(|s| (s.name.as_str(), s.kind.as_str()))
            .collect();
        assert_eq!(
            leaves,
            [
                ("m.gqa.input_norm", "residual_rms_norm"),
                ("m.gqa.qkv_quant", "static_fp8_quant"),
                ("m.gqa.q_proj", "single_gemm"),
                ("m.gqa.k_proj", "single_gemm"),
                ("m.gqa.v_proj", "single_gemm"),
                ("m.gqa.attn_prep", "gated_qk_rms_norm_rope"),
                ("m.gqa.attention", "tree_verify_attention"),
                ("m.gqa.gate_mul", "attn_output_gate"),
                ("m.gqa.o_quant", "static_fp8_quant"),
                ("m.gqa.o_proj", "single_gemm"),
            ]
        );
        // An empty round fills every slot with zero, no lookup.
        for leaf in eval(&worklet, &VerifyRound::default()) {
            assert_eq!(leaf.m.time_ms, 0.0);
            assert_eq!(leaf.coverage, CoverageFlags::EMPTY);
        }
    }

    /// Against the GB10 rows in `profiling/profile.db`, on verify rounds of
    /// `tf27b_n8.sqlite` (`tensorfold.engine.verify`, joined to
    /// `iter_records.jsonl`). `TF_GQA_ROUNDS=<json>` also prices a list of
    /// `{"streams": [[committed, rows], ...], "captured_ms": x}` rounds and
    /// prints the error over them. Run from the repo root with libpython on
    /// `LD_LIBRARY_PATH`:
    /// `PYTHONPATH=$PWD uv run cargo test -p simulator --lib tensorfold_qwen35_gated_gqa_verify -- --include-ignored`.
    #[test]
    #[ignore = "needs profile.db GB10 rows and the venv interpreter"]
    fn gb10_rows_price_captured_rounds() {
        let bridge = PerfApiBridge::new().expect("perf_api bridge");
        let worklet = build(&bridge);
        let price = |pairs: &[(u32, u32)]| -> Vec<LeafMetrics> {
            let round = VerifyRound {
                streams: pairs.iter().map(|&(c, r)| stream(c, r)).collect(),
            };
            eval(&worklet, &round)
        };
        // (committed, rows) per stream; captured ms over the 16 GQA layers.
        let rounds: [(&[(u32, u32)], f32); 4] = [
            (
                &[
                    (33912, 7),
                    (22213, 16),
                    (21288, 11),
                    (26766, 16),
                    (7771, 9),
                    (15555, 8),
                ],
                50.768,
            ),
            (
                &[
                    (33839, 8),
                    (27537, 10),
                    (22137, 16),
                    (21229, 14),
                    (26708, 8),
                    (7719, 10),
                    (9020, 12),
                    (15506, 10),
                ],
                62.964,
            ),
            (
                &[
                    (22328, 12),
                    (21458, 8),
                    (26923, 15),
                    (7902, 14),
                    (15780, 13),
                    (30338, 16),
                    (9327, 13),
                ],
                54.841,
            ),
            (
                &[
                    (22533, 16),
                    (21593, 16),
                    (27101, 12),
                    (16080, 16),
                    (30519, 15),
                    (9451, 16),
                    (33621, 16),
                    (8633, 16),
                ],
                67.698,
            ),
        ];
        for (pairs, captured) in rounds {
            let leaves = price(pairs);
            let total: f32 = leaves.iter().map(|l| l.m.time_ms).sum();
            let parts: Vec<String> = leaves
                .iter()
                .map(|l| format!("{:.4}", l.m.time_ms))
                .collect();
            let w: u32 = pairs.iter().map(|p| p.1).sum();
            eprintln!(
                "gqa verify W={w}: x16 {:.3} ms vs captured {captured:.3} ({:+.1}%) [{}]",
                16.0 * total,
                100.0 * (16.0 * total / captured - 1.0),
                parts.join(", ")
            );
            for leaf in &leaves {
                assert!(leaf.m.time_ms > 0.0 && leaf.m.time_ms.is_finite());
            }
        }
        if let Ok(path) = std::env::var("TF_GQA_ROUNDS") {
            let text = std::fs::read_to_string(path).unwrap();
            let rounds: Vec<serde_json::Value> = serde_json::from_str(&text).unwrap();
            let (mut predicted, mut captured, mut abs_err) = (0.0f64, 0.0f64, 0.0f64);
            for r in &rounds {
                let pairs: Vec<(u32, u32)> = r["streams"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| (p[0].as_u64().unwrap() as u32, p[1].as_u64().unwrap() as u32))
                    .collect();
                let p = 16.0
                    * price(&pairs)
                        .iter()
                        .map(|l| f64::from(l.m.time_ms))
                        .sum::<f64>();
                let c = r["captured_ms"].as_f64().unwrap();
                predicted += p;
                captured += c;
                abs_err += (p / c - 1.0).abs();
            }
            eprintln!(
                "gqa verify over {} rounds: predicted {predicted:.1} ms vs captured {captured:.1} ms \
                 ({:+.1}%), mean |err| {:.1}%",
                rounds.len(),
                100.0 * (predicted / captured - 1.0),
                100.0 * abs_err / rounds.len() as f64
            );
        }
    }
}
