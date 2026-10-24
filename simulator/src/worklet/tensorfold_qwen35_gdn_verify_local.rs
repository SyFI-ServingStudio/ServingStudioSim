//! One Qwen3.5-family Gated DeltaNet layer in TensorFold's batched verify
//! forward (`forward.py` `multi_tree_forward`, `layer.linear` branch), on one
//! GPU, over every decoding stream's window at once (W = Σ rows).
//!
//! The section owns the layer's input `_add_rmsnorm` (the residual add of the
//! previous layer's MLP output, then RMSNorm) and ends at the `out` projection
//! output. Layer 0 has no residual (`pending=None`): the arch builds that
//! instance with `residual_input_norm = false` and prices the first norm itself.
//!
//! ```text
//! _add_rmsnorm                      residual_rms_norm      m = W
//! quant8 (qkv and z share it)       static_fp8_quant       hidden k
//! lane qkv, lane z                  single_gemm lane FP8   m = W
//! b16 b|a pair                      single_gemm b16 bf16   n = 2 x value heads
//! torch.cat of the streams' conv    elementwise (pending)  only when streams > 1
//!   states (`forward.py` ~370)
//! _gdn_pre                          gdn_conv_norm_gates    num_tokens = W
//! tree_kernel                       gdn_tree_verify_recurrence (one launch)
//! _gated_norm                       gdn_gated_rms_norm     m = W x value heads
//! quant8                            static_fp8_quant       value width k
//! lane out                          single_gemm lane FP8   m = W
//! ```
//!
//! On NVFP4 checkpoints the GDN linears stay separate (`zba` is an MLX-only
//! stack): `_mm_group([qkv, z, b, a])` quantizes the rows once for the two FP8
//! linears (`checkpoint.matmul_group`), then runs the bf16 b/a pair in one
//! launch (`b16.matmul_pair`). The n8 capture has, per verify forward, 48 each
//! of `b16_kernel`, `_gdn_pre`, `tree_kernel`, `_gated_norm`, 48 x 3 lane and
//! 48 x 2 quant8 launches for the GDN layers, and 48 conv-state cats.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, GdnConvNormGatesKernel,
    GdnConvNormGatesKernelConfig, GdnConvNormGatesKernelInput, GdnGatedRmsNormKernel,
    GdnGatedRmsNormKernelConfig, GdnGatedRmsNormKernelInput, GdnTreeVerifyRecurrenceKernel,
    GdnTreeVerifyRecurrenceKernelConfig, ResidualRmsNormKernel, ResidualRmsNormKernelConfig,
    ResidualRmsNormKernelInput, SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
    StaticFp8QuantKernel, StaticFp8QuantKernelConfig, StaticFp8QuantKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge};

use super::tensorfold_common::{
    atomic, elementwise, gemm, push_or_zero, VerifyRound, TENSORFOLD_B16, TENSORFOLD_CUDA,
    TENSORFOLD_LANE, TENSORFOLD_TRITON,
};

/// One GDN layer's dimensions and precisions, from the checkpoint config
/// (`Qwen35ModelCfg`). Shared by the verify and prefill GDN worklets.
#[derive(Clone, Debug)]
pub struct TensorfoldQwen35GdnLocalWorkletConfig {
    pub hidden: u32,
    pub num_key_heads: u32,
    pub num_value_heads: u32,
    pub key_head_dim: u32,
    pub value_head_dim: u32,
    pub conv_kernel_size: u32,
    /// bf16: the residual stream, the conv state and the GDN tensors.
    pub activation_dtype: DType,
    /// The recurrent state (`mamba_ssm_dtype`, fp32).
    pub ssm_state_dtype: DType,
    /// `qkv`, `z` and `out` weights (FP8 per-tensor on Qwen3.8-27B-NVFP4).
    pub proj_dtype: DType,
    /// `b` / `a` weights (bf16).
    pub ba_dtype: DType,
    pub gpu_name: String,
    /// The section starts with the layer's residual `_add_rmsnorm`. `false`
    /// for layer 0, whose norm has no residual and belongs to the arch.
    pub residual_input_norm: bool,
    /// Backends of the conv-state `torch.cat` launches (tf-copies,
    /// `elementwise` / `tensorfold_torch_cat`). `None` until those rows are
    /// measured: the slots are then absent.
    pub cat_backends: Option<Vec<&'static str>>,
}

impl TensorfoldQwen35GdnLocalWorkletConfig {
    /// q | k | v channels: the conv's width and the `qkv` projection's n.
    pub fn conv_channels(&self) -> u32 {
        2 * self.num_key_heads * self.key_head_dim + self.num_value_heads * self.value_head_dim
    }

    /// `z`'s n and `out`'s k.
    pub fn value_width(&self) -> u32 {
        self.num_value_heads * self.value_head_dim
    }

    /// Rows of the conv state each stream keeps (`conv_kernel - 1`).
    pub fn conv_keep(&self) -> u32 {
        self.conv_kernel_size - 1
    }

    pub(crate) fn input_norm(&self) -> Option<ResidualRmsNormKernelConfig> {
        self.residual_input_norm
            .then(|| ResidualRmsNormKernelConfig {
                backends: vec![TENSORFOLD_TRITON],
                gpu_name: self.gpu_name.clone(),
                hidden: self.hidden.into(),
                dtype: self.activation_dtype,
            })
    }

    pub(crate) fn quant8(&self, hidden_dim: u32) -> StaticFp8QuantKernelConfig {
        StaticFp8QuantKernelConfig {
            backends: vec![TENSORFOLD_CUDA],
            gpu_name: self.gpu_name.clone(),
            hidden_dim: hidden_dim.into(),
            gated: false,
            dtype: self.activation_dtype,
        }
    }

    pub(crate) fn conv_gates(&self) -> GdnConvNormGatesKernelConfig {
        GdnConvNormGatesKernelConfig {
            backends: vec![TENSORFOLD_TRITON],
            gpu_name: self.gpu_name.clone(),
            num_qk_heads: self.num_key_heads.into(),
            num_value_heads: self.num_value_heads.into(),
            key_head_dim: self.key_head_dim.into(),
            value_head_dim: self.value_head_dim.into(),
            kernel_size: self.conv_kernel_size.into(),
            dtype: self.activation_dtype,
        }
    }

    pub(crate) fn gated_norm(&self) -> GdnGatedRmsNormKernelConfig {
        GdnGatedRmsNormKernelConfig {
            backends: vec![TENSORFOLD_TRITON],
            gpu_name: self.gpu_name.clone(),
            hidden: self.value_head_dim.into(),
            dtype: self.activation_dtype,
        }
    }

    /// A `torch.cat` whose rows are `bytes_per_row` wide (read once, written
    /// once).
    pub(crate) fn cat(&self, bytes_per_row: u32) -> Option<ElementwiseKernelConfig> {
        self.cat_backends
            .as_ref()
            .map(|backends| elementwise(backends, &self.gpu_name, bytes_per_row, bytes_per_row))
    }

    /// One conv-state row: every q/k/v channel at the activation dtype.
    pub(crate) fn conv_row_bytes(&self) -> u32 {
        self.conv_channels() * self.activation_dtype.size_bytes()
    }
}

#[derive(Clone, Debug)]
pub struct TensorfoldQwen35GdnVerifyLocalWorkletResolved {
    pub raw_cfg: TensorfoldQwen35GdnLocalWorkletConfig,
    pub input_norm: Option<ResidualRmsNormKernelConfig>,
    pub in_quant: StaticFp8QuantKernelConfig,
    pub qkv: SingleGemmKernelConfig,
    pub z: SingleGemmKernelConfig,
    /// `b16.matmul_pair`: b and a in one launch, n = 2 x value heads.
    pub ba: SingleGemmKernelConfig,
    /// `torch.cat([st.conv[i] for st in states])`, one launch per layer when
    /// two or more streams verify: S x (conv_kernel - 1) rows of
    /// `conv_channels` bf16, read once and written once.
    pub conv_cat: Option<ElementwiseKernelConfig>,
    pub conv_gates: GdnConvNormGatesKernelConfig,
    pub tree: GdnTreeVerifyRecurrenceKernelConfig,
    pub gated_norm: GdnGatedRmsNormKernelConfig,
    pub out_quant: StaticFp8QuantKernelConfig,
    pub out: SingleGemmKernelConfig,
}

pub type TensorfoldQwen35GdnVerifyLocalWorkletConfig = TensorfoldQwen35GdnLocalWorkletConfig;

/// The verify round every GDN layer sees.
pub type TensorfoldQwen35GdnVerifyLocalWorkletInput = VerifyRound;

pub struct TensorfoldQwen35GdnVerifyLocalWorklet {
    pub name: String,
    input_norm: Option<Op<ResidualRmsNormKernel>>,
    in_quant: Op<StaticFp8QuantKernel>,
    qkv: Op<SingleGemmKernel>,
    z: Op<SingleGemmKernel>,
    ba: Op<SingleGemmKernel>,
    conv_cat: Option<Op<ElementwiseKernel>>,
    conv_gates: Op<GdnConvNormGatesKernel>,
    tree: Op<GdnTreeVerifyRecurrenceKernel>,
    gated_norm: Op<GdnGatedRmsNormKernel>,
    out_quant: Op<StaticFp8QuantKernel>,
    out: Op<SingleGemmKernel>,
    resolved: TensorfoldQwen35GdnVerifyLocalWorkletResolved,
}

impl TensorfoldQwen35GdnVerifyLocalWorklet {
    pub fn resolve_config(
        cfg: &TensorfoldQwen35GdnLocalWorkletConfig,
    ) -> TensorfoldQwen35GdnVerifyLocalWorkletResolved {
        let gpu = cfg.gpu_name.as_str();
        TensorfoldQwen35GdnVerifyLocalWorkletResolved {
            input_norm: cfg.input_norm(),
            in_quant: cfg.quant8(cfg.hidden),
            qkv: gemm(
                TENSORFOLD_LANE,
                gpu,
                cfg.conv_channels(),
                cfg.hidden,
                cfg.proj_dtype,
            ),
            z: gemm(
                TENSORFOLD_LANE,
                gpu,
                cfg.value_width(),
                cfg.hidden,
                cfg.proj_dtype,
            ),
            ba: gemm(
                TENSORFOLD_B16,
                gpu,
                2 * cfg.num_value_heads,
                cfg.hidden,
                cfg.ba_dtype,
            ),
            conv_cat: cfg.cat(cfg.conv_row_bytes()),
            conv_gates: cfg.conv_gates(),
            tree: GdnTreeVerifyRecurrenceKernelConfig {
                backends: vec![TENSORFOLD_CUDA],
                gpu_name: cfg.gpu_name.clone(),
                num_qk_heads: cfg.num_key_heads.into(),
                num_value_heads: cfg.num_value_heads.into(),
                key_head_dim: cfg.key_head_dim.into(),
                value_head_dim: cfg.value_head_dim.into(),
                dtype: cfg.activation_dtype,
                state_dtype: cfg.ssm_state_dtype,
            },
            gated_norm: cfg.gated_norm(),
            out_quant: cfg.quant8(cfg.value_width()),
            out: gemm(
                TENSORFOLD_LANE,
                gpu,
                cfg.hidden,
                cfg.value_width(),
                cfg.proj_dtype,
            ),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: TensorfoldQwen35GdnVerifyLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let n = name.as_str();
        let gemm_op = |suffix: &str, config: SingleGemmKernelConfig| {
            atomic(n, suffix, config, SingleGemmKernel::build, bridge)
        };
        let quant_op = |suffix: &str, config: StaticFp8QuantKernelConfig| {
            atomic(n, suffix, config, StaticFp8QuantKernel::build, bridge)
        };
        Ok(Self {
            input_norm: r
                .input_norm
                .map(|c| atomic(n, "input_norm", c, ResidualRmsNormKernel::build, bridge))
                .transpose()?,
            in_quant: quant_op("in_quant", r.in_quant)?,
            qkv: gemm_op("qkv", r.qkv)?,
            z: gemm_op("z", r.z)?,
            ba: gemm_op("ba", r.ba)?,
            conv_cat: r
                .conv_cat
                .map(|c| atomic(n, "conv_cat", c, ElementwiseKernel::build, bridge))
                .transpose()?,
            conv_gates: atomic(
                n,
                "conv_gates",
                r.conv_gates,
                GdnConvNormGatesKernel::build,
                bridge,
            )?,
            tree: atomic(
                n,
                "tree",
                r.tree,
                GdnTreeVerifyRecurrenceKernel::build,
                bridge,
            )?,
            gated_norm: atomic(
                n,
                "gated_norm",
                r.gated_norm,
                GdnGatedRmsNormKernel::build,
                bridge,
            )?,
            out_quant: quant_op("out_quant", r.out_quant)?,
            out: gemm_op("out", r.out)?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        let mut children = Vec::with_capacity(11);
        if let Some(op) = &self.input_norm {
            children.push(op.compile(builder));
        }
        children.push(self.in_quant.compile(builder));
        children.push(self.qkv.compile(builder));
        children.push(self.z.compile(builder));
        children.push(self.ba.compile(builder));
        if let Some(op) = &self.conv_cat {
            children.push(op.compile(builder));
        }
        children.push(self.conv_gates.compile(builder));
        children.push(self.tree.compile(builder));
        children.push(self.gated_norm.compile(builder));
        children.push(self.out_quant.compile(builder));
        children.push(self.out.compile(builder));
        CostNode::Labeled {
            label: format!(
                "{} (TensorfoldQwen35GdnVerifyLocalWorklet) [local (1 GPU); Hk={}, Hv={}, K={}, V={}]",
                self.name,
                cfg.num_key_heads,
                cfg.num_value_heads,
                cfg.key_head_dim,
                cfg.value_head_dim
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    /// Every leaf is one launch over the round's W rows; an empty round is
    /// zero in every slot.
    pub fn eval(&self, round: &VerifyRound, ev: &mut Evaluator) {
        let cfg = &self.resolved.raw_cfg;
        let rows = round.total_rows();
        let empty = round.is_empty();
        let gemm_in = SingleGemmKernelInput { m: rows };
        let quant_in = StaticFp8QuantKernelInput { num_tokens: rows };
        if let Some(op) = &self.input_norm {
            push_or_zero(op, ResidualRmsNormKernelInput { m: rows }, empty, ev);
        }
        push_or_zero(&self.in_quant, quant_in.clone(), empty, ev);
        push_or_zero(&self.qkv, gemm_in.clone(), empty, ev);
        push_or_zero(&self.z, gemm_in.clone(), empty, ev);
        push_or_zero(&self.ba, gemm_in.clone(), empty, ev);
        if let Some(op) = &self.conv_cat {
            let rows = conv_cat_rows(round, cfg.conv_keep());
            push_or_zero(
                op,
                ElementwiseKernelInput {
                    num_tokens: rows.unwrap_or(0),
                },
                rows.is_none(),
                ev,
            );
        }
        push_or_zero(
            &self.conv_gates,
            GdnConvNormGatesKernelInput { num_tokens: rows },
            empty,
            ev,
        );
        push_or_zero(&self.tree, round.gdn_tree_input(), empty, ev);
        push_or_zero(
            &self.gated_norm,
            GdnGatedRmsNormKernelInput {
                m: rows * cfg.num_value_heads,
            },
            empty,
            ev,
        );
        push_or_zero(&self.out_quant, quant_in, empty, ev);
        push_or_zero(&self.out, gemm_in, empty, ev);
    }
}

/// Conv-state rows the layer's `torch.cat` copies: every stream's kept rows,
/// or `None` when fewer than two streams verify (one stream's state is read
/// in place).
fn conv_cat_rows(round: &VerifyRound, keep: u32) -> Option<u32> {
    (round.num_streams() >= 2).then(|| round.num_streams() * keep)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::timing::LeafMetrics;
    use crate::worklet::tensorfold_common::{VerifyStream, WindowMode};

    /// Qwen3.8-27B-NVFP4's GDN layer on GB10.
    pub(crate) fn cfg(
        residual_input_norm: bool,
        cat: bool,
    ) -> TensorfoldQwen35GdnLocalWorkletConfig {
        TensorfoldQwen35GdnLocalWorkletConfig {
            hidden: 5120,
            num_key_heads: 16,
            num_value_heads: 48,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel_size: 4,
            activation_dtype: DType::Bf16,
            ssm_state_dtype: DType::Fp32,
            proj_dtype: DType::Fp8E4m3,
            ba_dtype: DType::Bf16,
            gpu_name: "NVIDIA GB10".into(),
            residual_input_norm,
            cat_backends: cat.then(|| vec!["tensorfold_torch_cat"]),
        }
    }

    pub(crate) fn stream(committed_len: u32, rows: u32, tree_slots: u32) -> VerifyStream {
        VerifyStream {
            committed_len,
            rows,
            accepted: rows / 2,
            tree_slots,
            mode: if tree_slots > 0 {
                WindowMode::Tree
            } else {
                WindowMode::Copy
            },
        }
    }

    fn build(cfg: &TensorfoldQwen35GdnLocalWorkletConfig) -> TensorfoldQwen35GdnVerifyLocalWorklet {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        TensorfoldQwen35GdnVerifyLocalWorklet::build(
            "m.gdn".into(),
            TensorfoldQwen35GdnVerifyLocalWorklet::resolve_config(cfg),
            &bridge,
        )
        .unwrap()
    }

    fn leaves(worklet: &TensorfoldQwen35GdnVerifyLocalWorklet) -> Vec<(String, String)> {
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        let tree = builder.finish(root);
        tree.slots
            .iter()
            .map(|s| (s.name.clone(), s.kind.clone()))
            .collect()
    }

    #[test]
    fn shapes_follow_the_checkpoint_config() {
        let r = TensorfoldQwen35GdnVerifyLocalWorklet::resolve_config(&cfg(true, true));
        assert_eq!((r.qkv.n.get(), r.qkv.k.get()), (10240, 5120));
        assert_eq!((r.z.n.get(), r.z.k.get()), (6144, 5120));
        assert_eq!((r.ba.n.get(), r.ba.k.get()), (96, 5120));
        assert_eq!((r.out.n.get(), r.out.k.get()), (5120, 6144));
        assert_eq!(
            (r.in_quant.hidden_dim.get(), r.out_quant.hidden_dim.get()),
            (5120, 6144)
        );
        assert_eq!(r.qkv.backends, [TENSORFOLD_LANE]);
        assert_eq!(r.ba.backends, [TENSORFOLD_B16]);
        assert_eq!(r.gated_norm.hidden.get(), 128);
        assert_eq!(r.conv_cat.unwrap().input_bytes_per_token.get(), 20480);
    }

    #[test]
    fn leaves_follow_multi_tree_forward() {
        let names: Vec<_> = leaves(&build(&cfg(true, true)))
            .into_iter()
            .map(|(name, kind)| format!("{name}:{kind}"))
            .collect();
        assert_eq!(
            names,
            [
                "m.gdn.input_norm:residual_rms_norm",
                "m.gdn.in_quant:static_fp8_quant",
                "m.gdn.qkv:single_gemm",
                "m.gdn.z:single_gemm",
                "m.gdn.ba:single_gemm",
                "m.gdn.conv_cat:elementwise",
                "m.gdn.conv_gates:gdn_conv_norm_gates",
                "m.gdn.tree:gdn_tree_verify_recurrence",
                "m.gdn.gated_norm:gdn_gated_rms_norm",
                "m.gdn.out_quant:static_fp8_quant",
                "m.gdn.out:single_gemm",
            ]
        );
        // Layer 0 without its norm, and no cat before tf-copies lands.
        let first = leaves(&build(&cfg(false, false)));
        assert_eq!(first.len(), 9);
        assert_eq!(first[0].0, "m.gdn.in_quant");
    }

    #[test]
    fn empty_round_is_zero_and_one_stream_skips_the_cat() {
        let worklet = build(&cfg(true, true));
        let mut buf = vec![LeafMetrics::MISS; 11];
        let mut ev = Evaluator::new(&mut buf);
        worklet.eval(&VerifyRound::default(), &mut ev);
        assert_eq!(ev.filled(), 11);
        assert!(buf.iter().all(|m| m.m.time_ms == 0.0));

        let one = VerifyRound {
            streams: vec![stream(1000, 16, 4)],
        };
        assert_eq!(conv_cat_rows(&one, 3), None);
        let two = VerifyRound {
            streams: vec![stream(1000, 16, 4), stream(50, 3, 0)],
        };
        assert_eq!(conv_cat_rows(&two, 3), Some(6));
    }

    /// Captured n8 verify forwards: per stream `(committed_len, rows, accepted,
    /// tree_slots)`, and the measured kernel time of all 48 GDN layers (ms;
    /// nsys durations, `_add_rmsnorm` through the out lane, the conv cat
    /// included). Shapes from `iter_records.jsonl` joined to the
    /// `tensorfold.engine.verify` ranges of `tf27b_n8.sqlite`; slots from
    /// `slot_count_for_parents`.
    #[allow(clippy::type_complexity)]
    pub(crate) const N8_VERIFY_ROUNDS: &[(&[(u32, u32, u32, u32)], f64)] = &[
        (
            &[
                (33912, 7, 0, 1),
                (22213, 16, 11, 2),
                (21288, 11, 2, 2),
                (26766, 16, 6, 1),
                (7771, 9, 2, 3),
                (15555, 8, 2, 1),
            ],
            36.334,
        ),
        (
            &[
                (33913, 7, 0, 1),
                (22225, 16, 5, 3),
                (21291, 10, 2, 2),
                (26773, 16, 7, 2),
                (7774, 9, 2, 1),
                (15558, 15, 2, 1),
                (30212, 12, 1, 3),
            ],
            38.626,
        ),
        (
            &[
                (22328, 12, 3, 2),
                (21458, 8, 3, 1),
                (26923, 15, 2, 2),
                (7902, 14, 0, 1),
                (15780, 13, 11, 1),
                (30338, 16, 8, 1),
                (9327, 13, 1, 2),
            ],
            37.207,
        ),
        (
            &[
                (33862, 16, 2, 2),
                (27574, 16, 15, 0),
                (22169, 11, 2, 2),
                (21241, 9, 3, 2),
                (26730, 9, 3, 1),
                (7734, 16, 1, 3),
                (9037, 16, 1, 1),
                (15516, 11, 1, 2),
            ],
            41.117,
        ),
        (
            &[
                (22533, 16, 5, 0),
                (21593, 16, 2, 1),
                (27101, 12, 3, 2),
                (16080, 16, 0, 2),
                (30519, 15, 0, 2),
                (9451, 16, 2, 2),
                (33621, 16, 6, 2),
                (8633, 16, 3, 3),
            ],
            43.942,
        ),
    ];

    /// The per-layer GDN cost on captured n8 verify shapes, against the GB10
    /// rows in `profiling/profile.db`. Run from the repo root with libpython
    /// on `LD_LIBRARY_PATH`:
    /// `PYTHONPATH=$PWD uv run cargo test -p simulator --lib tensorfold_qwen35_gdn_verify -- --include-ignored --nocapture`.
    #[test]
    #[ignore = "needs profile.db GB10 rows and the venv interpreter"]
    fn gb10_rows_price_captured_verify_rounds() {
        let bridge = crate::timing::PerfApiBridge::new().expect("perf_api bridge");
        let worklet = TensorfoldQwen35GdnVerifyLocalWorklet::build(
            "m.gdn".into(),
            TensorfoldQwen35GdnVerifyLocalWorklet::resolve_config(&cfg(true, false)),
            &bridge,
        )
        .expect("GB10 rows for every GDN verify leaf");
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        let tree = builder.finish(root);
        for &(streams, measured_ms) in N8_VERIFY_ROUNDS {
            let round = VerifyRound {
                streams: streams
                    .iter()
                    .map(
                        |&(committed_len, rows, accepted, tree_slots)| VerifyStream {
                            committed_len,
                            rows,
                            accepted,
                            tree_slots,
                            mode: WindowMode::Tree,
                        },
                    )
                    .collect(),
            };
            let mut buf = vec![LeafMetrics::MISS; tree.n_slots()];
            let mut ev = Evaluator::new(&mut buf);
            worklet.eval(&round, &mut ev);
            let per_leaf: Vec<String> = tree
                .slots
                .iter()
                .zip(&buf)
                .map(|(s, m)| {
                    format!(
                        "{}={:.1}",
                        s.name.trim_start_matches("m.gdn."),
                        1000.0 * m.m.time_ms
                    )
                })
                .collect();
            let total: f32 = buf.iter().map(|m| m.m.time_ms).sum();
            let predicted = 48.0 * f64::from(total);
            eprintln!(
                "verify W={} S={} {:?}: layer {:.1} us, x48 {predicted:.3} ms vs capture {measured_ms:.3} ms ({:+.1}%) [{}]",
                round.total_rows(),
                round.num_streams(),
                round.slot_class(),
                1000.0 * total,
                100.0 * (predicted - measured_ms) / measured_ms,
                per_leaf.join(" ")
            );
            assert!(buf
                .iter()
                .all(|m| m.m.time_ms > 0.0 && m.m.time_ms.is_finite()));
        }
    }
}
