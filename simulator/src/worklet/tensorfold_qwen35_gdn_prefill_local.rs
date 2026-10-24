//! One Qwen3.5-family Gated DeltaNet layer in one TensorFold prefill step
//! (`prefill.py`: `prefill_chunk` for a single piece, `prefill_rows` for two or
//! more), on one GPU.
//!
//! Same boundary as the verify GDN worklet: it owns the layer's input
//! `_add_rmsnorm` (layer 0's no-residual norm belongs to the arch:
//! `residual_input_norm = false`) and ends at the `out` projection output.
//!
//! ```text
//! per forward (PrefillForward::forward_rows: each chunk, or every piece's rows at once)
//!   _add_rmsnorm                      residual_rms_norm      m = rows
//!   quant8 (qkv and z share it)       static_fp8_quant       hidden k
//!   ck qkv, ck z                      single_gemm ck FP8     m = rows
//!   b16_prompt b|a pair               single_gemm b16_prompt n = 2 x value heads
//!   _gdn_pre                          gdn_conv_norm_gates    num_tokens = rows
//!   _gated_norm                       gdn_gated_rms_norm     m = rows x value heads
//!   quant8                            static_fp8_quant       value width k
//!   ck out                            single_gemm ck FP8     m = rows
//! per sequence launch (PrefillForward::launches: each chunk, or each piece)
//!   chain_kernel                      TensorfoldGdnChainOp
//! torch.cat copies (pending tf-copies rows; absent without `cat_backends`)
//!   conv_cat        every conv-state cat: the per-launch state update
//!                   `torch.cat([st.conv[i], qkv[-keep:]])` (keep + min(rows, keep)
//!                   rows), plus on the batched path the gather of all pieces'
//!                   states ahead of `_gdn_pre` (pieces x keep rows)
//!   chain_out_cat   batched path only: `torch.cat(ys)` of the pieces' chain
//!                   outputs, every row at the value width
//! ```
//!
//! The n8 capture's `prefill_chunk` GDN layer is exactly `_add_rmsnorm, quant8,
//! gemm, gemm, b16_prompt, _gdn_pre, chain, _gated_norm, quant8, gemm, cat`.
//! Not modeled: a `keep_at` cut splits one launch's chain in two and adds the
//! cut state's cat (`PrefillForward` has no cut).

use crate::op::ssm::{TensorfoldGdnChainOp, TensorfoldGdnChainOpConfig, TensorfoldGdnChainOpInput};
use crate::op::Op;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, GdnConvNormGatesKernel,
    GdnConvNormGatesKernelConfig, GdnConvNormGatesKernelInput, GdnGatedRmsNormKernel,
    GdnGatedRmsNormKernelConfig, GdnGatedRmsNormKernelInput, ResidualRmsNormKernel,
    ResidualRmsNormKernelConfig, ResidualRmsNormKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput, StaticFp8QuantKernel,
    StaticFp8QuantKernelConfig, StaticFp8QuantKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge};

use super::tensorfold_common::{
    atomic, gemm, push_sum, PrefillForward, PrefillPath, TENSORFOLD_B16_PROMPT, TENSORFOLD_CHAIN,
    TENSORFOLD_CK,
};
pub use super::tensorfold_qwen35_gdn_verify_local::TensorfoldQwen35GdnLocalWorkletConfig;

pub type TensorfoldQwen35GdnPrefillLocalWorkletConfig = TensorfoldQwen35GdnLocalWorkletConfig;

/// The prefill step every GDN layer sees.
pub type TensorfoldQwen35GdnPrefillLocalWorkletInput = PrefillForward;

#[derive(Clone, Debug)]
pub struct TensorfoldQwen35GdnPrefillLocalWorkletResolved {
    pub raw_cfg: TensorfoldQwen35GdnLocalWorkletConfig,
    pub input_norm: Option<ResidualRmsNormKernelConfig>,
    pub in_quant: StaticFp8QuantKernelConfig,
    pub qkv: SingleGemmKernelConfig,
    pub z: SingleGemmKernelConfig,
    /// `b16.prompt_pair`: b and a in one launch, n = 2 x value heads.
    pub ba: SingleGemmKernelConfig,
    pub conv_gates: GdnConvNormGatesKernelConfig,
    pub chain: TensorfoldGdnChainOpConfig,
    pub gated_norm: GdnGatedRmsNormKernelConfig,
    pub out_quant: StaticFp8QuantKernelConfig,
    pub out: SingleGemmKernelConfig,
    /// Conv-state rows of `conv_channels` bf16, read once and written once.
    pub conv_cat: Option<ElementwiseKernelConfig>,
    /// Chain-output rows of the value width bf16.
    pub chain_out_cat: Option<ElementwiseKernelConfig>,
}

pub struct TensorfoldQwen35GdnPrefillLocalWorklet {
    pub name: String,
    input_norm: Option<Op<ResidualRmsNormKernel>>,
    in_quant: Op<StaticFp8QuantKernel>,
    qkv: Op<SingleGemmKernel>,
    z: Op<SingleGemmKernel>,
    ba: Op<SingleGemmKernel>,
    conv_gates: Op<GdnConvNormGatesKernel>,
    chain: TensorfoldGdnChainOp,
    gated_norm: Op<GdnGatedRmsNormKernel>,
    out_quant: Op<StaticFp8QuantKernel>,
    out: Op<SingleGemmKernel>,
    conv_cat: Option<Op<ElementwiseKernel>>,
    chain_out_cat: Option<Op<ElementwiseKernel>>,
    resolved: TensorfoldQwen35GdnPrefillLocalWorkletResolved,
}

impl TensorfoldQwen35GdnPrefillLocalWorklet {
    pub fn resolve_config(
        cfg: &TensorfoldQwen35GdnLocalWorkletConfig,
    ) -> TensorfoldQwen35GdnPrefillLocalWorkletResolved {
        let gpu = cfg.gpu_name.as_str();
        TensorfoldQwen35GdnPrefillLocalWorkletResolved {
            input_norm: cfg.input_norm(),
            in_quant: cfg.quant8(cfg.hidden),
            qkv: gemm(
                TENSORFOLD_CK,
                gpu,
                cfg.conv_channels(),
                cfg.hidden,
                cfg.proj_dtype,
            ),
            z: gemm(
                TENSORFOLD_CK,
                gpu,
                cfg.value_width(),
                cfg.hidden,
                cfg.proj_dtype,
            ),
            ba: gemm(
                TENSORFOLD_B16_PROMPT,
                gpu,
                2 * cfg.num_value_heads,
                cfg.hidden,
                cfg.ba_dtype,
            ),
            conv_gates: cfg.conv_gates(),
            chain: TensorfoldGdnChainOpConfig {
                backends: vec![TENSORFOLD_CHAIN],
                gpu_name: cfg.gpu_name.clone(),
                num_key_heads: cfg.num_key_heads.into(),
                num_heads: cfg.num_value_heads.into(),
                key_head_dim: cfg.key_head_dim.into(),
                value_head_dim: cfg.value_head_dim.into(),
                dtype: cfg.activation_dtype,
            },
            gated_norm: cfg.gated_norm(),
            out_quant: cfg.quant8(cfg.value_width()),
            out: gemm(
                TENSORFOLD_CK,
                gpu,
                cfg.hidden,
                cfg.value_width(),
                cfg.proj_dtype,
            ),
            conv_cat: cfg.cat(cfg.conv_row_bytes()),
            chain_out_cat: cfg.cat(cfg.value_width() * cfg.activation_dtype.size_bytes()),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: TensorfoldQwen35GdnPrefillLocalWorkletResolved,
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
        let cat_op = |suffix: &str, config: Option<ElementwiseKernelConfig>| {
            config
                .map(|c| atomic(n, suffix, c, ElementwiseKernel::build, bridge))
                .transpose()
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
            conv_gates: atomic(
                n,
                "conv_gates",
                r.conv_gates,
                GdnConvNormGatesKernel::build,
                bridge,
            )?,
            chain: TensorfoldGdnChainOp::build(format!("{n}.chain"), r.chain, bridge)?,
            gated_norm: atomic(
                n,
                "gated_norm",
                r.gated_norm,
                GdnGatedRmsNormKernel::build,
                bridge,
            )?,
            out_quant: quant_op("out_quant", r.out_quant)?,
            out: gemm_op("out", r.out)?,
            conv_cat: cat_op("conv_cat", r.conv_cat)?,
            chain_out_cat: cat_op("chain_out_cat", r.chain_out_cat)?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        let mut children = Vec::with_capacity(12);
        if let Some(op) = &self.input_norm {
            children.push(op.compile(builder));
        }
        children.push(self.in_quant.compile(builder));
        children.push(self.qkv.compile(builder));
        children.push(self.z.compile(builder));
        children.push(self.ba.compile(builder));
        children.push(self.conv_gates.compile(builder));
        children.push(self.chain.compile(builder));
        children.push(self.gated_norm.compile(builder));
        children.push(self.out_quant.compile(builder));
        children.push(self.out.compile(builder));
        for op in [&self.conv_cat, &self.chain_out_cat].into_iter().flatten() {
            children.push(op.compile(builder));
        }
        CostNode::Labeled {
            label: format!(
                "{} (TensorfoldQwen35GdnPrefillLocalWorklet) [local (1 GPU); Hk={}, Hv={}, K={}, V={}]",
                self.name,
                cfg.num_key_heads,
                cfg.num_value_heads,
                cfg.key_head_dim,
                cfg.value_head_dim
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    /// Row-wise leaves sum one launch per forward, the chain one per sequence
    /// launch; an empty step is zero in every slot.
    pub fn eval(&self, step: &PrefillForward, ev: &mut Evaluator) {
        let cfg = &self.resolved.raw_cfg;
        let w = Work::of(step, cfg.conv_keep());
        let gemm_in = per_forward(&w.forwards, |m| SingleGemmKernelInput { m });
        let quant_in = per_forward(&w.forwards, |num_tokens| StaticFp8QuantKernelInput {
            num_tokens,
        });
        if let Some(op) = &self.input_norm {
            push_sum(
                op,
                &per_forward(&w.forwards, |m| ResidualRmsNormKernelInput { m }),
                ev,
            );
        }
        push_sum(&self.in_quant, &quant_in, ev);
        push_sum(&self.qkv, &gemm_in, ev);
        push_sum(&self.z, &gemm_in, ev);
        push_sum(&self.ba, &gemm_in, ev);
        push_sum(
            &self.conv_gates,
            &per_forward(&w.forwards, |num_tokens| GdnConvNormGatesKernelInput {
                num_tokens,
            }),
            ev,
        );
        self.chain.eval(
            &TensorfoldGdnChainOpInput {
                sequence_lengths: w.chain.clone(),
            },
            ev,
        );
        let heads = cfg.num_value_heads;
        push_sum(
            &self.gated_norm,
            &per_forward(&w.forwards, |rows| GdnGatedRmsNormKernelInput {
                m: rows * heads,
            }),
            ev,
        );
        push_sum(&self.out_quant, &quant_in, ev);
        push_sum(&self.out, &gemm_in, ev);
        let cat =
            |rows: &[u32]| per_forward(rows, |num_tokens| ElementwiseKernelInput { num_tokens });
        if let Some(op) = &self.conv_cat {
            push_sum(op, &cat(&w.conv_cat), ev);
        }
        if let Some(op) = &self.chain_out_cat {
            push_sum(op, &cat(&w.chain_out_cat), ev);
        }
    }
}

/// The row counts of one step's launches, per leaf.
#[derive(Debug, PartialEq, Eq)]
struct Work {
    /// Rows of each forward: the row-wise leaves launch once per entry.
    forwards: Vec<u32>,
    /// Rows of each `chain_kernel` launch.
    chain: Vec<u32>,
    /// Conv-state rows of each conv-state cat: on the batched path the
    /// gather of every piece's state first, then each launch's update
    /// (`keep` state rows plus the launch's last `min(rows, keep)` rows).
    conv_cat: Vec<u32>,
    /// Rows of the batched path's chain-output cat.
    chain_out_cat: Vec<u32>,
}

impl Work {
    fn of(step: &PrefillForward, keep: u32) -> Self {
        let chain: Vec<u32> = step.launches().iter().map(|l| l.rows).collect();
        let batched = step.path() == Some(PrefillPath::Rows);
        let gather = batched.then(|| chain.len() as u32 * keep);
        Self {
            forwards: step.forward_rows(),
            conv_cat: gather
                .into_iter()
                .chain(chain.iter().map(|&rows| keep + rows.min(keep)))
                .collect(),
            chain_out_cat: batched.then(|| step.total_rows()).into_iter().collect(),
            chain,
        }
    }
}

/// One launch input per forward of the step.
fn per_forward<T>(forwards: &[u32], input: impl Fn(u32) -> T) -> Vec<T> {
    forwards.iter().map(|&rows| input(rows)).collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::timing::LeafMetrics;
    use crate::worklet::tensorfold_common::PrefillPiece;
    use crate::worklet::tensorfold_qwen35_gdn_verify_local::tests::cfg;

    fn build(
        cfg: &TensorfoldQwen35GdnLocalWorkletConfig,
    ) -> TensorfoldQwen35GdnPrefillLocalWorklet {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        TensorfoldQwen35GdnPrefillLocalWorklet::build(
            "m.gdn".into(),
            TensorfoldQwen35GdnPrefillLocalWorklet::resolve_config(cfg),
            &bridge,
        )
        .unwrap()
    }

    fn piece(prefix_len: u32, rows: u32) -> PrefillPiece {
        PrefillPiece {
            prefix_len,
            rows,
            finishes_prompt: false,
        }
    }

    #[test]
    fn leaves_follow_prefill_chunk() {
        let worklet = build(&cfg(true, true));
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        let tree = builder.finish(root);
        let names: Vec<_> = tree
            .slots
            .iter()
            .map(|s| format!("{}:{}", s.name, s.kind))
            .collect();
        assert_eq!(
            names,
            [
                "m.gdn.input_norm:residual_rms_norm",
                "m.gdn.in_quant:static_fp8_quant",
                "m.gdn.qkv:single_gemm",
                "m.gdn.z:single_gemm",
                "m.gdn.ba:single_gemm",
                "m.gdn.conv_gates:gdn_conv_norm_gates",
                "m.gdn.chain:gdn_chunk_delta_rule",
                "m.gdn.gated_norm:gdn_gated_rms_norm",
                "m.gdn.out_quant:static_fp8_quant",
                "m.gdn.out:single_gemm",
                "m.gdn.conv_cat:elementwise",
                "m.gdn.chain_out_cat:elementwise",
            ]
        );
        let r = TensorfoldQwen35GdnPrefillLocalWorklet::resolve_config(&cfg(true, true));
        assert_eq!(
            (r.qkv.n.get(), r.z.n.get(), r.out.k.get()),
            (10240, 6144, 6144)
        );
        assert_eq!(r.ba.backends, [TENSORFOLD_B16_PROMPT]);
        assert_eq!(r.chain_out_cat.unwrap().input_bytes_per_token.get(), 12288);
        // Layer 0, before tf-copies: no norm, no cats.
        let first = build(&cfg(false, false));
        let mut builder = CostTreeBuilder::new();
        let root = first.compile(&mut builder);
        assert_eq!(builder.finish(root).n_slots(), 9);
    }

    #[test]
    fn single_piece_runs_per_chunk_and_batched_step_runs_once() {
        // 9000 rows: three 3000-row chunks, each a forward, a chain launch and
        // a state update.
        let state = PrefillForward {
            pieces: vec![piece(1000, 9000)],
        };
        assert_eq!(
            Work::of(&state, 3),
            Work {
                forwards: vec![3000; 3],
                chain: vec![3000; 3],
                conv_cat: vec![6; 3],
                chain_out_cat: vec![],
            }
        );
        // Two pieces: one forward over every row, a chain launch per piece,
        // the state gather, each piece's update (a 2-row piece appends 2
        // rows), and the chain-output cat.
        let rows = PrefillForward {
            pieces: vec![piece(0, 1022), piece(500, 2)],
        };
        assert_eq!(
            Work::of(&rows, 3),
            Work {
                forwards: vec![1024],
                chain: vec![1022, 2],
                conv_cat: vec![6, 6, 5],
                chain_out_cat: vec![1024],
            }
        );
    }

    #[test]
    fn empty_step_is_zero() {
        let worklet = build(&cfg(true, true));
        let mut buf = vec![LeafMetrics::MISS; 12];
        let mut inputs = Vec::new();
        let mut ev = Evaluator::with_inputs(&mut buf, &mut inputs);
        worklet.eval(&PrefillForward::default(), &mut ev);
        assert_eq!(ev.filled(), 12);
        assert!(buf.iter().all(|m| m.m.time_ms == 0.0));
        assert_eq!(
            serde_json::to_value(&inputs).unwrap()[0],
            serde_json::json!({"launches": []})
        );
    }

    /// Captured n8 `prefill_chunk` shapes (`(prefix_len, rows)`) with the
    /// measured GDN-layer total over all 48 layers (ms; nsys kernel durations,
    /// `_add_rmsnorm` through the conv-state cat).
    pub(crate) const N8_PREFILL_CHUNKS: &[(u32, u32, f64)] = &[
        (8671, 5, 29.08),
        (8896, 85, 33.617),
        (7872, 799, 107.177),
        (2133, 1024, 127.324),
        (26709, 1024, 130.433),
    ];

    /// Per-GDN-layer prefill cost on captured n8 chunks, against the GB10
    /// rows. Run from the repo root with libpython on `LD_LIBRARY_PATH`:
    /// `PYTHONPATH=$PWD uv run cargo test -p simulator --lib tensorfold_qwen35_gdn_prefill -- --include-ignored --nocapture`.
    #[test]
    #[ignore = "needs profile.db GB10 rows and the venv interpreter"]
    fn gb10_rows_price_captured_prefill_chunks() {
        let bridge = PerfApiBridge::new().expect("perf_api bridge");
        let worklet = TensorfoldQwen35GdnPrefillLocalWorklet::build(
            "m.gdn".into(),
            TensorfoldQwen35GdnPrefillLocalWorklet::resolve_config(&cfg(true, false)),
            &bridge,
        )
        .expect("GB10 rows for every GDN prefill leaf");
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        let tree = builder.finish(root);
        for &(prefix_len, rows, measured_ms) in N8_PREFILL_CHUNKS {
            let step = PrefillForward {
                pieces: vec![piece(prefix_len, rows)],
            };
            let mut buf = vec![LeafMetrics::MISS; tree.n_slots()];
            let mut ev = Evaluator::new(&mut buf);
            worklet.eval(&step, &mut ev);
            let total: f32 = buf.iter().map(|m| m.m.time_ms).sum();
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
            let predicted = 48.0 * f64::from(total);
            eprintln!(
                "prefill rows={rows}: layer {:.1} us, x48 {predicted:.3} ms vs capture {measured_ms:.3} ms ({:+.1}%) [{}]",
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
