//! One Qwen3.8 gated full-attention layer of a TensorFold prefill step
//! (`prefill.py` `prefill_chunk` / `prefill_rows`), on one GPU, from the
//! layer's input norm through the `o` projection.
//!
//! ```text
//! per forward       _add_rmsnorm, quant8_kernel (k = hidden),
//!                   gemm_kernel x3 (FP8 q with its gate half, k, v), _attn_prep
//! per launch        K and V rows into the stream's cache buffer
//!                   (`kbuf[p0:p0+n] = key`: two device-to-device cudaMemcpyAsync),
//!                   pattn_kernel over (prefix, rows)
//! batched step      torch.cat of the per-piece attention outputs
//!                   (one CatArrayBatchedCopy; `prefill_rows` only)
//! per forward       _gate_mul, quant8_kernel (k = q heads x D), gemm_kernel (FP8 o)
//! ```
//!
//! A step's forwards and launches come from [`PrefillForward`]: a single piece
//! runs one forward per even chunk of at most 4096 rows, each with one
//! attention launch; two or more pieces run one forward over all rows with an
//! attention launch per piece. The row-wise leaves sum their forwards into one
//! slot each.
//!
//! The KV write and the output cat are torch copies whose L1 rows are still
//! being measured (lane tf-copies). Each is a slot built only when its
//! backends are given ([`TensorfoldQwen35GatedGqaPrefillCopies`]); until then
//! the layer has no leaf for them. Not modeled: `_grow`'s occasional
//! reallocation of a stream's KV buffer (a copy of the whole cached prefix).

use crate::op::attention::{
    TensorfoldPrefillAttentionOp, TensorfoldPrefillAttentionOpConfig,
    TensorfoldPrefillAttentionOpInput,
};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    AttnOutputGateKernel, AttnOutputGateKernelConfig, AttnOutputGateKernelInput, ElementwiseKernel,
    ElementwiseKernelConfig, ElementwiseKernelInput, FlashinferAttnPrefillKernelConfig,
    GatedQkRmsNormRopeKernel, GatedQkRmsNormRopeKernelConfig, GatedQkRmsNormRopeKernelInput,
    ResidualRmsNormKernel, ResidualRmsNormKernelConfig, ResidualRmsNormKernelInput,
    SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput, StaticFp8QuantKernel,
    StaticFp8QuantKernelConfig, StaticFp8QuantKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge};

use super::tensorfold_common::{
    atomic, elementwise, push_sum, PrefillForward, PrefillPath, TENSORFOLD_CK, TENSORFOLD_PATTN,
};
use super::tensorfold_qwen35_gated_gqa_verify_local::TensorfoldQwen35GatedGqaConfig;

/// Bytes of one bf16 element: the activations, attention output and KV cache.
const BF16_BYTES: u32 = 2;

/// Backends for the layer's torch copies, from lane tf-copies once measured.
/// `None` leaves the copy out of the layer.
#[derive(Clone, Debug, Default)]
pub struct TensorfoldQwen35GatedGqaPrefillCopies {
    /// `elementwise` backends for one of the two KV writes of each attention
    /// launch: `rows` bf16 rows of `kv_heads x head_dim` read and written
    /// (a contiguous slice assignment, which torch issues as a
    /// device-to-device `cudaMemcpyAsync`).
    pub kv_write: Option<Vec<&'static str>>,
    /// `elementwise` backends for the `torch.cat` of a batched step's
    /// per-piece attention outputs: one launch over all rows, bf16
    /// `q heads x head_dim` read and written a row
    /// (`CatArrayBatchedCopy_vectorized`).
    pub attention_out_cat: Option<Vec<&'static str>>,
}

#[derive(Clone, Debug)]
pub struct TensorfoldQwen35GatedGqaPrefillLocalWorkletConfig {
    pub gqa: TensorfoldQwen35GatedGqaConfig,
    pub copies: TensorfoldQwen35GatedGqaPrefillCopies,
}

#[derive(Clone, Debug)]
pub struct TensorfoldQwen35GatedGqaPrefillLocalWorkletResolved {
    pub raw_cfg: TensorfoldQwen35GatedGqaPrefillLocalWorkletConfig,
    pub input_norm: ResidualRmsNormKernelConfig,
    pub qkv_quant: StaticFp8QuantKernelConfig,
    pub q_proj: SingleGemmKernelConfig,
    pub k_proj: SingleGemmKernelConfig,
    pub v_proj: SingleGemmKernelConfig,
    pub attn_prep: GatedQkRmsNormRopeKernelConfig,
    pub kv_write: Option<ElementwiseKernelConfig>,
    pub attention: TensorfoldPrefillAttentionOpConfig,
    pub attention_out_cat: Option<ElementwiseKernelConfig>,
    pub gate_mul: AttnOutputGateKernelConfig,
    pub o_quant: StaticFp8QuantKernelConfig,
    pub o_proj: SingleGemmKernelConfig,
}

pub struct TensorfoldQwen35GatedGqaPrefillLocalWorklet {
    pub name: String,
    input_norm: Op<ResidualRmsNormKernel>,
    qkv_quant: Op<StaticFp8QuantKernel>,
    q_proj: Op<SingleGemmKernel>,
    k_proj: Op<SingleGemmKernel>,
    v_proj: Op<SingleGemmKernel>,
    attn_prep: Op<GatedQkRmsNormRopeKernel>,
    kv_write: Option<Op<ElementwiseKernel>>,
    attention: TensorfoldPrefillAttentionOp,
    attention_out_cat: Option<Op<ElementwiseKernel>>,
    gate_mul: Op<AttnOutputGateKernel>,
    o_quant: Op<StaticFp8QuantKernel>,
    o_proj: Op<SingleGemmKernel>,
    resolved: TensorfoldQwen35GatedGqaPrefillLocalWorkletResolved,
}

impl TensorfoldQwen35GatedGqaPrefillLocalWorklet {
    pub fn resolve_config(
        cfg: &TensorfoldQwen35GatedGqaPrefillLocalWorkletConfig,
    ) -> TensorfoldQwen35GatedGqaPrefillLocalWorkletResolved {
        let gqa = &cfg.gqa;
        let gpu = &gqa.gpu_name;
        let [q_proj, k_proj, v_proj, o_proj] = gqa.projections(TENSORFOLD_CK);
        let copy = |backends: &Option<Vec<&'static str>>, bytes_per_row: u32| {
            backends
                .as_ref()
                .map(|b| elementwise(b, gpu, bytes_per_row, bytes_per_row))
        };
        TensorfoldQwen35GatedGqaPrefillLocalWorkletResolved {
            input_norm: gqa.input_norm(),
            qkv_quant: gqa.quant8(gqa.hidden),
            q_proj,
            k_proj,
            v_proj,
            attn_prep: gqa.attn_prep(),
            kv_write: copy(&cfg.copies.kv_write, gqa.kv_width() * BF16_BYTES),
            attention: FlashinferAttnPrefillKernelConfig {
                backends: vec![TENSORFOLD_PATTN],
                gpu_name: gpu.clone(),
                num_qo_heads: gqa.num_q_heads.into(),
                num_kv_heads: gqa.num_kv_heads.into(),
                head_dim: gqa.head_dim.into(),
                q_dtype: DType::Bf16,
                kv_dtype: DType::Bf16,
                o_dtype: DType::Bf16,
            },
            attention_out_cat: copy(&cfg.copies.attention_out_cat, gqa.q_width() * BF16_BYTES),
            gate_mul: gqa.gate_mul(),
            o_quant: gqa.quant8(gqa.q_width()),
            o_proj,
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: TensorfoldQwen35GatedGqaPrefillLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let n = name.as_str();
        let copy = |suffix: &str, cfg: Option<ElementwiseKernelConfig>| {
            cfg.map(|cfg| atomic(n, suffix, cfg, ElementwiseKernel::build, bridge))
                .transpose()
        };
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
            kv_write: copy("kv_write", r.kv_write)?,
            attention: TensorfoldPrefillAttentionOp::build(
                format!("{n}.attention"),
                r.attention,
                bridge,
            )?,
            attention_out_cat: copy("attention_out_cat", r.attention_out_cat)?,
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
        let cfg = &self.resolved.raw_cfg.gqa;
        let mut children = vec![
            self.input_norm.compile(builder),
            self.qkv_quant.compile(builder),
            self.q_proj.compile(builder),
            self.k_proj.compile(builder),
            self.v_proj.compile(builder),
            self.attn_prep.compile(builder),
        ];
        if let Some(op) = &self.kv_write {
            children.push(op.compile(builder));
        }
        children.push(self.attention.compile(builder));
        if let Some(op) = &self.attention_out_cat {
            children.push(op.compile(builder));
        }
        children.extend([
            self.gate_mul.compile(builder),
            self.o_quant.compile(builder),
            self.o_proj.compile(builder),
        ]);
        CostNode::Labeled {
            label: format!(
                "{} (TensorfoldQwen35GatedGqaPrefillLocalWorklet) [local (1 GPU); Hq={}, Hkv={}, D={}]",
                self.name, cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    /// Row-wise leaves sum one launch per forward, the KV write two per
    /// attention launch, and the cat one over the step's rows when it is
    /// batched. An empty step is zero throughout.
    pub fn eval(&self, step: &PrefillForward, ev: &mut Evaluator) {
        let launches = StepLaunches::of(step);
        let rows = &launches.forwards;
        let gemm = per_forward(rows, |m| SingleGemmKernelInput { m });
        let quant = per_forward(rows, |num_tokens| StaticFp8QuantKernelInput { num_tokens });
        push_sum(
            &self.input_norm,
            &per_forward(rows, |m| ResidualRmsNormKernelInput { m }),
            ev,
        );
        push_sum(&self.qkv_quant, &quant, ev);
        push_sum(&self.q_proj, &gemm, ev);
        push_sum(&self.k_proj, &gemm, ev);
        push_sum(&self.v_proj, &gemm, ev);
        push_sum(
            &self.attn_prep,
            &per_forward(rows, |num_tokens| GatedQkRmsNormRopeKernelInput {
                num_tokens,
            }),
            ev,
        );
        if let Some(op) = &self.kv_write {
            push_sum(op, &copies(&launches.kv_writes), ev);
        }
        self.attention.eval(
            &TensorfoldPrefillAttentionOpInput {
                launches: launches.attention,
            },
            ev,
        );
        if let Some(op) = &self.attention_out_cat {
            push_sum(op, &copies(&launches.attention_out_cat), ev);
        }
        push_sum(
            &self.gate_mul,
            &per_forward(rows, |num_tokens| AttnOutputGateKernelInput { num_tokens }),
            ev,
        );
        push_sum(&self.o_quant, &quant, ev);
        push_sum(&self.o_proj, &gemm, ev);
    }
}

fn per_forward<T>(rows: &[u32], input: impl Fn(u32) -> T) -> Vec<T> {
    rows.iter().map(|&rows| input(rows)).collect()
}

fn copies(rows: &[u32]) -> Vec<ElementwiseKernelInput> {
    rows.iter()
        .map(|&num_tokens| ElementwiseKernelInput { num_tokens })
        .collect()
}

/// The launches of one GQA layer over a prefill step, by slot.
#[derive(Debug, PartialEq, Eq)]
struct StepLaunches {
    /// Rows of each forward: one launch of every row-wise leaf.
    forwards: Vec<u32>,
    /// Rows of each KV write: K then V for every attention launch.
    kv_writes: Vec<u32>,
    /// `(prefix_len, rows)` of each attention launch.
    attention: Vec<(u32, u32)>,
    /// Rows of the output cat: the step's rows on a batched step, else none.
    attention_out_cat: Vec<u32>,
}

impl StepLaunches {
    fn of(step: &PrefillForward) -> Self {
        let launches = step.launches();
        Self {
            forwards: step.forward_rows(),
            kv_writes: launches.iter().flat_map(|l| [l.rows, l.rows]).collect(),
            attention: launches.iter().map(|l| (l.prefix_len, l.rows)).collect(),
            attention_out_cat: match step.path() {
                Some(PrefillPath::Rows) => vec![step.total_rows()],
                _ => Vec::new(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::cache::interp::CoverageFlags;
    use crate::timing::LeafMetrics;
    use crate::worklet::tensorfold_common::PrefillPiece;
    use crate::worklet::tensorfold_qwen35_gated_gqa_verify_local::tests::qwen38_gqa;

    fn config(
        copies: TensorfoldQwen35GatedGqaPrefillCopies,
    ) -> TensorfoldQwen35GatedGqaPrefillLocalWorkletConfig {
        TensorfoldQwen35GatedGqaPrefillLocalWorkletConfig {
            gqa: qwen38_gqa("NVIDIA GB10"),
            copies,
        }
    }

    fn build(
        copies: TensorfoldQwen35GatedGqaPrefillCopies,
        bridge: &PerfApiBridge,
    ) -> TensorfoldQwen35GatedGqaPrefillLocalWorklet {
        let resolved = TensorfoldQwen35GatedGqaPrefillLocalWorklet::resolve_config(&config(copies));
        TensorfoldQwen35GatedGqaPrefillLocalWorklet::build("m.gqa".into(), resolved, bridge)
            .expect("prefill GQA worklet builds")
    }

    fn leaves(worklet: &TensorfoldQwen35GatedGqaPrefillLocalWorklet) -> Vec<(String, String)> {
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        builder
            .finish(root)
            .slots
            .iter()
            .map(|s| (s.name.clone(), s.kind.to_string()))
            .collect()
    }

    fn eval(
        worklet: &TensorfoldQwen35GatedGqaPrefillLocalWorklet,
        step: &PrefillForward,
    ) -> (Vec<LeafMetrics>, serde_json::Value) {
        let n = leaves(worklet).len();
        let mut buf = vec![LeafMetrics::MISS; n];
        let mut inputs = Vec::new();
        let mut ev = Evaluator::with_inputs(&mut buf, &mut inputs);
        worklet.eval(step, &mut ev);
        assert_eq!(ev.filled(), n);
        let logged = serde_json::to_value(&inputs).unwrap();
        (buf, logged)
    }

    fn piece(prefix_len: u32, rows: u32) -> PrefillPiece {
        PrefillPiece {
            prefix_len,
            rows,
            finishes_prompt: false,
        }
    }

    fn all_copies() -> TensorfoldQwen35GatedGqaPrefillCopies {
        TensorfoldQwen35GatedGqaPrefillCopies {
            kv_write: Some(vec!["copy_backend"]),
            attention_out_cat: Some(vec!["cat_backend"]),
        }
    }

    #[test]
    fn shapes_are_the_profiled_gb10_rows() {
        let r = TensorfoldQwen35GatedGqaPrefillLocalWorklet::resolve_config(&config(all_copies()));
        let nk = |g: &SingleGemmKernelConfig| (g.n.get(), g.k.get(), g.dtype, g.backends[0]);
        assert_eq!(nk(&r.q_proj), (12288, 5120, DType::Fp8E4m3, TENSORFOLD_CK));
        assert_eq!(nk(&r.k_proj), (1024, 5120, DType::Fp8E4m3, TENSORFOLD_CK));
        assert_eq!(nk(&r.v_proj), (1024, 5120, DType::Fp8E4m3, TENSORFOLD_CK));
        assert_eq!(nk(&r.o_proj), (5120, 6144, DType::Fp8E4m3, TENSORFOLD_CK));
        assert_eq!(r.attention.backends, [TENSORFOLD_PATTN]);
        let kv = r.kv_write.unwrap();
        assert_eq!(
            (
                kv.input_bytes_per_token.get(),
                kv.output_bytes_per_token.get()
            ),
            (2048, 2048)
        );
        let cat = r.attention_out_cat.unwrap();
        assert_eq!(
            (
                cat.input_bytes_per_token.get(),
                cat.output_bytes_per_token.get()
            ),
            (12288, 12288)
        );
        let none = TensorfoldQwen35GatedGqaPrefillLocalWorklet::resolve_config(&config(
            TensorfoldQwen35GatedGqaPrefillCopies::default(),
        ));
        assert!(none.kv_write.is_none() && none.attention_out_cat.is_none());
    }

    #[test]
    fn copy_slots_exist_only_with_backends() {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let names = |w: &TensorfoldQwen35GatedGqaPrefillLocalWorklet| -> Vec<String> {
            leaves(w)
                .into_iter()
                .map(|(n, _)| n.trim_start_matches("m.gqa.").to_string())
                .collect()
        };
        let bare = build(TensorfoldQwen35GatedGqaPrefillCopies::default(), &bridge);
        assert_eq!(
            names(&bare),
            [
                "input_norm",
                "qkv_quant",
                "q_proj",
                "k_proj",
                "v_proj",
                "attn_prep",
                "attention",
                "gate_mul",
                "o_quant",
                "o_proj"
            ]
        );
        let full = build(all_copies(), &bridge);
        let full_leaves = leaves(&full);
        assert_eq!(full_leaves.len(), 12);
        assert_eq!(
            full_leaves[6],
            ("m.gqa.kv_write".into(), "elementwise".into())
        );
        assert_eq!(full_leaves[7].1, "flashinfer_attn_prefill");
        assert_eq!(
            full_leaves[8],
            ("m.gqa.attention_out_cat".into(), "elementwise".into())
        );
    }

    #[test]
    fn launches_follow_the_prefill_path() {
        // One piece of 9000 rows: three forwards of 3000, a launch each.
        let state = StepLaunches::of(&PrefillForward {
            pieces: vec![piece(1000, 9000)],
        });
        assert_eq!(
            state,
            StepLaunches {
                forwards: vec![3000; 3],
                kv_writes: vec![3000; 6],
                attention: vec![(1000, 3000), (4000, 3000), (7000, 3000)],
                attention_out_cat: vec![],
            }
        );
        // Two pieces: one forward over 1024 rows, a launch per piece, one cat.
        let rows = StepLaunches::of(&PrefillForward {
            pieces: vec![piece(30207, 5), piece(7872, 1019)],
        });
        assert_eq!(
            rows,
            StepLaunches {
                forwards: vec![1024],
                kv_writes: vec![5, 5, 1019, 1019],
                attention: vec![(30207, 5), (7872, 1019)],
                attention_out_cat: vec![1024],
            }
        );
    }

    #[test]
    fn an_empty_step_is_zero_in_every_slot() {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let worklet = build(all_copies(), &bridge);
        let (buf, _) = eval(&worklet, &PrefillForward::default());
        assert_eq!(buf.len(), 12);
        for leaf in buf {
            assert_eq!(leaf.m.time_ms, 0.0);
            assert_eq!(leaf.coverage, CoverageFlags::EMPTY);
        }
    }

    /// Against the GB10 rows in `profiling/profile.db`, on prefill steps from
    /// `tf27b_n8.sqlite` (`tensorfold.prefill.step`), without the pending
    /// copy slots. Run from the repo root with libpython on `LD_LIBRARY_PATH`:
    /// `PYTHONPATH=$PWD uv run cargo test -p simulator --lib tensorfold_qwen35_gated_gqa_prefill -- --include-ignored`.
    #[test]
    #[ignore = "needs profile.db GB10 rows and the venv interpreter"]
    fn gb10_rows_price_captured_steps() {
        let bridge = PerfApiBridge::new().expect("perf_api bridge");
        let worklet = build(TensorfoldQwen35GatedGqaPrefillCopies::default(), &bridge);
        // (pieces, captured ms over the 16 GQA layers, copies excluded).
        let steps: [(&[(u32, u32)], f32); 6] = [
            (&[(8671, 5)], 20.232),
            (&[(6420, 148)], 19.718),
            (&[(1109, 1024)], 40.885),
            (&[(7872, 1024)], 79.144),
            (&[(29087, 1024)], 198.902),
            (&[(30207, 5), (7872, 1019)], 118.705 - 1.123),
        ];
        for (pieces, captured) in steps {
            let step = PrefillForward {
                pieces: pieces.iter().map(|&(p, r)| piece(p, r)).collect(),
            };
            let (buf, _) = eval(&worklet, &step);
            let total: f32 = buf.iter().map(|l| l.m.time_ms).sum();
            let parts: Vec<String> = buf.iter().map(|l| format!("{:.4}", l.m.time_ms)).collect();
            eprintln!(
                "gqa prefill {pieces:?}: x16 {:.3} ms vs captured {captured:.3} ({:+.1}%) [{}]",
                16.0 * total,
                100.0 * (16.0 * total / captured - 1.0),
                parts.join(", ")
            );
            for leaf in &buf {
                assert!(leaf.m.time_ms > 0.0 && leaf.m.time_ms.is_finite());
            }
        }
    }
}
