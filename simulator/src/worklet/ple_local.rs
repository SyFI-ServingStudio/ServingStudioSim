//! Per-layer n-gram embedding (PLE) on one GPU: the table lookup and the
//! gated short-convolution block that adds it to the multi-stream residual.
//!
//! Launches, in the vLLM fork's order (`ple_layer.py`, `ngram_embedding.py`):
//!
//! - `ple_ngram_ids`: the per-token n-gram hash ids (`num_heads = ngram_context
//!   x heads_per_ngram` ids per token);
//! - `embedding_row_gather`: the raw FP8 row gather from the table. With
//!   `residency = host_uva` (`cpu_offload=True`) it is the pinned-host UVA
//!   lookup the fork issues on a side stream before layer 0;
//! - `hc_combine` (only when a hyper-connection combine is pending at the PLE
//!   layer): PLE adds to the materialized multi-stream state, so the pending
//!   combine runs standalone first (`model.py:290-300`);
//! - `finalize_copy` (host_uva only): the join copies the prefetch buffer into
//!   graph-owned storage (`_finalize_prefetch`);
//! - dequant (only when the table dtype differs from the activations):
//!   `embeddings.to(dtype)` then `* weight_scale` (two elementwise launches);
//! - `kv_proj`: the merged key/value GEMM, N = `hc_count*hidden + hidden`,
//!   K = `ple_embed_dim`;
//! - `ple_gate`: normalized similarity gate over the streams;
//! - `ple_short_conv` per request type: `decode` (non-spec single-token
//!   decodes), `spec` (verify rows) and `prefill`. Each leaf is zero when its
//!   rows are absent; a mixed speculative step launches `spec` and `prefill`.
//!
//! Composition is a serial `Sum`, including the lookup. Measured on B200
//! (qf-ple job 6589/6594, isolated lookup beside a layer-0 proxy on the main
//! stream): from T = 64 the span is lookup + layer 0 (e.g. 268 vs 277 us
//! summed at T = 64, 1292 vs 1284 at T = 2048), so the prefetch is not hidden.
//! Known residual: at T <= 16 most of the lookup hides (exposed 2-6 us of an
//! 8-14 us lookup), so the serial sum over-predicts small verify steps by up
//! to ~8 us per step. This is the same composition as
//! `deepseek_v41_engram_prefetch_local`.
//!
//! Rows. The CUDA-graph-captured pieces (`hc_combine`, dequant, `kv_proj`,
//! `ple_gate`) run at `graph_rows`; the eager-break pieces (n-gram ids, the
//! lookup and its finalize copy) run at `eager_rows`; the short convolutions
//! run on the real per-request shapes. The arch chooses both row counts from
//! its graph mode.
//!
//! TODO(glue_map): the prefill branch's `query_start_loc` offset (one tiny
//! int elementwise launch) is left to the glue attribution.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, EmbeddingRowGatherKernel,
    EmbeddingRowGatherKernelConfig, EmbeddingRowGatherKernelInput, HcCombineKernel,
    MhcRmsNormKernelConfig, MhcRmsNormKernelInput, PleGateKernel, PleGateKernelConfig,
    PleGateKernelInput, PleNgramIdsKernel, PleNgramIdsKernelConfig, PleNgramIdsKernelInput,
    PleShortConvKernel, PleShortConvKernelConfig, PleShortConvKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge};
use crate::worklet::glm53_common::{atomic, elementwise, push_or_zero};

const HOST_UVA: &str = "host_uva";

#[derive(Clone, Debug)]
pub struct PleLocalWorkletConfig {
    pub hidden_size: u32,
    pub hc_count: u32,
    /// `ngram_size - 1`.
    pub ngram_context_len: u32,
    pub heads_per_ngram: u32,
    /// Width of one token's concatenated lookup (`num_heads * head_dim`), the
    /// `kv_proj` input.
    pub ple_embed_dim: u32,
    pub table_rows: u64,
    /// `host_uva` or `device`.
    pub residency: String,
    pub table_dtype: DType,
    pub conv_kernel_size: u32,
    pub conv_dilation: u32,
    /// A hyper-connection combine is pending when the PLE layer starts (any
    /// PLE layer after the first decoder layer).
    pub pending_combine: bool,
    pub dtype: DType,
    pub gpu_name: String,
    /// The PLE and HC Triton kernels.
    pub triton_backends: Vec<&'static str>,
    pub gather_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct PleLocalWorkletResolved {
    pub raw_cfg: PleLocalWorkletConfig,
    pub ngram_ids: PleNgramIdsKernelConfig,
    pub lookup: EmbeddingRowGatherKernelConfig,
    pub combine: Option<MhcRmsNormKernelConfig>,
    pub finalize_copy: Option<ElementwiseKernelConfig>,
    pub dequant_cast: Option<ElementwiseKernelConfig>,
    pub dequant_scale: Option<ElementwiseKernelConfig>,
    pub kv_proj: SingleGemmKernelConfig,
    pub gate: PleGateKernelConfig,
    pub conv_decode: PleShortConvKernelConfig,
    pub conv_spec: PleShortConvKernelConfig,
    pub conv_prefill: PleShortConvKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct PleLocalWorkletInput {
    /// Rows of the graph-captured pieces (padded T under a CUDA graph).
    pub graph_rows: u32,
    /// Rows of the eager-break lookup pieces.
    pub eager_rows: u32,
    /// Non-speculative single-token decode requests.
    pub decode_requests: u32,
    /// Speculative verify requests, priced at `spec_query_len` rows each.
    pub spec_requests: u32,
    pub spec_query_len: u32,
    /// Their real rows: `spec_query_len` each, fewer for a verify clipped at
    /// `max_model_len`.
    pub spec_tokens: u32,
    pub prefill_requests: u32,
    pub prefill_tokens: u32,
}

pub struct PleLocalWorklet {
    pub name: String,
    pub ngram_ids: Op<PleNgramIdsKernel>,
    pub lookup: Op<EmbeddingRowGatherKernel>,
    pub combine: Option<Op<HcCombineKernel>>,
    pub finalize_copy: Option<Op<ElementwiseKernel>>,
    pub dequant_cast: Option<Op<ElementwiseKernel>>,
    pub dequant_scale: Option<Op<ElementwiseKernel>>,
    pub kv_proj: Op<SingleGemmKernel>,
    pub gate: Op<PleGateKernel>,
    pub conv_decode: Op<PleShortConvKernel>,
    pub conv_spec: Op<PleShortConvKernel>,
    pub conv_prefill: Op<PleShortConvKernel>,
    resolved: PleLocalWorkletResolved,
}

/// Per-leaf inputs of one call; `None` means the leaf is zero this step.
#[derive(Debug, PartialEq)]
struct Work {
    graph_rows: u32,
    eager_rows: u32,
    num_reqs: u32,
    conv_decode: Option<(u32, u32)>,
    conv_spec: Option<(u32, u32)>,
    conv_prefill: Option<(u32, u32)>,
}

impl Work {
    fn of(input: &PleLocalWorkletInput) -> Self {
        let spec = input.spec_requests > 0;
        assert!(
            !spec || input.spec_query_len > 0,
            "speculative requests need spec_query_len > 0"
        );
        assert_eq!(
            input.prefill_requests == 0,
            input.prefill_tokens == 0,
            "prefill requests and prefill tokens must be both zero or both positive"
        );
        assert!(
            input.spec_tokens <= input.spec_requests * input.spec_query_len,
            "{} spec tokens for {} verify requests of {} rows",
            input.spec_tokens,
            input.spec_requests,
            input.spec_query_len
        );
        let real = input.decode_requests + input.spec_tokens + input.prefill_tokens;
        assert!(
            input.eager_rows >= real && input.graph_rows >= real,
            "PLE rows (graph {}, eager {}) below the {real} scheduled tokens",
            input.graph_rows,
            input.eager_rows
        );
        let nonzero = |batch: u32, query: u32| (batch > 0).then_some((batch, query));
        Self {
            graph_rows: input.graph_rows,
            eager_rows: input.eager_rows,
            num_reqs: (input.decode_requests + input.spec_requests + input.prefill_requests)
                .clamp(1, input.eager_rows.max(1)),
            conv_decode: nonzero(input.decode_requests, 1),
            conv_spec: nonzero(input.spec_requests, input.spec_query_len),
            // One call over all prefills, priced at their mean length.
            conv_prefill: nonzero(
                input.prefill_requests,
                ((f64::from(input.prefill_tokens) / f64::from(input.prefill_requests.max(1)))
                    .round() as u32)
                    .max(1),
            ),
        }
    }
}

impl PleLocalWorklet {
    pub fn resolve_config(cfg: &PleLocalWorkletConfig) -> PleLocalWorkletResolved {
        let num_heads = cfg.ngram_context_len * cfg.heads_per_ngram;
        assert!(
            num_heads > 0 && cfg.hidden_size > 0 && cfg.hc_count > 0,
            "PLE needs a positive n-gram and residual geometry"
        );
        assert_eq!(
            cfg.ple_embed_dim % num_heads,
            0,
            "ple_embed_dim {} must split over {num_heads} n-gram heads",
            cfg.ple_embed_dim
        );
        let head_dim = cfg.ple_embed_dim / num_heads;
        let streams = cfg.hidden_size * cfg.hc_count;
        let table_bytes = cfg.ple_embed_dim * cfg.table_dtype.size_bytes();
        let act_bytes = cfg.ple_embed_dim * cfg.dtype.size_bytes();
        let ew =
            |input, output| elementwise(&cfg.elementwise_backends, &cfg.gpu_name, input, output);
        let conv = |mode: &str| PleShortConvKernelConfig {
            backends: cfg.triton_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            channels: streams,
            kernel_size: cfg.conv_kernel_size,
            dilation: cfg.conv_dilation,
            mode: mode.to_string(),
            dtype: cfg.dtype,
        };
        let dequant = cfg.table_dtype != cfg.dtype;
        PleLocalWorkletResolved {
            ngram_ids: PleNgramIdsKernelConfig {
                backends: cfg.triton_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                ngram_context_len: cfg.ngram_context_len,
                heads_per_ngram: cfg.heads_per_ngram,
            },
            lookup: EmbeddingRowGatherKernelConfig {
                backends: cfg.gather_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads,
                head_dim,
                table_rows: cfg.table_rows,
                residency: cfg.residency.clone(),
                weight_dtype: cfg.table_dtype,
            },
            combine: cfg.pending_combine.then(|| MhcRmsNormKernelConfig {
                backends: cfg.triton_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden_size: cfg.hidden_size.into(),
                hc_mult: cfg.hc_count,
                hidden_dtype: cfg.dtype,
            }),
            finalize_copy: (cfg.residency == HOST_UVA).then(|| ew(table_bytes, table_bytes)),
            dequant_cast: dequant.then(|| ew(table_bytes, act_bytes)),
            dequant_scale: dequant.then(|| ew(act_bytes, act_bytes)),
            kv_proj: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: (streams + cfg.hidden_size).into(),
                k: cfg.ple_embed_dim.into(),
                dtype: cfg.dtype,
            },
            gate: PleGateKernelConfig {
                backends: cfg.triton_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden_size: cfg.hidden_size,
                hc_mult: cfg.hc_count,
                dtype: cfg.dtype,
            },
            conv_decode: conv("decode"),
            conv_spec: conv("spec"),
            conv_prefill: conv("prefill"),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: PleLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = &resolved;
        let ew = |suffix: &str, cfg: &Option<ElementwiseKernelConfig>| {
            cfg.clone()
                .map(|c| atomic(&name, suffix, c, ElementwiseKernel::build, bridge))
                .transpose()
        };
        Ok(Self {
            ngram_ids: atomic(
                &name,
                "ngram_ids",
                r.ngram_ids.clone(),
                PleNgramIdsKernel::build,
                bridge,
            )?,
            lookup: atomic(
                &name,
                "lookup",
                r.lookup.clone(),
                EmbeddingRowGatherKernel::build,
                bridge,
            )?,
            combine: r
                .combine
                .clone()
                .map(|c| atomic(&name, "hc_combine", c, HcCombineKernel::build, bridge))
                .transpose()?,
            finalize_copy: ew("finalize_copy", &r.finalize_copy)?,
            dequant_cast: ew("dequant_cast", &r.dequant_cast)?,
            dequant_scale: ew("dequant_scale", &r.dequant_scale)?,
            kv_proj: atomic(
                &name,
                "kv_proj",
                r.kv_proj.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            gate: atomic(&name, "gate", r.gate.clone(), PleGateKernel::build, bridge)?,
            conv_decode: atomic(
                &name,
                "conv_decode",
                r.conv_decode.clone(),
                PleShortConvKernel::build,
                bridge,
            )?,
            conv_spec: atomic(
                &name,
                "conv_spec",
                r.conv_spec.clone(),
                PleShortConvKernel::build,
                bridge,
            )?,
            conv_prefill: atomic(
                &name,
                "conv_prefill",
                r.conv_prefill.clone(),
                PleShortConvKernel::build,
                bridge,
            )?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let mut children = vec![
            self.ngram_ids.compile(builder),
            self.lookup.compile(builder),
        ];
        for op in [&self.combine].into_iter().flatten() {
            children.push(op.compile(builder));
        }
        for op in [&self.finalize_copy, &self.dequant_cast, &self.dequant_scale]
            .into_iter()
            .flatten()
        {
            children.push(op.compile(builder));
        }
        children.push(self.kv_proj.compile(builder));
        children.push(self.gate.compile(builder));
        children.push(self.conv_decode.compile(builder));
        children.push(self.conv_spec.compile(builder));
        children.push(self.conv_prefill.compile(builder));
        let cfg = &self.resolved.raw_cfg;
        CostNode::Labeled {
            label: format!(
                "{} (PleLocalWorklet) [local; heads={}x{}; residency={}; lookup=serial; pending_combine={}]",
                self.name,
                self.resolved.lookup.num_heads,
                self.resolved.lookup.head_dim,
                cfg.residency,
                cfg.pending_combine,
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    pub fn eval(&self, input: &PleLocalWorkletInput, ev: &mut Evaluator) {
        let w = Work::of(input);
        let (graph, eager) = (w.graph_rows, w.eager_rows);
        push_or_zero(
            &self.ngram_ids,
            PleNgramIdsKernelInput {
                num_tokens: eager,
                num_reqs: w.num_reqs,
            },
            eager == 0,
            ev,
        );
        push_or_zero(
            &self.lookup,
            EmbeddingRowGatherKernelInput { num_tokens: eager },
            eager == 0,
            ev,
        );
        if let Some(op) = &self.combine {
            push_or_zero(
                op,
                MhcRmsNormKernelInput { num_tokens: graph },
                graph == 0,
                ev,
            );
        }
        if let Some(op) = &self.finalize_copy {
            push_or_zero(
                op,
                ElementwiseKernelInput { num_tokens: eager },
                eager == 0,
                ev,
            );
        }
        for op in [&self.dequant_cast, &self.dequant_scale]
            .into_iter()
            .flatten()
        {
            push_or_zero(
                op,
                ElementwiseKernelInput { num_tokens: graph },
                graph == 0,
                ev,
            );
        }
        push_or_zero(
            &self.kv_proj,
            SingleGemmKernelInput { m: graph },
            graph == 0,
            ev,
        );
        push_or_zero(
            &self.gate,
            PleGateKernelInput { num_tokens: graph },
            graph == 0,
            ev,
        );
        for (op, shape) in [
            (&self.conv_decode, w.conv_decode),
            (&self.conv_spec, w.conv_spec),
            (&self.conv_prefill, w.conv_prefill),
        ] {
            let (batch_size, query_len) = shape.unwrap_or((1, 1));
            push_or_zero(
                op,
                PleShortConvKernelInput {
                    batch_size,
                    query_len,
                },
                shape.is_none(),
                ev,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(residency: &str, pending_combine: bool) -> PleLocalWorkletConfig {
        PleLocalWorkletConfig {
            hidden_size: 2560,
            hc_count: 4,
            ngram_context_len: 2,
            heads_per_ngram: 8,
            ple_embed_dim: 2560,
            table_rows: 320_001_536,
            residency: residency.into(),
            table_dtype: DType::Fp8E4m3,
            conv_kernel_size: 4,
            conv_dilation: 3,
            pending_combine,
            dtype: DType::Bf16,
            gpu_name: "NVIDIA B200".into(),
            triton_backends: vec!["vllm_triton"],
            gather_backends: vec!["vllm_triton"],
            elementwise_backends: vec!["triton"],
            gemm_backends: vec!["torch_linear_vllm"],
        }
    }

    fn input(decodes: u32, specs: u32, prefills: u32, prefill_tokens: u32) -> PleLocalWorkletInput {
        let real = decodes + 4 * specs + prefill_tokens;
        PleLocalWorkletInput {
            graph_rows: real,
            eager_rows: real,
            decode_requests: decodes,
            spec_requests: specs,
            spec_query_len: 4,
            spec_tokens: 4 * specs,
            prefill_requests: prefills,
            prefill_tokens,
        }
    }

    /// Config-derived shapes land on the profiled rows: 16 x 160 FP8 lookup,
    /// kv_proj 12800 x 2560, conv over the 10240 stream channels.
    #[test]
    fn shapes_are_the_profiled_rows() {
        let r = PleLocalWorklet::resolve_config(&cfg("host_uva", true));
        assert_eq!((r.lookup.num_heads, r.lookup.head_dim), (16, 160));
        assert_eq!((r.kv_proj.n.get(), r.kv_proj.k.get()), (12800, 2560));
        assert_eq!(r.conv_spec.channels, 10240);
        assert_eq!(r.gate.hc_mult, 4);
        let copy = r.finalize_copy.unwrap();
        assert_eq!(copy.input_bytes_per_token.get(), 2560);
        let cast = r.dequant_cast.unwrap();
        assert_eq!(cast.output_bytes_per_token.get(), 5120);
    }

    /// The finalize copy exists only for the host prefetch, and the standalone
    /// combine only when one is pending.
    #[test]
    fn residency_and_pending_combine_gate_their_leaves() {
        let device = PleLocalWorklet::resolve_config(&cfg("device", false));
        assert!(device.finalize_copy.is_none() && device.combine.is_none());
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        for (residency, pending, slots) in [("host_uva", true, 11), ("device", false, 9)] {
            let w = PleLocalWorklet::build(
                "ple".into(),
                PleLocalWorklet::resolve_config(&cfg(residency, pending)),
                &bridge,
            )
            .unwrap();
            let mut b = CostTreeBuilder::new();
            let root = w.compile(&mut b);
            assert_eq!(b.finish(root).n_slots(), slots);
        }
    }

    /// Pure verify launches only the spec conv; a mixed step launches spec and
    /// prefill; a plain decode step only decode.
    #[test]
    fn conv_mode_follows_the_request_types() {
        let verify = Work::of(&input(0, 8, 0, 0));
        assert_eq!(
            (verify.conv_decode, verify.conv_spec, verify.conv_prefill),
            (None, Some((8, 4)), None)
        );
        assert_eq!(verify.num_reqs, 8);
        let mixed = Work::of(&input(0, 128, 2, 1537));
        assert_eq!(mixed.conv_spec, Some((128, 4)));
        assert_eq!(mixed.conv_prefill, Some((2, 769)));
        assert_eq!(mixed.conv_decode, None);
        let decode = Work::of(&input(5, 0, 0, 0));
        assert_eq!(
            (decode.conv_decode, decode.conv_spec, decode.conv_prefill),
            (Some((5, 1)), None, None)
        );
    }

    #[test]
    #[should_panic(expected = "below")]
    fn rejects_rows_below_the_scheduled_tokens() {
        let mut i = input(0, 8, 0, 0);
        i.graph_rows = 16;
        i.eager_rows = 31;
        Work::of(&i);
    }
}
