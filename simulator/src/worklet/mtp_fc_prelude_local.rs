//! Local MTP prelude that projects the embedding and a multi-stream hidden
//! through two separate FC GEMMs.
//!
//! The draft layer's input is built from the next token's embedding and the
//! backbone's pre-mixer hidden, which holds `hc_count` residual streams per
//! token (`[T, hc_count * hidden]`). Source order (vLLM fork 892da082,
//! `models/qwen4_exp/nvidia/mtp.py:184-307`):
//!
//! 1. `token_embedding`: one `F.embedding` (index_select) over the
//!    device-resident BF16 vocabulary table, one row of `hidden` per token.
//! 2. `embedding_norm`: Gemma RMSNorm over `[T, hidden]`.
//! 3. `fc_embedding`: `[T, hidden] x [hidden, hidden]`.
//! 4. `hidden_norm`: Gemma RMSNorm over the flattened streams,
//!    `[T, hc_count * hidden]` (one norm across all streams).
//! 5. `fc_hidden`: one weight shared by every stream, applied to
//!    `[T, hc_count, hidden]`, so `M = hc_count * T`.
//!
//! The views/flatten between steps launch nothing. The embedding is added as a
//! residual inside the following decoder layer, not here. TP1: no collective.
//! Every dimension comes from the checkpoint config; backends are the arch's.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    EmbeddingRowGatherKernel, EmbeddingRowGatherKernelConfig, EmbeddingRowGatherKernelInput,
    RmsNormKernel, RmsNormKernelConfig, RmsNormKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};
use crate::worklet::glm53_common::{atomic, push_or_zero};

/// The vocabulary table lives in HBM (`VocabParallelEmbedding`).
const EMBEDDING_RESIDENCY: &str = "device";

#[derive(Clone, Debug)]
pub struct MtpFcPreludeLocalWorkletConfig {
    pub gpu_name: String,
    pub hidden: Dim,
    /// Residual streams carried per token by the backbone hidden.
    pub hc_count: u32,
    /// Rows of the token-embedding table.
    pub vocab_size: Dim,
    /// Activation and embedding-table dtype.
    pub activation_dtype: DType,
    /// Weight dtype of `fc_embedding` and `fc_hidden`.
    pub fc_dtype: DType,
    pub embedding_backends: Vec<&'static str>,
    pub rms_norm_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct MtpFcPreludeLocalWorkletResolved {
    pub raw_cfg: MtpFcPreludeLocalWorkletConfig,
    pub token_embedding: EmbeddingRowGatherKernelConfig,
    pub embedding_norm: RmsNormKernelConfig,
    pub fc_embedding: SingleGemmKernelConfig,
    pub hidden_norm: RmsNormKernelConfig,
    pub fc_hidden: SingleGemmKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct MtpFcPreludeLocalWorkletInput {
    /// Tokens entering the draft layer this pass.
    pub batch_tokens: u32,
}

pub struct MtpFcPreludeLocalWorklet {
    pub name: String,
    pub token_embedding: Op<EmbeddingRowGatherKernel>,
    pub embedding_norm: Op<RmsNormKernel>,
    pub fc_embedding: Op<SingleGemmKernel>,
    pub hidden_norm: Op<RmsNormKernel>,
    pub fc_hidden: Op<SingleGemmKernel>,
    resolved: MtpFcPreludeLocalWorkletResolved,
}

impl MtpFcPreludeLocalWorklet {
    pub fn resolve_config(
        cfg: &MtpFcPreludeLocalWorkletConfig,
    ) -> MtpFcPreludeLocalWorkletResolved {
        assert!(cfg.hidden.get() > 0, "hidden must be positive");
        assert!(cfg.hc_count >= 1, "hc_count must be at least 1");
        assert!(cfg.vocab_size.get() > 0, "vocab_size must be positive");
        let streams_width = cfg
            .hidden
            .get()
            .checked_mul(cfg.hc_count)
            .expect("hc_count * hidden overflows u32");

        let norm = |hidden: Dim| RmsNormKernelConfig {
            backends: cfg.rms_norm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            hidden,
            dtype: cfg.activation_dtype,
        };
        let fc = || SingleGemmKernelConfig {
            backends: cfg.gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n: cfg.hidden.clone(),
            k: cfg.hidden.clone(),
            dtype: cfg.fc_dtype,
        };
        MtpFcPreludeLocalWorkletResolved {
            token_embedding: EmbeddingRowGatherKernelConfig {
                backends: cfg.embedding_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads: 1,
                head_dim: cfg.hidden.get(),
                table_rows: u64::from(cfg.vocab_size.get()),
                residency: EMBEDDING_RESIDENCY.to_string(),
                weight_dtype: cfg.activation_dtype,
            },
            embedding_norm: norm(cfg.hidden.clone()),
            fc_embedding: fc(),
            hidden_norm: norm(Dim::param("hc_streams_width", streams_width)),
            fc_hidden: fc(),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: MtpFcPreludeLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = &resolved;
        Ok(Self {
            token_embedding: atomic(
                &name,
                "token_embedding",
                r.token_embedding.clone(),
                EmbeddingRowGatherKernel::build,
                bridge,
            )?,
            embedding_norm: atomic(
                &name,
                "embedding_norm",
                r.embedding_norm.clone(),
                RmsNormKernel::build,
                bridge,
            )?,
            fc_embedding: atomic(
                &name,
                "fc_embedding",
                r.fc_embedding.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            hidden_norm: atomic(
                &name,
                "hidden_norm",
                r.hidden_norm.clone(),
                RmsNormKernel::build,
                bridge,
            )?,
            fc_hidden: atomic(
                &name,
                "fc_hidden",
                r.fc_hidden.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        CostNode::Labeled {
            label: format!(
                "{} (MtpFcPreludeLocalWorklet) [local (1 GPU); hidden={}; hc_count={}; vocab={}]",
                self.name,
                cfg.hidden.get(),
                cfg.hc_count,
                cfg.vocab_size.get()
            ),
            child: Box::new(CostNode::Sum(vec![
                self.token_embedding.compile(builder),
                self.embedding_norm.compile(builder),
                self.fc_embedding.compile(builder),
                self.hidden_norm.compile(builder),
                self.fc_hidden.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &MtpFcPreludeLocalWorkletInput, ev: &mut Evaluator) {
        let work = derive_work(input, self.resolved.raw_cfg.hc_count);
        let zero = input.batch_tokens == 0;
        push_or_zero(&self.token_embedding, work.token_embedding, zero, ev);
        push_or_zero(&self.embedding_norm, work.embedding_norm, zero, ev);
        push_or_zero(&self.fc_embedding, work.fc_embedding, zero, ev);
        push_or_zero(&self.hidden_norm, work.hidden_norm, zero, ev);
        push_or_zero(&self.fc_hidden, work.fc_hidden, zero, ev);
    }
}

struct WorkInputs {
    token_embedding: EmbeddingRowGatherKernelInput,
    embedding_norm: RmsNormKernelInput,
    fc_embedding: SingleGemmKernelInput,
    hidden_norm: RmsNormKernelInput,
    fc_hidden: SingleGemmKernelInput,
}

fn hc_count_rows(tokens: u32, hc_count: u32) -> u32 {
    tokens
        .checked_mul(hc_count)
        .expect("hc_count * batch_tokens overflows u32")
}

fn derive_work(input: &MtpFcPreludeLocalWorkletInput, hc_count: u32) -> WorkInputs {
    let t = input.batch_tokens;
    WorkInputs {
        token_embedding: EmbeddingRowGatherKernelInput { num_tokens: t },
        embedding_norm: RmsNormKernelInput { m: t },
        fc_embedding: SingleGemmKernelInput { m: t },
        hidden_norm: RmsNormKernelInput { m: t },
        fc_hidden: SingleGemmKernelInput {
            m: hc_count_rows(t, hc_count),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> MtpFcPreludeLocalWorkletConfig {
        MtpFcPreludeLocalWorkletConfig {
            gpu_name: "NVIDIA B200".to_string(),
            hidden: Dim::param("hidden", 2560),
            hc_count: 4,
            vocab_size: Dim::param("vocab_size", 248_320),
            activation_dtype: DType::Bf16,
            fc_dtype: DType::Bf16,
            embedding_backends: vec!["torch"],
            rms_norm_backends: vec!["vllm_gemma_native"],
            gemm_backends: vec!["torch_linear_vllm"],
        }
    }

    /// The streams norm spans every stream at once, while the shared FC weight
    /// is applied per stream; the embedding is one device row per token.
    #[test]
    fn resolve_derives_stream_widths_and_embedding_table_from_config() {
        let r = MtpFcPreludeLocalWorklet::resolve_config(&cfg());
        assert_eq!(r.embedding_norm.hidden.get(), 2560);
        assert_eq!(r.hidden_norm.hidden.get(), 4 * 2560);
        for fc in [&r.fc_embedding, &r.fc_hidden] {
            assert_eq!((fc.n.get(), fc.k.get()), (2560, 2560));
        }
        let e = &r.token_embedding;
        assert_eq!(
            (e.num_heads, e.head_dim, e.table_rows, e.residency.as_str()),
            (1, 2560, 248_320, "device")
        );
        assert_eq!(e.weight_dtype, DType::Bf16);
    }

    #[test]
    fn fc_hidden_runs_on_every_stream_row() {
        let w = derive_work(&MtpFcPreludeLocalWorkletInput { batch_tokens: 17 }, 4);
        assert_eq!(w.token_embedding.num_tokens, 17);
        assert_eq!(w.embedding_norm.m, 17);
        assert_eq!(w.fc_embedding.m, 17);
        assert_eq!(w.hidden_norm.m, 17);
        assert_eq!(w.fc_hidden.m, 68);
    }

    #[test]
    #[should_panic(expected = "hc_count")]
    fn rejects_zero_streams() {
        let mut c = cfg();
        c.hc_count = 0;
        MtpFcPreludeLocalWorklet::resolve_config(&c);
    }
}
