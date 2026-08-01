//! GLM-5.2 local MTP prelude before the layer-78 decoder block.
//!
//! Pinned serving embeds the draft token, masks the position-zero embedding,
//! independently normalizes that embedding and the previous hidden state,
//! concatenates both rows, and projects the result back to hidden width. This
//! TP1 section has no embedding collective. Embedding lookup, masking, and
//! concatenation use measured elementwise traffic approximations; the two
//! RMSNorms and the final GEMM remain separate measured leaves.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, RmsNormKernel,
    RmsNormKernelConfig, RmsNormKernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, LeafMetrics, PerfApiBridge, Probe,
    SlotInput,
};

const HIDDEN_DIM: u32 = 6144;
const VOCAB_SIZE: u32 = 154880;
const TOKEN_ID_BYTES: u32 = 8;
const POSITION_BYTES: u32 = 8;

#[cfg(test)]
const SOURCE_ORDER: [&str; 6] = [
    "token_embedding",
    "position_zero_mask",
    "embedding_rms_norm",
    "previous_hidden_rms_norm",
    "concat_embedding_hidden",
    "eh_proj",
];

/// Raw GLM-5.2 MTP-prelude identity. This local TP1 worklet deliberately has
/// no parallelism or collective configuration.
#[derive(Clone, Debug)]
pub struct Glm52MtpPreludeLocalWorkletConfig {
    pub elementwise_backends: Vec<&'static str>,
    pub rms_norm_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_dim: Dim,
    pub vocab_size: Dim,
    pub dtype: DType,
    pub token_id_bytes: u32,
    pub position_bytes: u32,
}

/// Pure resolved data with all six atomic child configs fully baked.
#[derive(Clone, Debug)]
pub struct Glm52MtpPreludeLocalWorkletResolved {
    pub raw_cfg: Glm52MtpPreludeLocalWorkletConfig,
    pub token_embedding: ElementwiseKernelConfig,
    pub position_zero_mask: ElementwiseKernelConfig,
    pub embedding_rms_norm: RmsNormKernelConfig,
    pub previous_hidden_rms_norm: RmsNormKernelConfig,
    pub concat_embedding_hidden: ElementwiseKernelConfig,
    pub eh_proj: SingleGemmKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct Glm52MtpPreludeLocalWorkletInput {
    pub batch_tokens: u32,
}

pub struct Glm52MtpPreludeLocalWorklet {
    pub name: String,
    pub token_embedding: Op<ElementwiseKernel>,
    pub position_zero_mask: Op<ElementwiseKernel>,
    pub embedding_rms_norm: Op<RmsNormKernel>,
    pub previous_hidden_rms_norm: Op<RmsNormKernel>,
    pub concat_embedding_hidden: Op<ElementwiseKernel>,
    pub eh_proj: Op<SingleGemmKernel>,
    resolved: Glm52MtpPreludeLocalWorkletResolved,
}

impl Glm52MtpPreludeLocalWorklet {
    /// Resolve the one supported GLM-5.2 MTP-prelude identity without touching
    /// a bridge, GPU, cache, or `Arc`.
    pub fn resolve_config(
        cfg: &Glm52MtpPreludeLocalWorkletConfig,
    ) -> Glm52MtpPreludeLocalWorkletResolved {
        validate_config(cfg)
            .unwrap_or_else(|reason| panic!("invalid Glm52MtpPreludeLocalWorkletConfig: {reason}"));

        let hidden_bytes = checked_product(
            "hidden BF16 bytes",
            &[cfg.hidden_dim.get(), cfg.dtype.size_bytes()],
        )
        .expect("validated GLM-5.2 hidden byte width must fit u32");
        let token_embedding_input = checked_sum(
            "token_embedding.input_bytes_per_token",
            &[hidden_bytes, cfg.token_id_bytes],
        )
        .expect("validated GLM-5.2 embedding input byte rate must fit u32");
        let position_mask_input = checked_sum(
            "position_zero_mask.input_bytes_per_token",
            &[hidden_bytes, cfg.position_bytes],
        )
        .expect("validated GLM-5.2 position-mask input byte rate must fit u32");
        let concat_bytes = checked_product(
            "concat_embedding_hidden bytes per token",
            &[2, hidden_bytes],
        )
        .expect("validated GLM-5.2 concat byte rate must fit u32");
        let concat_width = checked_product("eh_proj.k", &[2, cfg.hidden_dim.get()])
            .expect("validated GLM-5.2 concat width must fit u32");

        Glm52MtpPreludeLocalWorkletResolved {
            token_embedding: ElementwiseKernelConfig {
                backends: cfg.elementwise_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: token_embedding_input.into(),
                output_bytes_per_token: hidden_bytes.into(),
            },
            position_zero_mask: ElementwiseKernelConfig {
                backends: cfg.elementwise_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: position_mask_input.into(),
                output_bytes_per_token: hidden_bytes.into(),
            },
            embedding_rms_norm: RmsNormKernelConfig {
                backends: cfg.rms_norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden_dim.clone(),
                dtype: cfg.dtype,
            },
            previous_hidden_rms_norm: RmsNormKernelConfig {
                backends: cfg.rms_norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden_dim.clone(),
                dtype: cfg.dtype,
            },
            concat_embedding_hidden: ElementwiseKernelConfig {
                backends: cfg.elementwise_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: concat_bytes.into(),
                output_bytes_per_token: concat_bytes.into(),
            },
            eh_proj: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.hidden_dim.clone(),
                k: concat_width.into(),
                dtype: cfg.dtype,
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: Glm52MtpPreludeLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let token_embedding = build_atomic(
            &name,
            "token_embedding",
            resolved.token_embedding.clone(),
            ElementwiseKernel::build,
            bridge,
        )?;
        let position_zero_mask = build_atomic(
            &name,
            "position_zero_mask",
            resolved.position_zero_mask.clone(),
            ElementwiseKernel::build,
            bridge,
        )?;
        let embedding_rms_norm = build_atomic(
            &name,
            "embedding_rms_norm",
            resolved.embedding_rms_norm.clone(),
            RmsNormKernel::build,
            bridge,
        )?;
        let previous_hidden_rms_norm = build_atomic(
            &name,
            "previous_hidden_rms_norm",
            resolved.previous_hidden_rms_norm.clone(),
            RmsNormKernel::build,
            bridge,
        )?;
        let concat_embedding_hidden = build_atomic(
            &name,
            "concat_embedding_hidden",
            resolved.concat_embedding_hidden.clone(),
            ElementwiseKernel::build,
            bridge,
        )?;
        let eh_proj = build_atomic(
            &name,
            "eh_proj",
            resolved.eh_proj.clone(),
            SingleGemmKernel::build,
            bridge,
        )?;

        Ok(Self {
            name,
            token_embedding,
            position_zero_mask,
            embedding_rms_norm,
            previous_hidden_rms_norm,
            concat_embedding_hidden,
            eh_proj,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: worklet_label(&self.name, &self.resolved.raw_cfg),
            child: Box::new(CostNode::Sum(vec![
                self.token_embedding.compile(builder),
                self.position_zero_mask.compile(builder),
                self.embedding_rms_norm.compile(builder),
                self.previous_hidden_rms_norm.compile(builder),
                self.concat_embedding_hidden.compile(builder),
                self.eh_proj.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &Glm52MtpPreludeLocalWorkletInput, ev: &mut Evaluator) {
        let work = work_inputs(input.batch_tokens);
        let zero = input.batch_tokens == 0;

        eval_atomic_or_zero(&self.token_embedding, work.token_embedding, zero, ev);
        eval_atomic_or_zero(&self.position_zero_mask, work.position_zero_mask, zero, ev);
        eval_atomic_or_zero(&self.embedding_rms_norm, work.embedding_rms_norm, zero, ev);
        eval_atomic_or_zero(
            &self.previous_hidden_rms_norm,
            work.previous_hidden_rms_norm,
            zero,
            ev,
        );
        eval_atomic_or_zero(
            &self.concat_embedding_hidden,
            work.concat_embedding_hidden,
            zero,
            ev,
        );
        eval_atomic_or_zero(&self.eh_proj, work.eh_proj, zero, ev);
    }
}

struct WorkInputs {
    token_embedding: ElementwiseKernelInput,
    position_zero_mask: ElementwiseKernelInput,
    embedding_rms_norm: RmsNormKernelInput,
    previous_hidden_rms_norm: RmsNormKernelInput,
    concat_embedding_hidden: ElementwiseKernelInput,
    eh_proj: SingleGemmKernelInput,
}

fn validate_config(cfg: &Glm52MtpPreludeLocalWorkletConfig) -> Result<(), String> {
    for (name, actual, required) in [
        ("hidden_dim", cfg.hidden_dim.get(), HIDDEN_DIM),
        ("vocab_size", cfg.vocab_size.get(), VOCAB_SIZE),
        ("token_id_bytes", cfg.token_id_bytes, TOKEN_ID_BYTES),
        ("position_bytes", cfg.position_bytes, POSITION_BYTES),
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
    Ok(())
}

fn checked_product(name: &str, factors: &[u32]) -> Result<u32, String> {
    factors.iter().try_fold(1_u32, |value, &factor| {
        value
            .checked_mul(factor)
            .ok_or_else(|| format!("{name} overflows u32"))
    })
}

fn checked_sum(name: &str, terms: &[u32]) -> Result<u32, String> {
    terms.iter().try_fold(0_u32, |value, &term| {
        value
            .checked_add(term)
            .ok_or_else(|| format!("{name} overflows u32"))
    })
}

fn work_inputs(batch_tokens: u32) -> WorkInputs {
    WorkInputs {
        token_embedding: ElementwiseKernelInput {
            num_tokens: batch_tokens,
        },
        position_zero_mask: ElementwiseKernelInput {
            num_tokens: batch_tokens,
        },
        embedding_rms_norm: RmsNormKernelInput { m: batch_tokens },
        previous_hidden_rms_norm: RmsNormKernelInput { m: batch_tokens },
        concat_embedding_hidden: ElementwiseKernelInput {
            num_tokens: batch_tokens,
        },
        eh_proj: SingleGemmKernelInput { m: batch_tokens },
    }
}

fn worklet_label(name: &str, cfg: &Glm52MtpPreludeLocalWorkletConfig) -> String {
    format!(
        "{name} (Glm52MtpPreludeLocalWorklet) [local (1 GPU); hidden={}; vocab={}; embedding/mask/concat=measured-traffic-approximations]",
        cfg.hidden_dim.get(),
        cfg.vocab_size.get()
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

    fn cfg() -> Glm52MtpPreludeLocalWorkletConfig {
        Glm52MtpPreludeLocalWorkletConfig {
            elementwise_backends: vec!["triton"],
            rms_norm_backends: vec!["flashinfer"],
            gemm_backends: vec!["torch_linear"],
            gpu_name: "NVIDIA H200".to_string(),
            hidden_dim: Dim::param("hidden_dim", HIDDEN_DIM),
            vocab_size: Dim::param("vocab_size", VOCAB_SIZE),
            dtype: DType::Bf16,
            token_id_bytes: TOKEN_ID_BYTES,
            position_bytes: POSITION_BYTES,
        }
    }

    #[test]
    fn source_order_is_exact_local_and_excludes_later_sections() {
        assert_eq!(
            SOURCE_ORDER,
            [
                "token_embedding",
                "position_zero_mask",
                "embedding_rms_norm",
                "previous_hidden_rms_norm",
                "concat_embedding_hidden",
                "eh_proj",
            ]
        );
        assert_eq!(SOURCE_ORDER.len(), 6);
        for forbidden in [
            "decoder",
            "attention",
            "indexer",
            "moe",
            "all_reduce",
            "communication",
            "head",
            "logits",
            "sampler",
        ] {
            assert!(!SOURCE_ORDER.iter().any(|child| child.contains(forbidden)));
        }
    }

    #[test]
    fn resolve_bakes_exact_glm_shapes_bytes_and_backend_roles() {
        let r = Glm52MtpPreludeLocalWorklet::resolve_config(&cfg());

        assert_eq!(r.raw_cfg.hidden_dim, 6144);
        assert_eq!(r.raw_cfg.vocab_size, 154880);
        assert_eq!(r.raw_cfg.dtype, DType::Bf16);
        assert_eq!(r.raw_cfg.token_id_bytes, 8);
        assert_eq!(r.raw_cfg.position_bytes, 8);

        assert_eq!(r.token_embedding.input_bytes_per_token, 12296);
        assert_eq!(r.token_embedding.output_bytes_per_token, 12288);
        assert_eq!(r.token_embedding.backends, vec!["triton"]);
        assert_eq!(r.position_zero_mask.input_bytes_per_token, 12296);
        assert_eq!(r.position_zero_mask.output_bytes_per_token, 12288);
        assert_eq!(r.position_zero_mask.backends, vec!["triton"]);

        assert_eq!(r.embedding_rms_norm.hidden, 6144);
        assert_eq!(r.embedding_rms_norm.dtype, DType::Bf16);
        assert_eq!(r.embedding_rms_norm.backends, vec!["flashinfer"]);
        assert_eq!(r.previous_hidden_rms_norm.hidden, 6144);
        assert_eq!(r.previous_hidden_rms_norm.dtype, DType::Bf16);
        assert_eq!(r.previous_hidden_rms_norm.backends, vec!["flashinfer"]);

        assert_eq!(r.concat_embedding_hidden.input_bytes_per_token, 24576);
        assert_eq!(r.concat_embedding_hidden.output_bytes_per_token, 24576);
        assert_eq!(r.concat_embedding_hidden.backends, vec!["triton"]);
        assert_eq!(r.eh_proj.n, 6144);
        assert_eq!(r.eh_proj.k, 12288);
        assert_eq!(r.eh_proj.dtype, DType::Bf16);
        assert_eq!(r.eh_proj.backends, vec!["torch_linear"]);
        assert_eq!(r.raw_cfg.gpu_name, "NVIDIA H200");
    }

    #[test]
    fn input_and_zero_case_feed_all_six_leaves_faithfully() {
        let work = work_inputs(73);
        assert_eq!(work.token_embedding.num_tokens, 73);
        assert_eq!(work.position_zero_mask.num_tokens, 73);
        assert_eq!(work.embedding_rms_norm.m, 73);
        assert_eq!(work.previous_hidden_rms_norm.m, 73);
        assert_eq!(work.concat_embedding_hidden.num_tokens, 73);
        assert_eq!(work.eh_proj.m, 73);

        let zero = work_inputs(0);
        assert_eq!(zero.token_embedding.num_tokens, 0);
        assert_eq!(zero.position_zero_mask.num_tokens, 0);
        assert_eq!(zero.embedding_rms_norm.m, 0);
        assert_eq!(zero.previous_hidden_rms_norm.m, 0);
        assert_eq!(zero.concat_embedding_hidden.num_tokens, 0);
        assert_eq!(zero.eh_proj.m, 0);
        assert_eq!(Glm52MtpPreludeLocalWorkletInput::default().batch_tokens, 0);
    }

    #[test]
    fn unsupported_glm_identity_fails_during_pure_resolution() {
        let mut bad_hidden = cfg();
        bad_hidden.hidden_dim = 4096.into();
        assert!(
            validate_config(&bad_hidden)
                .unwrap_err()
                .contains("hidden_dim must be 6144")
        );

        let mut bad_vocab = cfg();
        bad_vocab.vocab_size = 32000.into();
        assert!(
            validate_config(&bad_vocab)
                .unwrap_err()
                .contains("vocab_size must be 154880")
        );

        let mut bad_dtype = cfg();
        bad_dtype.dtype = DType::Fp16;
        assert!(
            validate_config(&bad_dtype)
                .unwrap_err()
                .contains("dtype must be bf16")
        );

        let mut bad_token_id = cfg();
        bad_token_id.token_id_bytes = 4;
        assert!(
            validate_config(&bad_token_id)
                .unwrap_err()
                .contains("token_id_bytes must be 8")
        );

        let mut bad_position = cfg();
        bad_position.position_bytes = 4;
        assert!(
            validate_config(&bad_position)
                .unwrap_err()
                .contains("position_bytes must be 8")
        );
    }

    #[test]
    fn checked_dimension_and_byte_math_rejects_overflow() {
        assert_eq!(checked_product("hidden", &[6144, 2]).unwrap(), 12288);
        assert_eq!(checked_sum("embedding", &[12288, 8]).unwrap(), 12296);
        assert_eq!(checked_product("concat", &[2, 12288]).unwrap(), 24576);
        assert_eq!(checked_product("eh k", &[2, 6144]).unwrap(), 12288);
        assert!(
            checked_product("product overflow", &[u32::MAX, 2])
                .unwrap_err()
                .contains("overflows u32")
        );
        assert!(
            checked_sum("sum overflow", &[u32::MAX, 1])
                .unwrap_err()
                .contains("overflows u32")
        );
    }

    #[test]
    fn label_documents_local_traffic_approximations() {
        let label = worklet_label("mtp.prelude", &cfg());
        assert!(label.contains("Glm52MtpPreludeLocalWorklet"));
        assert!(label.contains("local (1 GPU)"));
        assert!(label.contains("hidden=6144"));
        assert!(label.contains("vocab=154880"));
        assert!(label.contains("embedding/mask/concat=measured-traffic-approximations"));
        for forbidden in ["tp=", "allreduce", "collective"] {
            assert!(!label.contains(forbidden));
        }
    }
}
