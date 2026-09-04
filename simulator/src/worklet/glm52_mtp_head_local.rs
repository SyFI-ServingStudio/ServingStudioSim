//! GLM-5.2 local MTP output head after the layer-78 decoder block.
//!
//! Pinned serving adds the decoder's hidden and residual outputs, applies the
//! shared-head RMSNorm, and evaluates this rank's vocabulary shard. The
//! residual add uses a measured elementwise traffic approximation. Logits
//! processing, candidate sampling, and scheduler work remain outside this
//! local model-architecture boundary.
//!
//! `ParallelLMHead` is column-parallel, so the projection shards but needs no
//! collective inside this section: each rank produces its own logit shard and
//! the distributed sampler consumes them. The section therefore stays `Local`
//! while carrying a partition, like `glm52_shared_expert_local`.

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

#[cfg(test)]
const SOURCE_ORDER: [&str; 3] = ["residual_add", "shared_head_rms_norm", "lm_head"];

/// Raw GLM-5.2 MTP-head identity. This rank-local worklet partitions the
/// vocabulary projection but deliberately has no collective configuration.
#[derive(Clone, Debug)]
pub struct Glm52MtpHeadLocalWorkletConfig {
    pub elementwise_backends: Vec<&'static str>,
    pub rms_norm_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_dim: Dim,
    /// Global checkpoint vocabulary. `resolve_config` derives the local
    /// `ParallelLMHead` output width by dividing this by `tp_size`.
    pub vocab_size: Dim,
    /// Ranks the vocabulary is column-partitioned over. An arch that shards its
    /// main LM head must pass the same degree here: it is the same
    /// `ParallelLMHead`, one layer later.
    pub tp_size: u16,
    pub dtype: DType,
    pub gemm_dtype: DType,
}

/// Pure resolved data with all three atomic child configs fully baked.
#[derive(Clone, Debug)]
pub struct Glm52MtpHeadLocalWorkletResolved {
    pub raw_cfg: Glm52MtpHeadLocalWorkletConfig,
    pub residual_add: ElementwiseKernelConfig,
    pub shared_head_rms_norm: RmsNormKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub vocab_size_per_rank: Dim,
}

#[derive(Clone, Debug, Default)]
pub struct Glm52MtpHeadLocalWorkletInput {
    pub batch_tokens: u32,
}

pub struct Glm52MtpHeadLocalWorklet {
    pub name: String,
    pub residual_add: Op<ElementwiseKernel>,
    pub shared_head_rms_norm: Op<RmsNormKernel>,
    pub lm_head: Op<SingleGemmKernel>,
    resolved: Glm52MtpHeadLocalWorkletResolved,
}

impl Glm52MtpHeadLocalWorklet {
    /// Resolve the one supported GLM-5.2 MTP-head identity without touching a
    /// bridge, GPU, cache, or `Arc`.
    pub fn resolve_config(
        cfg: &Glm52MtpHeadLocalWorkletConfig,
    ) -> Glm52MtpHeadLocalWorkletResolved {
        validate_config(cfg)
            .unwrap_or_else(|reason| panic!("invalid Glm52MtpHeadLocalWorkletConfig: {reason}"));

        let hidden_bytes = checked_product(
            "hidden BF16 bytes",
            &[cfg.hidden_dim.get(), cfg.dtype.size_bytes()],
        )
        .expect("validated GLM-5.2 hidden byte width must fit u32");
        let residual_input_bytes =
            checked_product("residual_add.input_bytes_per_token", &[2, hidden_bytes])
                .expect("validated GLM-5.2 residual input byte rate must fit u32");

        let vocab_size_per_rank =
            cfg.vocab_size.clone() / Dim::param("lm_head_tp", u32::from(cfg.tp_size));

        Glm52MtpHeadLocalWorkletResolved {
            residual_add: ElementwiseKernelConfig {
                backends: cfg.elementwise_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: residual_input_bytes.into(),
                output_bytes_per_token: hidden_bytes.into(),
            },
            shared_head_rms_norm: RmsNormKernelConfig {
                backends: cfg.rms_norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden_dim.clone(),
                dtype: cfg.dtype,
            },
            lm_head: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: vocab_size_per_rank.clone(),
                k: cfg.hidden_dim.clone(),
                dtype: cfg.gemm_dtype,
            },
            vocab_size_per_rank,
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: Glm52MtpHeadLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let residual_add = build_atomic(
            &name,
            "residual_add",
            resolved.residual_add.clone(),
            ElementwiseKernel::build,
            bridge,
        )?;
        let shared_head_rms_norm = build_atomic(
            &name,
            "shared_head_rms_norm",
            resolved.shared_head_rms_norm.clone(),
            RmsNormKernel::build,
            bridge,
        )?;
        let lm_head = build_atomic(
            &name,
            "lm_head",
            resolved.lm_head.clone(),
            SingleGemmKernel::build,
            bridge,
        )?;

        Ok(Self {
            name,
            residual_add,
            shared_head_rms_norm,
            lm_head,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: worklet_label(&self.name, &self.resolved.raw_cfg),
            child: Box::new(CostNode::Sum(vec![
                self.residual_add.compile(builder),
                self.shared_head_rms_norm.compile(builder),
                self.lm_head.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &Glm52MtpHeadLocalWorkletInput, ev: &mut Evaluator) {
        let work = work_inputs(input.batch_tokens);
        let zero = input.batch_tokens == 0;

        eval_atomic_or_zero(&self.residual_add, work.residual_add, zero, ev);
        eval_atomic_or_zero(
            &self.shared_head_rms_norm,
            work.shared_head_rms_norm,
            zero,
            ev,
        );
        eval_atomic_or_zero(&self.lm_head, work.lm_head, zero, ev);
    }
}

struct WorkInputs {
    residual_add: ElementwiseKernelInput,
    shared_head_rms_norm: RmsNormKernelInput,
    lm_head: SingleGemmKernelInput,
}

fn validate_config(cfg: &Glm52MtpHeadLocalWorkletConfig) -> Result<(), String> {
    for (name, actual, required) in [
        ("hidden_dim", cfg.hidden_dim.get(), HIDDEN_DIM),
        ("vocab_size", cfg.vocab_size.get(), VOCAB_SIZE),
    ] {
        if actual != required {
            return Err(format!("{name} must be {required}, got {actual}"));
        }
    }
    if cfg.tp_size == 0 {
        return Err("tp_size must be positive".to_string());
    }
    if cfg.vocab_size.get() % u32::from(cfg.tp_size) != 0 {
        return Err(format!(
            "vocab_size {} must be divisible by tp_size {}",
            cfg.vocab_size, cfg.tp_size
        ));
    }
    if cfg.dtype != DType::Bf16 {
        return Err(format!(
            "dtype must be {}, got {}",
            DType::Bf16.as_str(),
            cfg.dtype.as_str()
        ));
    }
    if !matches!(cfg.gemm_dtype, DType::Bf16 | DType::Fp8E4m3) {
        return Err(format!(
            "gemm_dtype must be {} or {}, got {}",
            DType::Bf16.as_str(),
            DType::Fp8E4m3.as_str(),
            cfg.gemm_dtype.as_str()
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

fn work_inputs(batch_tokens: u32) -> WorkInputs {
    WorkInputs {
        residual_add: ElementwiseKernelInput {
            num_tokens: batch_tokens,
        },
        shared_head_rms_norm: RmsNormKernelInput { m: batch_tokens },
        lm_head: SingleGemmKernelInput { m: batch_tokens },
    }
}

fn worklet_label(name: &str, cfg: &Glm52MtpHeadLocalWorkletConfig) -> String {
    format!(
        "{name} (Glm52MtpHeadLocalWorklet) [rank-local; tp={}; hidden={}; vocab/rank={}; residual_add=measured-traffic-approximation]",
        cfg.tp_size,
        cfg.hidden_dim.get(),
        cfg.vocab_size.get() / u32::from(cfg.tp_size)
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

    fn cfg() -> Glm52MtpHeadLocalWorkletConfig {
        Glm52MtpHeadLocalWorkletConfig {
            elementwise_backends: vec!["triton"],
            rms_norm_backends: vec!["flashinfer"],
            gemm_backends: vec!["torch_linear"],
            gpu_name: "NVIDIA H200".to_string(),
            hidden_dim: Dim::param("hidden_dim", HIDDEN_DIM),
            vocab_size: Dim::param("vocab_size", VOCAB_SIZE),
            tp_size: 1,
            dtype: DType::Bf16,
            gemm_dtype: DType::Bf16,
        }
    }

    #[test]
    fn source_order_is_exact_local_and_excludes_other_sections() {
        assert_eq!(
            SOURCE_ORDER,
            ["residual_add", "shared_head_rms_norm", "lm_head"]
        );
        assert_eq!(SOURCE_ORDER.len(), 3);
        for forbidden in [
            "embedding",
            "prelude",
            "decoder",
            "attention",
            "indexer",
            "moe",
            "all_reduce",
            "communication",
            "logits_processor",
            "sampling",
            "sampler",
        ] {
            assert!(!SOURCE_ORDER.iter().any(|child| child.contains(forbidden)));
        }
    }

    #[test]
    fn resolve_bakes_exact_glm_shapes_bytes_and_backend_roles() {
        let r = Glm52MtpHeadLocalWorklet::resolve_config(&cfg());

        assert_eq!(r.raw_cfg.hidden_dim, 6144);
        assert_eq!(r.raw_cfg.vocab_size, 154880);
        assert_eq!(r.raw_cfg.dtype, DType::Bf16);
        assert_eq!(r.residual_add.input_bytes_per_token, 24576);
        assert_eq!(r.residual_add.output_bytes_per_token, 12288);
        assert_eq!(r.residual_add.backends, vec!["triton"]);
        assert_eq!(r.shared_head_rms_norm.hidden, 6144);
        assert_eq!(r.shared_head_rms_norm.dtype, DType::Bf16);
        assert_eq!(r.shared_head_rms_norm.backends, vec!["flashinfer"]);
        assert_eq!(r.lm_head.n, 154880);
        assert_eq!(r.vocab_size_per_rank, 154880);
        assert_eq!(r.lm_head.k, 6144);
        assert_eq!(r.lm_head.dtype, DType::Bf16);
        assert_eq!(r.lm_head.backends, vec!["torch_linear"]);
        assert_eq!(r.raw_cfg.gpu_name, "NVIDIA H200");
    }

    #[test]
    fn input_and_zero_case_feed_all_three_leaves_faithfully() {
        let work = work_inputs(73);
        assert_eq!(work.residual_add.num_tokens, 73);
        assert_eq!(work.shared_head_rms_norm.m, 73);
        assert_eq!(work.lm_head.m, 73);

        let zero = work_inputs(0);
        assert_eq!(zero.residual_add.num_tokens, 0);
        assert_eq!(zero.shared_head_rms_norm.m, 0);
        assert_eq!(zero.lm_head.m, 0);
        assert_eq!(Glm52MtpHeadLocalWorkletInput::default().batch_tokens, 0);
    }

    #[test]
    fn unsupported_glm_identity_fails_during_pure_resolution() {
        let mut bad_hidden = cfg();
        bad_hidden.hidden_dim = 4096.into();
        assert!(validate_config(&bad_hidden)
            .unwrap_err()
            .contains("hidden_dim must be 6144"));

        let mut bad_vocab = cfg();
        bad_vocab.vocab_size = 32000.into();
        assert!(validate_config(&bad_vocab)
            .unwrap_err()
            .contains("vocab_size must be 154880"));

        let mut bad_dtype = cfg();
        bad_dtype.dtype = DType::Fp16;
        assert!(validate_config(&bad_dtype)
            .unwrap_err()
            .contains("dtype must be bf16"));

        let mut zero_tp = cfg();
        zero_tp.tp_size = 0;
        assert!(validate_config(&zero_tp)
            .unwrap_err()
            .contains("tp_size must be positive"));

        let mut indivisible = cfg();
        indivisible.tp_size = 3;
        assert!(validate_config(&indivisible)
            .unwrap_err()
            .contains("must be divisible by tp_size 3"));
    }

    #[test]
    fn the_vocabulary_projection_shards_with_the_main_lm_head() {
        // `ParallelLMHead` is one column-parallel matrix; the MTP head is the
        // same matrix one layer later. Billing it whole while the arch's main
        // head is sharded charges this rank for every other rank's logits --
        // 8x on the TP8 pack.
        for tp_size in [1, 2, 4, 8] {
            let mut config = cfg();
            config.tp_size = tp_size;
            let r = Glm52MtpHeadLocalWorklet::resolve_config(&config);

            assert_eq!(r.lm_head.n, VOCAB_SIZE / u32::from(tp_size));
            assert_eq!(r.vocab_size_per_rank, VOCAB_SIZE / u32::from(tp_size));
            // Only the vocabulary axis moves: hidden is never sharded.
            assert_eq!(r.lm_head.k, 6144);
            assert_eq!(r.shared_head_rms_norm.hidden, 6144);
            assert_eq!(r.residual_add.input_bytes_per_token, 24576);
        }
    }

    #[test]
    fn checked_dimension_and_byte_math_rejects_overflow() {
        assert_eq!(checked_product("hidden", &[6144, 2]).unwrap(), 12288);
        assert_eq!(checked_product("residual", &[2, 12288]).unwrap(), 24576);
        assert!(checked_product("overflow", &[u32::MAX, 2])
            .unwrap_err()
            .contains("overflows u32"));
    }

    #[test]
    fn label_documents_the_rank_local_vocabulary_partition() {
        let label = worklet_label("mtp.head", &cfg());
        assert!(label.contains("Glm52MtpHeadLocalWorklet"));
        assert!(label.contains("rank-local"));
        assert!(label.contains("tp=1"));
        assert!(label.contains("hidden=6144"));
        assert!(label.contains("vocab/rank=154880"));
        assert!(label.contains("residual_add=measured-traffic-approximation"));
        // Column-parallel needs no reduction: the shard IS the answer for this
        // rank's slice of the vocabulary.
        for forbidden in ["allreduce", "collective"] {
            assert!(!label.contains(forbidden));
        }
    }
}
