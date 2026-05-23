//! `AttnLocalWorklet` — single-GPU attention section, wrapping the L2 compound
//! `FlashInferAttentionOp`. `Local` group suffix (L3 §1.5): 1 GPU, no HP split,
//! no collective.
//!
//! `FlashInferAttentionOp` predates the worklet shape — it exposes `new` /
//! `compile` / `eval` (no `*Resolved`). So this worklet's `resolve_config` just
//! bakes a `FlashInferAttentionConfig`; `init_ops` calls `FlashInferAttentionOp::new`.
//!
//! `gpu_name` rides in the config (L3 §1.6 / `gpu_name in *KernelConfig`); the
//! `*Input` is pure shape.

use crate::op::attention::{FlashInferAttentionConfig, FlashInferAttentionInput, FlashInferAttentionOp};
use crate::timing::bridge::DType;
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge};

/// Raw config; mirrors `FlashInferAttentionConfig` (no partition under `Local`).
#[derive(Clone, Debug)]
pub struct AttnLocalWorkletConfig {
    pub num_qo_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub q_dtype: DType,
    pub kv_dtype: DType,
    pub o_dtype: DType,
    pub gpu_name: String,
    pub backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct AttnLocalWorkletResolved {
    pub raw_cfg: AttnLocalWorkletConfig,
    pub attn: FlashInferAttentionConfig,
}

/// Per-call shape: every prefill/chunked request `(prefix_len, append_len)` and
/// every decode request's KV length (one `q = 1` token each).
#[derive(Clone, Debug, Default)]
pub struct AttnLocalWorkletInput {
    pub prefill_chunk_pairs: Vec<(u32, u32)>,
    pub decode_kv_lens: Vec<u32>,
}

pub struct AttnLocalWorklet {
    pub name: String,
    pub attn: FlashInferAttentionOp,
    resolved: AttnLocalWorkletResolved,
}

impl AttnLocalWorklet {
    pub fn resolve_config(cfg: &AttnLocalWorkletConfig) -> AttnLocalWorkletResolved {
        AttnLocalWorkletResolved {
            attn: FlashInferAttentionConfig {
                backends: cfg.backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_qo_heads: cfg.num_qo_heads,
                num_kv_heads: cfg.num_kv_heads,
                head_dim: cfg.head_dim,
                q_dtype: cfg.q_dtype,
                kv_dtype: cfg.kv_dtype,
                o_dtype: cfg.o_dtype,
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn init_ops(
        name: String,
        resolved: AttnLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let attn = FlashInferAttentionOp::new(format!("{name}.attn"), resolved.attn.clone(), bridge)?;
        Ok(Self {
            name,
            attn,
            resolved,
        })
    }

    /// CostTree compile: the attention op's two fixed prefill/decode leaves,
    /// wrapped in a `Labeled` node carrying the worklet identity + partition/shape
    /// annotation (the old `Describe` header lines).
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let label = format!(
            "{} (AttnLocalWorklet) [local (1 GPU); qo={}, kv={}, head_dim={}]",
            self.name,
            self.resolved.raw_cfg.num_qo_heads,
            self.resolved.raw_cfg.num_kv_heads,
            self.resolved.raw_cfg.head_dim
        );
        CostNode::Labeled {
            label,
            child: Box::new(self.attn.compile(builder)),
        }
    }

    /// CostTree eval: delegate to the attn op (its two prefill/decode slots),
    /// mapping the worklet input to the op input — mirrors `compile`/`lookup`.
    pub fn eval(&self, input: &AttnLocalWorkletInput, ev: &mut Evaluator) {
        let op_in = FlashInferAttentionInput {
            prefill_chunk_pairs: input.prefill_chunk_pairs.clone(),
            decode_kv_lens: input.decode_kv_lens.clone(),
        };
        self.attn.eval(&op_in, ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> AttnLocalWorkletConfig {
        AttnLocalWorkletConfig {
            num_qo_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            q_dtype: DType::Bf16,
            kv_dtype: DType::Bf16,
            o_dtype: DType::Bf16,
            gpu_name: "H100".to_string(),
            backends: vec!["fa2", "fa3"],
        }
    }

    #[test]
    fn resolve_bakes_attn_config() {
        let r = AttnLocalWorklet::resolve_config(&cfg());
        assert_eq!(r.attn.num_qo_heads, 32);
        assert_eq!(r.attn.num_kv_heads, 8);
        assert_eq!(r.attn.head_dim, 128);
        assert_eq!(r.attn.gpu_name, "H100");
    }
}
