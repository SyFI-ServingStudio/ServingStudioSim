//! `FlashInferAttentionOp` — compound L2 op (L2 design §3). One attention call
//! over a sim batch dispatches to two L1 kernels: `flashinfer_attn_prefill`
//! (causal, merges fresh + chunked) and `flashinfer_attn_decode`.
//!
//! **v1 cost model (deliberately simple).** The validated tuple/banding model
//! (shape × log2(kv) partition + 3-tuple collapse) is deferred. Instead:
//!   - prefill / chunked: look up EACH request on its own (no normalization) and
//!     sum — exact when a step has ≤1 prefill request, which is the common
//!     continuous-batching case (one sequence chunk-prefilling + a decode batch).
//!   - decode: collapse to ONE cell `(batch_size = count, total_tokens = Σ kv)`.
//!
//! This diverges from design.md §3.3's fresh/chunked normalization by explicit
//! user decision; per-request sum over-estimates 1.6–2× only when multiple small-q
//! prefills are batched (rare). See agent-trace/attention_cache_fidelity.md.
//!
//! A prefill / chunked request is `(prefix_len, append_len)` — exactly the cell
//! the prefill kernel caches on, so it passes straight through (no conversion).
//! `prefix_len` is the already-cached context, `append_len` the new tokens this
//! step; fresh prefill is `prefix_len == 0`. The Python runner derives
//! `kv_len = prefix_len + append_len`.

use std::sync::Arc;

use crate::timing::bridge::DType;
use crate::timing::kernels::{
    FlashinferAttnDecodeKernel, FlashinferAttnDecodeKernelConfig, FlashinferAttnDecodeKernelInput,
    FlashinferAttnPrefillKernel, FlashinferAttnPrefillKernelConfig,
    FlashinferAttnPrefillKernelInput,
};
use crate::timing::{BuildError, Describe, JitPlan, LookupResult, PerfApiBridge};

/// Single op-level config; expands into the two sub-kernel configs (their field
/// sets are identical, so this is their shared union). L2 design §3.5-1.
#[derive(Clone, Debug)]
pub struct FlashInferAttentionConfig {
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_qo_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub q_dtype: DType,
    pub kv_dtype: DType,
    pub o_dtype: DType,
}

/// Op-level input: raw per-request data for one attention call in the sim batch
/// (L2 design §3.2). Partition / dispatch happen inside `lookup` — L3 never sees
/// the sub-kernels.
#[derive(Clone, Debug, Default)]
pub struct FlashInferAttentionInput {
    /// Every prefill / chunked-prefill request as `(prefix_len, append_len)`:
    /// `prefix_len` is the already-cached context, `append_len` the new tokens
    /// this step. `prefix_len == 0` is fresh prefill; `prefix_len > 0` is chunked.
    pub prefill_chunk_pairs: Vec<(u32, u32)>,
    /// Every decode request's KV length (each contributes one `q = 1` token).
    pub decode_kv_lens: Vec<u32>,
}

pub struct FlashInferAttentionOp {
    pub name: String,
    pub prefill: Arc<FlashinferAttnPrefillKernel>,
    pub decode: Arc<FlashinferAttnDecodeKernel>,
}

impl FlashInferAttentionOp {
    pub fn new(
        name: String,
        cfg: FlashInferAttentionConfig,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let prefill = Arc::new(FlashinferAttnPrefillKernel::init(
            format!("{name}.prefill"),
            prefill_config(&cfg),
            bridge,
        )?);
        let decode = Arc::new(FlashinferAttnDecodeKernel::init(
            format!("{name}.decode"),
            decode_config(&cfg),
            bridge,
        )?);
        Ok(Self {
            name,
            prefill,
            decode,
        })
    }

    pub fn lookup(&self, input: &FlashInferAttentionInput) -> LookupResult {
        let mut parts = Vec::with_capacity(input.prefill_chunk_pairs.len() + 1);
        for &(prefix_len, append_len) in &input.prefill_chunk_pairs {
            parts.push(self.prefill.lookup(&FlashinferAttnPrefillKernelInput {
                prefix_len,
                append_len,
            }));
        }
        if let Some(decode_input) = decode_input(&input.decode_kv_lens) {
            parts.push(self.decode.lookup(&decode_input));
        }
        LookupResult::sum(self.name.clone(), parts)
    }

    /// Build-time sibling of `new`: borrows only, constructs nothing, returns the
    /// `JitPlan` tree summed over both sub-kernels (L2 design §3.7). Sub-kernel
    /// names share the `"{op}.{slot}"` formula used by `new`.
    pub fn dry_run_init(
        name: String,
        cfg: &FlashInferAttentionConfig,
        bridge: &PerfApiBridge,
    ) -> Result<JitPlan, BuildError> {
        let parts = vec![
            FlashinferAttnPrefillKernel::dry_run(
                &format!("{name}.prefill"),
                &prefill_config(cfg),
                bridge,
            )?,
            FlashinferAttnDecodeKernel::dry_run(
                &format!("{name}.decode"),
                &decode_config(cfg),
                bridge,
            )?,
        ];
        Ok(JitPlan::sum(name, parts))
    }
}

impl Describe for FlashInferAttentionOp {
    fn describe(&self, depth: usize, out: &mut String) {
        use std::fmt::Write;
        writeln!(out, "{}{}", "│  ".repeat(depth), self.name).unwrap();
        self.prefill.describe(depth + 1, out);
        self.decode.describe(depth + 1, out);
    }
}

// ─── internal helpers (pure; unit-tested without a bridge) ───────────────────

fn prefill_config(cfg: &FlashInferAttentionConfig) -> FlashinferAttnPrefillKernelConfig {
    FlashinferAttnPrefillKernelConfig {
        backends: cfg.backends.clone(),
        gpu_name: cfg.gpu_name.clone(),
        num_qo_heads: cfg.num_qo_heads,
        num_kv_heads: cfg.num_kv_heads,
        head_dim: cfg.head_dim,
        q_dtype: cfg.q_dtype,
        kv_dtype: cfg.kv_dtype,
        o_dtype: cfg.o_dtype,
    }
}

fn decode_config(cfg: &FlashInferAttentionConfig) -> FlashinferAttnDecodeKernelConfig {
    FlashinferAttnDecodeKernelConfig {
        backends: cfg.backends.clone(),
        gpu_name: cfg.gpu_name.clone(),
        num_qo_heads: cfg.num_qo_heads,
        num_kv_heads: cfg.num_kv_heads,
        head_dim: cfg.head_dim,
        q_dtype: cfg.q_dtype,
        kv_dtype: cfg.kv_dtype,
        o_dtype: cfg.o_dtype,
    }
}

/// All decode requests → one cell: `batch_size = count`, `total_tokens = Σ kv`.
/// `None` when there are no decode requests (skip the sub-kernel call).
fn decode_input(decode_kv_lens: &[u32]) -> Option<FlashinferAttnDecodeKernelInput> {
    if decode_kv_lens.is_empty() {
        return None;
    }
    Some(FlashinferAttnDecodeKernelInput {
        batch_size: decode_kv_lens.len() as u32,
        total_tokens: decode_kv_lens.iter().sum(),
    })
}

#[cfg(test)]
mod tests {
    use super::decode_input;

    #[test]
    fn decode_collapses_to_count_and_total_kv() {
        let input = decode_input(&[4096, 8192, 2048]).expect("non-empty");
        assert_eq!(input.batch_size, 3);
        assert_eq!(input.total_tokens, 14336);
    }

    #[test]
    fn decode_input_is_none_when_no_decode_requests() {
        assert!(decode_input(&[]).is_none());
    }
}
