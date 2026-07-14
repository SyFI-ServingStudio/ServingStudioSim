//! `FlashInferAttentionOp` — compound L2 op (L2 design §3). One attention call
//! over a sim batch dispatches to three L1 kernels: `kv_cache_append`,
//! `flashinfer_attn_prefill` (causal, merges fresh + chunked), and
//! `flashinfer_attn_decode`.
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
    FlashinferAttnPrefillKernelInput, KvCacheAppendKernel, KvCacheAppendKernelConfig,
    KvCacheAppendKernelInput,
};
use crate::timing::{
    AttnPrefillLog, BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, LeafMetrics,
    PerfApiBridge, Probe,
};

/// Single op-level config; expands into three sub-kernel configs. The op owns
/// the FP8 precision policy: `dtype` is the base (16-bit) dtype and `fp8` picks
/// the per-phase preset (mirrors ref `python_bridge.rs prefill_dtypes` /
/// `decode_dtypes`):
///   - prefill / chunked → **fp8816** (q=fp8, kv=fp8, o=bf16) on backend `fa3`.
///   - decode → **fp16816** (q=bf16, kv=fp8, o=bf16) on backend `fa2` — decode is
///     KV-bandwidth bound, so only the KV cache is fp8, the query stays 16-bit.
/// Non-fp8 keeps everything at `dtype` on the caller's `backends`. L2 design §3.5-1.
#[derive(Clone, Debug)]
pub struct FlashInferAttentionConfig {
    /// Backends for the non-fp8 path (best-of-N). In fp8 mode the op overrides
    /// them per phase (prefill=fa3, decode=fa2), so this is ignored there.
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_qo_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    /// Base (16-bit) dtype — the attention output and (fp16816) decode query. In
    /// non-fp8 it is also q/kv.
    pub dtype: DType,
    /// FP8 run: prefill q/kv → fp8 (fp8816), decode kv → fp8 (fp16816).
    pub fp8: bool,
    /// Separate cache-write implementation and physical paged-cache contract.
    pub kv_cache_append_backends: Vec<&'static str>,
    pub kv_cache_block_size: u32,
    pub kv_cache_layout: String,
    pub kv_scale_granularity: String,
}

/// Op-level input: raw per-request data for one attention call in the sim batch
/// (L2 design §3.2). Partition / dispatch happen inside `eval` — L3 never sees
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

impl FlashInferAttentionConfig {
    /// KV cache dtype: fp8 in an fp8 run (both prefill and decode read fp8 KV),
    /// else the base dtype. Used by the arch's KV-byte accounting.
    pub fn kv_dtype(&self) -> DType {
        if self.fp8 {
            DType::Fp8E4m3
        } else {
            self.dtype
        }
    }
}

pub struct FlashInferAttentionOp {
    pub name: String,
    pub kv_cache_append: Arc<KvCacheAppendKernel>,
    pub prefill: Arc<FlashinferAttnPrefillKernel>,
    pub decode: Arc<FlashinferAttnDecodeKernel>,
}

impl FlashInferAttentionOp {
    pub fn build(
        name: String,
        cfg: FlashInferAttentionConfig,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let kv_cache_append = Arc::new(KvCacheAppendKernel::build(
            format!("{name}.kv_cache_append"),
            kv_cache_append_config(&cfg),
            bridge,
        )?);
        let prefill = Arc::new(FlashinferAttnPrefillKernel::build(
            format!("{name}.prefill"),
            prefill_config(&cfg),
            bridge,
        )?);
        let decode = Arc::new(FlashinferAttnDecodeKernel::build(
            format!("{name}.decode"),
            decode_config(&cfg),
            bridge,
        )?);
        Ok(Self {
            name,
            kv_cache_append,
            prefill,
            decode,
        })
    }

    /// CostTree compile: three fixed leaves — append, prefill, decode — regardless
    /// of request count (INV-1: stable shape). The per-request prefill fan-out is
    /// NOT one slot per request; at eval the `prefill` slot is the aggregating leaf
    /// that sums `prefill.eval(prefix_i, append_i)` over
    /// `prefill_chunk_pairs` into that single slot (decode already collapses to one
    /// cell). Each leaf carries its sub-kernel's `kind`/`config` for the render.
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Sum(vec![
            builder.leaf(
                format!("{}.kv_cache_append", self.name),
                self.kv_cache_append.kind(),
                self.kv_cache_append.describe_config(),
                self.kv_cache_append.backends(),
            ),
            builder.leaf(
                format!("{}.prefill", self.name),
                self.prefill.kind(),
                self.prefill.describe_config(),
                self.prefill.backends(),
            ),
            builder.leaf(
                format!("{}.decode", self.name),
                self.decode.kind(),
                self.decode.describe_config(),
                self.decode.backends(),
            ),
        ])
    }

    /// CostTree eval: fill the three fixed slots `compile` minted — append,
    /// `prefill`, then `decode`. The prefill slot is the INV-1 aggregating leaf:
    /// sum the per-request `prefill.eval` over `prefill_chunk_pairs` into one slot. The decode
    /// slot collapses all decode requests to one cell (zero metrics when none).
    /// `aggregate(Sum[append, prefill, decode])` reproduces the streamed slot total.
    pub fn eval(&self, input: &FlashInferAttentionInput, ev: &mut Evaluator) {
        let cache_append = kv_cache_append_input(input);
        let cache_append_metrics = match &cache_append {
            Some(shape) => self.kv_cache_append.eval(shape),
            None => LeafMetrics::ZERO,
        };
        ev.push(cache_append_metrics, || {
            cache_append
                .unwrap_or(KvCacheAppendKernelInput { num_tokens: 0 })
                .into()
        });

        let mut prefill = LeafMetrics::ZERO;
        for &(prefix_len, append_len) in &input.prefill_chunk_pairs {
            // `add_fanin` (not `add`) so the aggregated prefill slot carries the
            // selected backend into `slot_backend` — otherwise it keeps the ZERO
            // accumulator's NO_BACKEND sentinel and reads as "never executed".
            prefill.add_fanin(self.prefill.eval(&FlashinferAttnPrefillKernelInput {
                prefix_len,
                append_len,
            }));
        }
        // The prefill slot is the INV-1 aggregating leaf: its faithful `slot_input`
        // is the whole `(prefix, append)` fan-out it summed over (cloned only when
        // recording — see `Evaluator::push`).
        ev.push(prefill, || {
            AttnPrefillLog {
                prefill_chunk_pairs: input.prefill_chunk_pairs.clone(),
            }
            .into()
        });

        // The decode slot's faithful input is the collapsed cell the kernel saw;
        // reuse the real `FlashinferAttnDecodeKernelInput` (zeros when no decode).
        let decode = decode_input(&input.decode_kv_lens);
        let decode_metrics = match &decode {
            Some(d) => self.decode.eval(d),
            None => LeafMetrics::ZERO,
        };
        ev.push(decode_metrics, || {
            decode
                .unwrap_or(FlashinferAttnDecodeKernelInput {
                    batch_size: 0,
                    total_tokens: 0,
                })
                .into()
        });
    }
}

// ─── internal helpers (pure; unit-tested without a bridge) ───────────────────

fn kv_cache_append_config(cfg: &FlashInferAttentionConfig) -> KvCacheAppendKernelConfig {
    KvCacheAppendKernelConfig {
        backends: cfg.kv_cache_append_backends.clone(),
        gpu_name: cfg.gpu_name.clone(),
        num_kv_heads: cfg.num_kv_heads.clone(),
        head_dim: cfg.head_dim.clone(),
        block_size: cfg.kv_cache_block_size,
        input_dtype: cfg.dtype,
        kv_dtype: cfg.kv_dtype(),
        cache_layout: cfg.kv_cache_layout.clone(),
        scale_granularity: cfg.kv_scale_granularity.clone(),
    }
}

/// vLLM appends every actual new token once per layer: each prefill contributes
/// its append length, and each decode request contributes one token.
fn kv_cache_append_input(input: &FlashInferAttentionInput) -> Option<KvCacheAppendKernelInput> {
    let prefill_tokens: u32 = input
        .prefill_chunk_pairs
        .iter()
        .map(|&(_, append_len)| append_len)
        .sum();
    let num_tokens = prefill_tokens + input.decode_kv_lens.len() as u32;
    (num_tokens > 0).then_some(KvCacheAppendKernelInput { num_tokens })
}

/// Prefill / chunked preset. fp8 → **fp8816** (q=fp8, kv=fp8, o=base) on `fa3`
/// (fa2 has no fp8-query kernel); else base dtype on the caller's backends.
fn prefill_config(cfg: &FlashInferAttentionConfig) -> FlashinferAttnPrefillKernelConfig {
    let (backends, q, kv) = if cfg.fp8 {
        (vec!["fa3"], DType::Fp8E4m3, DType::Fp8E4m3)
    } else {
        (cfg.backends.clone(), cfg.dtype, cfg.dtype)
    };
    FlashinferAttnPrefillKernelConfig {
        backends,
        gpu_name: cfg.gpu_name.clone(),
        num_qo_heads: cfg.num_qo_heads.clone(),
        num_kv_heads: cfg.num_kv_heads.clone(),
        head_dim: cfg.head_dim.clone(),
        q_dtype: q,
        kv_dtype: kv,
        o_dtype: cfg.dtype,
    }
}

/// Decode preset. fp8 → **fp16816** (q=base bf16, kv=fp8, o=base) on `fa2` —
/// decode is KV-bandwidth bound, so only the KV cache is fp8; else base dtype on
/// the caller's backends.
fn decode_config(cfg: &FlashInferAttentionConfig) -> FlashinferAttnDecodeKernelConfig {
    let (backends, kv) = if cfg.fp8 {
        (vec!["fa2"], DType::Fp8E4m3)
    } else {
        (cfg.backends.clone(), cfg.dtype)
    };
    FlashinferAttnDecodeKernelConfig {
        backends,
        gpu_name: cfg.gpu_name.clone(),
        num_qo_heads: cfg.num_qo_heads.clone(),
        num_kv_heads: cfg.num_kv_heads.clone(),
        head_dim: cfg.head_dim.clone(),
        q_dtype: cfg.dtype,
        kv_dtype: kv,
        o_dtype: cfg.dtype,
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
    use super::{
        decode_config, decode_input, kv_cache_append_config, kv_cache_append_input, prefill_config,
        FlashInferAttentionConfig, FlashInferAttentionInput,
    };
    use crate::timing::bridge::DType;

    fn cfg(fp8: bool) -> FlashInferAttentionConfig {
        FlashInferAttentionConfig {
            backends: vec!["fa2", "fa3"],
            gpu_name: "H200".to_string(),
            num_qo_heads: 8.into(),
            num_kv_heads: 2.into(),
            head_dim: 128.into(),
            dtype: DType::Bf16,
            fp8,
            kv_cache_append_backends: vec!["vllm_cuda"],
            kv_cache_block_size: 16,
            kv_cache_layout: "NHD".to_string(),
            kv_scale_granularity: "tensor".to_string(),
        }
    }

    #[test]
    fn fp8_prefill_is_fp8816_on_fa3() {
        let p = prefill_config(&cfg(true));
        assert_eq!(p.backends, vec!["fa3"]);
        assert_eq!(p.q_dtype, DType::Fp8E4m3);
        assert_eq!(p.kv_dtype, DType::Fp8E4m3);
        assert_eq!(p.o_dtype, DType::Bf16);
    }

    #[test]
    fn fp8_decode_is_fp16816_on_fa2() {
        let d = decode_config(&cfg(true));
        assert_eq!(d.backends, vec!["fa2"]);
        assert_eq!(d.q_dtype, DType::Bf16); // decode query stays 16-bit
        assert_eq!(d.kv_dtype, DType::Fp8E4m3);
        assert_eq!(d.o_dtype, DType::Bf16);
    }

    #[test]
    fn non_fp8_keeps_base_dtype_and_backends_both_phases() {
        let p = prefill_config(&cfg(false));
        let d = decode_config(&cfg(false));
        assert_eq!(p.backends, vec!["fa2", "fa3"]);
        assert_eq!(d.backends, vec!["fa2", "fa3"]);
        for dt in [
            p.q_dtype, p.kv_dtype, p.o_dtype, d.q_dtype, d.kv_dtype, d.o_dtype,
        ] {
            assert_eq!(dt, DType::Bf16);
        }
    }

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

    #[test]
    fn cache_append_counts_prefill_appends_and_decode_requests() {
        let input = FlashInferAttentionInput {
            prefill_chunk_pairs: vec![(0, 512), (1024, 128)],
            decode_kv_lens: vec![100, 200, 300],
        };
        assert_eq!(kv_cache_append_input(&input).unwrap().num_tokens, 643);
        assert!(kv_cache_append_input(&FlashInferAttentionInput::default()).is_none());
    }

    #[test]
    fn cache_append_config_uses_kv_precision_and_layout() {
        let c = kv_cache_append_config(&cfg(true));
        assert_eq!(c.backends, vec!["vllm_cuda"]);
        assert_eq!(c.input_dtype, DType::Bf16);
        assert_eq!(c.kv_dtype, DType::Fp8E4m3);
        assert_eq!(c.block_size, 16);
        assert_eq!(c.cache_layout, "NHD");
        assert_eq!(c.scale_granularity, "tensor");
    }
}
