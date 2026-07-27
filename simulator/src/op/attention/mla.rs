//! `MlaAttentionOp` — compound L2 op for MLA (multi-head latent attention,
//! DeepSeek/Kimi style) over a compressed-KV paged cache. One attention call
//! dispatches to three L1 kernels, mirroring `FlashInferAttentionOp` but with
//! **asymmetric prefill/decode geometry**:
//!
//!   - `kv_cache_append`: writes ONE compressed `kv_compressed_dim`-wide vector
//!     per token per layer (num_kv_heads=1; K and V share it — hence the
//!     factor-2-free KV accounting at the arch).
//!   - prefill: the EXISTING `flashinfer_attn_prefill` family at the
//!     **decompressed MHA** shape — `num_heads` qo heads × `num_heads` kv heads
//!     × `head_dim` (96/96/128 for Kimi-K3), profilable with the existing
//!     runners. The nope/rope qk split (192-wide q·k) is NOT modeled
//!     separately; the 128-wide MHA shape is the profiling proxy.
//!   - decode: the EXISTING `flashinfer_attn_decode` family at the
//!     **absorbed-weight MQA** shape — `num_heads` qo heads × 1 kv head ×
//!     `kv_compressed_dim` head_dim (96/1/576). Rows are profiled later; the
//!     Rust kernel spec carries no head_dim cap, so 576 needs no validation
//!     relaxation.
//!
//! The v1 batching cost model is inherited verbatim from `FlashInferAttentionOp`:
//! per-request prefill sum into one aggregating slot, decode collapsed to one
//! `(batch_size, total_tokens)` cell.

use std::sync::Arc;

use crate::timing::bridge::DType;
use crate::timing::kernels::{
    FlashinferAttnDecodeKernel, FlashinferAttnDecodeKernelConfig, FlashinferAttnDecodeKernelInput,
    FlashinferAttnPrefillKernel, FlashinferAttnPrefillKernelConfig,
    FlashinferAttnPrefillKernelInput, KvCacheAppendKernel, KvCacheAppendKernelConfig,
    KvCacheAppendKernelInput,
};
use crate::timing::{
    AttnPrefillLog, BuildError, CostNode, CostTreeBuilder, Evaluator, LeafMetrics, PerfApiBridge,
    Probe,
};

/// Single op-level config; expands into the three sub-kernel configs above.
/// FP8 policy mirrors `FlashInferAttentionOp`: prefill fp8816 on `fa3`, decode
/// fp16816 on `fa2`, non-fp8 keeps `dtype` on the caller's backend lists.
#[derive(Clone, Debug)]
pub struct MlaAttentionConfig {
    pub gpu_name: String,
    /// Query/output heads (96). Prefill also uses this as its kv-head count
    /// (decompressed MHA); decode always uses ONE shared compressed kv head.
    pub num_heads: u32,
    /// Decompressed per-head dim for prefill q/k/v (128).
    pub head_dim: u32,
    /// Compressed-KV vector width = decode MQA head_dim (kv_lora_rank +
    /// qk_rope_head_dim = 576).
    pub kv_compressed_dim: u32,
    /// Base (16-bit) dtype.
    pub dtype: DType,
    /// FP8 run: prefill q/kv → fp8 on fa3, decode kv → fp8 on fa2.
    pub fp8: bool,
    pub prefill_backends: Vec<&'static str>,
    pub decode_backends: Vec<&'static str>,
    pub kv_cache_append_backends: Vec<&'static str>,
    pub kv_cache_block_size: u32,
    pub kv_cache_layout: String,
    pub kv_scale_granularity: String,
}

/// Op-level input — identical shape to `FlashInferAttentionInput`.
#[derive(Clone, Debug, Default)]
pub struct MlaAttentionInput {
    /// Every prefill / chunked-prefill request as `(prefix_len, append_len)`.
    pub prefill_chunk_pairs: Vec<(u32, u32)>,
    /// Every decode request's KV length (one `q = 1` token each).
    pub decode_kv_lens: Vec<u32>,
}

impl MlaAttentionConfig {
    /// Compressed-KV cache dtype: fp8 in an fp8 run, else the base dtype.
    pub fn kv_dtype(&self) -> DType {
        if self.fp8 {
            DType::Fp8E4m3
        } else {
            self.dtype
        }
    }
}

pub struct MlaAttentionOp {
    pub name: String,
    pub kv_cache_append: Arc<KvCacheAppendKernel>,
    pub prefill: Arc<FlashinferAttnPrefillKernel>,
    pub decode: Arc<FlashinferAttnDecodeKernel>,
}

impl MlaAttentionOp {
    pub fn build(
        name: String,
        cfg: MlaAttentionConfig,
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

    /// CostTree compile: three fixed leaves — append, prefill, decode (INV-1).
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

    /// CostTree eval: fill the three slots in `compile` order (mirrors
    /// `FlashInferAttentionOp::eval`).
    pub fn eval(&self, input: &MlaAttentionInput, ev: &mut Evaluator) {
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
            prefill.add_fanin(self.prefill.eval(&FlashinferAttnPrefillKernelInput {
                prefix_len,
                append_len,
            }));
        }
        ev.push(prefill, || {
            AttnPrefillLog {
                prefill_chunk_pairs: input.prefill_chunk_pairs.clone(),
            }
            .into()
        });

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

/// Compressed-KV cache write: ONE `kv_compressed_dim`-wide vector per token
/// (num_kv_heads = 1) — K and V share the latent, so no factor 2.
fn kv_cache_append_config(cfg: &MlaAttentionConfig) -> KvCacheAppendKernelConfig {
    KvCacheAppendKernelConfig {
        backends: cfg.kv_cache_append_backends.clone(),
        gpu_name: cfg.gpu_name.clone(),
        num_kv_heads: 1,
        head_dim: cfg.kv_compressed_dim,
        block_size: cfg.kv_cache_block_size,
        input_dtype: cfg.dtype,
        kv_dtype: cfg.kv_dtype(),
        cache_layout: cfg.kv_cache_layout.clone(),
        scale_granularity: cfg.kv_scale_granularity.clone(),
    }
}

fn kv_cache_append_input(input: &MlaAttentionInput) -> Option<KvCacheAppendKernelInput> {
    let prefill_tokens: u32 = input
        .prefill_chunk_pairs
        .iter()
        .map(|&(_, append_len)| append_len)
        .sum();
    let num_tokens = prefill_tokens + input.decode_kv_lens.len() as u32;
    (num_tokens > 0).then_some(KvCacheAppendKernelInput { num_tokens })
}

/// Prefill: decompressed MHA — qo == kv == `num_heads`, `head_dim` per head.
/// fp8 → fp8816 on `fa3`, else base dtype on `prefill_backends`.
fn prefill_config(cfg: &MlaAttentionConfig) -> FlashinferAttnPrefillKernelConfig {
    let (backends, q, kv) = if cfg.fp8 {
        (vec!["fa3"], DType::Fp8E4m3, DType::Fp8E4m3)
    } else {
        (cfg.prefill_backends.clone(), cfg.dtype, cfg.dtype)
    };
    FlashinferAttnPrefillKernelConfig {
        backends,
        gpu_name: cfg.gpu_name.clone(),
        num_qo_heads: cfg.num_heads,
        num_kv_heads: cfg.num_heads,
        head_dim: cfg.head_dim,
        q_dtype: q,
        kv_dtype: kv,
        o_dtype: cfg.dtype,
    }
}

/// Decode: absorbed-weight MQA — `num_heads` qo heads, ONE compressed kv head,
/// head_dim = `kv_compressed_dim` (576). fp8 → fp16816 on `fa2`, else base
/// dtype on `decode_backends`.
fn decode_config(cfg: &MlaAttentionConfig) -> FlashinferAttnDecodeKernelConfig {
    let (backends, kv) = if cfg.fp8 {
        (vec!["fa2"], DType::Fp8E4m3)
    } else {
        (cfg.decode_backends.clone(), cfg.dtype)
    };
    FlashinferAttnDecodeKernelConfig {
        backends,
        gpu_name: cfg.gpu_name.clone(),
        num_qo_heads: cfg.num_heads,
        num_kv_heads: 1,
        head_dim: cfg.kv_compressed_dim,
        q_dtype: cfg.dtype,
        kv_dtype: kv,
        o_dtype: cfg.dtype,
    }
}

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
        MlaAttentionConfig, MlaAttentionInput,
    };
    use crate::timing::bridge::DType;

    fn cfg(fp8: bool) -> MlaAttentionConfig {
        MlaAttentionConfig {
            gpu_name: "NVIDIA H200".to_string(),
            num_heads: 96,
            head_dim: 128,
            kv_compressed_dim: 576,
            dtype: DType::Bf16,
            fp8,
            prefill_backends: vec!["fa2", "fa3"],
            decode_backends: vec!["fa2", "fa3"],
            kv_cache_append_backends: vec!["vllm_cuda"],
            kv_cache_block_size: 16,
            kv_cache_layout: "NHD".to_string(),
            kv_scale_granularity: "tensor".to_string(),
        }
    }

    #[test]
    fn prefill_is_decompressed_mha_96_96_128() {
        let p = prefill_config(&cfg(false));
        assert_eq!(p.num_qo_heads, 96);
        assert_eq!(p.num_kv_heads, 96);
        assert_eq!(p.head_dim, 128);
        assert_eq!(p.backends, vec!["fa2", "fa3"]);
    }

    #[test]
    fn decode_is_absorbed_mqa_96_1_576() {
        let d = decode_config(&cfg(false));
        assert_eq!(d.num_qo_heads, 96);
        assert_eq!(d.num_kv_heads, 1);
        assert_eq!(d.head_dim, 576);
        assert_eq!(d.backends, vec!["fa2", "fa3"]);
    }

    #[test]
    fn cache_append_writes_one_compressed_vector_per_token() {
        let c = kv_cache_append_config(&cfg(false));
        assert_eq!(c.num_kv_heads, 1);
        assert_eq!(c.head_dim, 576);
        assert_eq!(c.kv_dtype, DType::Bf16);
    }

    #[test]
    fn fp8_presets_mirror_the_flashinfer_op() {
        let p = prefill_config(&cfg(true));
        assert_eq!(p.backends, vec!["fa3"]);
        assert_eq!(p.q_dtype, DType::Fp8E4m3);
        let d = decode_config(&cfg(true));
        assert_eq!(d.backends, vec!["fa2"]);
        assert_eq!(d.q_dtype, DType::Bf16); // decode query stays 16-bit
        assert_eq!(d.kv_dtype, DType::Fp8E4m3);
    }

    #[test]
    fn append_and_decode_inputs_collapse_like_the_flashinfer_op() {
        let input = MlaAttentionInput {
            prefill_chunk_pairs: vec![(0, 512), (1024, 128)],
            decode_kv_lens: vec![100, 200, 300],
        };
        assert_eq!(kv_cache_append_input(&input).unwrap().num_tokens, 643);
        let d = decode_input(&input.decode_kv_lens).unwrap();
        assert_eq!(d.batch_size, 3);
        assert_eq!(d.total_tokens, 600);
        assert!(decode_input(&[]).is_none());
    }
}
