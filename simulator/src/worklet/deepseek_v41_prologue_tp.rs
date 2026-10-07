//! DeepSeek-V4.1-Flash forward prologue on one TP rank (vLLM): the
//! vocab-parallel embedding, its all-reduce, and the Engram hash.
//!
//! Capture 2 decode iteration 1337 (48 rows, device 0):
//! `vocab_parallel_embedding_kernel` 181.02-183.74 us, then
//! `trtllm_mnnvl_allreduce::twoshotAllreduceKernel` 183.74-195.39 (the one
//! all-reduce outside the 80 per-layer ones), then `_hash_ids_kernel`
//! 218.05-223.62 (fork `models/deepseek_v41/nvidia/model.py:777`), whose
//! output feeds both Engram lookups.
//!
//! The ~35 attention/indexer metadata launches before the embedding
//! (`_compute_swa_indices_and_lens_kernel`, `_indexer_decode_metadata_kernel`,
//! `sm100_paged_mqa_logits_metadata`, ...: 0-178 us at 48 rows) are
//! launch-bound, not byte-bound, so an elementwise placeholder understates
//! them badly; the optional `metadata` placeholder exists only so an L4 can
//! opt in, and the recommended treatment is the framework-overhead term.

use std::sync::Arc;

use super::deepseek_v41_common::{eval_or_zero, placeholder};
use crate::common::Fabric;
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    AllReduceFusionKernel, AllReduceFusionKernelConfig, AllReduceFusionKernelInput,
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

#[derive(Clone, Debug)]
pub struct DeepseekV41PrologueTpWorkletConfig {
    pub tp_size: u32,
    pub hidden_size: Dim,
    /// Engram hash columns (24).
    pub engram_num_heads: u32,
    /// `(input, output)` bytes per token of the optional metadata placeholder.
    pub metadata_bytes_per_token: Option<(u32, u32)>,
    pub gpu_name: String,
    pub all_reduce_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct DeepseekV41PrologueTpWorkletResolved {
    pub raw_cfg: DeepseekV41PrologueTpWorkletConfig,
    pub metadata: Option<ElementwiseKernelConfig>,
    pub embedding: ElementwiseKernelConfig,
    pub embedding_all_reduce: Option<AllReduceFusionKernelConfig>,
    pub engram_hash: ElementwiseKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct DeepseekV41PrologueTpWorkletInput {
    pub num_tokens: u32,
}

pub struct DeepseekV41PrologueTpWorklet {
    pub name: String,
    pub metadata: Option<Op<ElementwiseKernel>>,
    pub embedding: Op<ElementwiseKernel>,
    pub embedding_all_reduce: Option<Op<AllReduceFusionKernel>>,
    pub engram_hash: Op<ElementwiseKernel>,
    resolved: DeepseekV41PrologueTpWorkletResolved,
}

impl DeepseekV41PrologueTpWorklet {
    pub fn resolve_config(
        cfg: &DeepseekV41PrologueTpWorkletConfig,
    ) -> DeepseekV41PrologueTpWorkletResolved {
        assert!(cfg.tp_size > 0, "tp_size must be positive");
        let gpu = cfg.gpu_name.as_str();
        let ew = |input: u32, output: u32| placeholder(&cfg.elementwise_backends, gpu, input, output);
        let hidden = cfg.hidden_size.get();
        DeepseekV41PrologueTpWorkletResolved {
            metadata: cfg.metadata_bytes_per_token.map(|(i, o)| ew(i, o)),
            // layers/vocab_parallel_embedding.py: int32 id in, bf16 row out
            // (zero for ids outside this rank's vocab shard).
            embedding: ew(4, hidden * 2),
            embedding_all_reduce: (cfg.tp_size > 1).then(|| AllReduceFusionKernelConfig {
                backends: cfg.all_reduce_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_gpus: cfg.tp_size,
                hidden_dim: hidden,
                dtype: DType::Bf16,
                fabric: Fabric::Nvlink,
                fused_token_limit: None,
            }),
            // nvidia/model.py:777 `_hash_ids_kernel`: the current id and its
            // lookback window in, one int64 row id per hash column out.
            engram_hash: ew(16, cfg.engram_num_heads * 8),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: DeepseekV41PrologueTpWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        macro_rules! op {
            ($kernel:ty, $cfg:expr, $suffix:literal) => {{
                let op_name = format!("{name}.{}", $suffix);
                Op::new(
                    op_name.clone(),
                    Arc::new(<$kernel>::build(op_name, $cfg, bridge)?),
                )
            }};
        }
        let r = resolved.clone();
        Ok(Self {
            metadata: match r.metadata {
                Some(config) => Some(op!(ElementwiseKernel, config, "metadata")),
                None => None,
            },
            embedding: op!(ElementwiseKernel, r.embedding, "embedding"),
            embedding_all_reduce: match r.embedding_all_reduce {
                Some(config) => Some(op!(AllReduceFusionKernel, config, "embedding_all_reduce")),
                None => None,
            },
            engram_hash: op!(ElementwiseKernel, r.engram_hash, "engram_hash"),
            name,
            resolved,
        })
    }

    pub fn compile(&self, b: &mut CostTreeBuilder) -> CostNode {
        let mut children = Vec::new();
        if let Some(metadata) = &self.metadata {
            children.push(metadata.compile(b));
        }
        children.push(self.embedding.compile(b));
        if let Some(all_reduce) = &self.embedding_all_reduce {
            children.push(all_reduce.compile(b));
        }
        children.push(self.engram_hash.compile(b));
        CostNode::Labeled {
            label: format!(
                "{} (DeepseekV41PrologueTpWorklet) [tp={}; metadata_placeholder={}]",
                self.name,
                self.resolved.raw_cfg.tp_size,
                self.metadata.is_some(),
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    pub fn eval(&self, input: &DeepseekV41PrologueTpWorkletInput, ev: &mut Evaluator) {
        let rows = input.num_tokens;
        let zero = rows == 0;
        let ew = ElementwiseKernelInput { num_tokens: rows };
        if let Some(metadata) = &self.metadata {
            eval_or_zero(metadata, ew.clone(), zero, ev);
        }
        eval_or_zero(&self.embedding, ew.clone(), zero, ev);
        if let Some(all_reduce) = &self.embedding_all_reduce {
            eval_or_zero(all_reduce, AllReduceFusionKernelInput { num_tokens: rows }, zero, ev);
        }
        eval_or_zero(&self.engram_hash, ew, zero, ev);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn config() -> DeepseekV41PrologueTpWorkletConfig {
        DeepseekV41PrologueTpWorkletConfig {
            tp_size: 4,
            hidden_size: 5120.into(),
            engram_num_heads: 24,
            metadata_bytes_per_token: None,
            gpu_name: "NVIDIA B200".into(),
            all_reduce_backends: vec!["flashinfer_mnnvl"],
            elementwise_backends: vec!["triton"],
        }
    }

    #[test]
    fn metadata_placeholder_is_opt_in() {
        let r = DeepseekV41PrologueTpWorklet::resolve_config(&config());
        assert!(r.metadata.is_none());
        assert_eq!(r.embedding.output_bytes_per_token.get(), 10_240);
        assert_eq!(r.engram_hash.output_bytes_per_token.get(), 192);
        let mut with = config();
        with.metadata_bytes_per_token = Some((64, 64));
        assert!(DeepseekV41PrologueTpWorklet::resolve_config(&with).metadata.is_some());
    }
}
