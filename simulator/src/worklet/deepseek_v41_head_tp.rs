//! DeepSeek-V4.1-Flash forward tail on one TP rank (vLLM): the last FFN's hc
//! post, the terminal mHC collapse, the final RMSNorm, the vocab-parallel
//! `lm_head`, and the logits AllGather.
//!
//! Capture 2 decode iteration 1337 (48 rows, device 0): `mhc_post_tilelang`
//! 11259.81-11269.48 us, `_hc_head_reduce_store_kernel` 11269.80 (1.8 us),
//! `vllm::rms_norm_kernel` 11272.03 (2.9 us), all over every scheduled token;
//! then over the logits rows: the row gather 11286.79, `lm_head` nvjet bf16
//! 11289.67-11347.27 (57.6 us), `ncclDevKernel_AllGather_RING_LL`
//! 11349.28-11399.97 (50.7 us). Mixed 310 (2048 tokens): mhc_post 34 us,
//! head reduce 13.6, rms_norm 10.5 over all tokens; lm_head 66 us and
//! AllGather 56.5 us over the logits rows. Sampling follows but, as in the V4
//! arch, is not modelled.
//!
//! No mHC-head or RMSNorm L1 rows exist on B200 at hidden 5120 (profile.db
//! `rms_norm` has flashinfer B200 rows at 128/512/2048/6144 only), so both are
//! byte placeholders.

use std::sync::Arc;

use super::deepseek_v41_common::{
    all_gather_as_all_reduce_bytes, eval_or_zero, nccl_all_gather_proxy, placeholder,
};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    AllReduceKernel, AllReduceKernelConfig, AllReduceKernelInput, ElementwiseKernel,
    ElementwiseKernelConfig, ElementwiseKernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

#[derive(Clone, Debug)]
pub struct DeepseekV41HeadTpWorkletConfig {
    pub tp_size: u32,
    pub hidden_size: Dim,
    pub hc_mult: u32,
    pub vocab_size: u32,
    pub gpu_name: String,
    pub lm_head_backends: Vec<&'static str>,
    pub all_gather_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct DeepseekV41HeadTpWorkletResolved {
    pub raw_cfg: DeepseekV41HeadTpWorkletConfig,
    pub vocab_per_rank: u32,
    pub mhc_post: ElementwiseKernelConfig,
    pub hc_collapse: ElementwiseKernelConfig,
    pub final_norm: ElementwiseKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub logits_all_gather: Option<AllReduceKernelConfig>,
}

#[derive(Clone, Debug, Default)]
pub struct DeepseekV41HeadTpWorkletInput {
    /// Every scheduled token (the hc post / collapse / norm run over all).
    pub num_tokens: u32,
    /// Rows that produce logits (one per request, more with spec decode).
    pub logits_rows: u32,
}

pub struct DeepseekV41HeadTpWorklet {
    pub name: String,
    pub mhc_post: Op<ElementwiseKernel>,
    pub hc_collapse: Op<ElementwiseKernel>,
    pub final_norm: Op<ElementwiseKernel>,
    pub lm_head: Op<SingleGemmKernel>,
    pub logits_all_gather: Option<Op<AllReduceKernel>>,
    resolved: DeepseekV41HeadTpWorkletResolved,
}

impl DeepseekV41HeadTpWorklet {
    pub fn resolve_config(cfg: &DeepseekV41HeadTpWorkletConfig) -> DeepseekV41HeadTpWorkletResolved {
        assert!(cfg.tp_size > 0, "tp_size must be positive");
        assert_eq!(cfg.vocab_size % cfg.tp_size, 0, "vocab must divide tp_size");
        let gpu = cfg.gpu_name.as_str();
        let ew = |input: u32, output: u32| placeholder(&cfg.elementwise_backends, gpu, input, output);
        let hidden_bytes = cfg.hidden_size.get() * 2;
        let hc_bytes = cfg.hc_mult * hidden_bytes;
        let vocab_per_rank = cfg.vocab_size / cfg.tp_size;
        DeepseekV41HeadTpWorkletResolved {
            vocab_per_rank,
            // nvidia/model.py:858 last FFN's `mhc_post`.
            mhc_post: ew(hc_bytes + hidden_bytes, hc_bytes),
            // nvidia/model.py:890 `_hc_head_reduce_store`: hc streams and the
            // fp32 head mix in, one bf16 row out.
            hc_collapse: ew(hc_bytes + cfg.hc_mult * 4, hidden_bytes),
            // nvidia/model.py:891 final `RMSNorm`.
            final_norm: ew(hidden_bytes, hidden_bytes),
            lm_head: SingleGemmKernelConfig {
                backends: cfg.lm_head_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: vocab_per_rank.into(),
                k: cfg.hidden_size.clone(),
                dtype: DType::Bf16,
            },
            // layers/logits_processor.py:130 `tensor_model_parallel_all_gather`.
            logits_all_gather: (cfg.tp_size > 1).then(|| {
                nccl_all_gather_proxy(&cfg.all_gather_backends, gpu, cfg.tp_size)
            }),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: DeepseekV41HeadTpWorkletResolved,
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
            mhc_post: op!(ElementwiseKernel, r.mhc_post, "mhc_post"),
            hc_collapse: op!(ElementwiseKernel, r.hc_collapse, "hc_head_collapse"),
            final_norm: op!(ElementwiseKernel, r.final_norm, "final_rms_norm"),
            lm_head: op!(SingleGemmKernel, r.lm_head, "lm_head"),
            logits_all_gather: match r.logits_all_gather {
                Some(config) => Some(op!(AllReduceKernel, config, "logits_all_gather")),
                None => None,
            },
            name,
            resolved,
        })
    }

    pub fn compile(&self, b: &mut CostTreeBuilder) -> CostNode {
        let mut children = vec![
            self.mhc_post.compile(b),
            self.hc_collapse.compile(b),
            self.final_norm.compile(b),
            self.lm_head.compile(b),
        ];
        if let Some(all_gather) = &self.logits_all_gather {
            children.push(all_gather.compile(b));
        }
        CostNode::Labeled {
            label: format!(
                "{} (DeepseekV41HeadTpWorklet) [tp={}; vocab/rank={}]",
                self.name, self.resolved.raw_cfg.tp_size, self.resolved.vocab_per_rank,
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    pub fn eval(&self, input: &DeepseekV41HeadTpWorkletInput, ev: &mut Evaluator) {
        assert!(
            input.logits_rows <= input.num_tokens,
            "logits rows ({}) exceed scheduled tokens ({})",
            input.logits_rows,
            input.num_tokens
        );
        let rows = input.num_tokens;
        let logits = input.logits_rows;
        let ew = ElementwiseKernelInput { num_tokens: rows };
        eval_or_zero(&self.mhc_post, ew.clone(), rows == 0, ev);
        eval_or_zero(&self.hc_collapse, ew.clone(), rows == 0, ev);
        eval_or_zero(&self.final_norm, ew, rows == 0, ev);
        eval_or_zero(&self.lm_head, SingleGemmKernelInput { m: logits }, logits == 0, ev);
        if let Some(all_gather) = &self.logits_all_gather {
            let gathered = u64::from(logits) * u64::from(self.resolved.raw_cfg.vocab_size) * 2;
            eval_or_zero(
                all_gather,
                AllReduceKernelInput {
                    message_size_bytes: all_gather_as_all_reduce_bytes(gathered),
                },
                logits == 0,
                ev,
            );
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn config() -> DeepseekV41HeadTpWorkletConfig {
        DeepseekV41HeadTpWorkletConfig {
            tp_size: 4,
            hidden_size: 5120.into(),
            hc_mult: 4,
            vocab_size: 129_280,
            gpu_name: "NVIDIA B200".into(),
            lm_head_backends: vec!["torch_linear"],
            all_gather_backends: vec!["nccl"],
            elementwise_backends: vec!["triton"],
        }
    }

    #[test]
    fn lm_head_is_vocab_parallel() {
        let r = DeepseekV41HeadTpWorklet::resolve_config(&config());
        assert_eq!((r.lm_head.k.get(), r.lm_head.n.get()), (5120, 32_320));
        assert_eq!(r.hc_collapse.input_bytes_per_token.get(), 40_976);
    }
}
