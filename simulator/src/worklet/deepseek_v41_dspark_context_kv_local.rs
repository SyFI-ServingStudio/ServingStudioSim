//! DeepSeek-V4.1 DSpark context-KV precomputation on one rank (vLLM).
//!
//! The prefill-shaped half of a DSpark draft call, the DFlash2 context-KV
//! pattern ([`super::dflash2_context_kv_local`]) with the DSv4.1 projections
//! and insert. It runs once per engine step over **every target row the step
//! scheduled** (verify rows and prefill rows alike), not over the draft's own
//! query rows (fork `v1/worker/gpu/spec_decode/dflash/speculator.py:397-423`
//! `_store_hidden_states`, `models/deepseek_v41/nvidia/dspark.py:174-213`):
//!
//! 1. the target captures each `dspark_target_layer_ids` layer's post-mHC
//!    hidden as the mean over its `hc_mult` copies (`nvidia/model.py:594`,
//!    `:1274-1278`), and the speculator concatenates the three
//!    (`torch.cat`, one `at::native` launch);
//! 2. `main_proj` (`ReplicatedLinear`, MXFP8, K = 3 x 5120 -> 5120) and
//!    `main_norm`;
//! 3. `context_wkv_proj` (`MergedColumnParallelLinear(disable_tp=True)`,
//!    MXFP8, 5120 -> num_draft_layers x head_dim): every draft layer's KV
//!    in one GEMM;
//! 4. per draft layer: `kv_norm` (`vllm::rms_norm_kernel`) and
//!    `fused_deepseek_v4_kv_rope_insert`, the KV-only variant of the
//!    q-pad/KV-rope/MXFP8 insert (no query, `dspark.py:268-291`) into that
//!    layer's 32-token SWA pages.
//!
//! Both GEMMs are replicated, so every rank runs the whole section: there is
//! no collective, hence `Local`. The per-layer pair is identical across the
//! draft layers and folds with `Scale`.
//!
//! Captured TP4 decode iteration 938 (156 target rows, device 0): cat 3.7 us,
//! `main_proj` 26.6 us, `main_norm` 3.7 us, `context_wkv_proj` 9.2 us, then
//! 3 x (2.3 us norm + 3.0 us insert). In a 2048-row mixed step `main_proj`
//! is 89.7 us (`phase1_split.md` §1.1).

use std::sync::Arc;

use super::deepseek_v41_common::{eval_or_zero, placeholder};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput,
    QPadKvRopeMxfp8InsertKernel, QPadKvRopeMxfp8InsertKernelConfig,
    QPadKvRopeMxfp8InsertKernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

#[derive(Clone, Debug)]
pub struct DeepseekV41DsparkContextKvLocalWorkletConfig {
    pub hidden_size: Dim,
    pub hc_mult: u32,
    /// `len(dspark_target_layer_ids)`: target layers whose hidden feeds
    /// `main_proj`.
    pub num_target_layers: u32,
    /// Draft decoder layers, each with its own SWA cache.
    pub num_draft_layers: u32,
    /// The draft attention's KV width (`head_dim`, 512).
    pub head_dim: Dim,
    /// The KV insert's shard width: the KV-only launch requires the Q width it
    /// would pad to (64 at TP4) to be the live width.
    pub insert_num_heads: u32,
    pub swa_block_size: u32,
    pub gpu_name: String,
    pub gemm_backends: Vec<&'static str>,
    pub kv_insert_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct DeepseekV41DsparkContextKvLocalWorkletResolved {
    pub raw_cfg: DeepseekV41DsparkContextKvLocalWorkletConfig,
    pub aux_hc_mean: ElementwiseKernelConfig,
    pub aux_concat: ElementwiseKernelConfig,
    pub main_proj: SingleGemmKernelConfig,
    pub main_norm: ElementwiseKernelConfig,
    pub context_wkv_proj: SingleGemmKernelConfig,
    pub kv_norm: ElementwiseKernelConfig,
    pub kv_insert: QPadKvRopeMxfp8InsertKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct DeepseekV41DsparkContextKvLocalWorkletInput {
    /// Target rows this step scheduled (verify rows plus prefill rows).
    pub context_tokens: u32,
}

pub struct DeepseekV41DsparkContextKvLocalWorklet {
    pub name: String,
    pub aux_hc_mean: Op<ElementwiseKernel>,
    pub aux_concat: Op<ElementwiseKernel>,
    pub main_proj: Op<SingleGemmKernel>,
    pub main_norm: Op<ElementwiseKernel>,
    pub context_wkv_proj: Op<SingleGemmKernel>,
    pub kv_norm: Op<ElementwiseKernel>,
    pub kv_insert: Op<QPadKvRopeMxfp8InsertKernel>,
    resolved: DeepseekV41DsparkContextKvLocalWorkletResolved,
}

impl DeepseekV41DsparkContextKvLocalWorklet {
    pub fn resolve_config(
        cfg: &DeepseekV41DsparkContextKvLocalWorkletConfig,
    ) -> DeepseekV41DsparkContextKvLocalWorkletResolved {
        assert!(
            cfg.num_target_layers > 0,
            "DSpark reads at least one target layer"
        );
        assert!(
            cfg.num_draft_layers > 0,
            "DSpark has at least one draft layer"
        );
        let gpu = cfg.gpu_name.as_str();
        let ew =
            |input: u32, output: u32| placeholder(&cfg.elementwise_backends, gpu, input, output);
        let mxfp8 = |n: Dim, k: Dim| SingleGemmKernelConfig {
            backends: cfg.gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n,
            k,
            dtype: DType::Mxfp8E4m3,
        };
        let hidden_bytes = cfg.hidden_size.get() * 2;
        let aux_bytes = cfg.num_target_layers * hidden_bytes;
        let kv_bytes = cfg.head_dim.get() * 2;
        DeepseekV41DsparkContextKvLocalWorkletResolved {
            // `previous_post.mean(dim=1)` on each target layer: hc streams in,
            // one bf16 row out.
            aux_hc_mean: ew(cfg.hc_mult * aux_bytes, aux_bytes),
            aux_concat: ew(aux_bytes, aux_bytes),
            main_proj: mxfp8(
                cfg.hidden_size.clone(),
                cfg.hidden_size.clone() * cfg.num_target_layers,
            ),
            main_norm: ew(hidden_bytes, hidden_bytes),
            context_wkv_proj: mxfp8(
                cfg.head_dim.clone() * cfg.num_draft_layers,
                cfg.hidden_size.clone(),
            ),
            kv_norm: ew(kv_bytes, kv_bytes),
            kv_insert: QPadKvRopeMxfp8InsertKernelConfig {
                backends: cfg.kv_insert_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads: cfg.insert_num_heads,
                padded_heads: 0,
                block_size: cfg.swa_block_size,
                input_dtype: DType::Bf16,
                swa_cache_format: "mxfp8".to_string(),
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: DeepseekV41DsparkContextKvLocalWorkletResolved,
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
            aux_hc_mean: op!(ElementwiseKernel, r.aux_hc_mean, "aux_hc_mean"),
            aux_concat: op!(ElementwiseKernel, r.aux_concat, "aux_concat"),
            main_proj: op!(SingleGemmKernel, r.main_proj, "main_proj"),
            main_norm: op!(ElementwiseKernel, r.main_norm, "main_norm"),
            context_wkv_proj: op!(SingleGemmKernel, r.context_wkv_proj, "context_wkv_proj"),
            kv_norm: op!(ElementwiseKernel, r.kv_norm, "layer.kv_norm"),
            kv_insert: op!(QPadKvRopeMxfp8InsertKernel, r.kv_insert, "layer.kv_insert"),
            name,
            resolved,
        })
    }

    pub fn compile(&self, b: &mut CostTreeBuilder) -> CostNode {
        let layers = self.resolved.raw_cfg.num_draft_layers;
        let children = vec![
            self.aux_hc_mean.compile(b),
            self.aux_concat.compile(b),
            self.main_proj.compile(b),
            self.main_norm.compile(b),
            self.context_wkv_proj.compile(b),
            CostNode::Scale {
                n: layers,
                child: Box::new(CostNode::Sum(vec![
                    self.kv_norm.compile(b),
                    self.kv_insert.compile(b),
                ])),
            },
        ];
        CostNode::Labeled {
            label: format!(
                "{} (DeepseekV41DsparkContextKvLocalWorklet) [target layers={}; draft layers={}; \
                 main_proj K={}]",
                self.name,
                self.resolved.raw_cfg.num_target_layers,
                layers,
                self.resolved.main_proj.k.get(),
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    pub fn eval(&self, input: &DeepseekV41DsparkContextKvLocalWorkletInput, ev: &mut Evaluator) {
        let rows = input.context_tokens;
        let zero = rows == 0;
        let ew = ElementwiseKernelInput { num_tokens: rows };
        let gemm = SingleGemmKernelInput { m: rows };
        eval_or_zero(&self.aux_hc_mean, ew.clone(), zero, ev);
        eval_or_zero(&self.aux_concat, ew.clone(), zero, ev);
        eval_or_zero(&self.main_proj, gemm.clone(), zero, ev);
        eval_or_zero(&self.main_norm, ew.clone(), zero, ev);
        eval_or_zero(&self.context_wkv_proj, gemm, zero, ev);
        eval_or_zero(&self.kv_norm, ew, zero, ev);
        eval_or_zero(
            &self.kv_insert,
            QPadKvRopeMxfp8InsertKernelInput { num_tokens: rows },
            zero,
            ev,
        );
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn config() -> DeepseekV41DsparkContextKvLocalWorkletConfig {
        DeepseekV41DsparkContextKvLocalWorkletConfig {
            hidden_size: 5120.into(),
            hc_mult: 4,
            num_target_layers: 3,
            num_draft_layers: 3,
            head_dim: 512.into(),
            insert_num_heads: 64,
            swa_block_size: 32,
            gpu_name: "NVIDIA B200".into(),
            gemm_backends: vec!["flashinfer_mxfp8"],
            kv_insert_backends: vec!["vllm_cuda"],
            elementwise_backends: vec!["triton"],
        }
    }

    #[test]
    fn resolves_the_captured_projection_shapes() {
        let r = DeepseekV41DsparkContextKvLocalWorklet::resolve_config(&config());
        let kn = |c: &SingleGemmKernelConfig| (c.k.get(), c.n.get());
        assert_eq!(kn(&r.main_proj), (15_360, 5120));
        assert_eq!(kn(&r.context_wkv_proj), (5120, 1536));
        assert_eq!((r.kv_insert.num_heads, r.kv_insert.padded_heads), (64, 0));
    }
}
