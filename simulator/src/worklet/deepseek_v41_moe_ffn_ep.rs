//! DeepSeek-V4.1-Flash MoE FFN on one rank: TP4 shared expert, EP4 routed
//! experts (vLLM, TRT-LLM MXFP4 MoE on B200).
//!
//! One sync section from the FFN-entry `mega_mhc` to the TP all-reduce
//! (fork `models/deepseek_v41/nvidia/model.py:531`, `moe_runner.py`). Every
//! rank holds all `T` tokens after the attention all-reduce, routes them over
//! its 96 local experts, and the closing all-reduce sums the routed partials
//! with the TP-sharded shared expert.
//!
//! Streams (capture 2, job 1185, device 0). Decode 1337 (48 rows), layer 1:
//! router gate nvjet 670.75 on s19 while the shared expert's quant + gate_up
//! run 672.96-682.94 on s47910; `_dsv4_topk_kernel` 677.44, MoE input quant
//! 681.66, routing + fc1 + fc2 + finalize 683.39-813.60 on s19; shared
//! `act_and_mul` 683.39 and down 807.33-812.86 on s47910; the residual add at
//! 813.95 joins them. Mixed 288 (174 tokens) is also overlapped (routed on
//! s45352, shared on s19). Mixed 310 (2048 tokens) runs everything on s19.
//! That is `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD` (256, fork
//! `envs.py:299`, `fused_moe/runner/shared_experts.py:111-116`), so the shared
//! branch is a gated dual slot: `Max{routed, shared}` at `T <= 256`, serial
//! above.
//!
//! The busiest EP rank is `folded_rank_position = 0` of the shared
//! active-count-ranked fold inside the fused-MoE L1 kind; the L4 caller picks
//! the position (0 for the critical path).

use std::sync::Arc;

use super::deepseek_v41_common::{
    compile_serial_copy, eval_or_zero, gated_fanout, placeholder,
    SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD,
};
use crate::common::Fabric;
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::kernels::{
    AllReduceFusionKernel, AllReduceFusionKernelConfig, AllReduceFusionKernelInput,
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, GemmFp32OutputKernel,
    GemmFp32OutputKernelConfig, GemmFp32OutputKernelInput, MhcFusedPostPreRmsNormKernel,
    MhcRmsNormKernelConfig, MhcRmsNormKernelInput, Nvfp4FusedMoeKernel,
    Nvfp4FusedMoeKernelConfig, Nvfp4FusedMoeKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

#[derive(Clone, Debug)]
pub struct DeepseekV41MoeFfnEpWorkletConfig {
    pub tp_size: u32,
    pub ep_size: u32,
    pub serialize_streams: bool,
    pub hidden_size: Dim,
    pub hc_mult: u32,
    pub num_experts: Dim,
    pub top_k: u32,
    pub moe_intermediate_size: Dim,
    /// Global shared-expert intermediate size (sharded by TP).
    pub shared_intermediate_size: Dim,
    pub gpu_name: String,
    pub mhc_backends: Vec<&'static str>,
    pub router_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
    pub moe_backends: Vec<&'static str>,
    pub all_reduce_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
    pub moe_input_dtype: DType,
    pub weight_format: DType,
    pub group_size: u32,
    pub routing_method: String,
    pub n_group: u32,
    pub topk_group: u32,
    pub routed_scaling_numerator: u32,
    pub routed_scaling_denominator: u32,
    pub expert_demand: ExpertDemand,
    /// 0 = the busiest EP rank.
    pub folded_rank_position: u32,
}

#[derive(Clone, Debug)]
pub struct DeepseekV41MoeFfnEpWorkletResolved {
    pub raw_cfg: DeepseekV41MoeFfnEpWorkletConfig,
    pub entry: MhcRmsNormKernelConfig,
    pub router_gate: GemmFp32OutputKernelConfig,
    pub topk: ElementwiseKernelConfig,
    pub moe_input_quant: ElementwiseKernelConfig,
    pub fused_moe: Nvfp4FusedMoeKernelConfig,
    pub shared_gate_up: SingleGemmKernelConfig,
    pub shared_act: ElementwiseKernelConfig,
    pub shared_down: SingleGemmKernelConfig,
    pub residual_add: ElementwiseKernelConfig,
    pub all_reduce: Option<AllReduceFusionKernelConfig>,
}

#[derive(Clone, Debug, Default)]
pub struct DeepseekV41MoeFfnEpWorkletInput {
    pub num_tokens: u32,
}

pub struct DeepseekV41MoeFfnEpWorklet {
    pub name: String,
    pub entry: Op<MhcFusedPostPreRmsNormKernel>,
    pub router_gate: Op<GemmFp32OutputKernel>,
    pub topk: Op<ElementwiseKernel>,
    pub moe_input_quant: Op<ElementwiseKernel>,
    pub fused_moe: Op<Nvfp4FusedMoeKernel>,
    pub shared_gate_up: Op<SingleGemmKernel>,
    pub shared_act: Op<ElementwiseKernel>,
    pub shared_down: Op<SingleGemmKernel>,
    pub residual_add: Op<ElementwiseKernel>,
    pub all_reduce: Option<Op<AllReduceFusionKernel>>,
    resolved: DeepseekV41MoeFfnEpWorkletResolved,
}

impl DeepseekV41MoeFfnEpWorklet {
    pub fn resolve_config(cfg: &DeepseekV41MoeFfnEpWorkletConfig) -> DeepseekV41MoeFfnEpWorkletResolved {
        assert!(cfg.tp_size > 0 && cfg.ep_size > 0, "parallel sizes must be positive");
        assert_eq!(
            cfg.num_experts.get() % cfg.ep_size,
            0,
            "experts must divide ep_size"
        );
        assert_eq!(
            cfg.shared_intermediate_size.get() % cfg.tp_size,
            0,
            "shared intermediate must divide tp_size"
        );
        assert!(
            cfg.folded_rank_position < cfg.ep_size,
            "folded_rank_position must index an EP rank"
        );
        assert_eq!(
            cfg.expert_demand.num_experts(),
            cfg.num_experts.get() as usize,
            "demand source width must match num_experts"
        );
        let gpu = cfg.gpu_name.as_str();
        let ew = |input: u32, output: u32| placeholder(&cfg.elementwise_backends, gpu, input, output);
        let hidden = cfg.hidden_size.get();
        let shared_local = cfg.shared_intermediate_size.get() / cfg.tp_size;
        let mxfp8 = |n: Dim, k: Dim| SingleGemmKernelConfig {
            backends: cfg.gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n,
            k,
            dtype: DType::Mxfp8E4m3,
        };
        DeepseekV41MoeFfnEpWorkletResolved {
            entry: MhcRmsNormKernelConfig {
                backends: cfg.mhc_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden_size: cfg.hidden_size.clone(),
                hc_mult: cfg.hc_mult,
                hidden_dtype: DType::Bf16,
            },
            router_gate: GemmFp32OutputKernelConfig {
                backends: cfg.router_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.num_experts.clone(),
                k: cfg.hidden_size.clone(),
                input_dtype: DType::Bf16,
            },
            // layers/fused_moe/dsv4_topk.py:136 `_dsv4_topk_kernel`: fp32
            // logits in, top-k fp32 weights + int32 ids out.
            topk: ew(cfg.num_experts.get() * 4, cfg.top_k * 8),
            // experts/trtllm_mxfp4_moe.py:136 MXFP8 activation quant: bf16 in,
            // e4m3 data + one ue8m0 scale per 32 out.
            moe_input_quant: ew(hidden * 2, hidden + hidden / 32),
            // experts/trtllm_mxfp4_moe.py:361 routing + fc1 + fc2 + finalize.
            fused_moe: Nvfp4FusedMoeKernelConfig {
                backends: cfg.moe_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden_size: cfg.hidden_size.clone(),
                intermediate_size: cfg.moe_intermediate_size.clone(),
                num_experts: cfg.num_experts.clone(),
                num_local_experts: cfg.num_experts.clone() / cfg.ep_size,
                top_k: cfg.top_k,
                input_dtype: cfg.moe_input_dtype,
                weight_format: cfg.weight_format,
                group_size: cfg.group_size,
                routing_method: cfg.routing_method.clone(),
                n_group: cfg.n_group,
                topk_group: cfg.topk_group,
                routed_scaling_numerator: cfg.routed_scaling_numerator,
                routed_scaling_denominator: cfg.routed_scaling_denominator,
                expert_demand: cfg.expert_demand.clone(),
                folded_rank_position: cfg.folded_rank_position,
            },
            shared_gate_up: mxfp8((2 * shared_local).into(), cfg.hidden_size.clone()),
            // models/deepseek_v4/nvidia/model.py:187 `act_and_mul` (clamped
            // SwiGLU): gate|up bf16 in, bf16 out.
            shared_act: ew(2 * shared_local * 2, shared_local * 2),
            shared_down: mxfp8(cfg.hidden_size.clone(), shared_local.into()),
            // moe_runner.py:786 routed + shared output add.
            residual_add: ew(hidden * 4, hidden * 2),
            all_reduce: (cfg.tp_size > 1).then(|| AllReduceFusionKernelConfig {
                backends: cfg.all_reduce_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_gpus: cfg.tp_size,
                hidden_dim: hidden,
                dtype: DType::Bf16,
                fabric: Fabric::Nvlink,
            }),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: DeepseekV41MoeFfnEpWorkletResolved,
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
            entry: op!(MhcFusedPostPreRmsNormKernel, r.entry, "entry.mega_mhc_post_pre"),
            router_gate: op!(GemmFp32OutputKernel, r.router_gate, "routed.router_gate"),
            topk: op!(ElementwiseKernel, r.topk, "routed.topk"),
            moe_input_quant: op!(ElementwiseKernel, r.moe_input_quant, "routed.input_quant"),
            fused_moe: op!(Nvfp4FusedMoeKernel, r.fused_moe, "routed.fused_moe"),
            shared_gate_up: op!(SingleGemmKernel, r.shared_gate_up, "shared.gate_up"),
            shared_act: op!(ElementwiseKernel, r.shared_act, "shared.act_and_mul"),
            shared_down: op!(SingleGemmKernel, r.shared_down, "shared.down"),
            residual_add: op!(ElementwiseKernel, r.residual_add, "residual_add"),
            all_reduce: match r.all_reduce {
                Some(config) => Some(op!(AllReduceFusionKernel, config, "tp_all_reduce")),
                None => None,
            },
            name,
            resolved,
        })
    }

    pub fn compile(&self, b: &mut CostTreeBuilder) -> CostNode {
        let entry = self.entry.compile(b);
        let routed = vec![
            self.router_gate.compile(b),
            self.topk.compile(b),
            self.moe_input_quant.compile(b),
            self.fused_moe.compile(b),
        ];
        let shared = CostNode::Sum(vec![
            self.shared_gate_up.compile(b),
            self.shared_act.compile(b),
            self.shared_down.compile(b),
        ]);
        let shared_serial = vec![
            compile_serial_copy(&self.shared_gate_up, b),
            compile_serial_copy(&self.shared_act, b),
            compile_serial_copy(&self.shared_down, b),
        ];
        let mut children = vec![
            entry,
            gated_fanout(routed, vec![shared], shared_serial),
            self.residual_add.compile(b),
        ];
        if let Some(all_reduce) = &self.all_reduce {
            children.push(all_reduce.compile(b));
        }
        CostNode::Labeled {
            label: format!(
                "{} (DeepseekV41MoeFfnEpWorklet) [tp={}; ep={}; local_experts={}; rank_position={}]",
                self.name,
                self.resolved.raw_cfg.tp_size,
                self.resolved.raw_cfg.ep_size,
                self.resolved.fused_moe.num_local_experts.get(),
                self.resolved.raw_cfg.folded_rank_position,
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    pub fn eval(&self, input: &DeepseekV41MoeFfnEpWorkletInput, ev: &mut Evaluator) {
        let rows = input.num_tokens;
        let zero = rows == 0;
        let overlapped = shared_on_aux_stream(rows, self.resolved.raw_cfg.serialize_streams);
        let ew = ElementwiseKernelInput { num_tokens: rows };
        let gemm = SingleGemmKernelInput { m: rows };
        eval_or_zero(&self.entry, MhcRmsNormKernelInput { num_tokens: rows }, zero, ev);
        eval_or_zero(&self.router_gate, GemmFp32OutputKernelInput { m: rows }, zero, ev);
        eval_or_zero(&self.topk, ew.clone(), zero, ev);
        eval_or_zero(&self.moe_input_quant, ew.clone(), zero, ev);
        eval_or_zero(&self.fused_moe, Nvfp4FusedMoeKernelInput { num_tokens: rows }, zero, ev);
        for copy_overlapped in [true, false] {
            let off = zero || overlapped != copy_overlapped;
            eval_or_zero(&self.shared_gate_up, gemm.clone(), off, ev);
            eval_or_zero(&self.shared_act, ew.clone(), off, ev);
            eval_or_zero(&self.shared_down, gemm.clone(), off, ev);
        }
        eval_or_zero(&self.residual_add, ew, zero, ev);
        if let Some(all_reduce) = &self.all_reduce {
            eval_or_zero(all_reduce, AllReduceFusionKernelInput { num_tokens: rows }, zero, ev);
        }
    }
}

/// Whether the shared expert runs beside the routed experts on this call.
pub fn shared_on_aux_stream(num_tokens: u32, serialize_streams: bool) -> bool {
    !serialize_streams && num_tokens > 0 && num_tokens <= SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn config() -> DeepseekV41MoeFfnEpWorkletConfig {
        DeepseekV41MoeFfnEpWorkletConfig {
            tp_size: 4,
            ep_size: 4,
            serialize_streams: false,
            hidden_size: 5120.into(),
            hc_mult: 4,
            num_experts: 384.into(),
            top_k: 6,
            moe_intermediate_size: 2304.into(),
            shared_intermediate_size: 2304.into(),
            gpu_name: "NVIDIA B200".into(),
            mhc_backends: vec!["deepgemm_mega"],
            router_backends: vec!["torch_cublas_vllm_fork"],
            gemm_backends: vec!["flashinfer_mxfp8"],
            moe_backends: vec!["flashinfer_trtllm_sm100_mxfp4"],
            all_reduce_backends: vec!["flashinfer_mnnvl"],
            elementwise_backends: vec!["triton"],
            moe_input_dtype: DType::Bf16,
            weight_format: DType::Mxfp4E2m1,
            group_size: 32,
            routing_method: "precomputed_dsv4".into(),
            n_group: 1,
            topk_group: 1,
            routed_scaling_numerator: 1,
            routed_scaling_denominator: 1,
            expert_demand: ExpertDemand::Popularity {
                layerwise_global_ppm: vec![vec![1_000_000 / 384; 384]],
            },
            folded_rank_position: 0,
        }
    }

    #[test]
    fn resolves_the_captured_shapes() {
        let r = DeepseekV41MoeFfnEpWorklet::resolve_config(&config());
        assert_eq!(r.fused_moe.num_local_experts.get(), 96);
        assert_eq!((r.shared_gate_up.k.get(), r.shared_gate_up.n.get()), (5120, 1152));
        assert_eq!((r.shared_down.k.get(), r.shared_down.n.get()), (576, 5120));
        assert_eq!(r.router_gate.n.get(), 384);
        assert_eq!(r.moe_input_quant.output_bytes_per_token.get(), 5280);
    }

    #[test]
    fn shared_expert_overlaps_only_up_to_256_tokens() {
        assert!(shared_on_aux_stream(48, false));
        assert!(shared_on_aux_stream(256, false));
        assert!(!shared_on_aux_stream(257, false));
        assert!(!shared_on_aux_stream(48, true));
        assert!(!shared_on_aux_stream(0, false));
    }
}
