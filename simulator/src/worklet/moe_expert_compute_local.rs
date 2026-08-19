//! `MoeExpertComputeLocalWorklet` — per-EP-rank `MoE` expert compute (the work
//! between `MoE` dispatch and `MoE` combine): optional FP8 input quant →
//! grouped-up_gate → `SwiGLU` activation → optional FP8 input quant →
//! grouped-down. `Local` group suffix (L3 §1.5): one sync section on ONE GPU,
//! no collective inside. The arch wraps `ep_size` independent instances of this
//! worklet under a `Max{1.0}` so the slowest EP rank's expert compute is the
//! cell's wallclock (L4 §3.3 fan-out for per-GPU imbalance).
//!
//! `local_ppm` is this rank's SHARD of the global routing distribution (one
//! ppm value per local expert; `len() == num_experts / ep_size`). The grouped
//! GEMM kernel is **distribution-sensitive** (its cache identity bakes
//! `local_ppm` in), so different ranks with different shards become different
//! cached kernels — this is what makes the `Max` non-trivial under skewed
//! routing. For the uniform shape every rank's shard is identical and the Max
//! degenerates, but the same worklet contract handles skewed rank shards.
//!
//! Shape per leaf (`moe_intermediate = m_inter`, full-hidden = `h`):
//!   - `gate_up` : `(2·m_inter) × h` GEMM, one row per routed (token,expert) pair
//!     in this rank's experts; the runtime input is `global_expert_selections =
//!     num_tokens × top_k` and the kernel splits it across the local experts by
//!     `local_ppm`.
//!   - activation : `SwiGLU` elementwise on this rank's routed-pair count,
//!     apportioned from `global_expert_selections` by this rank's `local_ppm`
//!     (on `2·m_inter`-wide partials).
//!   - `gate_up_input_quant` / `down_input_quant`: BF16 → FP8 E4M3 1x128
//!     block quantization over the same final per-rank routed rows. These leaves
//!     exist only when the grouped GEMMs consume FP8.
//!   - down : `h × m_inter` GEMM, same global routed-pair input.

use std::sync::Arc;

use crate::op::moe::{
    GroupedFp8GemmWithQuantConfig, GroupedFp8GemmWithQuantInput, GroupedFp8GemmWithQuantOp,
    GroupedGemmConfig, GroupedQuantConfig, QuantRows,
};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, Fp8BlockQuantKernelConfig,
    Fp8BlockscaleGroupedGemmKernelConfig, GroupedGemmKernel, GroupedGemmKernelConfig,
    GroupedGemmKernelInput,
};
use crate::timing::routing::RoutingDistribution;
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

/// Raw config + this rank's `local_ppm` shard. Partition is derivable
/// (`experts_per_gpu = num_experts / ep_size`).
#[derive(Clone, Debug)]
pub struct MoeExpertComputeLocalWorkletConfig {
    pub hidden: Dim,
    pub moe_intermediate: Dim,
    pub num_experts: Dim,
    pub ep_size: u16,
    pub top_k: u32,
    /// Dtype of the grouped gate/up/down GEMMs.
    pub dtype: DType,
    /// Dtype of the `SwiGLU` activation traffic between the GEMMs.
    pub activation_dtype: DType,
    pub gpu_name: String,
    pub act_backends: Vec<&'static str>,
    pub fp8_quant_backends: Vec<&'static str>,
    pub grouped_gemm_backends: Vec<&'static str>,
    pub fp8_grouped_gemm_backends: Vec<&'static str>,
    /// Select the production FlashInfer/TensorRT-LLM block-scale recipe for
    /// FP8 expert GEMMs. When false, FP8 follows the original generic
    /// `grouped_gemm` backend contract (currently `DeepGEMM`).
    pub use_fp8_blockscale_grouped_gemm: bool,
    /// This rank's slice of the global routing distribution — one ppm value per
    /// local expert. `Σ(local_ppm) < TOTAL_PPM` (it is a shard). Uniform callers
    /// still produce equal shards; profile-backed callers preserve the measured
    /// raw values so grouped-GEMM cache identity remains rank-specific.
    pub local_ppm: Vec<u32>,
}

impl MoeExpertComputeLocalWorkletConfig {
    /// Materialize one local config per EP rank from a complete global ppm
    /// snapshot. The partition is contiguous and balanced (`E % ep == 0`), and
    /// the raw shard is deliberately not renormalized: its sum is that rank's
    /// absolute share of global routing mass and therefore part of the grouped
    /// GEMM cache identity.
    #[must_use]
    pub fn split_for_ep(
        mut template: Self,
        global_ppm: &[u32],
    ) -> Vec<MoeExpertComputeLocalWorkletConfig> {
        let ep = usize::from(template.ep_size);
        assert!(ep > 0, "ep_size must be non-zero");
        let num_experts = template.num_experts.get() as usize;
        assert_eq!(
            global_ppm.len(),
            num_experts,
            "global ppm len ({}) must equal num_experts ({})",
            global_ppm.len(),
            num_experts
        );
        assert_eq!(
            num_experts % ep,
            0,
            "num_experts {num_experts} must be divisible by ep_size {ep}"
        );
        let experts_per_rank = num_experts / ep;
        template.local_ppm.clear();
        (0..ep)
            .map(|rank| {
                let mut rank_config = template.clone();
                let start = rank * experts_per_rank;
                rank_config.local_ppm = global_ppm[start..start + experts_per_rank].to_vec();
                rank_config
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct MoeExpertComputeLocalWorkletResolved {
    pub raw_cfg: MoeExpertComputeLocalWorkletConfig,
    pub gate_up: Option<GroupedGemmKernelConfig>,
    pub gate_up_fp8: Option<GroupedFp8GemmWithQuantConfig>,
    pub act: ElementwiseKernelConfig,
    pub down: Option<GroupedGemmKernelConfig>,
    pub down_fp8: Option<GroupedFp8GemmWithQuantConfig>,
    /// Per-rank expert count, symbolic: `num_experts / ep`. Folds to the plain
    /// count at `.get()`; the grouped GEMM's real expert axis is `local_ppm`.
    pub experts_per_gpu: Dim,
    pub dtype_bytes: u32,
}

/// Per-call shape: the GLOBAL routed-pair count for the iteration (`num_tokens
/// × top_k`). Grouped GEMMs consume it directly; the activation consumes its
/// per-rank share, derived inside `eval` from the worklet's `local_ppm` shard.
#[derive(Clone, Debug, Default)]
pub struct MoeExpertComputeLocalWorkletInput {
    pub global_expert_selections: u32,
}

pub struct MoeExpertComputeLocalWorklet {
    pub name: String,
    pub gate_up: Option<Op<GroupedGemmKernel>>,
    pub gate_up_fp8: Option<GroupedFp8GemmWithQuantOp>,
    pub act: Op<ElementwiseKernel>,
    pub down: Option<Op<GroupedGemmKernel>>,
    pub down_fp8: Option<GroupedFp8GemmWithQuantOp>,
    resolved: MoeExpertComputeLocalWorkletResolved,
}

impl MoeExpertComputeLocalWorklet {
    #[must_use]
    pub fn resolve_config(
        cfg: &MoeExpertComputeLocalWorkletConfig,
    ) -> MoeExpertComputeLocalWorkletResolved {
        let ep = u32::from(cfg.ep_size);
        assert!(ep > 0, "ep_size must be non-zero");
        assert!(
            cfg.num_experts.get().is_multiple_of(ep),
            "num_experts {} not divisible by ep_size {}",
            cfg.num_experts,
            ep,
        );
        // A per-rank expert count — symbolic `num_experts / ep` so the derivation
        // survives (folds to the plain count at `.get()`). The grouped GEMM's real
        // expert axis is `local_ppm`, not this count.
        let experts_per_gpu = cfg.num_experts.clone() / Dim::param("ep", ep);
        assert_eq!(
            cfg.local_ppm.len() as u32,
            experts_per_gpu.get(),
            "local_ppm len ({}) must equal experts_per_gpu ({})",
            cfg.local_ppm.len(),
            experts_per_gpu.get(),
        );
        let activation_dtype_bytes = cfg.activation_dtype.size_bytes();
        let bytes = Dim::param("activation_bytes", activation_dtype_bytes);
        let uses_fp8_grouped_gemm =
            cfg.dtype == DType::Fp8E4m3 && cfg.use_fp8_blockscale_grouped_gemm;
        if uses_fp8_grouped_gemm {
            assert_eq!(
                cfg.activation_dtype,
                DType::Bf16,
                "fp8_block_quant currently supports BF16 activation input only"
            );
            assert!(
                !cfg.fp8_quant_backends.is_empty(),
                "FP8 grouped GEMM requires at least one fp8 quant backend"
            );
        }
        // Unchanged realization: this path pairs FlashInfer's grouped
        // `scale_1x128_kernel` with the TRT-LLM blockscale grouped GEMM. The
        // vLLM-mirroring worklet beside this one uses the per-token-group
        // kernel instead because that is what nsys observed on its EP1 path;
        // no capture of *this* path exists yet to move it either way.
        let quant_config = |hidden_size: Dim| {
            GroupedQuantConfig::Block(Fp8BlockQuantKernelConfig {
                backends: cfg.fp8_quant_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden_size,
                num_problems: experts_per_gpu.clone(),
                input_dtype: cfg.activation_dtype,
            })
        };
        let gate_up_gemm = GroupedGemmKernelConfig {
            backends: cfg.grouped_gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n: 2 * cfg.moe_intermediate.clone(),
            k: cfg.hidden.clone(),
            dtype: cfg.dtype,
            local_ppm: cfg.local_ppm.clone(),
        };
        let down_gemm = GroupedGemmKernelConfig {
            backends: cfg.grouped_gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n: cfg.hidden.clone(),
            k: cfg.moe_intermediate.clone(),
            dtype: cfg.dtype,
            local_ppm: cfg.local_ppm.clone(),
        };
        let direct_fp8_gemm = |n: Dim, k: Dim| Fp8BlockscaleGroupedGemmKernelConfig {
            backends: cfg.fp8_grouped_gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n,
            k,
            dtype: cfg.dtype,
            experts_per_token: cfg.top_k,
            local_ppm: cfg.local_ppm.clone(),
        };
        MoeExpertComputeLocalWorkletResolved {
            gate_up: (!uses_fp8_grouped_gemm).then(|| gate_up_gemm.clone()),
            gate_up_fp8: uses_fp8_grouped_gemm.then(|| GroupedFp8GemmWithQuantConfig {
                quant: quant_config(cfg.hidden.clone()),
                gemm: GroupedGemmConfig::TrtllmBlockscale(direct_fp8_gemm(
                    2 * cfg.moe_intermediate.clone(),
                    cfg.hidden.clone(),
                )),
                // Unchanged from before this field existed, and deliberately not
                // switched to PerToken with the vLLM worklet: this realization
                // permutes tokens into per-expert order before quantizing
                // (DeepEP into DeepGEMM), so the expanded layout is what the
                // quantize kernel sees. No nsys capture of this path has been
                // taken to confirm it; that is the open item, not a known bug.
                quant_rows: QuantRows::PerSelection,
            }),
            act: ElementwiseKernelConfig {
                // SwiGLU on the gate_up output: reads 2·moe_intermediate, writes
                // moe_intermediate elements per routed (token,expert) pair.
                backends: cfg.act_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: 2 * cfg.moe_intermediate.clone() * bytes.clone(),
                output_bytes_per_token: cfg.moe_intermediate.clone() * bytes.clone(),
            },
            down: (!uses_fp8_grouped_gemm).then(|| down_gemm.clone()),
            down_fp8: uses_fp8_grouped_gemm.then(|| GroupedFp8GemmWithQuantConfig {
                quant: quant_config(cfg.moe_intermediate.clone()),
                gemm: GroupedGemmConfig::TrtllmBlockscale(direct_fp8_gemm(
                    cfg.hidden.clone(),
                    cfg.moe_intermediate.clone(),
                )),
                quant_rows: QuantRows::PerSelection,
            }),
            experts_per_gpu,
            dtype_bytes: activation_dtype_bytes,
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: MoeExpertComputeLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let gu_name = format!("{name}.gate_up");
        let act_name = format!("{name}.activation");
        let dn_name = format!("{name}.down");
        let gate_up = resolved
            .gate_up
            .as_ref()
            .map(|config| {
                GroupedGemmKernel::build(gu_name.clone(), config.clone(), bridge)
                    .map(|kernel| Op::new(gu_name.clone(), Arc::new(kernel)))
            })
            .transpose()?;
        let gate_up_fp8 = resolved
            .gate_up_fp8
            .as_ref()
            .map(|config| GroupedFp8GemmWithQuantOp::build(gu_name, config.clone(), bridge))
            .transpose()?;
        let act = Op::new(
            act_name.clone(),
            Arc::new(ElementwiseKernel::build(
                act_name,
                resolved.act.clone(),
                bridge,
            )?),
        );
        let down = resolved
            .down
            .as_ref()
            .map(|config| {
                GroupedGemmKernel::build(dn_name.clone(), config.clone(), bridge)
                    .map(|kernel| Op::new(dn_name.clone(), Arc::new(kernel)))
            })
            .transpose()?;
        let down_fp8 = resolved
            .down_fp8
            .as_ref()
            .map(|config| GroupedFp8GemmWithQuantOp::build(dn_name, config.clone(), bridge))
            .transpose()?;
        Ok(Self {
            name,
            gate_up,
            gate_up_fp8,
            act,
            down,
            down_fp8,
            resolved,
        })
    }

    /// `CostTree` compile: `Sum([gate_up_quant], gate_up, activation,
    /// [down_quant], down)`; bracketed slots exist only for FP8 GEMMs.
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let r = &self.resolved;
        let label = format!(
            "{} (MoeExpertComputeLocalWorklet) [{:?}, {:?}, experts_per_gpu={:?}]",
            self.name, r.raw_cfg.hidden, r.raw_cfg.moe_intermediate, r.experts_per_gpu,
        );
        let mut children = Vec::with_capacity(5);
        if let Some(gate_up) = &self.gate_up {
            children.push(gate_up.compile(builder));
        }
        if let Some(gate_up_fp8) = &self.gate_up_fp8 {
            children.push(gate_up_fp8.compile(builder));
        }
        children.push(self.act.compile(builder));
        if let Some(down) = &self.down {
            children.push(down.compile(builder));
        }
        if let Some(down_fp8) = &self.down_fp8 {
            children.push(down_fp8.compile(builder));
        }
        CostNode::Labeled {
            label,
            child: Box::new(CostNode::Sum(children)),
        }
    }

    /// `CostTree` eval: fill slots in the same optional-quant order as `compile`.
    /// `global_expert_selections` flows straight into the grouped GEMMs (their
    /// caches split it by `local_ppm`); the activation gets this rank's share
    /// from `local_ppm`, matching the grouped-GEMM per-group apportionment.
    pub fn eval(&self, input: &MoeExpertComputeLocalWorkletInput, ev: &mut Evaluator) {
        let global = input.global_expert_selections;
        let routed_per_rank: u32 =
            RoutingDistribution::to_per_expert_counts(global, &self.resolved.raw_cfg.local_ppm)
                .into_iter()
                .sum();
        if let Some(gate_up) = &self.gate_up {
            gate_up.eval(
                &GroupedGemmKernelInput {
                    global_expert_selections: global,
                },
                ev,
            );
        }
        if let Some(gate_up_fp8) = &self.gate_up_fp8 {
            gate_up_fp8.eval(
                &GroupedFp8GemmWithQuantInput {
                    global_expert_selections: global,
                },
                ev,
            );
        }
        self.act.eval(
            &ElementwiseKernelInput {
                num_tokens: routed_per_rank,
            },
            ev,
        );
        if let Some(down) = &self.down {
            down.eval(
                &GroupedGemmKernelInput {
                    global_expert_selections: global,
                },
                ev,
            );
        }
        if let Some(down_fp8) = &self.down_fp8 {
            down_fp8.eval(
                &GroupedFp8GemmWithQuantInput {
                    global_expert_selections: global,
                },
                ev,
            );
        }
    }
}

/// Build a v1 uniform `local_ppm` shard: each of the `experts_per_gpu` experts
/// on this rank receives `TOTAL_PPM / num_experts` ppm (the global uniform per-
/// expert weight). Bake-time helper used by the L4 arch's `build_configs`; not
/// part of the worklet contract, but co-located so the v1 shape stays in one
/// file.
#[must_use]
pub fn uniform_local_ppm(num_experts: u32, ep_size: u16) -> Vec<u32> {
    assert!(ep_size > 0, "ep_size must be non-zero");
    let ep = u32::from(ep_size);
    assert!(
        num_experts.is_multiple_of(ep),
        "num_experts {num_experts} not divisible by ep_size {ep}"
    );
    let experts_per_gpu = (num_experts / ep) as usize;
    let per_expert = RoutingDistribution::TOTAL_PPM / num_experts;
    vec![per_expert; experts_per_gpu]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(ep_size: u16) -> MoeExpertComputeLocalWorkletConfig {
        let num_experts = 128u32;
        MoeExpertComputeLocalWorkletConfig {
            hidden: 4096.into(),
            moe_intermediate: 3072.into(),
            num_experts: num_experts.into(),
            ep_size,
            top_k: 8,
            dtype: DType::Bf16,
            activation_dtype: DType::Bf16,
            gpu_name: "H100".to_string(),
            act_backends: vec!["triton"],
            fp8_quant_backends: vec!["flashinfer_trtllm"],
            grouped_gemm_backends: vec!["deepgemm"],
            fp8_grouped_gemm_backends: vec!["flashinfer_trtllm"],
            use_fp8_blockscale_grouped_gemm: true,
            local_ppm: uniform_local_ppm(num_experts, ep_size),
        }
    }

    #[test]
    fn uniform_local_ppm_distributes_evenly() {
        let s = uniform_local_ppm(128, 8);
        // Each expert gets TOTAL_PPM / num_experts (rounded toward zero).
        assert_eq!(s.len(), 16); // 128 / 8
        assert_eq!(s[0], 1_000_000 / 128);
        assert!(s.iter().all(|&v| v == s[0]));
    }

    #[test]
    fn split_for_ep_derives_contiguous_rank_shards_from_full_ppm() {
        let template = cfg(4);
        let global_ppm = (1..=128).collect::<Vec<u32>>();
        let rank_configs = MoeExpertComputeLocalWorkletConfig::split_for_ep(template, &global_ppm);
        assert_eq!(rank_configs.len(), 4);
        assert_eq!(rank_configs[0].local_ppm, (1..=32).collect::<Vec<_>>());
        assert_eq!(rank_configs[1].local_ppm, (33..=64).collect::<Vec<_>>());
        assert_eq!(rank_configs[3].local_ppm, (97..=128).collect::<Vec<_>>());
    }

    #[test]
    fn resolve_shards_experts_and_threads_local_ppm_into_grouped_gemms() {
        let r = MoeExpertComputeLocalWorklet::resolve_config(&cfg(8));
        assert_eq!(r.experts_per_gpu, 16); // 128 / 8
        let gate_up = r.gate_up.as_ref().unwrap();
        let down = r.down.as_ref().unwrap();
        assert_eq!(gate_up.n, 2 * 3072); // fused gate||up output
        assert_eq!(gate_up.k, 4096); // hidden
        assert_eq!(down.n, 4096); // hidden
        assert_eq!(down.k, 3072); // moe_intermediate
        assert_eq!(gate_up.local_ppm.len(), 16);
        assert!(r.gate_up_fp8.is_none());
        assert!(r.down_fp8.is_none());
        // bf16 = 2 bytes/elem.
        assert_eq!(r.act.input_bytes_per_token, 2 * 3072 * 2);
        assert_eq!(r.act.output_bytes_per_token, 3072 * 2);
    }

    #[test]
    fn fp8_resolve_adds_two_bf16_input_quant_slots() {
        let mut config = cfg(4);
        config.dtype = DType::Fp8E4m3;
        let resolved = MoeExpertComputeLocalWorklet::resolve_config(&config);

        // This worklet keeps the grouped `scale_1x128_kernel` realization; the
        // assertion is on the variant as much as on the fields, so a silent
        // switch to the per-token-group kernel fails here rather than in a
        // downstream timing diff.
        let gate_up = resolved.gate_up_fp8.unwrap();
        let GroupedQuantConfig::Block(gate_up_quant) = &gate_up.quant else {
            panic!("this realization must use the grouped block quant");
        };
        assert_eq!(gate_up_quant.hidden_size, 4096);
        assert_eq!(gate_up_quant.num_problems, 32);
        assert_eq!(gate_up_quant.input_dtype, DType::Bf16);
        assert_eq!(gate_up_quant.backends, vec!["flashinfer_trtllm"]);
        let GroupedGemmConfig::TrtllmBlockscale(gate_up_gemm) = &gate_up.gemm else {
            panic!("this realization must use the TRT-LLM grouped GEMM");
        };
        assert_eq!(gate_up_gemm.n, 2 * 3072);
        assert_eq!(gate_up_gemm.experts_per_token, 8);
        assert_eq!(gate_up_gemm.backends, vec!["flashinfer_trtllm"]);

        let down = resolved.down_fp8.unwrap();
        let GroupedQuantConfig::Block(down_quant) = &down.quant else {
            panic!("this realization must use the grouped block quant");
        };
        assert_eq!(down_quant.hidden_size, 3072);
        assert_eq!(down_quant.num_problems, 32);
        assert_eq!(down_quant.input_dtype, DType::Bf16);
        let GroupedGemmConfig::TrtllmBlockscale(down_gemm) = &down.gemm else {
            panic!("this realization must use the TRT-LLM grouped GEMM");
        };
        assert_eq!(down_gemm.n, 4096);
    }

    #[test]
    #[should_panic(expected = "num_experts")]
    fn ep_indivisible_num_experts_panics() {
        let mut c = cfg(8);
        c.ep_size = 9; // 128 % 9 != 0
        c.local_ppm = vec![0; 1]; // ignored — panic fires first
        let _ = MoeExpertComputeLocalWorklet::resolve_config(&c);
    }

    #[test]
    #[should_panic(expected = "local_ppm len")]
    fn local_ppm_length_mismatch_panics() {
        let mut c = cfg(8);
        c.local_ppm = vec![0; 4]; // expected 16 for ep=8, num_experts=128
        let _ = MoeExpertComputeLocalWorklet::resolve_config(&c);
    }
}
