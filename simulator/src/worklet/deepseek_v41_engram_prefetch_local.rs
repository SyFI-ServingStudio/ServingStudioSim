//! DeepSeek-V4.1-Flash Engram lookups (layers 1 and 14) on one rank.
//!
//! Both layers' `_engram_lookup_kernel` launches (fork `common/engram.py`,
//! UVA row gather from the host-resident table) are issued right after the
//! iteration's `_hash_ids_kernel` on their own streams and consumed much later
//! by each Engram block's AllGather. Capture 2 (job 1185, device 0):
//!
//! - decode 1337 (48 rows): hash 218.05-223.62 us; lookups 224.00-245.34 and
//!   224.48-246.30 on two streams, i.e. a concurrent pair spanning ~22 us
//!   where one isolated 48-row lookup measures ~12 us (the pair roughly sums:
//!   both share the PCIe/C2C read path); the main path (layer-0 expand, layer-0
//!   attention and FFN) runs meanwhile on a third stream and reaches the
//!   layer-1 consumer AllGather at 520.99, ~297 us after the hash.
//! - mixed 310 (2048 tokens): lookups 451.17-1140.38 (s51) and
//!   616.93-1281.66 (s55); the main stream reaches the layer-1 AllGather at
//!   2325.12. The lookups finish before the consumer, but they are not free.
//!
//! Capture 3 (`logs/20260924_0_dsv41_flash_capture`, Phase-5 D3) shows why a
//! race (`Max`) under-states the window: the lookups share the device with the
//! concurrent main path and slow it down.
//!
//! - Iteration 160 (T=2048, device 0): the lookups run 2-570 us and 169-780 us
//!   after the hash. Beside them hc_expand takes 307 us, hc_prenorm 123 us,
//!   pre_big_fuse 123 us, qk_rmsnorm 93 us, the wq_b quant 90 us and kv_insert
//!   111 us; after the lookups end the same kernel types run at normal speed.
//! - Across iterations the hash -> layer-1 AllGather window grows ~0.70 ms per
//!   ms of lookup at T~2048 (r=0.66) and ~0.66 ms per ms in (1024,1536]
//!   (r=0.69): the pair is ~70% serial with the main path.
//! - Nothing ever waits on a lookup: the main path reaches the consumer with
//!   0.24-0.70 ms of slack at minimum, so the cost is contention, not a wait.
//!
//! So the composition is serial `Sum[main path from the hash to the layer-1
//! AllGather, lookup_l1, lookup_l14]`: the two lookups contend with each other
//! and with the main path. Same-device contention must not be a `CostNode::Max`
//! (`doc/analyzer.md`, "Concurrent CUDA streams"); `Sum` is the closest
//! existing node until a contention node exists, and it still under-predicts
//! the window in every captured bucket (the in-server pair is longer than the
//! isolated L1 lookup). Layer 14's consumer is further away still, so charging
//! both lookups against the layer-1 window is the conservative choice.
//!
//! Tree order: [`compile_joined`](DeepseekV41EngramPrefetchLocalWorklet::compile_joined)
//! takes the already-compiled main path, so its leaves are minted *after*
//! that path's leaves; the L4 caller must likewise call
//! [`eval`](DeepseekV41EngramPrefetchLocalWorklet::eval) after evaluating the
//! main-path worklets (INV-2).

use std::sync::Arc;

use super::deepseek_v41_common::eval_or_zero;
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    EngramLookupKernel, EngramLookupKernelConfig, EngramLookupKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge};

/// Engram layers whose lookups this worklet issues.
pub const ENGRAM_LOOKUP_LAYERS: [u32; 2] = [1, 14];

#[derive(Clone, Debug)]
pub struct DeepseekV41EngramPrefetchLocalWorkletConfig {
    pub tp_size: u32,
    pub engram_num_heads: u32,
    pub engram_head_dim: u32,
    pub quant_block_size: u32,
    pub table_rows: u64,
    /// `host_uva` (production) or `device`.
    pub residency: String,
    pub weight_dtype: DType,
    pub gpu_name: String,
    pub backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct DeepseekV41EngramPrefetchLocalWorkletResolved {
    pub raw_cfg: DeepseekV41EngramPrefetchLocalWorkletConfig,
    pub lookup: EngramLookupKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct DeepseekV41EngramPrefetchLocalWorkletInput {
    pub num_tokens: u32,
}

pub struct DeepseekV41EngramPrefetchLocalWorklet {
    pub name: String,
    /// One lookup per Engram layer, in [`ENGRAM_LOOKUP_LAYERS`] order.
    pub lookups: Vec<Op<EngramLookupKernel>>,
    resolved: DeepseekV41EngramPrefetchLocalWorkletResolved,
}

impl DeepseekV41EngramPrefetchLocalWorklet {
    pub fn resolve_config(
        cfg: &DeepseekV41EngramPrefetchLocalWorkletConfig,
    ) -> DeepseekV41EngramPrefetchLocalWorkletResolved {
        assert!(cfg.tp_size > 0, "tp_size must be positive");
        assert_eq!(
            cfg.engram_num_heads % cfg.tp_size,
            0,
            "Engram heads must divide tp_size"
        );
        DeepseekV41EngramPrefetchLocalWorkletResolved {
            lookup: EngramLookupKernelConfig {
                backends: cfg.backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                local_heads: cfg.engram_num_heads / cfg.tp_size,
                head_dim: cfg.engram_head_dim,
                quant_block_size: cfg.quant_block_size,
                table_rows: cfg.table_rows,
                residency: cfg.residency.clone(),
                weight_dtype: cfg.weight_dtype,
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: DeepseekV41EngramPrefetchLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let lookups = ENGRAM_LOOKUP_LAYERS
            .iter()
            .map(|layer| {
                let op_name = format!("{name}.layer{layer}.engram_lookup");
                Ok(Op::new(
                    op_name.clone(),
                    Arc::new(EngramLookupKernel::build(
                        op_name,
                        resolved.lookup.clone(),
                        bridge,
                    )?),
                ))
            })
            .collect::<Result<Vec<_>, BuildError>>()?;
        Ok(Self {
            name,
            lookups,
            resolved,
        })
    }

    /// The two lookups alone, contending with each other.
    pub fn compile(&self, b: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: format!(
                "{} (DeepseekV41EngramPrefetchLocalWorklet) [local_heads={}; residency={}]",
                self.name, self.resolved.lookup.local_heads, self.resolved.lookup.residency,
            ),
            child: Box::new(CostNode::Sum(
                self.lookups.iter().map(|op| op.compile(b)).collect(),
            )),
        }
    }

    /// The lookups serialized after `main_path` (hash through the layer-1
    /// consumer): same-device contention, not a race (see the module doc).
    pub fn compile_joined(&self, b: &mut CostTreeBuilder, main_path: CostNode) -> CostNode {
        let lookups = self.compile(b);
        CostNode::Sum(vec![main_path, lookups])
    }

    pub fn eval(&self, input: &DeepseekV41EngramPrefetchLocalWorkletInput, ev: &mut Evaluator) {
        for op in &self.lookups {
            eval_or_zero(
                op,
                EngramLookupKernelInput {
                    num_tokens: input.num_tokens,
                },
                input.num_tokens == 0,
                ev,
            );
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn config() -> DeepseekV41EngramPrefetchLocalWorkletConfig {
        DeepseekV41EngramPrefetchLocalWorkletConfig {
            tp_size: 4,
            engram_num_heads: 24,
            engram_head_dim: 256,
            quant_block_size: 32,
            table_rows: 96_000_564,
            residency: "host_uva".into(),
            weight_dtype: DType::Fp8E4m3,
            gpu_name: "NVIDIA B200".into(),
            backends: vec!["vllm_triton"],
        }
    }

    #[test]
    fn resolves_the_profiled_lookup_identity() {
        let r = DeepseekV41EngramPrefetchLocalWorklet::resolve_config(&config());
        assert_eq!(
            (r.lookup.local_heads, r.lookup.head_dim, r.lookup.quant_block_size),
            (6, 256, 32)
        );
        assert_eq!(r.lookup.table_rows, 96_000_564);
    }

    #[test]
    fn joined_lookups_are_serial_after_the_main_path() {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let w = DeepseekV41EngramPrefetchLocalWorklet::build(
            "model.engram_prefetch".into(),
            DeepseekV41EngramPrefetchLocalWorklet::resolve_config(&config()),
            &bridge,
        )
        .unwrap();
        let mut b = CostTreeBuilder::new();
        let main_path = CostNode::Sum(vec![]);
        let CostNode::Sum(children) = w.compile_joined(&mut b, main_path) else {
            panic!("the lookups must be serialized with the main path (Sum), not raced (Max)");
        };
        assert_eq!(children.len(), 2);
        assert!(matches!(children[0], CostNode::Sum(ref c) if c.is_empty()));
        let CostNode::Labeled { child, .. } = &children[1] else {
            panic!("second child is the labeled lookup pair");
        };
        assert!(matches!(**child, CostNode::Sum(ref leaves) if leaves.len() == 2));
    }
}
