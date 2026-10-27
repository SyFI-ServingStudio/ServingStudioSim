//! Collectives of a Trainium2 tensor-parallel group over NeuronLink: one cached
//! curve per `(op, num_ranks, fabric)` config, swept over the message size.
//!
//! The stock vLLM Neuron Llama forward closes its blocks with three collectives
//! (`vllm_neuron/model/llama3/model.py`, 0.29): prefill runs sequence-parallel
//! and brackets each block with `all_gather(dim=0)` before the projections and
//! `reduce_scatter(dim=0)` after them; decode closes each block with an
//! `all_reduce`. One kind with an `op` field keeps the three on one runner and
//! one measurement method (`profiling/runners/neuron/collective_bench.py`:
//! `torch.distributed` on the lite runtime's `neuron` backend, one rank per LNC2
//! logical core).
//!
//! Like `all_reduce`, the cost is size-keyed, not dtype-keyed: the curve is
//! `time_ms` over `message_size_bytes`, measured once at BF16. The message is
//! the FULL buffer of the operation: the whole tensor every rank holds after an
//! all-reduce, the gathered output of an all-gather, the un-scattered input of
//! a reduce-scatter. A TP block's `[tokens x hidden]` activation is therefore
//! `tokens x hidden x 2` bytes for every op. The sweep is a `pow2(10, 28)` byte
//! ladder (1 KB .. 256 MB: a decode step at bucket 1 moves 8 KB, a C8192
//! prefill 64 MB); `Cache1DLinear` interpolates between adjacent powers of two.

use crate::common::Fabric;
use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

/// The backend the Neuron collectives are measured with: `torch.distributed`
/// in the vLLM Neuron 0.29 image (Neuron SDK 2.33), one process per LNC2 logical core.
pub const NEURON_COLLECTIVE_BACKEND: &str = "vllm_neuron_dist";

/// The three collectives the stock forward issues; serialized as the `op`
/// row field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NeuronCollectiveOp {
    AllReduce,
    AllGather,
    ReduceScatter,
}

impl NeuronCollectiveOp {
    /// The serde wire token, for the `op` field of a row.
    pub fn as_str(self) -> &'static str {
        match self {
            NeuronCollectiveOp::AllReduce => "all_reduce",
            NeuronCollectiveOp::AllGather => "all_gather",
            NeuronCollectiveOp::ReduceScatter => "reduce_scatter",
        }
    }
}

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct NeuronCollectiveKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub op: NeuronCollectiveOp,
    /// Ranks of the group: the TP degree, one LNC2 logical core each.
    pub num_ranks: u32,
    pub fabric: Fabric,
}

impl NeuronCollectiveKernelConfig {
    /// One collective of a `num_ranks`-wide tensor-parallel group.
    pub fn tp_group(
        gpu_name: &str,
        backends: &[&'static str],
        fabric: Fabric,
        num_ranks: u16,
        op: NeuronCollectiveOp,
    ) -> Self {
        Self {
            backends: backends.to_vec(),
            gpu_name: gpu_name.to_string(),
            op,
            num_ranks: u32::from(num_ranks),
            fabric,
        }
    }
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct NeuronCollectiveKernelInput {
    /// The full buffer of the operation (see the module doc), never a per-rank
    /// shard.
    pub message_size_bytes: u64,
}

/// The kind's registry entry (`KIND`, grid, cache, row fields).
pub struct NeuronCollectiveSpec;

impl KernelSpec for NeuronCollectiveSpec {
    type Config = NeuronCollectiveKernelConfig;
    type Input = NeuronCollectiveKernelInput;

    const KIND: KernelKind = "neuron_collective";

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        anyhow::ensure!(
            config.backends == [NEURON_COLLECTIVE_BACKEND],
            "Neuron collectives require the {NEURON_COLLECTIVE_BACKEND} backend"
        );
        anyhow::ensure!(
            config.fabric == Fabric::NeuronLink,
            "Neuron collectives run over the neuron_link fabric"
        );
        anyhow::ensure!(
            config.num_ranks >= 2,
            "a collective needs at least two ranks"
        );
        Ok(())
    }

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // 1 KB .. 256 MB byte ladder (2^10 .. 2^28).
        SweepGrid::new(vec![Axis::pow2(10, 28)])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|message_size| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("op", config.op.as_str())
                .with("num_ranks", config.num_ranks)
                .with("message_size_bytes", message_size as u64)
                // Fixed profiling dtype: comm is size-keyed (see module doc).
                .with("dtype", DType::Bf16.as_str())
                .with("fabric", config.fabric.as_str())
        })
    }
}

register_kernel!(NeuronCollectiveKernel, NeuronCollectiveSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::SweepCoords;
    use serde_json::Value;

    fn cfg(op: NeuronCollectiveOp) -> NeuronCollectiveKernelConfig {
        NeuronCollectiveKernelConfig {
            backends: vec![NEURON_COLLECTIVE_BACKEND],
            gpu_name: "AWS Trainium2 LNC2".to_string(),
            op,
            num_ranks: 4,
            fabric: Fabric::NeuronLink,
        }
    }

    #[test]
    fn rows_carry_the_python_fields_over_the_byte_ladder() {
        let c = cfg(NeuronCollectiveOp::ReduceScatter);
        NeuronCollectiveSpec::validate_config(&c).unwrap();
        let grid = NeuronCollectiveSpec::sweep_grid(&c);
        assert_eq!(grid.axes()[0].first().copied(), Some(1024.0));
        assert_eq!(grid.axes()[0].last().copied(), Some((1u64 << 28) as f64));
        let rows = NeuronCollectiveSpec::enumerate(&c, &grid, NEURON_COLLECTIVE_BACKEND);
        assert_eq!(rows.len(), 19);
        let fields = rows[0].fields();
        // Wire schema: { backend, op, num_ranks, message_size_bytes, dtype, fabric },
        // aligned with NeuronCollectiveArgs in profiling/kernels/neuron_collective.py.
        assert_eq!(fields.len(), 6);
        assert_eq!(fields.get("op"), Some(&Value::from("reduce_scatter")));
        assert_eq!(fields.get("num_ranks"), Some(&Value::from(4_u32)));
        assert_eq!(fields.get("fabric"), Some(&Value::from("neuron_link")));
        assert_eq!(fields.get("dtype"), Some(&Value::from("bf16")));
        assert_eq!(
            fields.get("message_size_bytes"),
            Some(&Value::from(1024_u64))
        );
        assert_eq!(
            NeuronCollectiveSpec::cache_kind(NEURON_COLLECTIVE_BACKEND),
            CacheKind::Cache1DLinear
        );
        let input = NeuronCollectiveKernelInput {
            message_size_bytes: 1 << 20,
        };
        assert_eq!(&*input.coords(), &[(1u64 << 20) as f64]);
    }
}
