//! TensorFold's GDN prompt recurrence over one prefill step: one fixed leaf
//! summing a `gdn_chunk_delta_rule` / `tensorfold_chain` launch per sequence
//! chunk.
//!
//! `deltanet.chain` runs one sequence a launch: once per stream piece in a
//! batched `prefill_rows` forward, once per even chunk in a single-piece
//! `prefill_state` (`prefill.py`). Each launch is the one-sequence cell
//! `(num_tokens = rows, max_sequence_length = rows)`. The launch count varies
//! per step, so the launches sum into one slot (INV-1). The caller lists the
//! launches' rows (`worklet::tensorfold_common::PrefillForward::launches`).
//!
//! The `_gdn_pre` convolution / gates ahead of it is its own kind
//! (`gdn_conv_norm_gates`), one launch over every row, not part of this op.

use std::sync::Arc;

use crate::timing::kernels::{
    GdnChunkDeltaRuleKernel, GdnChunkDeltaRuleKernelConfig, GdnChunkDeltaRuleKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Evaluator, FanInLog, LeafMetrics, PerfApiBridge, Probe,
};

/// The `chain_kernel` config: backend `tensorfold_chain`, the GDN heads.
pub type TensorfoldGdnChainOpConfig = GdnChunkDeltaRuleKernelConfig;

/// Rows of every launch of the step, in order; each positive.
#[derive(Clone, Debug, Default)]
pub struct TensorfoldGdnChainOpInput {
    pub sequence_lengths: Vec<u32>,
}

pub struct TensorfoldGdnChainOp {
    pub name: String,
    pub chain: Arc<GdnChunkDeltaRuleKernel>,
}

impl TensorfoldGdnChainOp {
    pub fn build(
        name: String,
        config: TensorfoldGdnChainOpConfig,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let chain = Arc::new(GdnChunkDeltaRuleKernel::build(
            name.clone(),
            config,
            bridge,
        )?);
        Ok(Self { name, chain })
    }

    /// One leaf, whatever the launch count.
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        builder.leaf(
            self.name.clone(),
            self.chain.kind(),
            self.chain.describe_config(),
        )
    }

    /// Sum every launch into the slot; no launches is zero without a lookup.
    pub fn eval(&self, input: &TensorfoldGdnChainOpInput, ev: &mut Evaluator) {
        let launches = launch_inputs(&input.sequence_lengths);
        let mut metrics = LeafMetrics::ZERO;
        for launch in &launches {
            metrics.add_fanin(self.chain.eval(launch));
        }
        ev.push(metrics, || {
            FanInLog {
                launches: launches.into_iter().map(Into::into).collect(),
            }
            .into()
        });
    }
}

/// One single-sequence cell per launch.
fn launch_inputs(sequence_lengths: &[u32]) -> Vec<GdnChunkDeltaRuleKernelInput> {
    sequence_lengths
        .iter()
        .map(|&rows| {
            assert!(rows > 0, "a GDN chain launch has rows");
            GdnChunkDeltaRuleKernelInput {
                num_tokens: rows,
                max_sequence_length: rows,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::bridge::DType;
    use crate::timing::cache::interp::CoverageFlags;
    use crate::timing::kernels::gdn_chunk_delta_rule::TENSORFOLD_CHAIN;

    fn gb10_config() -> TensorfoldGdnChainOpConfig {
        GdnChunkDeltaRuleKernelConfig {
            backends: vec![TENSORFOLD_CHAIN],
            gpu_name: "NVIDIA GB10".to_string(),
            num_key_heads: 16.into(),
            num_heads: 48.into(),
            key_head_dim: 128.into(),
            value_head_dim: 128.into(),
            dtype: DType::Bf16,
        }
    }

    #[test]
    fn compile_is_one_leaf_and_empty_eval_is_zero() {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let op = TensorfoldGdnChainOp::build("m.gdn.chain".into(), gb10_config(), &bridge).unwrap();
        let mut builder = CostTreeBuilder::new();
        let root = op.compile(&mut builder);
        let tree = builder.finish(root);
        assert_eq!(tree.n_slots(), 1);
        assert_eq!(tree.slots[0].kind, "gdn_chunk_delta_rule");

        let mut buf = vec![LeafMetrics::MISS; 1];
        let mut inputs = Vec::new();
        let mut ev = Evaluator::with_inputs(&mut buf, &mut inputs);
        op.eval(&TensorfoldGdnChainOpInput::default(), &mut ev);
        assert_eq!(buf[0].m.time_ms, 0.0);
        assert_eq!(buf[0].coverage, CoverageFlags::EMPTY);
        assert_eq!(
            serde_json::to_value(&inputs).unwrap(),
            serde_json::json!([{"launches": []}])
        );
    }

    #[test]
    fn each_launch_is_one_sequence() {
        let cells: Vec<_> = launch_inputs(&[600, 424])
            .iter()
            .map(|c| (c.num_tokens, c.max_sequence_length))
            .collect();
        assert_eq!(cells, [(600, 600), (424, 424)]);
    }

    /// Against the GB10 `tensorfold_chain` rows in `profiling/profile.db`.
    /// Run from the repo root with libpython on `LD_LIBRARY_PATH`:
    /// `PYTHONPATH=$PWD uv run cargo test -p simulator --lib tensorfold_gdn_chain -- --ignored`.
    #[test]
    #[ignore = "needs profile.db GB10 rows and the venv interpreter"]
    fn gb10_rows_price_a_step() {
        let bridge = PerfApiBridge::new().expect("perf_api bridge");
        let op = TensorfoldGdnChainOp::build("m.gdn.chain".into(), gb10_config(), &bridge)
            .expect("tensorfold_chain rows on GB10");
        let eval = |sequence_lengths: Vec<u32>| {
            let mut buf = vec![LeafMetrics::MISS; 1];
            let mut ev = Evaluator::new(&mut buf);
            op.eval(&TensorfoldGdnChainOpInput { sequence_lengths }, &mut ev);
            buf[0]
        };
        let one = eval(vec![1024]);
        let four = eval(vec![1024; 4]);
        let ragged = eval(vec![600, 424]);
        let chunked = eval(vec![3000; 3]);
        // The profiled N-stream cell of the same launches, for comparison.
        let n4 = op.chain.eval(&GdnChunkDeltaRuleKernelInput {
            num_tokens: 4096,
            max_sequence_length: 1024,
        });
        eprintln!(
            "chain GB10 ms: 1024 {:.4}, 4x1024 summed {:.4} (N=4 cell {:.4}), 600+424 {:.4}, 3x3000 {:.4}",
            one.m.time_ms, four.m.time_ms, n4.m.time_ms, ragged.m.time_ms, chunked.m.time_ms
        );
        for m in [one, four, ragged, chunked] {
            assert!(m.m.time_ms > 0.0 && m.m.time_ms.is_finite());
            assert_eq!(m.coverage, CoverageFlags::EMPTY, "on the measured grid");
        }
        assert!((four.m.time_ms - 4.0 * one.m.time_ms).abs() < 1e-4);
    }
}
