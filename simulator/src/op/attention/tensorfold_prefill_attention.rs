//! TensorFold's prompt attention over one prefill step: one fixed leaf summing
//! a `flashinfer_attn_prefill` / `tensorfold_pattn` launch per sequence chunk.
//!
//! TensorFold launches `pattn_kernel` once per stream piece in a batched
//! `prefill_rows` forward, and once per even chunk in a single-piece
//! `prefill_state` (`prefill.py`). Each launch is exactly the kernel's cache
//! cell `(prefix_len, append_len)`: the rows already committed and the rows it
//! appends. The number of launches varies per step, so they sum into one slot
//! (INV-1). The caller lists the launches; the chunking rule lives with the
//! prefill shape (`worklet::tensorfold_common::PrefillForward::launches`).
//!
//! The KV write ahead of it is a copy into the stream's buffer, not part of
//! this op.

use std::sync::Arc;

use crate::timing::kernels::{
    FlashinferAttnPrefillKernel, FlashinferAttnPrefillKernelConfig,
    FlashinferAttnPrefillKernelInput,
};
use crate::timing::{
    AttnPrefillLog, BuildError, CostNode, CostTreeBuilder, Evaluator, LeafMetrics, PerfApiBridge,
    Probe,
};

/// The `pattn_kernel` config: backend `tensorfold_pattn`, the GQA heads, and
/// bf16 q/kv/o.
pub type TensorfoldPrefillAttentionOpConfig = FlashinferAttnPrefillKernelConfig;

/// Every launch of the step, in order, as `(prefix_len, append_len)`.
#[derive(Clone, Debug, Default)]
pub struct TensorfoldPrefillAttentionOpInput {
    pub launches: Vec<(u32, u32)>,
}

pub struct TensorfoldPrefillAttentionOp {
    pub name: String,
    pub prefill: Arc<FlashinferAttnPrefillKernel>,
}

impl TensorfoldPrefillAttentionOp {
    pub fn build(
        name: String,
        config: TensorfoldPrefillAttentionOpConfig,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let prefill = Arc::new(FlashinferAttnPrefillKernel::build(
            name.clone(),
            config,
            bridge,
        )?);
        Ok(Self { name, prefill })
    }

    /// One leaf, whatever the launch count.
    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        builder.leaf(
            self.name.clone(),
            self.prefill.kind(),
            self.prefill.describe_config(),
        )
    }

    /// Sum every launch into the slot; no launches is zero without a lookup.
    pub fn eval(&self, input: &TensorfoldPrefillAttentionOpInput, ev: &mut Evaluator) {
        let metrics = sum_launches(&input.launches, |shape| self.prefill.eval(shape));
        ev.push(metrics, || {
            AttnPrefillLog {
                prefill_chunk_pairs: input.launches.clone(),
            }
            .into()
        });
    }
}

fn sum_launches(
    launches: &[(u32, u32)],
    mut eval: impl FnMut(&FlashinferAttnPrefillKernelInput) -> LeafMetrics,
) -> LeafMetrics {
    let mut metrics = LeafMetrics::ZERO;
    for &(prefix_len, append_len) in launches {
        assert!(append_len > 0, "a prompt attention launch appends rows");
        metrics.add_fanin(eval(&FlashinferAttnPrefillKernelInput {
            prefix_len,
            append_len,
        }));
    }
    metrics
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::bridge::DType;
    use crate::timing::cache::interp::CoverageFlags;
    use crate::timing::kernels::flashinfer_attn_prefill::TENSORFOLD_PATTN;

    fn gb10_config() -> TensorfoldPrefillAttentionOpConfig {
        FlashinferAttnPrefillKernelConfig {
            backends: vec![TENSORFOLD_PATTN],
            gpu_name: "NVIDIA GB10".to_string(),
            num_qo_heads: 24.into(),
            num_kv_heads: 4.into(),
            head_dim: 256.into(),
            q_dtype: DType::Bf16,
            kv_dtype: DType::Bf16,
            o_dtype: DType::Bf16,
        }
    }

    #[test]
    fn compile_is_one_leaf_and_empty_eval_is_zero() {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let op =
            TensorfoldPrefillAttentionOp::build("m.attn.prefill".into(), gb10_config(), &bridge)
                .unwrap();
        let mut builder = CostTreeBuilder::new();
        let root = op.compile(&mut builder);
        let tree = builder.finish(root);
        assert_eq!(tree.n_slots(), 1);
        assert_eq!(tree.slots[0].name, "m.attn.prefill");
        assert_eq!(tree.slots[0].kind, "flashinfer_attn_prefill");

        let mut buf = vec![LeafMetrics::MISS; 1];
        let mut inputs = Vec::new();
        let mut ev = Evaluator::with_inputs(&mut buf, &mut inputs);
        op.eval(&TensorfoldPrefillAttentionOpInput::default(), &mut ev);
        assert_eq!(ev.filled(), 1);
        assert_eq!(buf[0].m.time_ms, 0.0);
        assert_eq!(buf[0].coverage, CoverageFlags::EMPTY);
        assert_eq!(
            serde_json::to_value(&inputs).unwrap(),
            serde_json::json!([{"prefill_chunk_pairs": []}])
        );
    }

    #[test]
    fn launches_pass_through_as_cache_cells_and_sum() {
        let mut seen = Vec::new();
        let metrics = sum_launches(&[(0, 600), (2048, 424)], |shape| {
            seen.push((shape.prefix_len, shape.append_len));
            let mut m = LeafMetrics::ZERO;
            m.m.time_ms = shape.append_len as f32;
            m.backend_index = 0;
            m
        });
        assert_eq!(seen, [(0, 600), (2048, 424)]);
        assert_eq!(metrics.m.time_ms, 1024.0);
        assert_eq!(metrics.backend_index, 0);
    }

    /// Against the GB10 `tensorfold_pattn` rows in `profiling/profile.db`.
    /// Run from the repo root with libpython on `LD_LIBRARY_PATH`:
    /// `PYTHONPATH=$PWD uv run cargo test -p simulator --lib tensorfold_prefill_attention -- --ignored`.
    #[test]
    #[ignore = "needs profile.db GB10 rows and the venv interpreter"]
    fn gb10_rows_price_a_step() {
        let bridge = PerfApiBridge::new().expect("perf_api bridge");
        let op =
            TensorfoldPrefillAttentionOp::build("m.attn.prefill".into(), gb10_config(), &bridge)
                .expect("tensorfold_pattn rows on GB10");
        let eval = |launches: Vec<(u32, u32)>| {
            let mut buf = vec![LeafMetrics::MISS; 1];
            let mut ev = Evaluator::new(&mut buf);
            op.eval(&TensorfoldPrefillAttentionOpInput { launches }, &mut ev);
            buf[0]
        };
        let fresh = eval(vec![(0, 4096)]);
        let chunked = eval(vec![(0, 3000), (3000, 3000), (6000, 3000)]);
        let batched = eval(vec![(0, 600), (2048, 424)]);
        let long = eval(vec![(65536, 1024)]);
        eprintln!(
            "pattn GB10 ms: fresh 4096 {:.4}, 3x3000 chunks {:.4}, 600+424 {:.4}, 1024@64k {:.4}",
            fresh.m.time_ms, chunked.m.time_ms, batched.m.time_ms, long.m.time_ms
        );
        for m in [fresh, chunked, batched, long] {
            assert!(m.m.time_ms > 0.0 && m.m.time_ms.is_finite());
        }
        // A cached prefix lands inside the grid. A fresh (k = 0) launch sits a
        // half token below its diagonal node and reads as extrapolated.
        for cell in [(3000, 3000), (6000, 3000), (2048, 424), (65536, 1024)] {
            assert_eq!(eval(vec![cell]).coverage, CoverageFlags::EMPTY, "{cell:?}");
        }
        // The chunks of a step sum: (0, 3000) + (3000, 3000) + (6000, 3000).
        let parts: f32 = [(0, 3000), (3000, 3000), (6000, 3000)]
            .into_iter()
            .map(|cell| eval(vec![cell]).m.time_ms)
            .sum();
        assert!((chunked.m.time_ms - parts).abs() < 1e-4);
        // A longer prefix reads more keys.
        assert!(long.m.time_ms > eval(vec![(0, 1024)]).m.time_ms);
    }
}
