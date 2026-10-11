//! KDA RecoverSSM accepted-state commit, one `KDARecoverSSMCommitContext.commit`
//! call (vLLM fork, Kimi-K3 ops; the GLM-5.3-Flash KDA layer is the same op).
//!
//! One semantic call, three launches: the commit plan, the conv-window
//! compaction of every layer, and the fused fold of every layer's accepted
//! records into its fp32 checkpoint (one state read and one write per request,
//! head and layer). vLLM makes the call once per KDA KV-cache group, and one
//! call covers all `num_layers` layers of the group, so the layer count is
//! static identity: billing a one-layer row per layer would repeat the plan and
//! the launch latencies (B200: 16 us for one layer vs 54.5 us for nine at 8
//! heads, batch 8, instead of 144 us), while the cost is linear in layers past
//! that fixed part.
//!
//! The runtime axes are the request count and the accepted length. The fold
//! loops over each request's accepted records, and the measured cost is linear
//! in that length (B200, 64 heads, one layer, batch 128: 247 / 408 / 629 us at
//! 1 / 4 / 8 accepted tokens), so it is a cache axis, not a constant. Requests
//! accept different lengths; each one's programs loop its own length, so the
//! batch costs its mean, which is what `mean_commit_len` carries.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::kda_recoverssm_verify::{batch_axis, value_blocks};
use crate::timing::sweep::{Axis, Coords, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KdaRecoverSSMCommitKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_heads: Dim,
    pub head_dim: Dim,
    pub spec_query_len: Dim,
    /// KDA layers in the KV-cache group one call commits.
    pub num_layers: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KdaRecoverSSMCommitKernelInput {
    /// Spec-decode requests committed.
    pub batch_size: u32,
    /// Mean accepted tokens per request, in `1..=spec_query_len` (spec decode
    /// always accepts the target token).
    pub mean_commit_len: f64,
}

impl SweepCoords for KdaRecoverSSMCommitKernelInput {
    fn coords(&self) -> Coords {
        Coords::new([f64::from(self.batch_size), self.mean_commit_len])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["batch_size", "mean_commit_len"]
    }
}

pub struct KdaRecoverSSMCommitSpec;

impl KernelSpec for KdaRecoverSSMCommitSpec {
    type Config = KdaRecoverSSMCommitKernelConfig;
    type Input = KdaRecoverSSMCommitKernelInput;

    const KIND: KernelKind = "kda_recoverssm_commit";

    /// Batch as the verify kind lays it out (every batch while the fold launch
    /// is a few waves deep, then powers of two to 512), by accepted length 1,
    /// 2, 4, 6 and the full window. The cost is linear in the length (within 3%
    /// between 1, 4 and 8 on B200), so a sparse length axis interpolates it.
    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        let programs =
            value_blocks(config.head_dim.get()) * config.num_heads.get() * config.num_layers.get();
        let window = config.spec_query_len.get();
        let mut lengths: Vec<u32> = [1, 2, 4, 6].into_iter().filter(|&c| c < window).collect();
        lengths.push(window);
        SweepGrid::new(vec![batch_axis(programs), Axis::values(lengths)])
    }

    /// Bilinear: the work is batch x (state read/write + accepted length x
    /// record fold), so the cross term is the physics past the batch edge.
    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|batch_size, commit_len| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("batch_size", batch_size as u32)
                .with("commit_len", commit_len as u32)
                .with("num_heads", config.num_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("spec_query_len", config.spec_query_len.get())
                .with("num_layers", config.num_layers.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(KdaRecoverSSMCommitKernel, KdaRecoverSSMCommitSpec);

#[cfg(test)]
mod tests {
    use super::{
        KdaRecoverSSMCommitKernelConfig, KdaRecoverSSMCommitKernelInput, KdaRecoverSSMCommitSpec,
    };
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{SlotInput, SweepCoords};
    use serde_json::Value;

    fn config(spec_query_len: u32) -> KdaRecoverSSMCommitKernelConfig {
        KdaRecoverSSMCommitKernelConfig {
            backends: vec!["vllm_triton"],
            gpu_name: "NVIDIA B200".to_string(),
            num_heads: 64.into(),
            head_dim: 128.into(),
            spec_query_len: spec_query_len.into(),
            num_layers: 9.into(),
            dtype: DType::Bf16,
        }
    }

    /// Catches a dtype tag drift that would hide the kind from launcher
    /// backend selection (Python BackendSupport is compute bf16, no KV axis).
    #[test]
    fn dtype_tag_is_compute_only() {
        let cfg = config(8);
        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        assert_eq!(cfg.kv_dtype(), None);
    }

    /// Catches a length axis that profiles a length the plan kernel would
    /// clamp (past the window) or drops the full window.
    #[test]
    fn length_axis_stays_inside_the_window() {
        let grid = KdaRecoverSSMCommitSpec::sweep_grid(&config(8));
        assert_eq!(grid.axes()[1], [1.0, 2.0, 4.0, 6.0, 8.0]);
        let grid = KdaRecoverSSMCommitSpec::sweep_grid(&config(4));
        assert_eq!(grid.axes()[1], [1.0, 2.0, 4.0]);
        let grid = KdaRecoverSSMCommitSpec::sweep_grid(&config(2));
        assert_eq!(grid.axes()[1], [1.0, 2.0]);
    }

    /// Catches a batch axis that does not count every layer's programs: eight
    /// heads over six layers are 192 programs per request, dense to batch 21.
    #[test]
    fn batch_axis_counts_layers_and_heads() {
        let narrow = KdaRecoverSSMCommitKernelConfig {
            num_heads: 8.into(),
            num_layers: 6.into(),
            ..config(8)
        };
        let grid = KdaRecoverSSMCommitSpec::sweep_grid(&narrow);
        let axis = &grid.axes()[0];
        assert_eq!(axis.len(), 26);
        assert_eq!(&axis[20..], &[21.0, 32.0, 64.0, 128.0, 256.0, 512.0]);
        let grid = KdaRecoverSSMCommitSpec::sweep_grid(&config(8));
        assert_eq!(grid.axes()[0].len(), 10);
    }

    /// Catches payload keys that would miss the Python KernelArgs DB key, or
    /// a layer count / length that is not forwarded.
    #[test]
    fn enumerate_emits_exact_python_fields() {
        let cfg = config(8);
        let grid = KdaRecoverSSMCommitSpec::sweep_grid(&cfg);
        let payloads = KdaRecoverSSMCommitSpec::enumerate(&cfg, &grid, "vllm_triton");
        assert_eq!(payloads.len(), 50);
        for payload in &payloads {
            let names: Vec<&str> = payload.fields().keys().map(String::as_str).collect();
            assert_eq!(
                names,
                [
                    "backend",
                    "batch_size",
                    "commit_len",
                    "dtype",
                    "head_dim",
                    "num_heads",
                    "num_layers",
                    "spec_query_len"
                ]
            );
        }
        let last = payloads[49].fields();
        assert_eq!(last["batch_size"], Value::from(512_u32));
        assert_eq!(last["commit_len"], Value::from(8_u32));
        assert_eq!(last["num_layers"], Value::from(9_u32));
    }

    /// Catches an Input that does not project onto (batch, mean length) or
    /// does not reach the cost-log slot payload.
    #[test]
    fn input_projects_batch_and_mean_length_and_logs_them() {
        let input = KdaRecoverSSMCommitKernelInput {
            batch_size: 24,
            mean_commit_len: 3.5,
        };
        assert_eq!(&*input.coords(), &[24.0, 3.5]);
        let slot: SlotInput = input.into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"batch_size": 24, "mean_commit_len": 3.5})
        );
    }
}
