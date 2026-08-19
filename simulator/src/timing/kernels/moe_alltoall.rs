//! `FlashInfer` MNNVL two-sided `MoE` all-to-all timing leaf.
//!
//! This is the transfer vLLM runs for expert parallelism under
//! `--all2all-backend=flashinfer_nvlink_two_sided`. Both legs of the round trip
//! are the same `moeAllToAllKernel` over a workspace staged in fabric memory,
//! reached through `MnnvlMoe.mnnvl_moe_alltoallv` on the way out and
//! `mnnvl_moe_alltoallv_combine` on the way back.
//!
//! It is a distinct kind from `p2p_intra` because it is not a point-to-point
//! send. Every rank sends and receives at once, the rows are gathered by index
//! rather than streamed contiguously, and the achieved bandwidth sits well below
//! the two-rank NCCL curve — pricing a measured 8k-token GLM-5.2 prefill layer
//! off `p2p_intra` under-predicted it by ~1.4x even after the payload width was
//! corrected.
//!
//! ## The leaf is the transfer, and only the transfer
//!
//! The runner times `moe_comm` itself, not the Python wrapper around it. That
//! matters for combine, whose wrapper also allocates a zeroed
//! `token_count x top_k` staging buffer and reduces over the top-k axis
//! afterwards. Those two are token-count work, not row work: at 8k tokens the
//! fill alone writes 805 MB per layer, comparable to the transfer beside it, and
//! it scales with a quantity this leaf's axes do not carry. Leaving them inside
//! the measured call would have forced a third axis. The arch prices them as two
//! elementwise leaves instead.
//!
//! ## Two axes, because the two imbalances are not interchangeable
//!
//! A collective ends when its slowest rank ends, and a rank is loaded from two
//! sides at once: the rows it sends (set by how the tokens are spread over DP
//! ranks) and the rows it receives (set by how the experts are spread over EP
//! ranks). Between dispatch and combine those two are transposes of each other,
//! so neither can stand in for the other.
//!
//! Measured on H200 x8 under graph replay, holding the total constant at 8x1024
//! tokens: against a single-axis fit on `max(send, recv)`, configurations skewed
//! on the input side land at -0.1%/0.0% while configurations skewed on the
//! output side land at -9.5%/-8.3%. For combine the signs flip (+2.6%/+4.5%),
//! which is exactly what a transpose predicts. Send-side skew costs more than
//! receive-side skew at the same row count — egress leaves one GPU, whereas
//! fan-in arrives from `ep_size` senders in parallel.
//!
//! It is not a third axis. Growing `max_token_count_per_rank` from 1024 to 8192
//! (an 8x larger receive allocation) while holding the moved rows fixed changed
//! dispatch by -2.3%..0% and combine by -0.3%..0% — both negative, i.e. noise.
//! Padding is free; only live rows cost.
//!
//! ## The infeasible corner
//!
//! Every row one rank sends is a row some rank receives, so
//! `sum(send) == sum(recv)`, and therefore `max_send <= total` while
//! `max_recv >= total / ep_size`. The ratio between the two axes is bounded by
//! `ep_size` in both directions, and the rectangular cache grid's far corners
//! describe transfers that cannot exist. `infeasible_mask` strips them so the
//! cache renormalizes over the feasible corners rather than storing a
//! fabricated time.
//!
//! ## Shape split
//!
//! Static config is the parallel recipe `(ep_size, top_k, slot_count,
//! hidden_bytes, direction, fabric)`; the runtime axes are the two row counts.
//! `top_k`/`slot_count` stay in the config because they set the fan-out — how
//! many distinct ranks one token reaches — which the runner reproduces when it
//! builds the send indices, and which a serving deployment never changes
//! mid-run.
//!
//! `direction` stays a config field rather than collapsing into the transpose of
//! the two axes because the index structure is not symmetric: dispatch gathers
//! its send rows out of a token buffer with reuse (one token row leaves for
//! every distinct destination it reaches), while combine reads each row of the
//! expert output exactly once. Same kernel, different reuse on the read side.

use crate::common::Fabric;
use crate::timing::bridge::{de_backends, ArgsPayload, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

/// Which leg of the round trip a row is on.
#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoeAlltoallDirection {
    /// `mnnvl_moe_alltoallv`: this rank's tokens out to the experts that chose
    /// them. One row leaves per (token, distinct destination rank), so the same
    /// token row is read once per destination it reaches.
    Dispatch,
    /// `mnnvl_moe_alltoallv_combine`: expert outputs back to the owning rank.
    /// Each row of the expert output is read exactly once. The wrapper's zero
    /// fill and top-k reduction are NOT part of this leaf — see the module doc.
    Combine,
}

impl MoeAlltoallDirection {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dispatch => "dispatch",
            Self::Combine => "combine",
        }
    }
}

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MoeAlltoallKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Ranks in the expert-parallel group. Every one is a real GPU the profiler
    /// must reserve; the kernel asserts the workspace spans them all. Also the
    /// bound on how far the two row axes may diverge.
    pub ep_size: u32,
    pub top_k: u32,
    pub slot_count: u32,
    /// One row's payload. The vLLM block-scale path defers activation
    /// quantisation until after the transfer, so this is the bf16 hidden width
    /// even when the experts are fp8 (see the arch that builds this config).
    pub hidden_bytes: Dim,
    pub direction: MoeAlltoallDirection,
    /// Row/cache key only: MNNVL is `NVLink` by construction.
    pub fabric: Fabric,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct MoeAlltoallKernelInput {
    /// Rows leaving the busiest sender. For dispatch that is one row per
    /// (token, distinct destination rank) on the DP rank holding the most
    /// tokens; for combine it is the rows the busiest EP rank's experts
    /// produced.
    pub max_send_rows: u32,
    /// Rows arriving at the busiest receiver — the transpose of the above.
    pub max_recv_rows: u32,
}

/// The row ladder both axes sweep.
///
/// Decode asks for tens of rows per rank (64 tokens across 8 DP ranks, each
/// token reaching ~5 of 8 destinations), a DP8 replica scheduling 8k-token
/// prefills asks for tens of thousands.
///
/// Doubling steps all the way up was the first shape, on the argument that the
/// curve is affine above the floor and bilinear interpolation is exact on an
/// affine segment however wide the step. The measurement says otherwise, and the
/// reason is geometric rather than statistical: the cost tracks `max(send, recv)
/// * hidden_bytes / bandwidth`, and `max` has a KINK along `send == recv` that a
/// bilinear patch cannot represent. Interpolating `max` itself over the
///   32768..65536 square gives 58031 rows at (43374, 54441) against a true 54441 —
///   +6.6%, and +16.7% dead in the middle of that cell. The GLM-5.2 DP8 prefill
///   point lands there in BOTH axes at once, which is why 8k-token steps read
///   +13.8% (dispatch) / +18.5% (combine) against vLLM while decode, whose cells
///   are only ~1.1x wide, was already within a few percent.
///
/// A cell spanning `lo..hi` overshoots by `0.5 * (r - 1) / (r + 1)` at its
/// centre, where `r = hi / lo` — so the budget is set by the RATIO, and a fixed
/// stride cannot hold a relative bound at the bottom of the range (an 8192-row
/// stride is 16.7% at 8192 and 2.9% at 32768). The ladder therefore doubles only
/// while the curve is launch-bound and flat — where the kink costs nothing and
/// decode already read within a few percent — then goes geometric at `r = 1.25`,
/// which holds every bandwidth-bound cell at <=5.6% for 10 points instead of the
/// 21 a fixed stride would need.
fn row_axis() -> Vec<f64> {
    Axis::chain([
        Axis::values([1, 4, 8, 16]),
        Axis::pow2(5, 13),
        Axis::values([
            10_240, 12_800, 16_000, 20_000, 25_088, 31_232, 39_168, 48_896, 61_184, 65_536,
        ]),
    ])
}

pub struct MoeAlltoallSpec;

impl KernelSpec for MoeAlltoallSpec {
    type Config = MoeAlltoallKernelConfig;
    type Input = MoeAlltoallKernelInput;

    const KIND: KernelKind = "moe_alltoall";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![row_axis(), row_axis()])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        // A rank is loaded from two sides at once and the loads ADD: rows it
        // sends and rows it receives are separate traffic, not a product. The
        // per-axis weights are fitted, not declared — in-grid the two slopes
        // measure 1.02:1 while off-grid ground truth is ~8:1 (fan-in arrives
        // from `ep_size` senders in parallel, egress leaves one GPU), and that
        // ratio is not identifiable from the grid. Fitting under-predicts the
        // far field; the bilinear cross term over-predicted it by 78x.
        CacheKind::Cache2DLinear(Extrapolation::Weighted)
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        // Conservation (`sum(send) == sum(recv)`) bounds the ratio between the
        // axes by `ep_size`, and the runner also has to hold both buffers in
        // device memory at once.
        const BUFFER_BUDGET_BYTES: f64 = 32.0 * 1024.0 * 1024.0 * 1024.0;
        let ratio_bound = f64::from(config.ep_size);
        let row_bytes = f64::from(config.hidden_bytes.get());
        grid.expand_2d(|send_rows, recv_rows| {
            let ratio = if send_rows >= recv_rows {
                send_rows / recv_rows
            } else {
                recv_rows / send_rows
            };
            ratio > ratio_bound || (send_rows + recv_rows) * row_bytes > BUFFER_BUDGET_BYTES
        })
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        assert!(
            config.top_k <= config.slot_count,
            "top_k {} cannot exceed slot_count {}",
            config.top_k,
            config.slot_count
        );
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "max_send_rows/max_recv_rows are non-negative sweep-grid coordinates, far below u32::MAX"
        )]
        grid.expand_2d(|max_send_rows, max_recv_rows| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("ep_size", config.ep_size)
                .with("max_send_rows", max_send_rows as u32)
                .with("max_recv_rows", max_recv_rows as u32)
                .with("top_k", config.top_k)
                .with("slot_count", config.slot_count)
                .with("hidden_bytes", config.hidden_bytes.get())
                .with("direction", config.direction.as_str())
                .with("fabric", config.fabric.as_str())
        })
    }
}

register_kernel!(MoeAlltoallKernel, MoeAlltoallSpec);

#[cfg(test)]
mod tests {
    use super::{
        MoeAlltoallDirection, MoeAlltoallKernelConfig, MoeAlltoallKernelInput, MoeAlltoallSpec,
    };
    use crate::common::Fabric;
    use crate::timing::cache::{CacheKind, Extrapolation};
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{SlotInput, SweepCoords, SweepGrid};
    use serde_json::Value;

    fn config() -> MoeAlltoallKernelConfig {
        MoeAlltoallKernelConfig {
            backends: vec!["flashinfer_mnnvl"],
            gpu_name: "NVIDIA H200".to_string(),
            ep_size: 8,
            top_k: 8,
            slot_count: 256,
            // GLM-5.2's 6144 hidden in bf16.
            hidden_bytes: 12_288.into(),
            direction: MoeAlltoallDirection::Dispatch,
            fabric: Fabric::Nvlink,
        }
    }

    #[test]
    fn config_identity_and_description_include_all_static_axes() {
        let base = config();
        assert_eq!(base, base.clone());

        let mut changed_ep_size = config();
        changed_ep_size.ep_size = 4;
        assert_ne!(base, changed_ep_size);

        let mut changed_direction = config();
        changed_direction.direction = MoeAlltoallDirection::Combine;
        assert_ne!(base, changed_direction);

        let mut changed_hidden = config();
        changed_hidden.hidden_bytes = 8_192.into();
        assert_ne!(base, changed_hidden);

        let mut changed_top_k = config();
        changed_top_k.top_k = 4;
        assert_ne!(base, changed_top_k);

        assert_eq!(
            base.describe_config(),
            serde_json::json!({
                "backends": ["flashinfer_mnnvl"],
                "gpu_name": "NVIDIA H200",
                "ep_size": 8,
                "top_k": 8,
                "slot_count": 256,
                "hidden_bytes": {"value": 12_288, "expression": null, "bindings": {}},
                "direction": "dispatch",
                "fabric": "nvlink",
            })
        );
    }

    #[test]
    fn input_projects_to_both_row_axes_and_converts_to_slot_input() {
        let input = MoeAlltoallKernelInput {
            max_send_rows: 8_190,
            max_recv_rows: 4_100,
        };
        assert_eq!(&*input.coords(), &[8_190.0, 4_100.0]);
        assert_eq!(
            MoeAlltoallKernelInput::coord_field_names(),
            &["max_send_rows", "max_recv_rows"]
        );

        let slot_input: SlotInput = input.into();
        assert!(matches!(slot_input, SlotInput::MoeAlltoall(_)));
    }

    #[test]
    fn grid_is_two_row_ladders_spanning_decode_to_a_dp8_prefill_step() {
        let grid = MoeAlltoallSpec::sweep_grid(&config());
        assert_eq!(grid.axes().len(), 2);
        assert_eq!(grid.axes()[0], grid.axes()[1]);
        assert_eq!(&grid.axes()[0][..5], &[1.0, 4.0, 8.0, 16.0, 32.0]);
        assert_eq!(grid.axes()[0].last().copied(), Some(65_536.0));
        assert_eq!(
            MoeAlltoallSpec::cache_kind("flashinfer_mnnvl"),
            CacheKind::Cache2DLinear(Extrapolation::Weighted)
        );
    }

    /// The bilinear patch cannot represent `max(send, recv)`'s kink along the
    /// diagonal, so a cell's width is an error budget, not a free parameter.
    ///
    /// Interpolating `max` over a cell whose sides run `lo..hi` overshoots by
    /// `0.25 * (hi - lo)` at the centre. Doubling steps make that 16.7% of the
    /// value, which is what the GLM-5.2 DP8 prefill point paid: the sim read
    /// +13.8% (dispatch) / +18.5% (combine) against vLLM at (43374, 54441),
    /// while a direct measurement of that exact shape was within +3.3%.
    ///
    /// Small rows may keep doubling — that segment is launch-bound and flat, so
    /// the kink costs nothing. The guard applies where the curve is
    /// bandwidth-bound and the prefill point lives.
    #[test]
    fn no_cell_in_the_bandwidth_bound_range_is_wide_enough_to_hide_the_diagonal_kink() {
        let grid = MoeAlltoallSpec::sweep_grid(&config());
        let axis = &grid.axes()[0];
        for window in axis.windows(2) {
            let (low, high) = (window[0], window[1]);
            if low < 8_192.0 {
                continue;
            }
            let ratio = high / low;
            let centre_error = 0.5 * (ratio - 1.0) / (ratio + 1.0);
            assert!(
                centre_error <= 0.06,
                "cell {low}..{high} (ratio {ratio:.3}) interpolates max() {:.1}% high at its \
                 centre; the prefill point sits in this range",
                centre_error * 100.0
            );
        }
        // And the ladder still brackets the DP8 8k-prefill point tightly in both
        // axes: (43374, 54441) is what the arch asks for at 8192 tokens/rank.
        let brackets = |value: f64| {
            axis.windows(2).any(|window| {
                window[0] <= value && value <= window[1] && window[1] / window[0] < 1.3
            })
        };
        assert!(brackets(43_374.0) && brackets(54_441.0));
    }

    #[test]
    fn infeasible_mask_strips_exactly_the_cells_conservation_forbids() {
        let config = config();
        let grid = MoeAlltoallSpec::sweep_grid(&config);
        let mask = MoeAlltoallSpec::infeasible_mask(&config, &grid);
        let axis = &grid.axes()[0];
        assert_eq!(mask.len(), axis.len() * axis.len());

        let feasible = |send: f64, recv: f64| {
            let send_index = axis.iter().position(|&v| v == send).unwrap();
            let recv_index = axis.iter().position(|&v| v == recv).unwrap();
            !mask[send_index * axis.len() + recv_index]
        };
        // Balanced, and both extremes of the ratio the group can reach.
        assert!(feasible(8_192.0, 8_192.0));
        assert!(feasible(8_192.0, 1_024.0));
        assert!(feasible(1_024.0, 8_192.0));
        // One more doubling and the ratio exceeds `ep_size`: a rank cannot send
        // 8192 rows into a group whose busiest receiver only takes 512.
        assert!(!feasible(8_192.0, 512.0));
        assert!(!feasible(512.0, 8_192.0));
        // The mask must not swallow the whole grid.
        assert!(mask.iter().filter(|&&stripped| !stripped).count() > axis.len());
    }

    #[test]
    fn a_narrower_ep_group_forbids_more_of_the_grid() {
        let wide = config();
        let mut narrow = config();
        narrow.ep_size = 2;
        let grid = MoeAlltoallSpec::sweep_grid(&wide);
        let wide_feasible = MoeAlltoallSpec::infeasible_mask(&wide, &grid)
            .iter()
            .filter(|&&stripped| !stripped)
            .count();
        let narrow_feasible = MoeAlltoallSpec::infeasible_mask(&narrow, &grid)
            .iter()
            .filter(|&&stripped| !stripped)
            .count();
        assert!(
            narrow_feasible < wide_feasible,
            "ep_size 2 kept {narrow_feasible} cells, ep_size 8 kept {wide_feasible}"
        );
    }

    #[test]
    fn a_payload_too_wide_for_the_runner_is_stripped_even_when_balanced() {
        let mut wide = config();
        // 512 KiB per row: 65,536 + 65,536 rows is 64 GiB of buffers.
        wide.hidden_bytes = 524_288.into();
        let grid = MoeAlltoallSpec::sweep_grid(&wide);
        let mask = MoeAlltoallSpec::infeasible_mask(&wide, &grid);
        let axis = &grid.axes()[0];
        let top = axis.len() - 1;
        assert!(
            mask[top * axis.len() + top],
            "the top corner must be stripped"
        );
        // ...while a shape the runner can hold survives.
        let small = axis.iter().position(|&v| v == 1_024.0).unwrap();
        assert!(!mask[small * axis.len() + small]);
    }

    #[test]
    fn enumerate_emits_exactly_the_python_wire_schema() {
        let config = config();
        let grid = SweepGrid::new(vec![vec![8_190.0], vec![4_100.0]]);
        let payloads = MoeAlltoallSpec::enumerate(&config, &grid, "flashinfer_mnnvl");
        let fields = payloads[0].fields();

        // Must stay aligned with `MoeAlltoallArgs` in
        // profiling/kernels/moe_alltoall.py.
        assert_eq!(fields.len(), 9);
        assert_eq!(
            fields.get("backend"),
            Some(&Value::from("flashinfer_mnnvl"))
        );
        assert_eq!(fields.get("ep_size"), Some(&Value::from(8_u32)));
        assert_eq!(fields.get("max_send_rows"), Some(&Value::from(8_190_u32)));
        assert_eq!(fields.get("max_recv_rows"), Some(&Value::from(4_100_u32)));
        assert_eq!(fields.get("top_k"), Some(&Value::from(8_u32)));
        assert_eq!(fields.get("slot_count"), Some(&Value::from(256_u32)));
        assert_eq!(fields.get("hidden_bytes"), Some(&Value::from(12_288_u32)));
        assert_eq!(fields.get("direction"), Some(&Value::from("dispatch")));
        assert_eq!(fields.get("fabric"), Some(&Value::from("nvlink")));
        assert_eq!(payloads[0].backend(), Some("flashinfer_mnnvl"));
        assert!(fields.get("tokens_per_rank").is_none());
    }
}
