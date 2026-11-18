//! DSA decode state update from an fp32 split-GEMM row, in one launch.
//!
//! Reads the fp32 output of the DSA layer's hidden-state GEMM ([q latent | kv
//! latent | indexer k | gate score | fp32 head weights]) and writes every
//! decode state that depends on it: the RMSNormed q latent (bf16), the
//! RMSNormed kv latent as an FP8 MLA cache row, the LayerNormed indexer k and
//! gate score (bf16), and the pooled index cache / tail ring (as
//! `dsa_pooled_index_cache_update`). Two row programs per row, then one program
//! per request. Static identity is the row geometry, `next_n`, pool size, tail
//! ring and index-cache page size; runtime is the number of requests.
//!
//! Backend `triton` is a repo-vendored kernel whose row geometry is compiled
//! in; its runner refuses any other geometry.

use crate::timing::bridge::{de_backends, ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DsaFp32SplitStateUpdateKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Rows per request in one step (1 for plain decode).
    pub next_n: u32,
    /// Leading q-latent columns, RMSNormed into bf16.
    pub q_lora_cols: Dim,
    /// kv-latent columns, RMSNormed into the FP8 MLA cache.
    pub kv_lora_cols: Dim,
    /// Indexer k columns (and as many gate-score columns after them).
    pub index_dim: Dim,
    /// fp32 columns left in place (head weights), not read.
    pub fp32_tail_cols: Dim,
    /// Tokens compressed into one cached index key.
    pub pool_size: u32,
    /// Slots in each request's raw-token tail ring.
    pub tail_ring: u32,
    /// Pooled keys per index-cache page.
    pub block_size: u32,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct DsaFp32SplitStateUpdateKernelInput {
    /// Decode requests in the step (rows = requests x next_n).
    pub num_requests: u32,
}

pub struct DsaFp32SplitStateUpdateSpec;

impl KernelSpec for DsaFp32SplitStateUpdateSpec {
    type Config = DsaFp32SplitStateUpdateKernelConfig;
    type Input = DsaFp32SplitStateUpdateKernelInput;
    const KIND: KernelKind = "dsa_fp32_split_state_update";

    fn sweep_grid(_: &Self::Config) -> SweepGrid {
        // The request axis of dsa_pooled_index_cache_update: the per-request
        // program carries the serial pool chain, the row programs add with
        // rows. 1..2048 requests covers decode and verify batches past 1024
        // rows at every next_n; SM-multiple points mark CTA waves.
        SweepGrid::new(vec![Axis::values([
            1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 148, 192, 256, 296, 384, 512, 592,
            768, 1024, 1184, 1536, 2048,
        ])])
    }

    fn cache_kind(_: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|num_requests| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_requests", num_requests as u32)
                .with("next_n", config.next_n)
                .with("q_lora_cols", config.q_lora_cols.get())
                .with("kv_lora_cols", config.kv_lora_cols.get())
                .with("index_dim", config.index_dim.get())
                .with("fp32_tail_cols", config.fp32_tail_cols.get())
                .with("pool_size", config.pool_size)
                .with("tail_ring", config.tail_ring)
                .with("block_size", config.block_size)
        })
    }
}

register_kernel!(DsaFp32SplitStateUpdateKernel, DsaFp32SplitStateUpdateSpec);

#[cfg(test)]
mod tests {
    use super::{
        DsaFp32SplitStateUpdateKernelConfig, DsaFp32SplitStateUpdateKernelInput,
        DsaFp32SplitStateUpdateSpec,
    };
    use crate::timing::kernels::engine::KernelSpec;
    use crate::timing::{SlotInput, SweepCoords};

    fn config() -> DsaFp32SplitStateUpdateKernelConfig {
        DsaFp32SplitStateUpdateKernelConfig {
            backends: vec!["triton"],
            gpu_name: "NVIDIA B200".to_string(),
            next_n: 8,
            q_lora_cols: 1536.into(),
            kv_lora_cols: 512.into(),
            index_dim: 128.into(),
            fp32_tail_cols: 32.into(),
            pool_size: 4,
            tail_ring: 4,
            block_size: 64,
        }
    }

    #[test]
    fn input_is_exactly_the_request_count() {
        // Catches a re-axis onto rows: the request programs set the time.
        let input = DsaFp32SplitStateUpdateKernelInput { num_requests: 16 };
        assert_eq!(&*input.coords(), &[16.0]);
        let slot: SlotInput = input.into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"num_requests": 16})
        );
    }

    /// Catches a field sent under another name: the runner would refuse the
    /// row (or time the wrong geometry).
    #[test]
    fn payload_carries_each_field_under_its_python_name() {
        let cfg = config();
        let grid = DsaFp32SplitStateUpdateSpec::sweep_grid(&cfg);
        let axis = &grid.axes()[0];
        assert!(axis.windows(2).all(|w| w[0] < w[1]));
        // c128 / c256 / c384 / c512 at next_n 8, past 1024 rows at next_n 1.
        for requests in [1.0, 16.0, 32.0, 48.0, 64.0, 2048.0] {
            assert!(axis.contains(&requests), "missing num_requests={requests}");
        }
        let payload = DsaFp32SplitStateUpdateSpec::enumerate(&cfg, &grid, "triton").remove(0);
        let fields = payload.fields();
        let names: Vec<&str> = fields.keys().map(String::as_str).collect();
        assert_eq!(
            names,
            [
                "backend",
                "block_size",
                "fp32_tail_cols",
                "index_dim",
                "kv_lora_cols",
                "next_n",
                "num_requests",
                "pool_size",
                "q_lora_cols",
                "tail_ring",
            ]
        );
        assert_eq!(fields["num_requests"], 1);
        assert_eq!(fields["q_lora_cols"], 1536);
        assert_eq!(fields["kv_lora_cols"], 512);
        assert_eq!(fields["index_dim"], 128);
        assert_eq!(fields["fp32_tail_cols"], 32);
        assert_eq!(fields["next_n"], 8);
        assert_eq!(fields["tail_ring"], 4);
    }
}
