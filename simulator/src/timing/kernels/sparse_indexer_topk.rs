//! Row-level top-k over sparse-indexer logits, decode and prefill alike.
//!
//! Reads the `dsa_persistent_topk_decode` rows (same args, same backends) on
//! its own grid. A prefill chunk selects one top-k per new token, so it is the
//! same call with `batch_size` = tokens, `next_n` = 1 and `context_len` = the
//! compressed keys a row sees; this grid therefore reaches 16384 rows, where
//! the decode grid stops at 256.
//!
//! The radix select's work follows the score distribution, so the op passes
//! `context_mode: "uniform_random"` (independent normal scores, like a live
//! indexer's): on B200 the increasing `uniform` template, whose valid values
//! crowd together when a row is short of its padded width, was 20% slow at
//! 1024 rows of 8192 and 10% fast at 1024 rows of 30000 against normal
//! scores, which match the fork's own selection call within 3% on a long
//! prefill chunk. The row stride only spaces the
//! rows once the values are random, so the op passes the model's compressed
//! context range instead of the deployment width.
//!
//! The backend is the op's choice, never the faster of two: vLLM's indexer
//! dispatch (`model_executor/layers/indexer_topk.py:215-254`, mirrored by
//! the Qwen4Exp QSA indexer `ops/qsa_indexer.py:471-499` of the alignment
//! fork) runs `cooperative_topk` for at most 64 rows on SM90+ outside SM12x
//! and `persistent_topk` otherwise. A config naming `vllm_cuda_cooperative`
//! masks every cell past 64 rows.
//!
//! The persistent kernel's cliff sits at one row per SM (B200, 148 SMs, 8192
//! logits: 37 requests of 4 rows 6.2 µs, 38 requests 9.8 µs), so the batch
//! axis comes from a rows axis that brackets the 132- and 148-SM waves,
//! divided by `next_n` both ways. A context of 513 holds one candidate past
//! top-512 and runs faster than longer rows (152 rows: 4.1 µs, against 6.3 µs
//! at 600), so 600 opens the long-row plateau.

use crate::timing::bridge::{ArgsPayload, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::dsa_persistent_topk_decode::{
    DsaPersistentTopkDecodeKernelConfig, DsaPersistentTopkDecodeKernelInput,
    DsaPersistentTopkDecodeSpec,
};
use crate::timing::kernels::engine::{register_kernel, KernelSpec, OffGrid};
use crate::timing::sweep::{Axis, SweepGrid};

/// The cooperative backend and its row ceiling (`cooperative_topk.cu:63`).
pub const COOPERATIVE_BACKEND: &str = "vllm_cuda_cooperative";
pub const COOPERATIVE_MAX_ROWS: u32 = 64;

const ROWS_AXIS: [u32; 31] = [
    1, 2, 4, 8, 12, 16, 17, 24, 32, 48, 64, 65, 96, 128, 132, 133, 148, 149, 192, 256, 384, 512,
    768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 16384,
];
const CONTEXT_AXIS: [u32; 14] = [
    0, 1, 128, 511, 512, 513, 600, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
];
const MAX_ROWS: u32 = 16384;

/// Requests whose rows land on each side of every rows-axis point.
fn batch_axis(next_n: u32) -> Vec<u32> {
    let next_n = next_n.max(1);
    let mut batches: Vec<u32> = ROWS_AXIS
        .iter()
        .flat_map(|&rows| [(rows / next_n).max(1), rows.div_ceil(next_n)])
        .filter(|&batch| batch * next_n <= MAX_ROWS)
        .collect();
    batches.sort_unstable();
    batches.dedup();
    batches
}

pub struct SparseIndexerTopkSpec;

impl KernelSpec for SparseIndexerTopkSpec {
    type Config = DsaPersistentTopkDecodeKernelConfig;
    type Input = DsaPersistentTopkDecodeKernelInput;

    const KIND: KernelKind = "sparse_indexer_topk";

    fn profile_kind() -> KernelKind {
        DsaPersistentTopkDecodeSpec::KIND
    }

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![
            Axis::values(batch_axis(config.next_n)),
            Axis::values(CONTEXT_AXIS),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        let next_n = f64::from(config.next_n);
        let min_context = f64::from(config.next_n.saturating_sub(1));
        let max_context = f64::from(config.max_model_len.get());
        let max_rows = f64::from(if config.backends.contains(&COOPERATIVE_BACKEND) {
            COOPERATIVE_MAX_ROWS
        } else {
            MAX_ROWS
        });
        grid.expand_2d(|batch, context| {
            context < min_context || context > max_context || batch * next_n > max_rows
        })
    }

    /// Rows are independent: a call past the grid runs as full-edge pieces.
    fn off_grid(
        config: &Self::Config,
        input: &Self::Input,
        _backend: &'static str,
    ) -> OffGrid<Self::Input> {
        let max_batch = *batch_axis(config.next_n).last().expect("non-empty axis");
        if input.batch_size <= max_batch {
            return OffGrid::Cache;
        }
        let mut pieces = Vec::new();
        let mut batch = input.batch_size;
        while batch > 0 {
            let piece = batch.min(max_batch);
            pieces.push(DsaPersistentTopkDecodeKernelInput {
                batch_size: piece,
                context_len: input.context_len,
            });
            batch -= piece;
        }
        OffGrid::Launches(pieces)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        DsaPersistentTopkDecodeSpec::enumerate(config, grid, backend)
    }
}

register_kernel!(SparseIndexerTopkKernel, SparseIndexerTopkSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::bridge::DType;
    use crate::timing::Dim;

    fn config(backend: &'static str, next_n: u32) -> DsaPersistentTopkDecodeKernelConfig {
        DsaPersistentTopkDecodeKernelConfig {
            backends: vec![backend],
            gpu_name: "NVIDIA B200".into(),
            next_n,
            max_model_len: Dim::param("max_model_len", 65536),
            top_k: 512,
            logits_row_stride: Dim::param("logits_row_stride", 65536),
            logits_dtype: DType::Fp32,
            index_dtype: "int32".into(),
            context_mode: "uniform_random".into(),
        }
    }

    fn feasible(cfg: &DsaPersistentTopkDecodeKernelConfig) -> Vec<(f64, f64)> {
        let grid = SparseIndexerTopkSpec::sweep_grid(cfg);
        let mask = SparseIndexerTopkSpec::infeasible_mask(cfg, &grid);
        grid.expand_2d(|b, c| (b, c))
            .into_iter()
            .zip(mask)
            .filter(|(_, m)| !m)
            .map(|(p, _)| p)
            .collect()
    }

    #[test]
    fn grid_stays_under_the_point_ceiling_and_reads_the_decode_rows() {
        assert_eq!(
            SparseIndexerTopkSpec::profile_kind(),
            "dsa_persistent_topk_decode"
        );
        let cfg = config("vllm_cuda", 1);
        assert!(feasible(&cfg).len() <= 500);
        let grid = SparseIndexerTopkSpec::sweep_grid(&cfg);
        assert_eq!(
            SparseIndexerTopkSpec::enumerate(&cfg, &grid, "vllm_cuda")[0]
                .fields()
                .len(),
            10
        );
    }

    #[test]
    fn batch_axis_brackets_each_rows_point_for_any_verify_width() {
        let four = batch_axis(4);
        for batch in [37, 38, 33, 34, 4096] {
            assert!(four.contains(&batch), "next_n 4 misses batch {batch}");
        }
        assert_eq!(batch_axis(1), ROWS_AXIS.to_vec());
        assert!(SparseIndexerTopkSpec::sweep_grid(&config("vllm_cuda", 4)).axes()[0].len() <= 35);
    }

    #[test]
    fn cooperative_masks_past_its_row_ceiling() {
        let rows = |cfg: &DsaPersistentTopkDecodeKernelConfig| {
            feasible(cfg)
                .iter()
                .map(|&(b, _)| b * f64::from(cfg.next_n))
                .fold(0.0, f64::max)
        };
        assert_eq!(rows(&config(COOPERATIVE_BACKEND, 4)), 64.0);
        assert_eq!(rows(&config(COOPERATIVE_BACKEND, 1)), 64.0);
        assert_eq!(rows(&config("vllm_cuda", 4)), 16384.0);
        assert!(feasible(&config("vllm_cuda", 4))
            .iter()
            .all(|&(_, c)| c >= 3.0));
    }
}
