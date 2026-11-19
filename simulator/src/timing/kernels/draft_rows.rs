//! Shared by the DFlash2 draft-chain kinds: the row axis they are measured on.

/// Row counts of one launch: a DP rank's draft block rows
/// (`requests * (1 + draft_tokens)`) and context rows, dense where the
/// per-rank batches sit.
pub(crate) const DRAFT_ROW_AXIS: [u32; 18] = [
    8, 16, 32, 64, 128, 192, 256, 384, 512, 640, 768, 1024, 1280, 1536, 2048, 3072, 4096, 8192,
];

pub(crate) fn draft_row_axis() -> Vec<f64> {
    DRAFT_ROW_AXIS.iter().map(|&x| f64::from(x)).collect()
}
