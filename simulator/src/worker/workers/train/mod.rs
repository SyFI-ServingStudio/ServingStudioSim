//! Training-cadence workers: a block held for one chunk at a time.
//!
//! Its own family because the cadence is neither iter-wise nor layer-wise — a
//! training block runs one fused forward+backward over a batch of finished
//! prompt groups and is free again when it lands.

pub(crate) mod chunk_worker;
