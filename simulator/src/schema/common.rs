//! Cross-deployment shared flags.
//!
//! Per L7 design.md §1.8 the actual flat-CLI sharing happens via *pool
//! fragments* in `schema/common_pool.rs` (ModelCommon / ParallelismCommon /
//! WorkloadCommon / IoCommon) flattened into each deployment's `Args`. Pool
//! fragments themselves land in Phase 3 alongside the first concrete
//! deployment file; this Phase-0 module only reserves the location and an
//! empty `COMMON_PARAMS` so downstream code can already import `crate::schema`
//! without a stub error.

use super::param_def::ParamDef;

pub const COMMON_PARAMS: &[ParamDef] = &[];
