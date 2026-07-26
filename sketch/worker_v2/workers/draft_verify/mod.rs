//! S6 draft/verify cadence family.
//!
//! This family owns the tentative-proposal → target-verification → commit/discard
//! timeline. It is separate from the S0 whole-iteration cadence because the
//! execution result must survive across the simulated compute window.

mod build_speculative_decode_worker;
mod draft_verify_worker;

pub use build_speculative_decode_worker::build_speculative_decode_worker;
pub use draft_verify_worker::DraftVerifyWorker;
