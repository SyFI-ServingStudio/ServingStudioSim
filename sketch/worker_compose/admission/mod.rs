//! Admission policy implementations.
//!
//! A concrete Worker selects one policy from this family. New batching or
//! admission behavior belongs in a sibling module instead of widening
//! `PrefillDecode` with mode branches.

mod prefill_decode;

pub(super) use prefill_decode::PrefillDecode;
