//! One item in an omni model's heterogeneous input sequence.
//!
//! Defined in the shared trace crate: it is the JSON schema of the
//! `input_segments` column, so the program that writes that column and the
//! program that reads it cannot be allowed to disagree about it.

pub use req_frontend::schema::OmniInputSegment;
