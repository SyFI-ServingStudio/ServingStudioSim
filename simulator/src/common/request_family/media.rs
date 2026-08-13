//! Physical media shapes.
//!
//! Defined in the shared trace crate, because a generator writing these columns
//! and this simulator reading them must mean the same thing by them. Re-exported
//! here so a family consumer still finds them beside the families that use them.

pub use req_frontend::schema::media::{AudioExtent, ImageExtent, VideoExtent};
