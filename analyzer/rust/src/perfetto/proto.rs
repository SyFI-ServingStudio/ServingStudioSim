//! The prost-generated Perfetto proto types (`build.rs` → `$OUT_DIR`). Kept in
//! its own module so the generated code is the whole file — `mod.rs` builds the
//! ergonomic writer on top of these raw message structs.

include!(concat!(env!("OUT_DIR"), "/perfetto.protos.rs"));
