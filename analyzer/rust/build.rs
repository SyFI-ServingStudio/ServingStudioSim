//! Codegen the trimmed Perfetto proto types from `proto/perfetto_trace.proto`
//! into `$OUT_DIR/perfetto.protos.rs`, which `src/perfetto/proto.rs` `include!`s.
//! Needs `protoc` on PATH (prost-build shells out to it).

use std::io::Result;

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=proto/perfetto_trace.proto");
    prost_build::compile_protos(&["proto/perfetto_trace.proto"], &["proto"])
}
