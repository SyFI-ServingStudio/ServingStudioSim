//! Bake an rpath to the embedded interpreter's libpython into the binary so the
//! PyO3 `auto-initialize` build can `dlopen` libpython3.x.so without the caller
//! exporting LD_LIBRARY_PATH. Without this, running `./target/<p>/simulator`
//! directly fails with "libpython3.12.so.1.0: cannot open shared object file".
//!
//! We don't hardcode the path (unlike ref/moesim-rs's .cargo/config.toml): we
//! ask the *same* interpreter PyO3 will link against for its LIBDIR, so the
//! rpath follows whichever Python built the binary and survives version bumps.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");

    // Prefer the interpreter PyO3 itself uses (PYO3_PYTHON), else `python3` on
    // PATH — under `uv run` that resolves to the project venv.
    let python = std::env::var("PYO3_PYTHON").unwrap_or_else(|_| "python3".to_string());

    let output = Command::new(&python)
        .args(["-c", "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')"])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let libdir = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !libdir.is_empty() {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
            } else {
                println!("cargo:warning=could not resolve Python LIBDIR; libpython rpath not baked in (set LD_LIBRARY_PATH to run the binary directly)");
            }
        }
        _ => {
            println!("cargo:warning=could not run `{python}` to resolve libpython rpath; set LD_LIBRARY_PATH to run the binary directly");
        }
    }
}
