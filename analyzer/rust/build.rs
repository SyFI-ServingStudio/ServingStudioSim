//! Build the analyzer's generated contracts.
//!
//! Besides the trimmed Perfetto proto, this embeds the source revision in the
//! executable.  The launcher must learn provenance from the executable it will
//! run; consulting the launcher's current checkout later can attribute an old
//! binary to an unrelated HEAD.

use std::env;
use std::io::{Error, ErrorKind, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

fn git_stdout(repo_root: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(args)
        .output()?;
    if !output.status.success() {
        return Err(Error::new(
            ErrorKind::Other,
            format!(
                "git {} failed while embedding analyzer provenance: {}",
                args.join(" "),
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        ));
    }
    let value = String::from_utf8(output.stdout)
        .map_err(|error| Error::new(ErrorKind::InvalidData, error))?
        .trim()
        .to_string();
    if value.is_empty() {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("git {} returned an empty value", args.join(" ")),
        ));
    }
    Ok(value)
}

fn tracked_git_path(repo_root: &Path, git_path: &str) -> PathBuf {
    let path = PathBuf::from(git_path);
    if path.is_absolute() {
        path
    } else {
        repo_root.join(path)
    }
}

fn embed_source_revision(repo_root: &Path) -> Result<()> {
    let revision = git_stdout(repo_root, &["rev-parse", "--verify", "HEAD"])?;
    if revision.len() != 40
        || !revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("git returned a non-full analyzer revision {revision:?}"),
        ));
    }
    println!("cargo:rustc-env=VIBESIM_ANALYZER_REVISION={revision}");

    // A worktree's `.git` is a pointer file and its HEAD may be symbolic. Track
    // both the worktree HEAD and its branch ref so advancing the branch rebuilds
    // the identity even when no analyzer source timestamp otherwise changes.
    let head_path = git_stdout(repo_root, &["rev-parse", "--git-path", "HEAD"])?;
    println!(
        "cargo:rerun-if-changed={}",
        tracked_git_path(repo_root, &head_path).display()
    );
    if let Ok(symbolic_ref) = git_stdout(repo_root, &["symbolic-ref", "-q", "HEAD"]) {
        let ref_path = git_stdout(repo_root, &["rev-parse", "--git-path", &symbolic_ref])?;
        println!(
            "cargo:rerun-if-changed={}",
            tracked_git_path(repo_root, &ref_path).display()
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=proto/perfetto_trace.proto");
    let manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").map_err(|error| Error::new(ErrorKind::NotFound, error))?,
    );
    let repo_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| Error::new(ErrorKind::NotFound, "analyzer repo root is missing"))?;
    embed_source_revision(repo_root)?;
    prost_build::compile_protos(&["proto/perfetto_trace.proto"], &["proto"])
}
