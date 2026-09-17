//! Records the commit `kern` was built from, for `kern --version`.
//!
//! `git describe` on a checkout; `KERN_COMMIT` when set (a release build
//! passes the tagged commit); `unknown` when neither is available (a crate
//! built outside the repository).

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=KERN_COMMIT");
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/refs");
    let commit =
        std::env::var("KERN_COMMIT").ok().filter(|c| !c.is_empty()).or_else(describe).unwrap_or("unknown".into());
    println!("cargo:rustc-env=KERN_COMMIT={commit}");
}

/// The short commit id, `-dirty` when the tree has uncommitted changes.
fn describe() -> Option<String> {
    let out = Command::new("git").args(["describe", "--always", "--dirty", "--exclude", "*"]).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}
