//! Launch the independent HTTP server using a resolved kern.toml target.
//! No serving implementation or model knowledge enters the runtime workspace.

use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::process::Command;

use anyhow::{ensure, Context, Result};

use crate::config::Target;

/// Replace this process with kern-serve on Unix, preserving signals and status.
/// Explicit artifact flags override the target's corresponding defaults.
pub fn run(target: &Target, args: &[OsString]) -> Result<()> {
    let executable = match std::env::var_os("KERN_SERVE_BIN") {
        Some(path) => {
            ensure!(!path.is_empty(), "KERN_SERVE_BIN must name a server executable");
            PathBuf::from(path)
        }
        None => {
            let sibling =
                std::env::current_exe()?.with_file_name(format!("kern-serve{}", std::env::consts::EXE_SUFFIX));
            if sibling.is_file() {
                sibling
            } else {
                PathBuf::from("kern-serve")
            }
        }
    };
    let mut command = Command::new(&executable);
    command.args(arguments(target, args));
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        Err(command.exec()).with_context(|| {
            format!(
                "starting {}: install kern-serve beside kern or on PATH, or set KERN_SERVE_BIN",
                executable.display()
            )
        })
    }
    #[cfg(not(unix))]
    {
        let status = command.status().with_context(|| format!("starting {}", executable.display()))?;
        std::process::exit(status.code().unwrap_or(1));
    }
}

fn arguments(target: &Target, args: &[OsString]) -> Vec<OsString> {
    let supplied = |flag: &str| {
        args.iter()
            .take_while(|a| a.as_os_str() != OsStr::new("--"))
            .any(|arg| arg == flag || arg.to_str().is_some_and(|a| a.starts_with(&format!("{flag}="))))
    };
    let mut out = Vec::new();
    for (flag, path) in [("--manifest", &target.manifest), ("--kernels", &target.kernels)] {
        if !supplied(flag) {
            out.push(flag.into());
            out.push(path.as_os_str().into());
        }
    }
    if !supplied("--weights") {
        for w in &target.weights {
            out.push("--weights".into());
            out.push(w.into());
        }
    }
    out.extend_from_slice(args);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overrides_keep_argument_boundaries_and_replace_all_weights() {
        let target: Target = toml::from_str(
            r#"manifest = "a manifest.json"
               kernels = "kernels"
               weights = ["base", "draft"]"#,
        )
        .unwrap();
        let args = ["--weights=alternate weights", "--port", "8123"].map(OsString::from);
        let actual = arguments(&target, &args);
        let expected =
            ["--manifest", "a manifest.json", "--kernels", "kernels", "--weights=alternate weights", "--port", "8123"]
                .map(OsString::from);
        assert_eq!(actual, expected);
    }
}
