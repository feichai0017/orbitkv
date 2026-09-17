#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::ExitStatusExt;
use std::path::PathBuf;
use std::process::Command;

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!("kern-server-cli-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("child")).unwrap();
        std::fs::write(
            dir.join("kern.toml"),
            "[targets.demo]\nmanifest='a manifest.json'\nkernels='cubins'\nweights=['base weights','draft']\n",
        )
        .unwrap();
        let server = dir.join("mock-server");
        std::fs::write(&server, "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$KERN_TEST_ARGV\"\nif [ \"$KERN_TEST_SIGNAL\" = yes ]; then kill -TERM $$; fi\nexit 37\n").unwrap();
        std::fs::set_permissions(&server, std::fs::Permissions::from_mode(0o755)).unwrap();
        Self(dir)
    }

    fn command(&self) -> Command {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_kern"));
        cmd.current_dir(self.0.join("child"))
            .env("KERN_SERVE_BIN", self.0.join("mock-server"))
            .env("KERN_TEST_ARGV", self.0.join("args"));
        cmd
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn server_forwards_target_args_status_and_signal() {
    let f = Fixture::new();
    let output =
        f.command().args(["server", "demo", "--port", "8123", "--model-name", "model with spaces"]).output().unwrap();
    assert_eq!(output.status.code(), Some(37), "{}", String::from_utf8_lossy(&output.stderr));
    let actual = std::fs::read_to_string(f.0.join("args")).unwrap();
    let expected =
        format!(
        "--manifest\n{}\n--kernels\n{}\n--weights\n{}\n--weights\n{}\n--port\n8123\n--model-name\nmodel with spaces\n",
        f.0.join("a manifest.json").display(), f.0.join("cubins").display(),
        f.0.join("base weights").display(), f.0.join("draft").display()
    );
    assert_eq!(actual, expected);
    let status =
        f.command().env("KERN_TEST_SIGNAL", "yes").args(["server", "demo", "--", "--port=8124"]).status().unwrap();
    assert_eq!(status.signal(), Some(15));
    assert!(std::fs::read_to_string(f.0.join("args")).unwrap().ends_with("--port=8124\n"));
}
