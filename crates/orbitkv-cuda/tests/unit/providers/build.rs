use super::*;

#[test]
fn compiler_diagnostics_retain_a_bounded_tail() {
    let mut input = vec![b'a'; MAX_DIAGNOSTIC_BYTES * 2];
    input.extend_from_slice(b"final diagnostic");
    let output = diagnostic_tail(&input[..]).unwrap();
    assert_eq!(output.len(), MAX_DIAGNOSTIC_BYTES);
    assert!(output.ends_with(b"final diagnostic"));
}

#[test]
#[cfg(unix)]
fn compiler_failure_preserves_exit_and_diagnostics() {
    let error = run_compiler(
        Command::new("sh").args(["-c", "printf 'bad compilation' >&2; exit 7"]),
        Duration::from_secs(2),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("7"));
    assert!(error.contains("bad compilation"));
}

#[test]
#[cfg(unix)]
fn compiler_timeout_reaps_the_owned_process_group() {
    let started = Instant::now();
    let error = run_compiler(
        Command::new("sh").args(["-c", "sleep 30 & wait"]),
        Duration::from_millis(100),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("timed out"));
    assert!(started.elapsed() < Duration::from_secs(5));
}

#[test]
#[cfg(unix)]
fn compiler_deadline_includes_inherited_diagnostic_pipes() {
    let started = Instant::now();
    let error = run_compiler(
        Command::new("sh").args(["-c", "sleep 30 & exit 0"]),
        Duration::from_millis(100),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("timed out"));
    assert!(started.elapsed() < Duration::from_secs(5));
}
