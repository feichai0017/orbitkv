use super::*;

#[test]
fn cache_policy_controls_validate_protection_and_admission() {
    assert!(
        Cli::try_parse_from(["orbitkv-cache-manager", "--cache-protected-percent", "101"]).is_err()
    );
    assert!(
        Cli::try_parse_from(["orbitkv-cache-manager", "--ssd-write-policy", "unknown"]).is_err()
    );
    let cli = Cli::try_parse_from([
        "orbitkv-cache-manager",
        "--cache-protected-percent",
        "80",
        "--ssd-write-policy",
        "reuse",
        "--ssd-cache-path",
        "/tmp/policy-test",
    ])
    .unwrap();
    assert_eq!(cli.cache_protected_percent, 80);
    assert_eq!(cli.ssd_write_policy, orbitkv_core::SsdWritePolicy::Reuse);
}

#[test]
fn cli_membership_requires_stable_node_identity_and_catalog_placement() {
    let flags = [
        "orbitkv-cache-manager",
        "--etcd-endpoints",
        "http://127.0.0.1:2379",
    ];
    assert!(Cli::try_parse_from(flags).is_err());
    assert!(Cli::try_parse_from(flags.into_iter().chain(["--node-id", "node-a"])).is_err());
    let cli = Cli::try_parse_from(flags.into_iter().chain([
        "--node-id",
        "node-a",
        "--catalog-nodes",
        "node-a,node-b",
    ]))
    .unwrap();
    assert_eq!(cli.node_id.as_deref(), Some("node-a"));
    assert_eq!(cli.membership_ttl_secs, 30);
    assert!(Cli::try_parse_from(["orbitkv-cache-manager", "--membership-ttl-secs", "0"]).is_err());
}

#[test]
fn parse_hll_windows_canonicalizes_labels() {
    let windows = parse_hll_windows("15m,60m,24h").unwrap();

    assert_eq!(windows, expected_hll_windows());
}

#[test]
fn cli_default_metric_hll_windows_parses_without_panic() {
    let cli = Cli::try_parse_from(["orbitkv-cache-manager"]).unwrap();

    assert_eq!(
        parse_hll_windows(&cli.metric_hll_windows).unwrap(),
        expected_hll_windows()
    );
    assert_eq!(cli.metric_hll_bucket_bits, 16);
}

#[test]
fn cli_accepts_channel_options() {
    let cli = Cli::try_parse_from([
        "orbitkv-cache-manager",
        "--channel-service",
        "orbitkv/test/server",
        "--channel-session-epoch",
        "42",
        "--bootstrap-socket",
        "/tmp/orbitkv-test.sock",
        "--descriptor-arena-size",
        "4mb",
        "--descriptor-slot-size",
        "32kb",
    ])
    .unwrap();
    assert_eq!(cli.channel_service.as_deref(), Some("orbitkv/test/server"));
    assert_eq!(cli.channel_session_epoch, Some(42));
    assert_eq!(
        cli.bootstrap_socket.as_deref(),
        Some(std::path::Path::new("/tmp/orbitkv-test.sock"))
    );
    assert_eq!(cli.descriptor_arena_size, 4 * 1024 * 1024);
    assert_eq!(cli.descriptor_slot_size, 32 * 1024);
    assert!(Cli::try_parse_from(["orbitkv-cache-manager", "--enable-grpc"]).is_err());
    assert!(Cli::try_parse_from(["orbitkv-cache-manager", "--disable-channel"]).is_err());
}

#[test]
fn cli_nics_accepts_comma_separated_values() {
    let cli =
        Cli::try_parse_from(["orbitkv-cache-manager", "--nics", "mlx5_0,mlx5_1,mlx5_2"]).unwrap();

    assert_eq!(
        cli.nics.unwrap(),
        [
            "mlx5_0".to_string(),
            "mlx5_1".to_string(),
            "mlx5_2".to_string()
        ]
    );
}

#[test]
fn cli_nics_trims_comma_separated_values() {
    let cli =
        Cli::try_parse_from(["orbitkv-cache-manager", "--nics", "mlx5_0, mlx5_1, mlx5_2"]).unwrap();

    assert_eq!(
        cli.nics.unwrap(),
        [
            "mlx5_0".to_string(),
            "mlx5_1".to_string(),
            "mlx5_2".to_string()
        ]
    );
}

#[test]
fn cli_nics_rejects_empty_comma_separated_values() {
    let err =
        Cli::try_parse_from(["orbitkv-cache-manager", "--nics", "mlx5_0,,mlx5_1"]).unwrap_err();

    assert!(err.to_string().contains("empty NIC name"), "{err}");
}

#[test]
fn cli_nics_accepts_repeated_values_after_flag() {
    let cli = Cli::try_parse_from([
        "orbitkv-cache-manager",
        "--nics",
        "mlx5_0",
        "mlx5_1",
        "mlx5_2",
    ])
    .unwrap();

    assert_eq!(
        cli.nics.unwrap(),
        [
            "mlx5_0".to_string(),
            "mlx5_1".to_string(),
            "mlx5_2".to_string()
        ]
    );
}

#[test]
fn cli_explicit_metric_hll_windows_parses_without_panic() {
    let cli = Cli::try_parse_from([
        "orbitkv-cache-manager",
        "--metric-hll-windows",
        "15m,1h,24h",
    ])
    .unwrap();

    assert_eq!(
        parse_hll_windows(&cli.metric_hll_windows).unwrap(),
        expected_hll_windows()
    );
}

#[test]
fn cli_rejects_invalid_metric_hll_windows() {
    let err = Cli::try_parse_from(["orbitkv-cache-manager", "--metric-hll-windows", "15m,,1h"])
        .unwrap_err();

    assert!(err.to_string().contains("empty window"), "{err}");
}

fn expected_hll_windows() -> Vec<(String, Duration)> {
    vec![
        ("15m".to_string(), Duration::from_secs(15 * 60)),
        ("1h".to_string(), Duration::from_secs(60 * 60)),
        ("1d".to_string(), Duration::from_secs(24 * 60 * 60)),
    ]
}

#[test]
fn parse_hll_windows_rejects_empty_tokens() {
    let err = parse_hll_windows("15m,,1h").unwrap_err();

    assert!(err.contains("empty window"), "{err}");
}

#[test]
fn parse_hll_windows_rejects_duplicate_durations() {
    let err = parse_hll_windows("1h,60m").unwrap_err();

    assert!(err.contains("duplicate HLL window duration"), "{err}");
}
