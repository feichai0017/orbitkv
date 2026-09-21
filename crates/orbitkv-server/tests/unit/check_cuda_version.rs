use super::*;

#[test]
fn compatible_runtime_must_match_major_and_not_be_older() {
    assert!(is_compatible_cuda_runtime(12080, 12080));
    assert!(is_compatible_cuda_runtime(12080, 12090));
    assert!(is_compatible_cuda_runtime(13000, 13010));

    assert!(!is_compatible_cuda_runtime(12080, 12070));
    assert!(!is_compatible_cuda_runtime(12080, 13000));
    assert!(!is_compatible_cuda_runtime(13000, 12080));
}

#[test]
fn cuda_versions_are_formatted_for_logs_and_errors() {
    assert_eq!(format_cuda_version(12080), "12.8 (12080)");
    assert_eq!(format_cuda_version(13000), "13.0 (13000)");
    assert_eq!(format_cuda_version(13010), "13.1 (13010)");
}
