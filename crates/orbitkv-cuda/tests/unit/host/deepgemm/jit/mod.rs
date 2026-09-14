use super::super::tiling::candidates;
use super::*;

#[test]
fn upstream_heuristic_is_first_but_not_the_only_candidate() {
    let choices = candidates(1, 5120, 5120, 78);
    assert!(choices.len() >= SEARCH_VARIANTS);
    assert_eq!((choices[0].block_m, choices[0].block_n), (16, 80));
    assert!(
        choices[..SEARCH_VARIANTS]
            .windows(2)
            .all(|pair| pair[0] != pair[1])
    );
}

#[test]
fn source_records_exact_provider_identity_and_tile() {
    let config = candidates(128, 5120, 5120, 78)[0];
    let source = config.source();
    assert!(source.contains(DEEPGEMM_REVISION));
    assert!(source.contains("sm90_fp8_gemm_1d2d_impl"));
    assert!(!source.contains('@'));
}
